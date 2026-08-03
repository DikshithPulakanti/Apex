# orchestrator.py
# APEX Full Pipeline — runs all 4 agents in sequence

import sys
import os
import time

from database.neo4j_client import Neo4jClient
from database.weaviate_client import WeaviateClient
from events.agent_events import (
    emit_agent_status,
    emit_papers_ingested,
    emit_hypothesis_created,
    emit_hypothesis_validated,
    emit_hypothesis_rejected,
    emit_research_plan_created,
)

from dotenv import load_dotenv
load_dotenv()


def run_pipeline(seed_concept: str = "large language models"):
    """
    Full APEX pipeline:
    1. Harvester — scrape papers (skip if already ingested)
    2. Reasoner — find gaps, generate up to 5 candidate hypotheses
    3. Skeptic — debate and validate each candidate
    4. Inventor — draft a research plan for each validated candidate
    """

    print("=" * 60)
    print("  APEX — Autonomous Research Pipeline")
    print("=" * 60)
    start = time.time()

    # ── Check infrastructure ─────────────────────────────────────
    print("\n[1/5] Checking infrastructure...")
    neo4j = Neo4jClient()
    emit_agent_status('orchestrator', 'pipeline_started', {'seed': seed_concept})

    with neo4j.driver.session() as session:
        result = session.run("MATCH (p:Paper) RETURN count(p) AS count")
        paper_count = result.single()['count']

    print(f"  Papers in graph: {paper_count}")

    if paper_count < 50:
        print("  ⚠️  Low paper count — run pipeline/ingest.py first")
        print("  Continuing with existing data...")

    neo4j.close()

    # ── Reasoner ─────────────────────────────────────────────────
    print(f"\n[2/5] Reasoner — generating candidate directions from '{seed_concept}'...")
    emit_agent_status('reasoner', 'starting', {'seed': seed_concept})

    from agents.reasoner import build_reasoner, get_resources as reasoner_resources

    r_resources = reasoner_resources()
    reasoner = build_reasoner(r_resources)

    r_state = reasoner.invoke({
        'seed_concept': seed_concept,
        'gaps_found':   [],
        'candidates':   [],
        'stored':       [],
        'status':       'starting',
        'error':        ''
    })

    stored = r_state.get('stored', [])
    r_resources['neo4j'].close()
    r_resources['weaviate'].close()

    if not stored:
        print("  ❌ Reasoner failed to generate any hypotheses")
        emit_agent_status('reasoner', 'failed')
        return

    print(f"  ✅ Generated {len(stored)} candidate(s)")
    for item in stored:
        emit_hypothesis_created(
            item['hypothesis_id'], item['hypothesis']['statement'],
            item['hypothesis']['testability_score'],
        )

    # ── Skeptic + Inventor, once per candidate ────────────────────
    # Each candidate is independent — a crash debating or drafting a plan
    # for one (e.g. a malformed JSON response from Claude) must not lose
    # the others, several of which may have already succeeded.
    results = []
    for item in stored:
        hypothesis_id = item['hypothesis_id']
        try:
            print(f"\n[3/5] Skeptic — debating {hypothesis_id}...")
            emit_agent_status('skeptic', 'starting', {'hypothesis_id': hypothesis_id})

            from agents.skeptic import build_skeptic, get_resources as skeptic_resources

            s_resources = skeptic_resources()
            try:
                skeptic = build_skeptic(s_resources)
                s_state = skeptic.invoke({
                    'hypothesis_id':    hypothesis_id,
                    'hypothesis':       {},
                    'counterarguments': [],
                    'rebuttal':         '',
                    'debate_score':     0.0,
                    'rounds_completed': 0,
                    'verdict':          '',
                    'status':           'starting',
                    'error':            '',
                    'bert_confidence':  0.0,
                    'bert_verdict':     '',
                    'claude_score':     0.0,
                    'claude_reasoning': '',
                    'scoring_method':   '',
                    'mlflow_run_id':    '',
                })
            finally:
                s_resources['neo4j'].close()

            verdict = s_state.get('verdict', 'rejected')
            debate_score = s_state.get('debate_score', 0.0)

            if verdict == 'approved':
                print(f"  ✅ Validated (score: {debate_score})")
                emit_hypothesis_validated(hypothesis_id, debate_score,
                    method=s_state.get('scoring_method', 'claude'))
                plan_id, novelty, title = run_inventor_phase(hypothesis_id)
                results.append({'hypothesis_id': hypothesis_id, 'verdict': verdict, 'plan_id': plan_id})
            elif verdict == 'pending_review':
                print(f"  ⏸️  Flagged for human review (BERT: {s_state.get('bert_confidence', 0.0):.2f}, "
                      f"Claude: {s_state.get('claude_score', 0.0):.2f})")
                emit_agent_status('orchestrator', 'pipeline_paused', {
                    'hypothesis_id': hypothesis_id,
                    'reason': 'pending_human_review',
                })
                results.append({'hypothesis_id': hypothesis_id, 'verdict': verdict, 'plan_id': ''})
            else:
                print(f"  ❌ Rejected (score: {debate_score})")
                emit_hypothesis_rejected(hypothesis_id, debate_score)
                results.append({'hypothesis_id': hypothesis_id, 'verdict': verdict, 'plan_id': ''})

        except Exception as e:
            print(f"  ❌ Error processing {hypothesis_id}: {e}")
            emit_agent_status('orchestrator', 'candidate_failed', {
                'hypothesis_id': hypothesis_id, 'error': str(e),
            })
            results.append({'hypothesis_id': hypothesis_id, 'verdict': 'error', 'plan_id': ''})

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - start
    approved = sum(1 for r in results if r['verdict'] == 'approved')
    pending = sum(1 for r in results if r['verdict'] == 'pending_review')
    rejected = sum(1 for r in results if r['verdict'] == 'rejected')
    failed = sum(1 for r in results if r['verdict'] == 'error')

    print("\n" + "=" * 60)
    print("  APEX Pipeline Complete")
    print("=" * 60)
    print(f"  Seed concept:  {seed_concept}")
    print(f"  Candidates:    {len(results)} ({approved} approved, {pending} pending review, "
          f"{rejected} rejected, {failed} failed)")
    print(f"  Time:          {elapsed:.1f}s")
    print("=" * 60)

    emit_agent_status('orchestrator', 'pipeline_complete', {
        'seed_concept': seed_concept,
        'results': results,
        'elapsed_seconds': round(elapsed, 1),
    })


def run_inventor_phase(hypothesis_id: str):
    """
    Runs the Inventor agent on a validated hypothesis and publishes the
    research_plan.created event. Factored out so both the main pipeline
    (after an auto-approval) and the human-review UI (after a manual
    approval) can trigger it the same way.

    Returns (plan_id, novelty_score, methodology_sketch) — plan_id is ''
    on failure or skip.
    """
    print(f"\n[4/5] Inventor — checking prior art and drafting a research plan for {hypothesis_id}...")
    emit_agent_status('inventor', 'starting', {'hypothesis_id': hypothesis_id})

    from agents.inventor import build_inventor, get_resources as inventor_resources

    i_resources = inventor_resources()
    inventor = build_inventor(i_resources)

    i_state = inventor.invoke({
        'hypothesis_id': hypothesis_id,
        'hypothesis':    {},
        'novelty_score': 0.0,
        'prior_art':     [],
        'sim_result':    {},
        'plan_draft':    {},
        'plan_id':       '',
        'status':        'starting',
        'error':         ''
    })

    i_resources['neo4j'].close()
    i_resources['weaviate'].close()

    plan_id = i_state.get('plan_id', '')
    novelty = i_state.get('novelty_score', 0.0)
    methodology_sketch = i_state.get('plan_draft', {}).get('methodology_sketch', '')

    if plan_id:
        print(f"  ✅ Research plan: {plan_id}")
        print(f"  Methodology: {methodology_sketch[:80]}")
        emit_research_plan_created(plan_id, hypothesis_id, methodology_sketch, novelty)
    else:
        print("  ❌ Research plan drafting failed or skipped")

    return plan_id, novelty, methodology_sketch


if __name__ == '__main__':
    seed = sys.argv[1] if len(sys.argv) > 1 else 'large language models'
    run_pipeline(seed_concept=seed)