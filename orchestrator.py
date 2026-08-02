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
    emit_patent_drafted,
)

from dotenv import load_dotenv
load_dotenv()


def run_pipeline(seed_concept: str = "large language models"):
    """
    Full APEX pipeline:
    1. Harvester — scrape papers (skip if already ingested)
    2. Reasoner — find gaps, generate hypothesis
    3. Skeptic — debate and validate
    4. Inventor — draft patent from validated hypothesis
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

    # ── Reasoner ─────────────────────────────────────────────────
    print(f"\n[2/5] Reasoner — generating hypothesis from '{seed_concept}'...")
    emit_agent_status('reasoner', 'starting', {'seed': seed_concept})

    from agents.reasoner import build_reasoner, get_resources as reasoner_resources

    r_resources = reasoner_resources()
    reasoner = build_reasoner(r_resources)

    r_state = reasoner.invoke({
        'seed_concept':    seed_concept,
        'research_gaps':   [],
        'context_papers':  [],
        'hypothesis':      {},
        'hypothesis_id':   '',
        'testability':     0.0,
        'status':          'starting',
        'error':           ''
    })

    hypothesis_id = r_state.get('hypothesis_id', '')
    hypothesis_text = r_state.get('hypothesis', {}).get('statement', '')
    testability = r_state.get('testability', 0.0)

    if not hypothesis_id:
        print("  ❌ Reasoner failed to generate hypothesis")
        emit_agent_status('reasoner', 'failed')
        return

    print(f"  ✅ Hypothesis: {hypothesis_id}")
    emit_hypothesis_created(hypothesis_id, hypothesis_text, testability)
    r_resources['neo4j'].close()

    # ── Skeptic ──────────────────────────────────────────────────
    print(f"\n[3/5] Skeptic — debating {hypothesis_id}...")
    emit_agent_status('skeptic', 'starting', {'hypothesis_id': hypothesis_id})

    from agents.skeptic import build_skeptic, get_resources as skeptic_resources

    s_resources = skeptic_resources()
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

    verdict = s_state.get('verdict', 'rejected')
    debate_score = s_state.get('debate_score', 0.0)
    s_resources['neo4j'].close()

    if verdict == 'approved':
        print(f"  ✅ Validated (score: {debate_score})")
        emit_hypothesis_validated(hypothesis_id, debate_score,
            method=s_state.get('scoring_method', 'claude'))
    elif verdict == 'pending_review':
        print(f"  ⏸️  Flagged for human review (BERT: {s_state.get('bert_confidence', 0.0):.2f}, "
              f"Claude: {s_state.get('claude_score', 0.0):.2f})")
        emit_agent_status('orchestrator', 'pipeline_paused', {
            'hypothesis_id': hypothesis_id,
            'reason': 'pending_human_review',
        })
        return
    else:
        print(f"  ❌ Rejected (score: {debate_score})")
        emit_hypothesis_rejected(hypothesis_id, debate_score)
        emit_agent_status('orchestrator', 'pipeline_complete', {'result': 'rejected'})
        return

    # ── Inventor ─────────────────────────────────────────────────
    patent_id, novelty, title = run_inventor_phase(hypothesis_id)

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("  APEX Pipeline Complete")
    print("=" * 60)
    print(f"  Seed concept:  {seed_concept}")
    print(f"  Hypothesis:    {hypothesis_id}")
    print(f"  Verdict:       {verdict} ({debate_score})")
    print(f"  Patent:        {patent_id}")
    print(f"  Time:          {elapsed:.1f}s")
    print("=" * 60)

    emit_agent_status('orchestrator', 'pipeline_complete', {
        'hypothesis_id': hypothesis_id,
        'patent_id': patent_id,
        'verdict': verdict,
        'elapsed_seconds': round(elapsed, 1),
    })


def run_inventor_phase(hypothesis_id: str):
    """
    Runs the Inventor agent on a validated hypothesis and publishes the
    patent.drafted event. Factored out so both the main pipeline (after an
    auto-approval) and the human-review UI (after a manual approval) can
    trigger patent drafting the same way.

    Returns (patent_id, novelty_score, title) — patent_id is '' on failure
    or skip.
    """
    print(f"\n[4/5] Inventor — drafting patent for {hypothesis_id}...")
    emit_agent_status('inventor', 'starting', {'hypothesis_id': hypothesis_id})

    from agents.inventor import build_inventor, get_resources as inventor_resources

    i_resources = inventor_resources()
    inventor = build_inventor(i_resources)

    i_state = inventor.invoke({
        'hypothesis_id': hypothesis_id,
        'hypothesis':    {},
        'novelty_score': 0.0,
        'sim_result':    {},
        'patent_draft':  {},
        'patent_id':     '',
        'status':        'starting',
        'error':         ''
    })

    i_resources['neo4j'].close()

    patent_id = i_state.get('patent_id', '')
    novelty = i_state.get('novelty_score', 0.0)
    title = i_state.get('patent_draft', {}).get('title', '')

    if patent_id:
        print(f"  ✅ Patent: {patent_id}")
        print(f"  Title: {title[:80]}")
        emit_patent_drafted(patent_id, hypothesis_id, title, novelty)
    else:
        print("  ❌ Patent drafting failed")

    return patent_id, novelty, title


if __name__ == '__main__':
    seed = sys.argv[1] if len(sys.argv) > 1 else 'large language models'
    run_pipeline(seed_concept=seed)