# agents/inventor.py
# APEX Inventor Agent — checks prior art and drafts a next-steps research
# plan for validated hypotheses

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uuid
import anthropic
from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from database.neo4j_client import Neo4jClient
from database.weaviate_client import WeaviateClient
from database.embedder import Embedder
from events.node_tracing import node_tracer

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


# ── State ─────────────────────────────────────────────────────────────────

class InventorState(TypedDict):
    hypothesis_id:   str
    hypothesis:      dict
    novelty_score:   float
    prior_art:       list
    sim_result:      dict
    plan_draft:      dict
    plan_id:         str
    status:          str
    error:           str


# ── Resources ─────────────────────────────────────────────────────────────

def get_resources():
    return {
        'neo4j':    Neo4jClient(),
        'weaviate': WeaviateClient(),
        'embedder': Embedder(),
        'claude':   anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')),
    }


# ── Node 1: Load and Check Novelty ────────────────────────────────────────

def check_novelty(state: InventorState, resources: dict) -> dict:
    """
    Loads the hypothesis and checks it against the real paper corpus via
    semantic search — "is this already been done?" rather than the old
    self-referential check of whether APEX itself had generated something
    similar before. Runs (and persists) regardless of whether the hypothesis
    ends up passing the patent/plan gate, since the closest existing papers
    are exactly what a student needs to see when a direction gets filtered out.
    """
    print(f'\n[Inventor:check_novelty] Loading: {state["hypothesis_id"]}')

    neo4j = resources['neo4j']
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (h:Hypothesis {id: $id})
            RETURN h.statement AS statement,
                   h.rationale AS rationale,
                   h.testability_score AS testability_score,
                   h.predicted_impact AS predicted_impact,
                   h.debate_score AS debate_score,
                   h.status AS verdict,
                   h.rebuttal AS rebuttal,
                   h.counterarguments AS counterarguments
        """, id=state['hypothesis_id'])
        record = result.single()

    if not record:
        return {
            'hypothesis': {},
            'error':      'Hypothesis not found',
            'status':     'failed'
        }

    hypothesis = dict(record)

    # Get supporting concepts
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (h:Hypothesis {id: $id})-[:DERIVED_FROM]->(c:Concept)
            RETURN c.name AS name
        """, id=state['hypothesis_id'])
        concepts = [r['name'] for r in result]

    hypothesis['supporting_concepts'] = concepts

    # Real prior-art check: semantic search the actual corpus for papers
    # closest to this hypothesis's statement.
    embedder = resources['embedder']
    weaviate = resources['weaviate']
    query_vec = embedder.embed_text(hypothesis.get('statement', ''))
    matches = weaviate.vector_search(query_vec, limit=5)

    prior_art = [
        {
            'paper_id':   m.get('paper_id', ''),
            'title':      m.get('title', ''),
            'year':       m.get('year', 0),
            'similarity': round(max(0.0, 1.0 - m.get('distance', 1.0)), 4),
        }
        for m in matches
    ]
    top_similarity = prior_art[0]['similarity'] if prior_art else 0.0
    novelty_score = round(max(0.0, 1.0 - top_similarity), 4) if prior_art else 1.0

    with neo4j.driver.session() as session:
        session.run(
            'MATCH (h:Hypothesis {id: $id}) SET h.novelty_score = $novelty',
            id=state['hypothesis_id'], novelty=novelty_score,
        )
        for pa in prior_art:
            if not pa['paper_id']:
                continue
            session.run("""
                MATCH (h:Hypothesis {id: $hyp_id})
                MATCH (p:Paper {id: $paper_id})
                MERGE (h)-[r:SIMILAR_TO]->(p)
                SET r.similarity = $similarity
            """, hyp_id=state['hypothesis_id'], paper_id=pa['paper_id'], similarity=pa['similarity'])

    print(f'[Inventor:check_novelty] Novelty: {novelty_score} | Closest match: '
          f'{prior_art[0]["title"][:60] if prior_art else "none found"}')

    return {
        'hypothesis':    hypothesis,
        'novelty_score': novelty_score,
        'prior_art':     prior_art,
        'status':        f'novelty checked: {novelty_score}'
    }


# ── Node 2: Run Simulation ────────────────────────────────────────────────

def run_simulation(state: InventorState, resources: dict) -> dict:
    """Runs Monte Carlo simulation to estimate hypothesis validity."""
    print(f'\n[Inventor:run_simulation] Running simulation...')

    import math
    import random

    h           = state['hypothesis']
    testability = h.get('testability_score', 0.5) or 0.5

    random.seed(42)
    n_simulations = 1000
    successes     = sum(
        1 for _ in range(n_simulations)
        if random.random() < (testability * 0.8 + 0.1)
    )

    probability = successes / n_simulations
    std_error   = math.sqrt(probability * (1 - probability) / n_simulations)

    sim_result = {
        'success_rate':  round(probability, 4),
        'ci_lower':      round(max(0, probability - 1.96 * std_error), 4),
        'ci_upper':      round(min(1, probability + 1.96 * std_error), 4),
        'recommendation': 'proceed' if probability > 0.6 else 'needs more evidence'
    }

    print(f'[Inventor:run_simulation] Success rate: {sim_result["success_rate"]}')
    print(f'[Inventor:run_simulation] Recommendation: {sim_result["recommendation"]}')

    return {'sim_result': sim_result, 'status': 'simulation complete'}


# ── Routing: Should We Draft a Plan? ──────────────────────────────────────

def should_draft_plan(state: InventorState) -> str:
    """Only proceed to research-plan drafting if novelty and simulation pass."""
    novelty  = state.get('novelty_score', 0.0)
    sim_rate = state.get('sim_result', {}).get('success_rate', 0.0)

    if novelty >= 0.5 and sim_rate >= 0.6:
        print(f'  [router] Novelty {novelty:.2f} + Sim {sim_rate:.2f} → drafting research plan')
        return 'draft'
    else:
        print(f'  [router] Novelty {novelty:.2f} or Sim {sim_rate:.2f} too low → skip')
        return 'skip'


def skip_plan(state: InventorState, resources: dict) -> dict:
    return {'status': 'research plan skipped — insufficient novelty or simulation score'}


# ── Node 3: Draft Research Plan ───────────────────────────────────────────

def draft_research_plan(state: InventorState, resources: dict) -> dict:
    """
    Uses Claude to synthesize a practical next-steps research plan for a
    student — not a patent claim. Grounded in the hypothesis, the adversarial
    debate that already happened, and the real prior-art search from
    check_novelty, so the advice is specific rather than generic encouragement.
    """
    print(f'\n[Inventor:draft_research_plan] Drafting research plan...')

    h      = state['hypothesis']
    claude = resources['claude']

    counterarguments = h.get('counterarguments') or []
    counterargs_text = '; '.join(counterarguments) if counterarguments else 'none recorded'

    prior_art_text = '\n'.join(
        f'- "{pa["title"]}" ({pa["year"]}) — similarity {pa["similarity"]}'
        for pa in state.get('prior_art', [])
    ) or 'No close matches found in the corpus.'

    prompt = f"""A student is considering this research direction:

Hypothesis: {h.get('statement', '')}
Rationale: {h.get('rationale', '')}
Predicted impact if validated: {h.get('predicted_impact', '')}
Testability score (0-1, from initial screening): {h.get('testability_score', 0.0)}

Adversarial review already completed:
- Counterarguments raised: {counterargs_text}
- Rebuttal: {h.get('rebuttal', '')}
- Debate verdict: {h.get('verdict', '')} (score {h.get('debate_score', 0.0)})

Closest existing papers found via semantic search of the corpus (NOT an exhaustive
literature review — treat as a starting point):
{prior_art_text}

Given all of this, write a practical next-steps research plan for the student. Be
concrete and specific to this hypothesis, not generic advice.

Return ONLY a JSON object with these exact fields:
{{
    "methodology_sketch": "2-4 sentences: the general experimental or analytical approach that would test this hypothesis",
    "resources_needed": ["3-6 short items: datasets, tools, compute, or domain expertise required"],
    "first_experiment": "one concrete, scoped experiment or analysis the student could start within the next few weeks",
    "key_related_papers": ["2-4 short notes on which of the papers above matter most and why, or gaps in what's known"],
    "open_risks": ["2-4 specific risks: concrete failure modes, not generic caveats"],
    "novelty_assessment": "1-3 sentences: given the closest existing papers above, is this direction meaningfully different from what's already been done, and how?"
}}"""

    message = claude.messages.create(
        model      = 'claude-sonnet-5',
        max_tokens = 2500,
        system     = (
            'You are an experienced research advisor helping a graduate student '
            'evaluate whether a research direction is worth pursuing. Be honest and '
            'specific — flag real risks and gaps, not just encouragement. Always '
            'return valid JSON only.'
        ),
        messages   = [{'role': 'user', 'content': prompt}]
    )

    text = next((b.text for b in message.content if b.type == 'text'), '').strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    plan_draft = json.loads(text)
    print(f'[Inventor:draft_research_plan] First experiment: {plan_draft.get("first_experiment", "")[:80]}')

    return {'plan_draft': plan_draft, 'status': 'research plan drafted'}


# ── Node 4: Store Research Plan ───────────────────────────────────────────

def store_research_plan(state: InventorState, resources: dict) -> dict:
    """Stores a ResearchPlan node in Neo4j linked to the hypothesis, keeping
    every field the prompt asked for (the old patent-draft code silently
    dropped 2 of 6 fields here — don't repeat that)."""
    print(f'\n[Inventor:store_research_plan] Storing research plan...')

    neo4j   = resources['neo4j']
    plan_id = f'plan_{uuid.uuid4().hex[:12]}'
    draft   = state['plan_draft']

    with neo4j.driver.session() as session:
        session.run("""
            MERGE (pl:ResearchPlan {id: $id})
            SET pl.methodology_sketch  = $methodology_sketch,
                pl.resources_needed    = $resources_needed,
                pl.first_experiment    = $first_experiment,
                pl.key_related_papers  = $key_related_papers,
                pl.open_risks          = $open_risks,
                pl.novelty_assessment  = $novelty_assessment,
                pl.novelty_score       = $novelty,
                pl.feasibility_score   = $feasibility_score,
                pl.status              = 'draft'
            RETURN pl
        """,
            id                  = plan_id,
            methodology_sketch  = draft.get('methodology_sketch', ''),
            resources_needed    = draft.get('resources_needed', []),
            first_experiment    = draft.get('first_experiment', ''),
            key_related_papers  = draft.get('key_related_papers', []),
            open_risks          = draft.get('open_risks', []),
            novelty_assessment  = draft.get('novelty_assessment', ''),
            novelty             = state['novelty_score'],
            feasibility_score   = state['sim_result'].get('success_rate', 0)
        )

        # Link ResearchPlan to Hypothesis
        session.run("""
            MATCH (h:Hypothesis {id: $hyp_id})
            MATCH (pl:ResearchPlan {id: $plan_id})
            MERGE (h)-[:HAS_PLAN]->(pl)
        """, hyp_id=state['hypothesis_id'], plan_id=plan_id)

    print(f'[Inventor:store_research_plan] Stored: {plan_id}')
    return {'plan_id': plan_id, 'status': f'research plan stored: {plan_id}'}


# ── Build Graph ───────────────────────────────────────────────────────────

def build_inventor(resources: dict):
    def node_novelty(state):
        return check_novelty(state, resources)

    def node_sim(state):
        return run_simulation(state, resources)

    def node_draft(state):
        return draft_research_plan(state, resources)

    def node_store(state):
        return store_research_plan(state, resources)

    def node_skip(state):
        return skip_plan(state, resources)

    graph = StateGraph(InventorState)
    trace = node_tracer('inventor')

    trace(graph, 'check_novelty',        node_novelty, watch=['status', 'novelty_score', 'error'])
    trace(graph, 'run_simulation',       node_sim,     watch=['status'])
    trace(graph, 'draft_research_plan',  node_draft,   watch=['status'])
    trace(graph, 'store_research_plan',  node_store,   watch=['status', 'plan_id'])
    trace(graph, 'skip_plan',            node_skip,    watch=['status'])

    graph.add_edge('check_novelty', 'run_simulation')
    graph.add_conditional_edges(
        'run_simulation',
        should_draft_plan,
        {
            'draft': 'draft_research_plan',
            'skip':  'skip_plan',
        }
    )
    graph.add_edge('draft_research_plan', 'store_research_plan')
    graph.add_edge('store_research_plan', END)
    graph.add_edge('skip_plan',           END)

    graph.set_entry_point('check_novelty')
    return graph.compile()


# ── Run directly ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== APEX Inventor Agent ===\n')

    resources = get_resources()
    neo4j     = resources['neo4j']

    # Get the validated hypothesis
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (h:Hypothesis)
            WHERE h.status = 'validated'
            RETURN h.id AS id
            LIMIT 1
        """)
        record = result.single()

    if not record:
        print('No validated hypotheses found. Run the Skeptic first.')
        neo4j.close()
        exit()

    hypothesis_id = record['id']
    print(f'Processing hypothesis: {hypothesis_id}\n')

    inventor    = build_inventor(resources)
    final_state = inventor.invoke({
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

    print(f'\n=== Inventor Complete ===')
    print(f'Hypothesis ID: {hypothesis_id}')
    print(f'Plan ID:       {final_state["plan_id"]}')
    print(f'Novelty:       {final_state["novelty_score"]}')
    print(f'Sim Score:     {final_state["sim_result"].get("success_rate", 0)}')
    print(f'Status:        {final_state["status"]}')

    if final_state['plan_draft']:
        print(f'\nFirst experiment: {final_state["plan_draft"].get("first_experiment", "")[:100]}')

    resources['neo4j'].close()
    resources['weaviate'].close()