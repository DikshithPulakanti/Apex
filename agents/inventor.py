# agents/inventor.py
# APEX Inventor Agent — drafts patents from validated hypotheses

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

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


# ── State ─────────────────────────────────────────────────────────────────

class InventorState(TypedDict):
    hypothesis_id:   str
    hypothesis:      dict
    novelty_score:   float
    sim_result:      dict
    patent_draft:    dict
    patent_id:       str
    status:          str
    error:           str


# ── Resources ─────────────────────────────────────────────────────────────

def get_resources():
    return {
        'neo4j':  Neo4jClient(),
        'claude': anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')),
    }


# ── Node 1: Load and Check Novelty ────────────────────────────────────────

def check_novelty(state: InventorState, resources: dict) -> dict:
    """Loads hypothesis and computes novelty score."""
    print(f'\n[Inventor:check_novelty] Loading: {state["hypothesis_id"]}')

    neo4j = resources['neo4j']
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (h:Hypothesis {id: $id})
            RETURN h.statement AS statement,
                   h.rationale AS rationale,
                   h.testability_score AS testability_score,
                   h.predicted_impact AS predicted_impact,
                   h.debate_score AS debate_score
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

    # Compute novelty based on concept overlap with existing hypotheses
    shared = 0
    for concept in concepts:
        with neo4j.driver.session() as session:
            result = session.run("""
                MATCH (h:Hypothesis)-[:DERIVED_FROM]->(c:Concept {name: $name})
                WHERE h.id <> $hyp_id
                RETURN count(h) AS n
            """, name=concept, hyp_id=state['hypothesis_id'])
            shared += result.single()['n']

    novelty_score = max(0.1, 1.0 - (shared * 0.1))
    print(f'[Inventor:check_novelty] Novelty: {novelty_score} | Concepts: {concepts}')

    return {
        'hypothesis':   hypothesis,
        'novelty_score': novelty_score,
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


# ── Routing: Should We Patent? ────────────────────────────────────────────

def should_patent(state: InventorState) -> str:
    """Only proceed to patent drafting if novelty and simulation pass."""
    novelty  = state.get('novelty_score', 0.0)
    sim_rate = state.get('sim_result', {}).get('success_rate', 0.0)

    if novelty >= 0.5 and sim_rate >= 0.6:
        print(f'  [router] Novelty {novelty:.2f} + Sim {sim_rate:.2f} → drafting patent')
        return 'draft'
    else:
        print(f'  [router] Novelty {novelty:.2f} or Sim {sim_rate:.2f} too low → skip')
        return 'skip'


def skip_patent(state: InventorState, resources: dict) -> dict:
    return {'status': 'patent skipped — insufficient novelty or simulation score'}


# ── Node 3: Draft Patent ──────────────────────────────────────────────────

def draft_patent(state: InventorState, resources: dict) -> dict:
    """Uses Claude to draft structured patent claims."""
    print(f'\n[Inventor:draft_patent] Drafting patent...')

    h      = state['hypothesis']
    claude = resources['claude']

    concepts = h.get('supporting_concepts', [])
    impact   = h.get('predicted_impact', '')

    prompt = f"""Draft a patent application for this invention:

Invention: {h.get('statement', '')}
Key Concepts: {', '.join(concepts)}
Expected Impact: {impact}

Return ONLY a JSON object:
{{
    "title": "Patent title",
    "background": "Background (2-3 sentences)",
    "summary": "Summary (2-3 sentences)",
    "independent_claim_1": "Broadest independent claim",
    "dependent_claim_2": "Dependent claim adding specificity",
    "abstract": "Abstract (100 words max)"
}}"""

    message = claude.messages.create(
        model      = 'claude-sonnet-4-20250514',
        max_tokens = 800,
        messages   = [{'role': 'user', 'content': prompt}]
    )

    text = message.content[0].text.strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    patent_draft = json.loads(text)
    print(f'[Inventor:draft_patent] Title: {patent_draft.get("title", "")}')

    return {'patent_draft': patent_draft, 'status': 'patent drafted'}


# ── Node 4: Store Patent ──────────────────────────────────────────────────

def store_patent(state: InventorState, resources: dict) -> dict:
    """Stores Patent node in Neo4j linked to the hypothesis."""
    print(f'\n[Inventor:store_patent] Storing patent...')

    neo4j      = resources['neo4j']
    patent_id  = f'pat_{uuid.uuid4().hex[:12]}'
    draft      = state['patent_draft']

    with neo4j.driver.session() as session:
        session.run("""
            MERGE (p:Patent {id: $id})
            SET p.title                = $title,
                p.background           = $background,
                p.independent_claim_1  = $claim1,
                p.abstract             = $abstract,
                p.novelty_score        = $novelty,
                p.simulation_score     = $sim_score,
                p.status               = 'draft'
            RETURN p
        """,
            id         = patent_id,
            title      = draft.get('title', ''),
            background = draft.get('background', ''),
            claim1     = draft.get('independent_claim_1', ''),
            abstract   = draft.get('abstract', ''),
            novelty    = state['novelty_score'],
            sim_score  = state['sim_result'].get('success_rate', 0)
        )

        # Link Patent to Hypothesis
        session.run("""
            MATCH (h:Hypothesis {id: $hyp_id})
            MATCH (p:Patent {id: $pat_id})
            MERGE (h)-[:LED_TO]->(p)
        """, hyp_id=state['hypothesis_id'], pat_id=patent_id)

    print(f'[Inventor:store_patent] Stored: {patent_id}')
    return {'patent_id': patent_id, 'status': f'patent stored: {patent_id}'}


# ── Build Graph ───────────────────────────────────────────────────────────

def build_inventor(resources: dict):
    def node_novelty(state):
        return check_novelty(state, resources)

    def node_sim(state):
        return run_simulation(state, resources)

    def node_draft(state):
        return draft_patent(state, resources)

    def node_store(state):
        return store_patent(state, resources)

    def node_skip(state):
        return skip_patent(state, resources)

    graph = StateGraph(InventorState)

    graph.add_node('check_novelty',  node_novelty)
    graph.add_node('run_simulation', node_sim)
    graph.add_node('draft_patent',   node_draft)
    graph.add_node('store_patent',   node_store)
    graph.add_node('skip_patent',    node_skip)

    graph.add_edge('check_novelty', 'run_simulation')
    graph.add_conditional_edges(
        'run_simulation',
        should_patent,
        {
            'draft': 'draft_patent',
            'skip':  'skip_patent',
        }
    )
    graph.add_edge('draft_patent', 'store_patent')
    graph.add_edge('store_patent', END)
    graph.add_edge('skip_patent',  END)

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
        'sim_result':    {},
        'patent_draft':  {},
        'patent_id':     '',
        'status':        'starting',
        'error':         ''
    })

    print(f'\n=== Inventor Complete ===')
    print(f'Hypothesis ID: {hypothesis_id}')
    print(f'Patent ID:     {final_state["patent_id"]}')
    print(f'Novelty:       {final_state["novelty_score"]}')
    print(f'Sim Score:     {final_state["sim_result"].get("success_rate", 0)}')
    print(f'Status:        {final_state["status"]}')

    if final_state['patent_draft']:
        print(f'\nPatent Title:  {final_state["patent_draft"].get("title", "")}')
        print(f'Claim 1:       {final_state["patent_draft"].get("independent_claim_1", "")[:100]}')

    resources['neo4j'].close()