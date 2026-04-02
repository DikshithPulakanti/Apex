# agents/skeptic.py
# APEX Skeptic Agent — adversarially challenges hypotheses
# Now with HypothesisValidityBERT for fast local scoring

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import anthropic
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from database.neo4j_client import Neo4jClient
from training.predictor import HypothesisPredictor

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


# ── State ─────────────────────────────────────────────────────────────────

class SkepticState(TypedDict):
    hypothesis_id:      str
    hypothesis:         dict
    counterarguments:   List[str]
    rebuttal:           str
    debate_score:       float
    rounds_completed:   int
    verdict:            str
    status:             str
    error:              str


# ── Resources ─────────────────────────────────────────────────────────────

def get_resources():
    return {
        'neo4j':     Neo4jClient(),
        'claude':    anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')),
        'predictor': HypothesisPredictor(),
    }


# ── Node 1: Load Hypothesis ───────────────────────────────────────────────

def load_hypothesis(state: SkepticState, resources: dict) -> dict:
    """Loads hypothesis from Neo4j by ID."""
    print(f'\n[Skeptic:load_hypothesis] Loading: {state["hypothesis_id"]}')

    neo4j = resources['neo4j']
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (h:Hypothesis {id: $id})
            RETURN h.statement AS statement,
                   h.rationale AS rationale,
                   h.testability_score AS testability_score,
                   h.predicted_impact AS predicted_impact
        """, id=state['hypothesis_id'])
        record = result.single()

    if not record:
        return {
            'hypothesis': {},
            'error':      f'Hypothesis not found: {state["hypothesis_id"]}',
            'status':     'failed'
        }

    hypothesis = dict(record)
    print(f'[Skeptic:load_hypothesis] Loaded: {hypothesis["statement"][:60]}')
    return {'hypothesis': hypothesis, 'status': 'hypothesis loaded'}


# ── Node 2: Generate Counterarguments ────────────────────────────────────

def generate_counterarguments(state: SkepticState, resources: dict) -> dict:
    """
    Claude plays the Skeptic — finds 3 specific flaws in the hypothesis.
    """
    print(f'\n[Skeptic:generate_counterarguments] Round {state["rounds_completed"] + 1}')

    h      = state['hypothesis']
    claude = resources['claude']

    prompt = f"""You are a rigorous scientific reviewer. Critically analyze this hypothesis and find its weaknesses.

Hypothesis: {h.get('statement', '')}
Rationale: {h.get('rationale', '')}

Generate exactly 3 specific counterarguments. Each must:
1. Identify a specific methodological flaw, missing evidence, or alternative explanation
2. Reference what would need to be true for this hypothesis to be valid
3. Be scientifically grounded

Return ONLY a JSON object:
{{
    "counterargument_1": "specific flaw or missing evidence",
    "counterargument_2": "alternative explanation",
    "counterargument_3": "methodological concern"
}}"""

    message = claude.messages.create(
        model      = 'claude-sonnet-4-20250514',
        max_tokens = 500,
        messages   = [{'role': 'user', 'content': prompt}]
    )

    text = message.content[0].text.strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    data = json.loads(text)
    counterarguments = [
        data.get('counterargument_1', ''),
        data.get('counterargument_2', ''),
        data.get('counterargument_3', ''),
    ]

    print(f'[Skeptic:generate_counterarguments] Generated {len(counterarguments)} counterarguments')
    for i, c in enumerate(counterarguments, 1):
        print(f'  {i}. {c[:80]}')

    return {
        'counterarguments':  counterarguments,
        'rounds_completed':  state.get('rounds_completed', 0) + 1,
        'status':            'counterarguments generated'
    }


# ── Node 3: Generate Rebuttal ─────────────────────────────────────────────

def generate_rebuttal(state: SkepticState, resources: dict) -> dict:
    """
    Claude plays the Reasoner — defends the hypothesis against counterarguments.
    """
    print(f'\n[Skeptic:generate_rebuttal] Generating rebuttal...')

    h      = state['hypothesis']
    claude = resources['claude']

    counterargs_text = '\n'.join([
        f'{i+1}. {c}'
        for i, c in enumerate(state['counterarguments'])
    ])

    prompt = f"""You are defending a scientific hypothesis against criticism.

Hypothesis: {h.get('statement', '')}

Counterarguments raised:
{counterargs_text}

Write a concise rebuttal (3-4 sentences) that addresses these specific criticisms.
Focus on evidence and logical reasoning. Be honest about limitations.

Return ONLY a JSON object:
{{
    "rebuttal": "your defense of the hypothesis"
}}"""

    message = claude.messages.create(
        model      = 'claude-sonnet-4-20250514',
        max_tokens = 300,
        messages   = [{'role': 'user', 'content': prompt}]
    )

    text = message.content[0].text.strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    data     = json.loads(text)
    rebuttal = data.get('rebuttal', '')

    print(f'[Skeptic:generate_rebuttal] Rebuttal: {rebuttal[:100]}')
    return {'rebuttal': rebuttal, 'status': 'rebuttal generated'}


# ── Node 4: Score Debate (BERT + Claude fallback) ─────────────────────────

def score_debate(state: SkepticState, resources: dict) -> dict:
    """
    Two-stage scoring:
    1. HypothesisValidityBERT (instant, free)
    2. Claude Judge (only if BERT is uncertain)
    """
    print(f'\n[Skeptic:score_debate] Scoring debate...')

    h         = state['hypothesis']
    predictor = resources['predictor']
    statement = h.get('statement', '')

    # ── Stage 1: BERT fast-pass ──────────────────────────────────────
    bert_result = predictor.predict(statement)
    confidence  = bert_result['confidence']
    verdict_bert = bert_result['verdict']

    print(f'[Skeptic:score_debate] BERT → {verdict_bert} (confidence: {confidence})')

    if confidence >= 0.95:
        # High confidence — trust BERT, skip Claude
        debate_score = confidence if verdict_bert == 'valid' else (1 - confidence)
        verdict = 'approved' if verdict_bert == 'valid' else 'rejected'

        print(f'[Skeptic:score_debate] BERT confident enough — skipping Claude')
        print(f'[Skeptic:score_debate] Score: {debate_score:.4f} | Verdict: {verdict}')

        return {
            'debate_score': round(debate_score, 4),
            'verdict':      verdict,
            'status':       f'debate scored (BERT): {verdict} ({debate_score:.4f})'
        }

    # ── Stage 2: Claude Judge (BERT uncertain) ───────────────────────
    print(f'[Skeptic:score_debate] BERT uncertain — calling Claude Judge...')

    claude = resources['claude']
    counterargs_text = '\n'.join([
        f'{i+1}. {c}'
        for i, c in enumerate(state['counterarguments'])
    ])

    prompt = f"""You are an impartial scientific judge scoring a debate.

Hypothesis: {statement}

Counterarguments:
{counterargs_text}

Rebuttal: {state['rebuttal']}

Score the hypothesis from 0.0 to 1.0 based on:
- Scientific plausibility (0-0.4 points)
- Quality of evidence cited (0-0.3 points)  
- How well rebuttal addressed criticisms (0-0.3 points)

Return ONLY a JSON object:
{{
    "score": 0.0 to 1.0,
    "reasoning": "brief explanation of score",
    "verdict": "approved" or "rejected"
}}"""

    message = claude.messages.create(
        model      = 'claude-sonnet-4-20250514',
        max_tokens = 300,
        messages   = [{'role': 'user', 'content': prompt}]
    )

    text = message.content[0].text.strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    data         = json.loads(text)
    debate_score = float(data.get('score', 0.5))
    verdict      = data.get('verdict', 'rejected')

    print(f'[Skeptic:score_debate] Claude → Score: {debate_score} | Verdict: {verdict}')
    print(f'  Reasoning: {data.get("reasoning", "")[:100]}')

    return {
        'debate_score': debate_score,
        'verdict':      verdict,
        'status':       f'debate scored (Claude): {verdict} ({debate_score})'
    }


# ── Node 5: Update Hypothesis Status ─────────────────────────────────────

def update_hypothesis_status(state: SkepticState, resources: dict) -> dict:
    """Update hypothesis status in Neo4j based on debate outcome."""
    print(f'\n[Skeptic:update_hypothesis_status] Updating Neo4j...')

    neo4j  = resources['neo4j']
    status = 'validated' if state['verdict'] == 'approved' else 'rejected'

    with neo4j.driver.session() as session:
        session.run("""
            MATCH (h:Hypothesis {id: $id})
            SET h.status       = $status,
                h.debate_score = $score,
                h.rebuttal     = $rebuttal
        """,
            id       = state['hypothesis_id'],
            status   = status,
            score    = state['debate_score'],
            rebuttal = state['rebuttal']
        )

    print(f'[Skeptic:update_hypothesis_status] Status set to: {status}')
    return {'status': f'hypothesis {status} after debate'}


# ── Build Graph ───────────────────────────────────────────────────────────

def build_skeptic(resources: dict):
    def node_load(state):
        return load_hypothesis(state, resources)

    def node_counter(state):
        return generate_counterarguments(state, resources)

    def node_rebuttal(state):
        return generate_rebuttal(state, resources)

    def node_score(state):
        return score_debate(state, resources)

    def node_update(state):
        return update_hypothesis_status(state, resources)

    graph = StateGraph(SkepticState)

    graph.add_node('load_hypothesis',           node_load)
    graph.add_node('generate_counterarguments', node_counter)
    graph.add_node('generate_rebuttal',         node_rebuttal)
    graph.add_node('score_debate',              node_score)
    graph.add_node('update_status',             node_update)

    graph.add_edge('load_hypothesis',           'generate_counterarguments')
    graph.add_edge('generate_counterarguments', 'generate_rebuttal')
    graph.add_edge('generate_rebuttal',         'score_debate')
    graph.add_edge('score_debate',              'update_status')
    graph.add_edge('update_status',             END)

    graph.set_entry_point('load_hypothesis')
    return graph.compile()


# ── Run directly ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== APEX Skeptic Agent ===\n')

    resources = get_resources()

    # Get a hypothesis ID from Neo4j
    neo4j = resources['neo4j']
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (h:Hypothesis)
            WHERE h.status = 'proposed'
            RETURN h.id AS id
            LIMIT 1
        """)
        record = result.single()

    if not record:
        print('No proposed hypotheses found. Run the Reasoner first.')
        neo4j.close()
        exit()

    hypothesis_id = record['id']
    print(f'Testing hypothesis: {hypothesis_id}\n')

    skeptic = build_skeptic(resources)

    final_state = skeptic.invoke({
        'hypothesis_id':    hypothesis_id,
        'hypothesis':       {},
        'counterarguments': [],
        'rebuttal':         '',
        'debate_score':     0.0,
        'rounds_completed': 0,
        'verdict':          '',
        'status':           'starting',
        'error':            ''
    })

    print(f'\n=== Skeptic Complete ===')
    print(f'Hypothesis ID: {hypothesis_id}')
    print(f'Debate Score:  {final_state["debate_score"]}')
    print(f'Verdict:       {final_state["verdict"]}')
    print(f'Status:        {final_state["status"]}')

    resources['neo4j'].close()