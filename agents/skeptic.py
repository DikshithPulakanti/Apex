# agents/skeptic.py
# APEX Skeptic Agent — adversarially challenges hypotheses
# Now with HypothesisValidityBERT for fast local scoring

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import anthropic
import mlflow
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from database.neo4j_client import Neo4jClient
from database.postgres_client import PostgresClient
from training.predictor import HypothesisPredictor
from events.node_tracing import node_tracer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO_ROOT, '.env'))

# Reuse the same local tracking store train_bert.py writes to, unless the
# caller has already pointed MLflow somewhere else (e.g. a shared server).
if not os.getenv('MLFLOW_TRACKING_URI'):
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(REPO_ROOT, 'mlflow.db')}")

MLFLOW_EXPERIMENT = 'skeptic-evaluations'

# ── Confidence-gating thresholds ─────────────────────────────────────────
# Tier 1: BERT is confident enough to trust on its own — fast, free, auto.
BERT_CONFIDENCE_THRESHOLD = 0.95
# Tier 2: BERT was unsure, but Claude's score is decisive (and agrees with
# its own verdict) — still fully automated, just slower.
CLAUDE_APPROVE_THRESHOLD = 0.70
CLAUDE_REJECT_THRESHOLD = 0.30
# Tier 3: neither model was confident (or Claude's call/parse failed) —
# flag for structured human review instead of guessing.


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
    bert_confidence:    float
    bert_verdict:       str
    claude_score:       float
    claude_reasoning:   str
    scoring_method:     str
    mlflow_run_id:      str


# ── Resources ─────────────────────────────────────────────────────────────

def get_resources():
    return {
        'neo4j':     Neo4jClient(),
        'claude':    anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')),
        'predictor': HypothesisPredictor(),
        'postgres':  PostgresClient(),
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
        model      = 'claude-sonnet-5',
        max_tokens = 2000,
        messages   = [{'role': 'user', 'content': prompt}]
    )

    text = next((b.text for b in message.content if b.type == 'text'), '').strip()
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
        model      = 'claude-sonnet-5',
        max_tokens = 2000,
        messages   = [{'role': 'user', 'content': prompt}]
    )

    text = next((b.text for b in message.content if b.type == 'text'), '').strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    data     = json.loads(text)
    rebuttal = data.get('rebuttal', '')

    print(f'[Skeptic:generate_rebuttal] Rebuttal: {rebuttal[:100]}')
    return {'rebuttal': rebuttal, 'status': 'rebuttal generated'}


# ── Node 4: Score Debate (BERT + Claude fallback + human review gate) ─────

def _call_claude_judge(state: SkepticState, claude) -> dict:
    """Asks Claude to score the debate. Raises on API/parse failure — the
    caller decides what that means (here: fall through to human review)."""
    h = state['hypothesis']
    counterargs_text = '\n'.join([
        f'{i+1}. {c}'
        for i, c in enumerate(state['counterarguments'])
    ])

    prompt = f"""You are an impartial scientific judge scoring a debate.

Hypothesis: {h.get('statement', '')}

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
        model      = 'claude-sonnet-5',
        max_tokens = 2000,
        messages   = [{'role': 'user', 'content': prompt}]
    )

    text = next((b.text for b in message.content if b.type == 'text'), '').strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    data = json.loads(text)
    return {
        'score':     float(data.get('score', 0.5)),
        'verdict':   data.get('verdict', 'rejected'),
        'reasoning': data.get('reasoning', ''),
    }


def score_debate(state: SkepticState, resources: dict) -> dict:
    """
    Three-tier confidence-gated scoring:
    1. HypothesisValidityBERT (instant, free) — trusted if confident enough.
    2. Claude Judge (only if BERT is uncertain) — trusted if decisive.
    3. Neither was confident/decisive, or Claude's call failed — flagged
       for structured human review instead of guessing.

    Every decision is logged as its own MLflow run under the
    'skeptic-evaluations' experiment, so any hypothesis's verdict can be
    traced back to exactly what BERT/Claude saw and scored.
    """
    print(f'\n[Skeptic:score_debate] Scoring debate...')

    h         = state['hypothesis']
    predictor = resources['predictor']
    statement = h.get('statement', '')

    bert_result  = predictor.predict(statement)
    bert_confidence = bert_result['confidence']
    bert_verdict    = bert_result['verdict']
    print(f'[Skeptic:score_debate] BERT → {bert_verdict} (confidence: {bert_confidence})')

    claude_score = None
    claude_verdict = None
    claude_reasoning = ''

    if bert_confidence >= BERT_CONFIDENCE_THRESHOLD:
        # ── Tier 1: trust BERT, skip Claude entirely ─────────────────
        debate_score = bert_confidence if bert_verdict == 'valid' else (1 - bert_confidence)
        verdict = 'approved' if bert_verdict == 'valid' else 'rejected'
        scoring_method = 'bert_auto'
        print(f'[Skeptic:score_debate] BERT confident enough — skipping Claude')
    else:
        # ── Tier 2/3: BERT uncertain — ask Claude, then judge Claude ─
        print(f'[Skeptic:score_debate] BERT uncertain — calling Claude Judge...')
        try:
            claude_result = _call_claude_judge(state, resources['claude'])
            claude_score = claude_result['score']
            claude_verdict = claude_result['verdict']
            claude_reasoning = claude_result['reasoning']
            print(f'[Skeptic:score_debate] Claude → Score: {claude_score} | Verdict: {claude_verdict}')

            decisive = (
                (claude_score >= CLAUDE_APPROVE_THRESHOLD and claude_verdict == 'approved') or
                (claude_score <= CLAUDE_REJECT_THRESHOLD and claude_verdict == 'rejected')
            )
        except Exception as e:
            print(f'[Skeptic:score_debate] Claude Judge call failed: {e}')
            decisive = False
            claude_reasoning = f'Claude Judge call failed: {e}'

        if decisive:
            # ── Tier 2: Claude was decisive — still fully automated ──
            debate_score = claude_score
            verdict = claude_verdict
            scoring_method = 'claude_auto'
        else:
            # ── Tier 3: nobody was confident — flag for a human ──────
            debate_score = claude_score if claude_score is not None else bert_confidence
            verdict = 'pending_review'
            scoring_method = 'human_review'
            print(f'[Skeptic:score_debate] Neither model was decisive — flagging for human review')

    mlflow_run_id = ''
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"debate-{state['hypothesis_id']}") as run:
            mlflow.log_param('hypothesis_id', state['hypothesis_id'])
            mlflow.log_param('bert_verdict', bert_verdict)
            mlflow.log_param('claude_verdict', claude_verdict or 'n/a')
            mlflow.log_param('tier', scoring_method)
            mlflow.log_param('final_verdict', verdict)
            mlflow.log_metric('bert_confidence', bert_confidence)
            if claude_score is not None:
                mlflow.log_metric('claude_score', claude_score)
            mlflow.log_metric('debate_score', debate_score)
            mlflow_run_id = run.info.run_id
    except Exception as e:
        print(f'[Skeptic:score_debate] MLflow logging failed (non-fatal): {e}')

    print(f'[Skeptic:score_debate] Score: {debate_score:.4f} | Verdict: {verdict} | Method: {scoring_method}')

    return {
        'debate_score':     round(debate_score, 4),
        'verdict':          verdict,
        'bert_confidence':  bert_confidence,
        'bert_verdict':     bert_verdict,
        'claude_score':     claude_score if claude_score is not None else 0.0,
        'claude_reasoning': claude_reasoning,
        'scoring_method':   scoring_method,
        'mlflow_run_id':    mlflow_run_id,
        'status':           f'debate scored ({scoring_method}): {verdict} ({debate_score:.4f})',
    }


# ── Node 5: Update Hypothesis Status ─────────────────────────────────────

_VERDICT_TO_STATUS = {
    'approved':       'validated',
    'rejected':       'rejected',
    'pending_review': 'pending_review',
}


def update_hypothesis_status(state: SkepticState, resources: dict) -> dict:
    """
    Update hypothesis status in Neo4j based on the debate outcome, and —
    when the Skeptic couldn't confidently auto-decide — enqueue it in the
    Postgres human-review queue instead of guessing.
    """
    print(f'\n[Skeptic:update_hypothesis_status] Updating Neo4j...')

    neo4j  = resources['neo4j']
    status = _VERDICT_TO_STATUS.get(state['verdict'], 'rejected')

    with neo4j.driver.session() as session:
        session.run("""
            MATCH (h:Hypothesis {id: $id})
            SET h.status           = $status,
                h.debate_score     = $score,
                h.rebuttal         = $rebuttal,
                h.bert_confidence  = $bert_confidence,
                h.bert_verdict     = $bert_verdict,
                h.claude_score     = $claude_score,
                h.claude_reasoning = $claude_reasoning,
                h.scoring_method   = $scoring_method,
                h.mlflow_run_id    = $mlflow_run_id
        """,
            id               = state['hypothesis_id'],
            status           = status,
            score            = state['debate_score'],
            rebuttal         = state['rebuttal'],
            bert_confidence  = state.get('bert_confidence', 0.0),
            bert_verdict     = state.get('bert_verdict', ''),
            claude_score     = state.get('claude_score', 0.0),
            claude_reasoning = state.get('claude_reasoning', ''),
            scoring_method   = state.get('scoring_method', ''),
            mlflow_run_id    = state.get('mlflow_run_id', ''),
        )

    if status == 'pending_review':
        postgres = resources.get('postgres')
        if postgres:
            postgres.enqueue_review(
                hypothesis_id      = state['hypothesis_id'],
                statement_snapshot = state['hypothesis'].get('statement', ''),
                bert_confidence    = state.get('bert_confidence', 0.0),
                bert_verdict       = state.get('bert_verdict', ''),
                claude_score       = state.get('claude_score', 0.0),
                claude_verdict     = state.get('verdict', ''),
                claude_reasoning   = state.get('claude_reasoning', ''),
                mlflow_run_id      = state.get('mlflow_run_id', ''),
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
    trace = node_tracer('skeptic')

    trace(graph, 'load_hypothesis',           node_load,     watch=['status', 'error'])
    trace(graph, 'generate_counterarguments', node_counter,  watch=['status', 'rounds_completed'])
    trace(graph, 'generate_rebuttal',         node_rebuttal, watch=['status'])
    trace(graph, 'score_debate',              node_score,    watch=['status', 'verdict', 'debate_score', 'scoring_method'])
    trace(graph, 'update_status',             node_update,   watch=['status'])

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
        'error':            '',
        'bert_confidence':  0.0,
        'bert_verdict':     '',
        'claude_score':     0.0,
        'claude_reasoning': '',
        'scoring_method':   '',
        'mlflow_run_id':    '',
    })

    print(f'\n=== Skeptic Complete ===')
    print(f'Hypothesis ID: {hypothesis_id}')
    print(f'Debate Score:  {final_state["debate_score"]}')
    print(f'Verdict:       {final_state["verdict"]}')
    print(f'Status:        {final_state["status"]}')

    resources['neo4j'].close()