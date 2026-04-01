# agents/reasoner.py
# APEX Reasoner Agent — traverses knowledge graph and generates hypotheses

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import json
import anthropic
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from database.neo4j_client import Neo4jClient
from database.weaviate_client import WeaviateClient
from database.embedder import Embedder

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


# ── Pydantic-style hypothesis (plain dict for Python 3.9 compatibility) ───

def empty_hypothesis():
    return {
        'statement':           '',
        'rationale':           '',
        'supporting_concepts': [],
        'testability_score':   0.0,
        'predicted_impact':    ''
    }


# ── State ─────────────────────────────────────────────────────────────────

class ReasonerState(TypedDict):
    seed_concept:    str        # starting concept for traversal
    gaps_found:      List[dict] # research gaps from Neo4j
    context_papers:  List[dict] # papers fetched from Weaviate
    hypothesis:      dict       # generated hypothesis
    hypothesis_id:   str        # Neo4j id after storing
    attempts:        int        # retry counter
    status:          str        # current status
    error:           str        # error if something failed


# ── Resources ─────────────────────────────────────────────────────────────

def get_resources():
    return {
        'neo4j':    Neo4jClient(),
        'weaviate': WeaviateClient(),
        'embedder': Embedder(),
        'claude':   anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')),
    }


# ── Node 1: Select Seed Concept ───────────────────────────────────────────

def select_seed(state: ReasonerState, resources: dict) -> dict:
    """
    Finds research gaps starting from the seed concept.
    Uses PageRank + community scores from GDS.
    """
    print(f'\n[Reasoner:select_seed] Seed concept: "{state["seed_concept"]}"')

    try:
        neo4j = resources['neo4j']
        gaps  = neo4j.find_research_gaps(min_pagerank=0.3, limit=5)

        if not gaps:
            return {
                'gaps_found': [],
                'status':     'no gaps found',
                'error':      'no research gaps detected'
            }

        print(f'[Reasoner:select_seed] Found {len(gaps)} research gaps')
        for g in gaps[:3]:
            print(f'  → {g["concept1"]} ↔ {g["concept2"]} (score: {g["gap_score"]:.2f})')

        return {
            'gaps_found': gaps,
            'status':     f'Found {len(gaps)} research gaps'
        }

    except Exception as e:
        print(f'[Reasoner:select_seed] Error: {e}')
        return {'gaps_found': [], 'error': str(e), 'status': 'seed selection failed'}


# ── Node 2: Gather Context ────────────────────────────────────────────────

def gather_context(state: ReasonerState, resources: dict) -> dict:
    """
    For the top research gap, fetches relevant papers from Weaviate.
    These papers give Claude context for hypothesis generation.
    """
    print(f'\n[Reasoner:gather_context] Gathering context papers...')

    if not state['gaps_found']:
        return {'context_papers': [], 'status': 'no gaps to gather context for'}

    try:
        weaviate = resources['weaviate']
        embedder = resources['embedder']

        # Use the top gap
        top_gap   = state['gaps_found'][0]
        gap_query = f'{top_gap["concept1"]} {top_gap["concept2"]}'

        print(f'[Reasoner:gather_context] Searching for: "{gap_query}"')

        query_vec = embedder.embed_text(gap_query)
        papers    = weaviate.hybrid_search(
            query_text   = gap_query,
            query_vector = query_vec,
            limit        = 5,
            alpha        = 0.6
        )

        print(f'[Reasoner:gather_context] Found {len(papers)} context papers')
        for p in papers[:3]:
            print(f'  → {p.get("title", "")[:60]}')

        return {
            'context_papers': papers,
            'status':         f'Gathered {len(papers)} context papers'
        }

    except Exception as e:
        print(f'[Reasoner:gather_context] Error: {e}')
        return {'context_papers': [], 'error': str(e), 'status': 'context gathering failed'}


# ── Node 3: Generate Hypothesis ───────────────────────────────────────────

def generate_hypothesis(state: ReasonerState, resources: dict) -> dict:
    """
    Sends the research gap + context papers to Claude.
    Claude generates a structured hypothesis.
    """
    print(f'\n[Reasoner:generate_hypothesis] Generating hypothesis...')

    attempts = state.get('attempts', 0) + 1

    if not state['gaps_found']:
        return {
            'hypothesis': empty_hypothesis(),
            'attempts':   attempts,
            'status':     'no gap available for hypothesis'
        }

    try:
        claude   = resources['claude']
        top_gap  = state['gaps_found'][0]

        # Build context string from papers
        paper_context = ''
        for i, paper in enumerate(state['context_papers'][:5], 1):
            title    = paper.get('title', 'Unknown')
            abstract = paper.get('abstract', '')[:200]
            paper_context += f'\nPaper {i}: {title}\n{abstract}\n'

        prompt = f"""Research Gap Analysis:

Concept 1: "{top_gap['concept1']}" (PageRank: {top_gap.get('pagerank1', 0):.2f})
Concept 2: "{top_gap['concept2']}" (PageRank: {top_gap.get('pagerank2', 0):.2f})
Community 1: {top_gap.get('community1', 'unknown')}
Community 2: {top_gap.get('community2', 'unknown')}
Co-occurrences: {top_gap.get('co_occurrence', 0)} (low = unexplored connection)
Gap Score: {top_gap.get('gap_score', 0):.2f}

Related Papers:{paper_context}

Generate a novel, testable scientific hypothesis that bridges these two concept domains.
Return ONLY a JSON object with these exact fields:
{{
    "statement": "one clear hypothesis sentence",
    "rationale": "why this is plausible based on existing research",
    "supporting_concepts": ["concept1", "concept2", "concept3"],
    "testability_score": 0.0 to 1.0,
    "predicted_impact": "what happens if validated"
}}"""

        message = claude.messages.create(
            model      = 'claude-sonnet-4-20250514',
            max_tokens = 800,
            system     = (
                'You are a scientific research assistant specializing in cross-domain '
                'hypothesis generation. Generate novel, testable hypotheses that bridge '
                'different research areas. Always return valid JSON only.'
            ),
            messages   = [{'role': 'user', 'content': prompt}]
        )

        text = message.content[0].text.strip()

        # Parse JSON
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        data       = json.loads(text)
        hypothesis = {
            'statement':           data.get('statement', ''),
            'rationale':           data.get('rationale', ''),
            'supporting_concepts': data.get('supporting_concepts', []),
            'testability_score':   float(data.get('testability_score', 0.5)),
            'predicted_impact':    data.get('predicted_impact', ''),
        }

        print(f'[Reasoner:generate_hypothesis] Generated (attempt {attempts}):')
        print(f'  Statement:   {hypothesis["statement"][:80]}')
        print(f'  Testability: {hypothesis["testability_score"]}')

        return {
            'hypothesis': hypothesis,
            'attempts':   attempts,
            'status':     f'Generated hypothesis (attempt {attempts})'
        }

    except Exception as e:
        print(f'[Reasoner:generate_hypothesis] Error: {e}')
        return {
            'hypothesis': empty_hypothesis(),
            'attempts':   attempts,
            'error':      str(e),
            'status':     'hypothesis generation failed'
        }


# ── Node 4: Store Hypothesis ──────────────────────────────────────────────

def store_hypothesis(state: ReasonerState, resources: dict) -> dict:
    """
    Stores the validated hypothesis as a node in Neo4j.
    Links it to its source concepts.
    """
    print(f'\n[Reasoner:store_hypothesis] Storing hypothesis...')

    h = state['hypothesis']
    if not h or not h.get('statement'):
        return {'status': 'nothing to store', 'hypothesis_id': ''}

    try:
        neo4j        = resources['neo4j']
        hypothesis_id = f'hyp_{uuid.uuid4().hex[:12]}'

        # Store hypothesis node
        query = """
            MERGE (h:Hypothesis {id: $id})
            SET h.statement         = $statement,
                h.rationale         = $rationale,
                h.testability_score = $testability_score,
                h.predicted_impact  = $predicted_impact,
                h.status            = 'proposed',
                h.created_by        = 'Reasoner'
            RETURN h
        """
        with neo4j.driver.session() as session:
            session.run(query,
                id                = hypothesis_id,
                statement         = h['statement'],
                rationale         = h['rationale'],
                testability_score = h['testability_score'],
                predicted_impact  = h['predicted_impact']
            )

        # Link to source concepts
        for concept_name in h.get('supporting_concepts', []):
            link_query = """
                MATCH (h:Hypothesis {id: $hyp_id})
                MATCH (c:Concept {name: $concept_name})
                MERGE (h)-[:DERIVED_FROM]->(c)
            """
            with neo4j.driver.session() as session:
                session.run(link_query,
                    hyp_id       = hypothesis_id,
                    concept_name = concept_name.lower()
                )

        print(f'[Reasoner:store_hypothesis] Stored as: {hypothesis_id}')

        return {
            'hypothesis_id': hypothesis_id,
            'status':        f'Hypothesis stored: {hypothesis_id}'
        }

    except Exception as e:
        print(f'[Reasoner:store_hypothesis] Error: {e}')
        return {
            'hypothesis_id': '',
            'error':         str(e),
            'status':        'storage failed'
        }


# ── Routing Logic ─────────────────────────────────────────────────────────

def should_retry_hypothesis(state: ReasonerState) -> str:
    """
    If hypothesis quality is too low, retry generation.
    Max 3 attempts.
    """
    h = state.get('hypothesis', {})
    score    = h.get('testability_score', 0.0)
    attempts = state.get('attempts', 0)

    if score >= 0.6:
        print(f'  [router] Score {score:.2f} >= 0.6 → storing')
        return 'store'
    elif attempts >= 3:
        print(f'  [router] Max attempts reached → storing anyway')
        return 'store'
    else:
        print(f'  [router] Score {score:.2f} < 0.6 → retrying')
        return 'retry'


# ── Build Graph ───────────────────────────────────────────────────────────

def build_reasoner(resources: dict):
    def node_seed(state):
        return select_seed(state, resources)

    def node_context(state):
        return gather_context(state, resources)

    def node_generate(state):
        return generate_hypothesis(state, resources)

    def node_store(state):
        return store_hypothesis(state, resources)

    graph = StateGraph(ReasonerState)

    graph.add_node('select_seed',        node_seed)
    graph.add_node('gather_context',     node_context)
    graph.add_node('generate_hypothesis', node_generate)
    graph.add_node('store_hypothesis',   node_store)

    graph.add_edge('select_seed',    'gather_context')
    graph.add_edge('gather_context', 'generate_hypothesis')

    graph.add_conditional_edges(
        'generate_hypothesis',
        should_retry_hypothesis,
        {
            'store': 'store_hypothesis',
            'retry': 'generate_hypothesis',
        }
    )

    graph.add_edge('store_hypothesis', END)
    graph.set_entry_point('select_seed')

    return graph.compile()


# ── Run directly ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== APEX Reasoner Agent ===\n')

    resources = get_resources()
    reasoner  = build_reasoner(resources)

    initial_state = {
        'seed_concept':   'large language models',
        'gaps_found':     [],
        'context_papers': [],
        'hypothesis':     empty_hypothesis(),
        'hypothesis_id':  '',
        'attempts':       0,
        'status':         'starting',
        'error':          ''
    }

    final_state = reasoner.invoke(initial_state)

    print(f'\n=== Reasoner Complete ===')
    print(f'Status:       {final_state["status"]}')
    print(f'Hypothesis ID: {final_state["hypothesis_id"]}')
    if final_state['hypothesis']['statement']:
        print(f'\nHypothesis:')
        print(f'  {final_state["hypothesis"]["statement"]}')
        print(f'\nTestability: {final_state["hypothesis"]["testability_score"]}')
        print(f'Impact:      {final_state["hypothesis"]["predicted_impact"][:100]}')

    resources['neo4j'].close()
    resources['weaviate'].close()