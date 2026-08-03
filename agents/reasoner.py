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
from events.node_tracing import node_tracer

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
    seed_concept: str        # starting concept for traversal
    gaps_found:   List[dict] # research gaps from Neo4j
    candidates:   List[dict] # one per gap: {gap, context_papers, hypothesis, attempts}
    stored:       List[dict] # one per stored hypothesis: {hypothesis_id, hypothesis}
    status:       str        # current status
    error:        str        # error if something failed


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
        gaps  = neo4j.find_research_gaps_for_seed(state['seed_concept'], min_pagerank=0.1, limit=5)

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
    For every research gap found (not just the top one), fetches relevant
    papers from Weaviate. These papers give Claude context for hypothesis
    generation, one candidate hypothesis per gap.
    """
    gaps = state['gaps_found']
    print(f'\n[Reasoner:gather_context] Gathering context for {len(gaps)} gap(s)...')

    if not gaps:
        return {'candidates': [], 'status': 'no gaps to gather context for'}

    weaviate   = resources['weaviate']
    embedder   = resources['embedder']
    candidates = []

    for gap in gaps:
        gap_query = f'{gap["concept1"]} {gap["concept2"]}'
        print(f'[Reasoner:gather_context] Searching for: "{gap_query}"')

        try:
            query_vec = embedder.embed_text(gap_query)
            papers    = weaviate.hybrid_search(
                query_text   = gap_query,
                query_vector = query_vec,
                limit        = 5,
                alpha        = 0.6
            )
            print(f'[Reasoner:gather_context]   → {len(papers)} context papers')
        except Exception as e:
            print(f'[Reasoner:gather_context]   → Error: {e}')
            papers = []

        candidates.append({
            'gap':            gap,
            'context_papers': papers,
            'hypothesis':     empty_hypothesis(),
            'attempts':       0,
        })

    return {
        'candidates': candidates,
        'status':     f'Gathered context for {len(candidates)} candidate(s)'
    }


# ── Node 3: Generate Hypothesis ───────────────────────────────────────────

def _generate_one_hypothesis(gap: dict, context_papers: List[dict], claude) -> dict:
    """
    Builds the prompt for a single gap + its context papers, calls Claude,
    and parses the JSON hypothesis. Raises on failure — the caller decides
    what to do with a bad candidate rather than this helper swallowing it.
    """
    paper_context = ''
    for i, paper in enumerate(context_papers[:5], 1):
        title    = paper.get('title', 'Unknown')
        abstract = paper.get('abstract', '')[:200]
        paper_context += f'\nPaper {i}: {title}\n{abstract}\n'

    prompt = f"""Research Gap Analysis:

Concept 1: "{gap['concept1']}" (PageRank: {gap.get('pagerank1', 0):.2f})
Concept 2: "{gap['concept2']}" (PageRank: {gap.get('pagerank2', 0):.2f})
Community 1: {gap.get('community1', 'unknown')}
Community 2: {gap.get('community2', 'unknown')}
Co-occurrences: {gap.get('co_occurrence', 0)} (low = unexplored connection)
Gap Score: {gap.get('gap_score', 0):.2f}

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
        model      = 'claude-sonnet-5',
        max_tokens = 3000,
        system     = (
            'You are a scientific research assistant specializing in cross-domain '
            'hypothesis generation. Generate novel, testable hypotheses that bridge '
            'different research areas. Always return valid JSON only.'
        ),
        messages   = [{'role': 'user', 'content': prompt}]
    )

    text = next((b.text for b in message.content if b.type == 'text'), '').strip()

    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    data = json.loads(text)
    return {
        'statement':           data.get('statement', ''),
        'rationale':           data.get('rationale', ''),
        'supporting_concepts': data.get('supporting_concepts', []),
        'testability_score':   float(data.get('testability_score', 0.5)),
        'predicted_impact':    data.get('predicted_impact', ''),
    }


def generate_hypothesis(state: ReasonerState, resources: dict) -> dict:
    """
    Generates one hypothesis per candidate gap, retrying low-scoring ones up
    to 3 attempts each. A single candidate's failure (bad Claude response,
    malformed JSON) does not stop the others from being generated — each
    candidate is independent.
    """
    claude     = resources['claude']
    candidates = state.get('candidates', [])
    print(f'\n[Reasoner:generate_hypothesis] Generating {len(candidates)} hypothes(es)...')

    for c in candidates:
        while True:
            c['attempts'] += 1
            try:
                c['hypothesis'] = _generate_one_hypothesis(c['gap'], c['context_papers'], claude)
                print(f'[Reasoner:generate_hypothesis] "{c["gap"]["concept1"]} ↔ {c["gap"]["concept2"]}" '
                      f'(attempt {c["attempts"]}): {c["hypothesis"]["statement"][:80]}')
            except Exception as e:
                print(f'[Reasoner:generate_hypothesis] Error on '
                      f'"{c["gap"]["concept1"]} ↔ {c["gap"]["concept2"]}": {e}')
                c['hypothesis'] = empty_hypothesis()
                break

            if should_retry_hypothesis({'hypothesis': c['hypothesis'], 'attempts': c['attempts']}) == 'store':
                break

    return {
        'candidates': candidates,
        'status':     f'Generated {len(candidates)} hypothes(es)'
    }


# ── Node 4: Store Hypothesis ──────────────────────────────────────────────

def store_hypothesis(state: ReasonerState, resources: dict) -> dict:
    """
    Stores every generated candidate with a real statement as a Hypothesis
    node in Neo4j, linking it to its source concepts (DERIVED_FROM) and the
    papers it was built from (CITES), so a student can jump into the
    literature a hypothesis actually came from.
    """
    neo4j  = resources['neo4j']
    stored = []

    for c in state.get('candidates', []):
        h = c['hypothesis']
        if not h or not h.get('statement'):
            continue

        hypothesis_id = f'hyp_{uuid.uuid4().hex[:12]}'

        query = """
            MERGE (h:Hypothesis {id: $id})
            SET h.statement         = $statement,
                h.rationale         = $rationale,
                h.testability_score = $testability_score,
                h.predicted_impact  = $predicted_impact,
                h.seed_concept      = $seed_concept,
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
                predicted_impact  = h['predicted_impact'],
                seed_concept      = state['seed_concept'],
            )

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

        for rank, paper in enumerate(c['context_papers'][:5], start=1):
            paper_id = paper.get('paper_id', '')
            if not paper_id:
                continue
            cite_query = """
                MATCH (h:Hypothesis {id: $hyp_id})
                MATCH (p:Paper {id: $paper_id})
                MERGE (h)-[r:CITES]->(p)
                SET r.rank = $rank, r.hybrid_score = $score
            """
            with neo4j.driver.session() as session:
                session.run(cite_query,
                    hyp_id   = hypothesis_id,
                    paper_id = paper_id,
                    rank     = rank,
                    score    = paper.get('score', 0.0),
                )

        print(f'[Reasoner:store_hypothesis] Stored as: {hypothesis_id}')
        stored.append({'hypothesis_id': hypothesis_id, 'hypothesis': h})

    return {
        'stored': stored,
        'status': f'Stored {len(stored)} hypothes(es)'
    }


# ── Routing Logic ─────────────────────────────────────────────────────────

def should_retry_hypothesis(state: dict) -> str:
    """
    If hypothesis quality is too low, retry generation. Max 3 attempts.
    Takes a small {hypothesis, attempts} dict rather than the full
    ReasonerState so it works as a pure per-candidate decision function.
    """
    h        = state.get('hypothesis', {})
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
    trace = node_tracer('reasoner')

    trace(graph, 'select_seed',         node_seed,     watch=['status', 'error'])
    trace(graph, 'gather_context',      node_context,  watch=['status', 'error'])
    trace(graph, 'generate_hypothesis', node_generate, watch=['status', 'error'])
    trace(graph, 'store_hypothesis',    node_store,    watch=['status', 'error'])

    graph.add_edge('select_seed',         'gather_context')
    graph.add_edge('gather_context',      'generate_hypothesis')
    graph.add_edge('generate_hypothesis', 'store_hypothesis')
    graph.add_edge('store_hypothesis',    END)
    graph.set_entry_point('select_seed')

    return graph.compile()


# ── Run directly ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== APEX Reasoner Agent ===\n')

    resources = get_resources()
    reasoner  = build_reasoner(resources)

    initial_state = {
        'seed_concept': 'large language models',
        'gaps_found':   [],
        'candidates':   [],
        'stored':       [],
        'status':       'starting',
        'error':        ''
    }

    final_state = reasoner.invoke(initial_state)

    print(f'\n=== Reasoner Complete ===')
    print(f'Status: {final_state["status"]}')
    print(f'Generated {len(final_state.get("stored", []))} hypothes(es):')
    for item in final_state.get('stored', []):
        h = item['hypothesis']
        print(f'\n  [{item["hypothesis_id"]}]')
        print(f'  {h["statement"]}')
        print(f'  Testability: {h["testability_score"]}')

    resources['neo4j'].close()
    resources['weaviate'].close()
