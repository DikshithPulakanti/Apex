# agents/harvester.py
# APEX Harvester Agent — finds and processes scientific papers

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from database.neo4j_client import Neo4jClient
from database.weaviate_client import WeaviateClient
from database.embedder import Embedder
from scrapers.concept_extractor import ConceptExtractor

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


# ── State ─────────────────────────────────────────────────────────────────

class HarvesterState(TypedDict):
    query:              str        # the research topic to search for
    papers_found:       List[dict] # papers returned from Weaviate search
    concepts_extracted: List[str]  # concepts extracted from papers
    papers_processed:   int        # count of papers successfully processed
    status:             str        # current status message
    error:              str        # error message if something went wrong


# ── Shared resources (created once, reused across nodes) ──────────────────

def get_resources():
    """Creates database clients and NLP models."""
    return {
        'neo4j':     Neo4jClient(),
        'weaviate':  WeaviateClient(),
        'embedder':  Embedder(),
        'extractor': ConceptExtractor(),
    }


# ── Node 1: Search Papers ─────────────────────────────────────────────────

def search_papers(state: HarvesterState, resources: dict) -> dict:
    """
    Searches Weaviate for papers semantically similar to the query.
    Uses hybrid search — combines vector similarity + keyword matching.
    """
    print(f'\n[Harvester:search_papers] Query: "{state["query"]}"')

    try:
        embedder = resources['embedder']
        weaviate = resources['weaviate']

        # Convert query to vector
        query_vector = embedder.embed_text(state['query'])

        # Hybrid search — semantic + keyword
        papers = weaviate.hybrid_search(
            query_text   = state['query'],
            query_vector = query_vector,
            limit        = 10,
            alpha        = 0.7  # 70% semantic, 30% keyword
        )

        print(f'[Harvester:search_papers] Found {len(papers)} relevant papers')
        for p in papers[:3]:
            print(f'  → {p.get("title", "")[:60]}')

        return {
            'papers_found': papers,
            'status':       f'Found {len(papers)} papers for: {state["query"]}'
        }

    except Exception as e:
        print(f'[Harvester:search_papers] Error: {e}')
        return {
            'papers_found': [],
            'error':        str(e),
            'status':       'search failed'
        }


# ── Node 2: Extract Concepts ──────────────────────────────────────────────

def extract_concepts(state: HarvesterState, resources: dict) -> dict:
    """
    Extracts technical concepts from the found papers' abstracts.
    Adds these as Concept nodes to Neo4j.
    """
    print(f'\n[Harvester:extract_concepts] Processing {len(state["papers_found"])} papers')

    if not state['papers_found']:
        return {
            'concepts_extracted': [],
            'status': 'no papers to extract from'
        }

    try:
        extractor = resources['extractor']
        neo4j     = resources['neo4j']

        all_concepts = []

        for paper in state['papers_found']:
            abstract = paper.get('abstract', '') or paper.get('title', '')
            if not abstract:
                continue

            concepts = extractor.extract_concepts(abstract, max_concepts=6)
            all_concepts.extend(concepts)

            # Store concepts in Neo4j
            paper_id = paper.get('paper_id', '')
            if paper_id:
                for concept in concepts:
                    neo4j.upsert_concept(concept, domain='harvester')
                    neo4j.link_paper_to_concept(paper_id, concept)

        # Deduplicate
        unique_concepts = list(set(all_concepts))
        print(f'[Harvester:extract_concepts] Extracted {len(unique_concepts)} unique concepts')

        return {
            'concepts_extracted': unique_concepts,
            'status':             f'Extracted {len(unique_concepts)} concepts'
        }

    except Exception as e:
        print(f'[Harvester:extract_concepts] Error: {e}')
        return {
            'concepts_extracted': [],
            'error':              str(e),
            'status':             'concept extraction failed'
        }


# ── Node 3: Insert to Graph ───────────────────────────────────────────────

def insert_to_graph(state: HarvesterState, resources: dict) -> dict:
    """
    Updates co-occurrence relationships between newly extracted concepts.
    This keeps the knowledge graph fresh after every Harvester run.
    """
    print(f'\n[Harvester:insert_to_graph] Updating knowledge graph...')

    try:
        neo4j = resources['neo4j']

        # Rebuild co-occurrence relationships
        neo4j.build_concept_cooccurrence()

        return {
            'papers_processed': len(state['papers_found']),
            'status':           'knowledge graph updated'
        }

    except Exception as e:
        print(f'[Harvester:insert_to_graph] Error: {e}')
        return {
            'error':  str(e),
            'status': 'graph update failed'
        }


# ── Node 4: Update Status ─────────────────────────────────────────────────

def update_status(state: HarvesterState, resources: dict) -> dict:
    """
    Final node — logs completion and returns summary.
    """
    print(f'\n[Harvester:update_status] Run complete')
    print(f'  Papers found:     {len(state["papers_found"])}')
    print(f'  Concepts found:   {len(state["concepts_extracted"])}')
    print(f'  Papers processed: {state.get("papers_processed", 0)}')

    return {
        'status': (
            f'Harvester complete: {len(state["papers_found"])} papers, '
            f'{len(state["concepts_extracted"])} concepts'
        )
    }


# ── Build the Graph ───────────────────────────────────────────────────────

def build_harvester(resources: dict):
    """
    Assembles the Harvester agent graph.
    Returns a compiled, runnable LangGraph app.
    """

    # Wrap each node to inject resources
    # LangGraph nodes only receive state — we use closures to pass resources
    def node_search(state):
        return search_papers(state, resources)

    def node_extract(state):
        return extract_concepts(state, resources)

    def node_insert(state):
        return insert_to_graph(state, resources)

    def node_status(state):
        return update_status(state, resources)

    graph = StateGraph(HarvesterState)

    graph.add_node('search_papers',    node_search)
    graph.add_node('extract_concepts', node_extract)
    graph.add_node('insert_to_graph',  node_insert)
    graph.add_node('update_status',    node_status)

    graph.add_edge('search_papers',    'extract_concepts')
    graph.add_edge('extract_concepts', 'insert_to_graph')
    graph.add_edge('insert_to_graph',  'update_status')
    graph.add_edge('update_status',    END)

    graph.set_entry_point('search_papers')

    return graph.compile()


# ── Run directly ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== APEX Harvester Agent ===\n')

    resources = get_resources()
    harvester = build_harvester(resources)

    # Run on a test query
    initial_state = {
        'query':              'graph neural networks for drug discovery',
        'papers_found':       [],
        'concepts_extracted': [],
        'papers_processed':   0,
        'status':             'starting',
        'error':              ''
    }

    final_state = harvester.invoke(initial_state)

    print(f'\n--- Final Status ---')
    print(f'{final_state["status"]}')

    # Clean up
    resources['neo4j'].close()
    resources['weaviate'].close()