# tests/test_week2.py
# Week 2 integration test — semantic layer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.neo4j_client import Neo4jClient
from database.weaviate_client import WeaviateClient
from database.embedder import Embedder
from scrapers.concept_extractor import ConceptExtractor


if __name__ == '__main__':
    print('=== APEX Week 2 Integration Test ===\n')

    neo4j    = Neo4jClient()
    weaviate = WeaviateClient()
    embedder = Embedder()
    extractor = ConceptExtractor()

    # ── Test 1: Embeddings exist on papers ───────────────────────────────
    print('[Test 1] Paper embeddings...')
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            WHERE p.embedding IS NOT NULL
            RETURN count(p) AS n
        """)
        embedded_count = result.single()['n']
    assert embedded_count > 0, 'Papers should have embeddings'
    print(f'  ✓ {embedded_count} papers have embeddings')

    # ── Test 2: Weaviate has papers ───────────────────────────────────────
    print('[Test 2] Weaviate vector store...')
    weaviate_count = weaviate.get_paper_count()
    assert weaviate_count > 0, 'Weaviate should have papers'
    print(f'  ✓ {weaviate_count} papers in Weaviate')

    # ── Test 3: Semantic search works ────────────────────────────────────
    print('[Test 3] Semantic search...')
    query_vec = embedder.embed_text('transformer attention mechanism')
    results   = weaviate.vector_search(query_vec, limit=3)
    assert len(results) > 0, 'Vector search should return results'
    print(f'  ✓ Vector search returned {len(results)} results')
    print(f'  Top result: {results[0]["title"][:60]}')

    # ── Test 4: Concept nodes exist ───────────────────────────────────────
    print('[Test 4] Concept nodes...')
    with neo4j.driver.session() as session:
        result = session.run('MATCH (c:Concept) RETURN count(c) AS n')
        concept_count = result.single()['n']
    assert concept_count > 0, 'Concept nodes should exist'
    print(f'  ✓ {concept_count} concept nodes in Neo4j')

    # ── Test 5: Co-occurrence relationships exist ─────────────────────────
    print('[Test 5] Co-occurrence graph...')
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH ()-[r:CO_OCCURS_WITH]->()
            RETURN count(r) AS n
        """)
        cooccur_count = result.single()['n']
    assert cooccur_count > 0, 'Co-occurrence relationships should exist'
    print(f'  ✓ {cooccur_count} co-occurrence relationships')

    # ── Test 6: GDS scores exist ──────────────────────────────────────────
    print('[Test 6] GDS scores...')
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (c:Concept)
            WHERE c.pagerank IS NOT NULL
            RETURN count(c) AS n
        """)
        scored_count = result.single()['n']
    assert scored_count > 0, 'Concepts should have PageRank scores'
    print(f'  ✓ {scored_count} concepts have PageRank scores')

    # ── Test 7: Research gaps detectable ─────────────────────────────────
    print('[Test 7] Research gap detection...')
    gaps = neo4j.find_research_gaps(min_pagerank=0.5, limit=5)
    assert len(gaps) > 0, 'Should find at least one research gap'
    print(f'  ✓ Found {len(gaps)} research gaps')
    print(f'  Top gap: {gaps[0]["concept1"]} ↔ {gaps[0]["concept2"]}')

    # ── Test 8: Concept extraction works ─────────────────────────────────
    print('[Test 8] Concept extractor...')
    concepts = extractor.extract_concepts(
        'graph neural networks for molecular property prediction in drug discovery'
    )
    assert len(concepts) > 0
    print(f'  ✓ Extracted {len(concepts)} concepts: {concepts[:2]}')

    # ── Summary ───────────────────────────────────────────────────────────
    print('\n=== Week 2 Stats ===')
    stats = neo4j.get_stats()
    print(f'  Papers:        {stats["papers"]}')
    print(f'  Authors:       {stats["authors"]}')
    print(f'  Relationships: {stats["relationships"]}')
    print(f'  Concepts:      {concept_count}')
    print(f'  Weaviate:      {weaviate_count} papers')

    print('\n✅ All Week 2 tests passed.')

    neo4j.close()
    weaviate.close()