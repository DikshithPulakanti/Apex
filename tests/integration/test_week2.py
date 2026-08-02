# tests/integration/test_week2.py
# Integration test — semantic layer (embeddings, Weaviate, concepts, GDS).
# Needs live Neo4j (with GDS scores written) + Weaviate.
# Run explicitly with: pytest -m integration

import pytest

from database.embedder import Embedder
from database.neo4j_client import Neo4jClient
from database.weaviate_client import WeaviateClient
from scrapers.concept_extractor import ConceptExtractor


@pytest.mark.integration
def test_semantic_layer():
    neo4j = Neo4jClient()
    weaviate = WeaviateClient()
    embedder = Embedder()
    extractor = ConceptExtractor()

    with neo4j.driver.session() as session:
        embedded_count = session.run(
            'MATCH (p:Paper) WHERE p.embedding IS NOT NULL RETURN count(p) AS n'
        ).single()['n']
    assert embedded_count > 0, 'Papers should have embeddings'

    weaviate_count = weaviate.get_paper_count()
    assert weaviate_count > 0, 'Weaviate should have papers'

    query_vec = embedder.embed_text('transformer attention mechanism')
    results = weaviate.vector_search(query_vec, limit=3)
    assert len(results) > 0, 'Vector search should return results'

    with neo4j.driver.session() as session:
        concept_count = session.run('MATCH (c:Concept) RETURN count(c) AS n').single()['n']
    assert concept_count > 0, 'Concept nodes should exist'

    with neo4j.driver.session() as session:
        cooccur_count = session.run(
            'MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) AS n'
        ).single()['n']
    assert cooccur_count > 0, 'Co-occurrence relationships should exist'

    with neo4j.driver.session() as session:
        scored_count = session.run(
            'MATCH (c:Concept) WHERE c.pagerank IS NOT NULL RETURN count(c) AS n'
        ).single()['n']
    assert scored_count > 0, 'Concepts should have PageRank scores'

    gaps = neo4j.find_research_gaps(min_pagerank=0.5, limit=5)
    assert len(gaps) > 0, 'Should find at least one research gap'

    concepts = extractor.extract_concepts(
        'graph neural networks for molecular property prediction in drug discovery'
    )
    assert len(concepts) > 0

    neo4j.close()
    weaviate.close()
