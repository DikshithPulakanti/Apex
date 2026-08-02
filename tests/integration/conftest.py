# tests/integration/conftest.py
# Seeds a small amount of synthetic-but-real data (real embeddings, real
# Cypher co-occurrence) before the integration suite runs, so test_search.py
# and test_week2.py don't silently depend on someone having already run the
# full ingest -> extract_concepts -> run_gds pipeline against this database.
# On a fresh CI-provisioned Neo4j/Weaviate, that data would never exist
# otherwise. Every id/name is prefixed `seedtest` and every write is a
# MERGE, so this is also safe to run against an already-populated dev
# database — it just adds a handful of obviously-fake nodes alongside the
# real ones.

import pytest

from database.embedder import Embedder
from database.neo4j_client import Neo4jClient
from database.weaviate_client import WeaviateClient

SEED_PAPERS = [
    {
        'id': 'seedtest:paper:1',
        'title': 'Graph Neural Networks for Drug Discovery',
        'abstract': 'We apply graph neural networks to predict drug-target binding affinity, encoding molecular structure as a graph.',
        'year': 2024,
        'categories': ['cs.LG'],
        'concepts': ['seedtest-graph-neural-networks', 'seedtest-drug-discovery'],
    },
    {
        'id': 'seedtest:paper:2',
        'title': 'Large Language Models for Biological Sequence Understanding',
        'abstract': 'Large language models pretrained on protein sequences learn representations useful for biological structure prediction.',
        'year': 2024,
        'categories': ['cs.CL'],
        'concepts': ['seedtest-large-language-models', 'seedtest-protein-sequences'],
    },
    {
        'id': 'seedtest:paper:3',
        'title': 'Transformer Attention Mechanisms for Molecular Property Prediction',
        'abstract': 'Transformer attention mechanisms applied to molecular property prediction improve accuracy over prior graph-based methods.',
        'year': 2024,
        'categories': ['cs.LG'],
        'concepts': ['seedtest-transformer-attention', 'seedtest-molecular-property-prediction'],
    },
]

# Two synthetic "communities" — concepts that share a paper above also share
# a community, and the two clusters never co-occur with each other. That's
# exactly the cross-community, high-pagerank, low-co-occurrence shape
# find_research_gaps looks for, so it's guaranteed to find at least one gap.
CONCEPT_COMMUNITIES = {
    'seedtest-graph-neural-networks':         (1, 0.85),
    'seedtest-drug-discovery':                (1, 0.75),
    'seedtest-large-language-models':         (2, 0.80),
    'seedtest-protein-sequences':              (2, 0.70),
    'seedtest-transformer-attention':          (1, 0.65),
    'seedtest-molecular-property-prediction':  (2, 0.60),
}


@pytest.fixture(scope='session', autouse=True)
def seed_integration_data():
    """Runs once before the integration suite. Inserts synthetic papers with
    real embeddings into Neo4j + Weaviate, links synthetic concepts to them,
    builds real co-occurrence edges, and manually assigns pagerank/community
    (standing in for a full GDS run, which is overkill on a 6-node graph)."""
    neo4j = Neo4jClient()
    weaviate = WeaviateClient()
    embedder = Embedder()

    weaviate_batch = []
    for paper in SEED_PAPERS:
        embedding = embedder.embed_text(paper['abstract'])

        neo4j.upsert_paper(paper)
        neo4j.set_paper_embedding(paper['id'], embedding)

        for concept in paper['concepts']:
            neo4j.upsert_concept(concept, domain='seedtest')
            neo4j.link_paper_to_concept(paper['id'], concept)

        weaviate_batch.append({
            'paper_id': paper['id'],
            'title': paper['title'],
            'abstract': paper['abstract'],
            'year': paper['year'],
            'categories': paper['categories'],
            'embedding': embedding,
        })

    weaviate.upsert_papers_batch(weaviate_batch)
    neo4j.build_concept_cooccurrence()

    with neo4j.driver.session() as session:
        for concept, (community, pagerank) in CONCEPT_COMMUNITIES.items():
            session.run(
                'MATCH (c:Concept {name: $name}) SET c.community = $community, c.pagerank = $pagerank',
                name=concept, community=community, pagerank=pagerank,
            )

    neo4j.close()
    weaviate.close()

    yield
