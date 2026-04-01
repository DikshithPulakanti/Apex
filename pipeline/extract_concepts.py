# pipeline/extract_concepts.py
# Extracts concepts from papers and stores them in Neo4j

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.neo4j_client import Neo4jClient
from scrapers.concept_extractor import ConceptExtractor


def extract_and_store_concepts(batch_size: int = 50):
    """
    Fetches papers from Neo4j, extracts concepts from their abstracts,
    creates Concept nodes, and links papers to their concepts.
    """
    neo4j     = Neo4jClient()
    extractor = ConceptExtractor()

    print('\n[ConceptPipeline] Starting concept extraction...')

    # Get papers that don't have concepts yet
    query = """
        MATCH (p:Paper)
        WHERE NOT (p)-[:MENTIONS]->(:Concept)
        AND p.abstract IS NOT NULL
        RETURN p.id AS id, p.title AS title, p.abstract AS abstract
        LIMIT $limit
    """

    total_processed = 0
    total_concepts  = 0

    while True:
        with neo4j.driver.session() as session:
            result  = session.run(query, limit=batch_size)
            papers  = [dict(r) for r in result]

        if not papers:
            print('[ConceptPipeline] All papers processed.')
            break

        print(f'[ConceptPipeline] Processing {len(papers)} papers...')

        # Extract concepts in batch
        abstracts = [p['abstract'] for p in papers]
        all_concepts = extractor.extract_batch(abstracts, max_concepts=8)

        # Store concepts and link to papers
        for paper, concepts in zip(papers, all_concepts):
            for concept in concepts:
                neo4j.upsert_concept(concept, domain='extracted')
                neo4j.link_paper_to_concept(paper['id'], concept)
                total_concepts += 1

        total_processed += len(papers)
        print(f'[ConceptPipeline] Progress: {total_processed} papers, '
              f'{total_concepts} concept links created.')

    print(f'\n✅ Concept extraction complete.')
    print(f'   Papers processed : {total_processed}')
    print(f'   Concept links    : {total_concepts}')

    stats = neo4j.get_stats()
    neo4j.close()


if __name__ == '__main__':
    extract_and_store_concepts(batch_size=50)