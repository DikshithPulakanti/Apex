# pipeline/load_weaviate.py
# Loads all papers from Neo4j into Weaviate for vector search

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.neo4j_client import Neo4jClient
from database.weaviate_client import WeaviateClient


def load_papers_to_weaviate(batch_size: int = 100):
    """
    Reads all papers with embeddings from Neo4j
    and loads them into Weaviate for vector search.
    """
    neo4j    = Neo4jClient()
    weaviate = WeaviateClient()
    weaviate.delete_collection()

    print('\n[LoadWeaviate] Starting...')

    # Count how many papers already in Weaviate
    existing = weaviate.get_paper_count()
    print(f'[LoadWeaviate] Papers already in Weaviate: {existing}')

    # Fetch all papers with embeddings from Neo4j
    print('[LoadWeaviate] Fetching papers from Neo4j...')
    query = """
        MATCH (p:Paper)
        WHERE p.embedding IS NOT NULL
        RETURN p
    """
    papers = []
    with neo4j.driver.session() as session:
        result = session.run(query)
        for record in result:
            papers.append(dict(record['p']))

    print(f'[LoadWeaviate] Found {len(papers)} papers with embeddings in Neo4j.')

    if not papers:
        print('[LoadWeaviate] No papers to load.')
        return

    # Insert in batches
    total_inserted = 0
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]

        weaviate_batch = []
        for p in batch:
            weaviate_batch.append({
                'paper_id':   p.get('id', ''),
                'title':      p.get('title', ''),
                'abstract':   p.get('abstract', ''),
                'year':       p.get('year', 0),
                'categories': p.get('categories', []),
                'embedding':  p.get('embedding', []),
            })

        weaviate.upsert_papers_batch(weaviate_batch)
        total_inserted += len(batch)
        print(f'[LoadWeaviate] Progress: {total_inserted}/{len(papers)}')

    print(f'\n✅ Done. Total papers in Weaviate: {weaviate.get_paper_count()}')

    neo4j.close()
    weaviate.close()


if __name__ == '__main__':
    load_papers_to_weaviate(batch_size=100)