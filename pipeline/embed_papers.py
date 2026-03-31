# pipeline/embed_papers.py
# Embeds paper abstracts and stores vectors in Neo4j

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.neo4j_client import Neo4jClient
from database.embedder import Embedder


def embed_papers(batch_size: int = 100):
    """
    Fetches papers without embeddings from Neo4j,
    embeds their abstracts, and stores the vectors back.
    """
    neo4j   = Neo4jClient()
    embedder = Embedder()

    total_embedded = 0

    print('\n[EmbedPipeline] Starting embedding pipeline...')

    while True:
        # Step 1: Get papers without embeddings
        papers = neo4j.get_papers_without_embeddings(limit=batch_size)

        if not papers:
            print('[EmbedPipeline] All papers embedded.')
            break

        print(f'[EmbedPipeline] Embedding {len(papers)} papers...')

        # Step 2: Extract abstracts
        abstracts = [p.get('abstract', '') or p.get('title', '') for p in papers]

        # Step 3: Embed all abstracts in one batch — fast
        embeddings = embedder.embed_batch(abstracts)

        # Step 4: Store each embedding back on its Paper node
        for paper, embedding in zip(papers, embeddings):
            neo4j.set_paper_embedding(paper['id'], embedding)

        total_embedded += len(papers)
        print(f'[EmbedPipeline] Progress: {total_embedded} papers embedded so far.')

    print(f'\n✅ Embedding complete. Total: {total_embedded} papers embedded.')
    neo4j.close()


if __name__ == '__main__':
    embed_papers(batch_size=100)