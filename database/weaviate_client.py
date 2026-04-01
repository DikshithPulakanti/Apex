# database/weaviate_client.py
# APEX Weaviate Client — vector search layer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv

load_dotenv()


class WeaviateClient:
    """
    Handles all vector search operations for APEX.

    Weaviate stores paper embeddings and enables two types of search:
    1. Vector search  — find semantically similar papers
    2. Hybrid search  — combine vector + keyword search
    """

    COLLECTION_NAME = 'Paper'

    def __init__(self):
        self.client = weaviate.connect_to_local(
            host = os.getenv('WEAVIATE_HOST', 'localhost'),
            port = int(os.getenv('WEAVIATE_PORT', '8080')),
        )
        print('[WeaviateClient] Connected to Weaviate.')
        self._create_collection()

    def _create_collection(self):
        """
        Creates the Paper collection in Weaviate if it doesn't exist.
        """
        if self.client.collections.exists(self.COLLECTION_NAME):
            print(f'[WeaviateClient] Collection "{self.COLLECTION_NAME}" already exists.')
            return

        self.client.collections.create(
            name = self.COLLECTION_NAME,
            vectorizer_config = wvc.config.Configure.Vectorizer.none(),
            properties = [
                wvc.config.Property(name='paper_id',   data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name='title',      data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name='abstract',   data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name='year',       data_type=wvc.config.DataType.INT),
                wvc.config.Property(name='categories', data_type=wvc.config.DataType.TEXT_ARRAY),
            ]
        )
        print(f'[WeaviateClient] Collection "{self.COLLECTION_NAME}" created.')

    def upsert_paper(self, paper_id: str, title: str, abstract: str,
                     year: int, categories: list, embedding: list) -> None:
        """Inserts a paper in Weaviate with its embedding vector."""
        collection = self.client.collections.get(self.COLLECTION_NAME)
        collection.data.insert(
            properties = {
                'paper_id':   paper_id,
                'title':      title,
                'abstract':   abstract,
                'year':       year,
                'categories': categories,
            },
            vector = embedding
        )

    def upsert_papers_batch(self, papers: list) -> int:
        """
        Inserts many papers at once.

        PARAMETERS:
            papers: list of dicts with:
                    paper_id, title, abstract, year, categories, embedding
        RETURNS:
            number of papers inserted
        """
        collection = self.client.collections.get(self.COLLECTION_NAME)

        with collection.batch.dynamic() as batch:
            for paper in papers:
                batch.add_object(
                    properties = {
                        'paper_id':   paper['paper_id'],
                        'title':      paper['title'],
                        'abstract':   paper['abstract'],
                        'year':       paper['year'],
                        'categories': paper.get('categories', []),
                    },
                    vector = paper['embedding']
                )

        print(f'[WeaviateClient] Inserted {len(papers)} papers.')
        return len(papers)

    def vector_search(self, query_vector: list, limit: int = 10) -> list:
        """
        Finds papers most similar to the query vector.
        Pure semantic search — meaning based, not keyword based.
        """
        collection = self.client.collections.get(self.COLLECTION_NAME)
        results    = collection.query.near_vector(
            near_vector     = query_vector,
            limit           = limit,
            return_metadata = wvc.query.MetadataQuery(distance=True)
        )

        papers = []
        for obj in results.objects:
            paper = dict(obj.properties)
            paper['distance'] = obj.metadata.distance
            papers.append(paper)

        return papers

    def hybrid_search(self, query_text: str, query_vector: list,
                      limit: int = 10, alpha: float = 0.5) -> list:
        """
        Combines vector search + keyword search.

        alpha=1.0 → pure vector search
        alpha=0.0 → pure keyword search
        alpha=0.5 → equal mix
        """
        collection = self.client.collections.get(self.COLLECTION_NAME)
        results    = collection.query.hybrid(
            query           = query_text,
            vector          = query_vector,
            limit           = limit,
            alpha           = alpha,
            return_metadata = wvc.query.MetadataQuery(score=True)
        )

        papers = []
        for obj in results.objects:
            paper = dict(obj.properties)
            paper['score'] = obj.metadata.score
            papers.append(paper)

        return papers

    def get_paper_count(self) -> int:
        """Returns total number of papers in Weaviate."""
        collection = self.client.collections.get(self.COLLECTION_NAME)
        result     = collection.aggregate.over_all(total_count=True)
        return result.total_count

    def delete_collection(self) -> None:
        """Deletes and recreates the Paper collection. Use to clear all data."""
        if self.client.collections.exists(self.COLLECTION_NAME):
            self.client.collections.delete(self.COLLECTION_NAME)
            print(f'[WeaviateClient] Collection "{self.COLLECTION_NAME}" deleted.')
        self._create_collection()

    def close(self):
        self.client.close()
        print('[WeaviateClient] Connection closed.')


# ── Test it directly ───────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from embedder import Embedder

    client   = WeaviateClient()
    embedder = Embedder()

    print('\n--- Inserting 3 test papers ---')
    test_papers = [
        {
            'paper_id':   'test:w001',
            'title':      'Attention Is All You Need',
            'abstract':   'We propose a new architecture based on attention mechanisms.',
            'year':       2017,
            'categories': ['cs.LG', 'cs.AI'],
            'embedding':  embedder.embed_text('attention mechanism transformer architecture')
        },
        {
            'paper_id':   'test:w002',
            'title':      'BERT: Pre-training of Deep Bidirectional Transformers',
            'abstract':   'We introduce BERT, a language representation model.',
            'year':       2018,
            'categories': ['cs.CL'],
            'embedding':  embedder.embed_text('BERT language model pre-training')
        },
        {
            'paper_id':   'test:w003',
            'title':      'Protein Structure Prediction with AlphaFold',
            'abstract':   'We present AlphaFold, a system for protein structure prediction.',
            'year':       2021,
            'categories': ['q-bio.BM'],
            'embedding':  embedder.embed_text('protein structure prediction deep learning')
        },
    ]
    client.upsert_papers_batch(test_papers)
    print(f'  Total papers in Weaviate: {client.get_paper_count()}')

    print('\n--- Testing vector search ---')
    query_vec = embedder.embed_text('transformer attention neural network')
    results   = client.vector_search(query_vec, limit=2)
    print(f'  Found {len(results)} results:')
    for r in results:
        print(f'  → {r["title"]} (distance: {r["distance"]:.4f})')

    assert len(results) > 0
    print('  ✓ Vector search working')

    print('\n--- Testing hybrid search ---')
    results = client.hybrid_search('attention transformer', query_vec, limit=2)
    print(f'  Found {len(results)} results:')
    for r in results:
        print(f'  → {r["title"]} (score: {r["score"]:.4f})')

    assert len(results) > 0
    print('  ✓ Hybrid search working')

    print('\n✅ Weaviate client working correctly.')
    client.close()