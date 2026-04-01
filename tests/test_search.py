# tests/test_search.py
# Test Weaviate semantic search

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.weaviate_client import WeaviateClient
from database.embedder import Embedder


if __name__ == '__main__':
    client   = WeaviateClient()
    embedder = Embedder()

    print(f'\nTotal papers in Weaviate: {client.get_paper_count()}')

    # Test 1 — vector search
    print('\n--- Vector Search: "graph neural networks for drug discovery" ---')
    query_vec = embedder.embed_text('graph neural networks for drug discovery')
    results   = client.vector_search(query_vec, limit=5)
    for r in results:
        print(f'  [{r["year"]}] {r["title"][:70]} (dist: {r["distance"]:.3f})')

    # Test 2 — hybrid search
    print('\n--- Hybrid Search: "large language models biology" ---')
    query_vec2 = embedder.embed_text('large language models biology')
    results2   = client.hybrid_search('large language models biology', query_vec2, limit=5)
    for r in results2:
        print(f'  [{r["year"]}] {r["title"][:70]} (score: {r["score"]:.3f})')

    # Test 3 — compare alpha values
    print('\n--- Pure keyword search (alpha=0.0) ---')
    results3 = client.hybrid_search('transformer attention', query_vec, limit=3, alpha=0.0)
    for r in results3:
        print(f'  {r["title"][:70]}')

    print('\n--- Pure vector search (alpha=1.0) ---')
    results4 = client.hybrid_search('transformer attention', query_vec, limit=3, alpha=1.0)
    for r in results4:
        print(f'  {r["title"][:70]}')

    print('\n✅ Search tests complete.')
    client.close()