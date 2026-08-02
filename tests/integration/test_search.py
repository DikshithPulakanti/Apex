# tests/integration/test_search.py
# Integration test — Weaviate hybrid/vector search. Needs live Weaviate
# with an already-populated Paper collection (docker-compose up + ingest).
# Run explicitly with: pytest -m integration

import pytest

from database.embedder import Embedder
from database.weaviate_client import WeaviateClient


@pytest.mark.integration
def test_vector_and_hybrid_search():
    client = WeaviateClient()
    embedder = Embedder()

    assert client.get_paper_count() > 0

    query_vec = embedder.embed_text('graph neural networks for drug discovery')
    results = client.vector_search(query_vec, limit=5)
    assert len(results) > 0

    hybrid_results = client.hybrid_search('large language models biology', query_vec, limit=5)
    assert len(hybrid_results) > 0

    keyword_only = client.hybrid_search('transformer attention', query_vec, limit=3, alpha=0.0)
    vector_only = client.hybrid_search('transformer attention', query_vec, limit=3, alpha=1.0)
    assert isinstance(keyword_only, list)
    assert isinstance(vector_only, list)

    client.close()
