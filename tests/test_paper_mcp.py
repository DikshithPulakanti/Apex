# tests/test_paper_mcp.py
# Unit tests for paper-mcp tool handlers — Weaviate/Neo4j/Embedder mocked.

import json

from mcp_servers import paper_mcp


async def test_search_papers(fake_weaviate, fake_embedder, monkeypatch):
    monkeypatch.setattr(paper_mcp, 'get_embedder', lambda: fake_embedder)
    monkeypatch.setattr(paper_mcp, 'get_weaviate', lambda: fake_weaviate)
    fake_weaviate.hybrid_search.return_value = [
        {'paper_id': '2301.00001', 'title': 'Attention Is All You Need',
         'abstract': 'We propose a new architecture...', 'year': 2017, 'score': 0.93},
    ]

    result = await paper_mcp.handle_search_papers({'query': 'transformer attention mechanism', 'limit': 3})
    data = json.loads(result[0].text)

    assert data['count'] == 1
    assert data['papers'][0]['title'] == 'Attention Is All You Need'
    fake_embedder.embed_text.assert_called_once_with('transformer attention mechanism')


async def test_search_papers_no_results(fake_weaviate, fake_embedder, monkeypatch):
    monkeypatch.setattr(paper_mcp, 'get_embedder', lambda: fake_embedder)
    monkeypatch.setattr(paper_mcp, 'get_weaviate', lambda: fake_weaviate)
    fake_weaviate.hybrid_search.return_value = []

    result = await paper_mcp.handle_search_papers({'query': 'nonexistent topic'})
    data = json.loads(result[0].text)

    assert data['count'] == 0
    assert data['papers'] == []


async def test_get_paper_details_found(fake_neo4j, monkeypatch):
    monkeypatch.setattr(paper_mcp, 'get_neo4j', lambda: fake_neo4j)
    fake_neo4j.get_paper.return_value = {'id': '2301.00001', 'title': 'Test Paper', 'embedding': [0.1, 0.2]}

    result = await paper_mcp.handle_get_paper_details({'paper_id': '2301.00001'})
    data = json.loads(result[0].text)

    assert data['title'] == 'Test Paper'
    assert 'embedding' not in data


async def test_get_paper_details_not_found(fake_neo4j, monkeypatch):
    monkeypatch.setattr(paper_mcp, 'get_neo4j', lambda: fake_neo4j)
    fake_neo4j.get_paper.return_value = None

    result = await paper_mcp.handle_get_paper_details({'paper_id': 'missing'})

    assert 'not found' in result[0].text.lower()


async def test_get_paper_concepts(fake_neo4j, monkeypatch):
    monkeypatch.setattr(paper_mcp, 'get_neo4j', lambda: fake_neo4j)
    fake_neo4j.get_concepts_for_paper.return_value = ['graph neural networks', 'drug discovery']

    result = await paper_mcp.handle_get_paper_concepts({'paper_id': '2301.00001'})
    data = json.loads(result[0].text)

    assert data['concepts'] == ['graph neural networks', 'drug discovery']


async def test_get_papers_by_year(fake_neo4j, monkeypatch):
    monkeypatch.setattr(paper_mcp, 'get_neo4j', lambda: fake_neo4j)
    fake_neo4j.get_papers_by_year.return_value = [
        {'id': f'2026.{i:05d}', 'title': f'Paper {i}', 'embedding': [0.1]} for i in range(3)
    ]

    result = await paper_mcp.handle_get_papers_by_year({'year': 2026})
    data = json.loads(result[0].text)

    assert data['count'] == 3
    assert all('embedding' not in p for p in data['papers'])
