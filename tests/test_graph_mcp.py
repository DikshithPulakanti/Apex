# tests/test_graph_mcp.py
# Unit tests for graph-mcp tool handlers — Neo4j is mocked, no live DB needed.

import json

from tests.conftest import FakeResult
from mcp_servers import graph_mcp


async def test_find_research_gaps(fake_neo4j, monkeypatch):
    monkeypatch.setattr(graph_mcp, 'get_neo4j', lambda: fake_neo4j)
    fake_neo4j.find_research_gaps.return_value = [
        {'concept1': 'graph neural networks', 'concept2': 'drug discovery', 'gap_score': 0.42}
    ]

    result = await graph_mcp.handle_find_gaps({'min_pagerank': 0.3, 'limit': 5})
    data = json.loads(result[0].text)

    assert data['count'] == 1
    assert data['gaps'][0]['concept1'] == 'graph neural networks'
    fake_neo4j.find_research_gaps.assert_called_once_with(min_pagerank=0.3, limit=5)


async def test_create_hypothesis(fake_neo4j, monkeypatch):
    monkeypatch.setattr(graph_mcp, 'get_neo4j', lambda: fake_neo4j)

    result = await graph_mcp.handle_create_hypothesis({
        'statement': 'GNNs predict binding affinity better than docking.',
        'rationale': 'GNNs capture molecular topology traditional methods miss.',
        'supporting_concepts': ['graph neural networks', 'drug discovery'],
        'testability_score': 0.9,
    })
    data = json.loads(result[0].text)

    assert data['status'] == 'created'
    assert data['hypothesis_id'].startswith('hyp_')


async def test_get_graph_stats(fake_neo4j, monkeypatch):
    monkeypatch.setattr(graph_mcp, 'get_neo4j', lambda: fake_neo4j)
    fake_neo4j.get_stats.return_value = {'papers': 1490, 'authors': 300, 'relationships': 5000}
    fake_neo4j.fake_session.run.return_value = FakeResult(single={'n': 12})

    result = await graph_mcp.handle_get_stats({})
    data = json.loads(result[0].text)

    assert data['papers'] == 1490
    assert data['hypotheses'] == 12


async def test_get_top_concepts(fake_neo4j, monkeypatch):
    monkeypatch.setattr(graph_mcp, 'get_neo4j', lambda: fake_neo4j)
    fake_neo4j.fake_session.run.return_value = FakeResult(records=[
        {'name': 'transformers', 'pagerank': 0.8, 'community': 1, 'betweenness': 3.2},
        {'name': 'protein folding', 'pagerank': 0.6, 'community': 2, 'betweenness': 1.1},
    ])

    result = await graph_mcp.handle_get_top_concepts({'limit': 5})
    data = json.loads(result[0].text)

    assert len(data['concepts']) == 2
    assert data['concepts'][0]['name'] == 'transformers'


async def test_get_hypotheses(fake_neo4j, monkeypatch):
    monkeypatch.setattr(graph_mcp, 'get_neo4j', lambda: fake_neo4j)
    fake_neo4j.fake_session.run.return_value = FakeResult(records=[
        {'id': 'hyp_abc123', 'statement': 'Test hypothesis', 'testability_score': 0.8,
         'status': 'proposed', 'created_by': 'graph-mcp'},
    ])

    result = await graph_mcp.handle_get_hypotheses({'limit': 5})
    data = json.loads(result[0].text)

    assert len(data['hypotheses']) == 1
    assert data['hypotheses'][0]['id'] == 'hyp_abc123'
