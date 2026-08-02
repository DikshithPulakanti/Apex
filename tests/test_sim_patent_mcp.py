# tests/test_sim_patent_mcp.py
# Unit tests for sim-mcp (pure compute, no mocking needed) and patent-mcp
# (Claude + Neo4j mocked) tool handlers.

import json

from tests.conftest import make_claude_json_message, FakeResult
from mcp_servers import sim_mcp, patent_mcp

HYPOTHESIS = (
    'Graph neural networks can predict drug-protein binding affinity more '
    'accurately than traditional docking methods by encoding molecular topology.'
)
CONCEPTS = ['graph neural networks', 'drug discovery', 'molecular property prediction']


# ── sim-mcp ──────────────────────────────────────────────────────────────

async def test_run_hypothesis_simulation():
    result = await sim_mcp.handle_simulation({
        'hypothesis_statement': HYPOTHESIS,
        'testability_score': 0.9,
        'n_simulations': 1000,
    })
    data = json.loads(result[0].text)

    assert data['n_simulations'] == 1000
    assert 0.0 <= data['success_rate'] <= 1.0
    assert data['confidence_interval']['lower'] <= data['confidence_interval']['upper']


async def test_generate_synthetic_data():
    result = await sim_mcp.handle_synthetic_data({'hypothesis_statement': HYPOTHESIS, 'n_samples': 50})
    data = json.loads(result[0].text)

    assert data['n_samples'] == 50
    assert len(data['sample_data']) == 5


async def test_validate_against_known_flags_absolute_claims():
    result = await sim_mcp.handle_validation({
        'hypothesis_statement': 'This will always work and can never fail.',
        'supporting_concepts': [],
    })
    data = json.loads(result[0].text)

    assert data['contradiction_score'] > 0.3
    assert 'no supporting concepts provided' in data['flags']


# ── patent-mcp ───────────────────────────────────────────────────────────

async def test_draft_patent_claims(fake_claude, monkeypatch):
    monkeypatch.setattr(patent_mcp, 'get_claude', lambda: fake_claude)
    fake_claude.messages.create.return_value = make_claude_json_message({
        'title': 'Method for Predicting Binding Affinity via Graph Neural Networks',
        'background': '...',
        'summary': '...',
        'independent_claim_1': 'A method comprising...',
        'dependent_claim_2': '...',
        'dependent_claim_3': '...',
        'abstract': '...',
    })

    result = await patent_mcp.handle_draft_patent({
        'hypothesis_statement': HYPOTHESIS,
        'supporting_concepts': CONCEPTS,
        'predicted_impact': 'Could accelerate drug discovery by 10x.',
    })
    data = json.loads(result[0].text)

    assert 'title' in data
    assert data['status'] == 'draft'


async def test_check_prior_art_low_risk(fake_neo4j, monkeypatch):
    monkeypatch.setattr(patent_mcp, 'get_neo4j', lambda: fake_neo4j)
    fake_neo4j.fake_session.run.return_value = FakeResult(records=[])

    result = await patent_mcp.handle_prior_art({'invention_description': HYPOTHESIS})
    data = json.loads(result[0].text)

    assert data['prior_art_risk'] == 'low'


async def test_compute_novelty_score(fake_neo4j, monkeypatch):
    monkeypatch.setattr(patent_mcp, 'get_neo4j', lambda: fake_neo4j)
    fake_neo4j.fake_session.run.return_value = FakeResult(single={'n': 0})

    result = await patent_mcp.handle_novelty_score({
        'hypothesis_statement': HYPOTHESIS,
        'supporting_concepts': CONCEPTS,
    })
    data = json.loads(result[0].text)

    assert 0.0 <= data['novelty_score'] <= 1.0
    assert data['shared_concepts'] == 0
