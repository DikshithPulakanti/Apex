# tests/test_skeptic.py
# Unit tests for the Skeptic agent's node logic — BERT predictor, Claude,
# and MLflow are all mocked so these tests don't need live infra, don't
# spend real API tokens, and don't write to the real local MLflow store.

from unittest.mock import MagicMock

import pytest

from tests.conftest import FakeResult, make_claude_json_message
from agents import skeptic


@pytest.fixture(autouse=True)
def fake_mlflow(monkeypatch):
    """Autouse — score_debate always logs to MLflow, so every test here
    needs this mocked regardless of what it's actually testing."""
    fake = MagicMock()
    fake.start_run.return_value.__enter__.return_value.info.run_id = 'fake-run-id'
    monkeypatch.setattr(skeptic, 'mlflow', fake)
    return fake


def test_load_hypothesis_found(fake_neo4j):
    fake_neo4j.fake_session.run.return_value = FakeResult(single={
        'statement': 'Test hypothesis', 'rationale': 'Because reasons',
        'testability_score': 0.7, 'predicted_impact': 'Big impact',
    })
    resources = {'neo4j': fake_neo4j}
    state = {'hypothesis_id': 'hyp_1', 'hypothesis': {}, 'counterarguments': [],
             'rebuttal': '', 'debate_score': 0.0, 'rounds_completed': 0,
             'verdict': '', 'status': '', 'error': ''}

    result = skeptic.load_hypothesis(state, resources)

    assert result['hypothesis']['statement'] == 'Test hypothesis'
    assert result['status'] == 'hypothesis loaded'


def test_load_hypothesis_not_found(fake_neo4j):
    fake_neo4j.fake_session.run.return_value = FakeResult(single=None)
    resources = {'neo4j': fake_neo4j}
    state = {'hypothesis_id': 'missing', 'hypothesis': {}, 'counterarguments': [],
             'rebuttal': '', 'debate_score': 0.0, 'rounds_completed': 0,
             'verdict': '', 'status': '', 'error': ''}

    result = skeptic.load_hypothesis(state, resources)

    assert result['status'] == 'failed'
    assert 'not found' in result['error']


def _base_state(**overrides):
    state = {'hypothesis_id': 'hyp_1', 'hypothesis': {'statement': 'x'},
             'counterarguments': ['c1', 'c2', 'c3'], 'rebuttal': 'my rebuttal',
             'debate_score': 0.0, 'rounds_completed': 1, 'verdict': '', 'status': '', 'error': ''}
    state.update(overrides)
    return state


def test_score_debate_tier1_trusts_high_confidence_bert(fake_predictor, fake_claude):
    fake_predictor.predict.return_value = {'label': 1, 'confidence': 0.99, 'verdict': 'valid'}
    resources = {'predictor': fake_predictor, 'claude': fake_claude}

    result = skeptic.score_debate(_base_state(), resources)

    assert result['verdict'] == 'approved'
    assert result['scoring_method'] == 'bert_auto'
    fake_claude.messages.create.assert_not_called()


def test_score_debate_tier2_trusts_decisive_claude(fake_predictor, fake_claude):
    fake_predictor.predict.return_value = {'label': 1, 'confidence': 0.6, 'verdict': 'valid'}
    fake_claude.messages.create.return_value = make_claude_json_message({
        'score': 0.85, 'reasoning': 'Well supported by rebuttal.', 'verdict': 'approved',
    })
    resources = {'predictor': fake_predictor, 'claude': fake_claude}

    result = skeptic.score_debate(_base_state(), resources)

    assert result['verdict'] == 'approved'
    assert result['scoring_method'] == 'claude_auto'
    assert result['debate_score'] == 0.85
    fake_claude.messages.create.assert_called_once()


def test_score_debate_tier3_flags_ambiguous_claude_score_for_review(fake_predictor, fake_claude):
    fake_predictor.predict.return_value = {'label': 1, 'confidence': 0.6, 'verdict': 'valid'}
    fake_claude.messages.create.return_value = make_claude_json_message({
        'score': 0.5, 'reasoning': 'Mixed evidence, hard to call.', 'verdict': 'approved',
    })
    resources = {'predictor': fake_predictor, 'claude': fake_claude}

    result = skeptic.score_debate(_base_state(), resources)

    assert result['verdict'] == 'pending_review'
    assert result['scoring_method'] == 'human_review'


def test_score_debate_tier3_flags_score_verdict_disagreement_for_review(fake_predictor, fake_claude):
    fake_predictor.predict.return_value = {'label': 1, 'confidence': 0.6, 'verdict': 'valid'}
    # High score but verdict says rejected — inconsistent, shouldn't auto-decide.
    fake_claude.messages.create.return_value = make_claude_json_message({
        'score': 0.9, 'reasoning': 'Contradictory response.', 'verdict': 'rejected',
    })
    resources = {'predictor': fake_predictor, 'claude': fake_claude}

    result = skeptic.score_debate(_base_state(), resources)

    assert result['verdict'] == 'pending_review'


def test_score_debate_tier3_flags_claude_failure_for_review(fake_predictor, fake_claude):
    fake_predictor.predict.return_value = {'label': 1, 'confidence': 0.6, 'verdict': 'valid'}
    fake_claude.messages.create.side_effect = RuntimeError('API timeout')
    resources = {'predictor': fake_predictor, 'claude': fake_claude}

    result = skeptic.score_debate(_base_state(), resources)

    assert result['verdict'] == 'pending_review'
    assert result['scoring_method'] == 'human_review'
    assert 'API timeout' in result['claude_reasoning']


def test_update_hypothesis_status_writes_validated_verdict(fake_neo4j):
    resources = {'neo4j': fake_neo4j}
    state = _base_state(verdict='approved', debate_score=0.9, bert_confidence=0.99,
                         bert_verdict='valid', scoring_method='bert_auto')

    result = skeptic.update_hypothesis_status(state, resources)

    fake_neo4j.fake_session.run.assert_called_once()
    _, kwargs = fake_neo4j.fake_session.run.call_args
    assert kwargs['status'] == 'validated'
    assert kwargs['counterarguments'] == ['c1', 'c2', 'c3']
    assert 'validated' in result['status']


def test_update_hypothesis_status_enqueues_review_when_pending(fake_neo4j):
    fake_postgres = MagicMock()
    resources = {'neo4j': fake_neo4j, 'postgres': fake_postgres}
    state = _base_state(
        verdict='pending_review', debate_score=0.5, bert_confidence=0.6,
        bert_verdict='valid', claude_score=0.5, claude_reasoning='Mixed evidence.',
        scoring_method='human_review', mlflow_run_id='run-123',
    )

    result = skeptic.update_hypothesis_status(state, resources)

    _, kwargs = fake_neo4j.fake_session.run.call_args
    assert kwargs['status'] == 'pending_review'
    fake_postgres.enqueue_review.assert_called_once()
    _, review_kwargs = fake_postgres.enqueue_review.call_args
    assert review_kwargs['hypothesis_id'] == 'hyp_1'
    assert review_kwargs['claude_verdict'] == 'pending_review'
    assert 'pending_review' in result['status']
