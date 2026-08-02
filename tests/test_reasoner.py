# tests/test_reasoner.py
# Unit tests for the Reasoner agent's node logic and routing — DB/Claude mocked.

from tests.conftest import make_claude_json_message
from agents import reasoner


def test_select_seed_returns_gaps(fake_neo4j):
    fake_neo4j.find_research_gaps_for_seed.return_value = [
        {'concept1': 'automatic parallelization', 'concept2': 'large language models', 'gap_score': 0.55},
    ]
    resources = {'neo4j': fake_neo4j}
    state = {'seed_concept': 'large language models', 'gaps_found': [], 'context_papers': [],
             'hypothesis': reasoner.empty_hypothesis(), 'hypothesis_id': '', 'attempts': 0,
             'status': '', 'error': ''}

    result = reasoner.select_seed(state, resources)

    assert len(result['gaps_found']) == 1
    fake_neo4j.find_research_gaps_for_seed.assert_called_once_with(
        'large language models', min_pagerank=0.1, limit=5
    )


def test_select_seed_handles_no_gaps(fake_neo4j):
    fake_neo4j.find_research_gaps_for_seed.return_value = []
    resources = {'neo4j': fake_neo4j}
    state = {'seed_concept': 'x', 'gaps_found': [], 'context_papers': [],
             'hypothesis': reasoner.empty_hypothesis(), 'hypothesis_id': '', 'attempts': 0,
             'status': '', 'error': ''}

    result = reasoner.select_seed(state, resources)

    assert result['gaps_found'] == []
    assert result['error']


def test_generate_hypothesis_parses_claude_response(fake_claude):
    fake_claude.messages.create.return_value = make_claude_json_message({
        'statement': 'Automatic parallelization techniques can accelerate LLM training.',
        'rationale': 'Both fields optimize compute graphs but rarely intersect.',
        'supporting_concepts': ['automatic parallelization', 'large language models'],
        'testability_score': 0.8,
        'predicted_impact': 'Faster training pipelines.',
    })
    resources = {'claude': fake_claude}
    state = {
        'seed_concept': 'large language models',
        'gaps_found': [{'concept1': 'automatic parallelization', 'concept2': 'large language models',
                         'gap_score': 0.55, 'pagerank1': 0.4, 'pagerank2': 0.6,
                         'community1': 1, 'community2': 2, 'co_occurrence': 0}],
        'context_papers': [], 'hypothesis': reasoner.empty_hypothesis(), 'hypothesis_id': '',
        'attempts': 0, 'status': '', 'error': '',
    }

    result = reasoner.generate_hypothesis(state, resources)

    assert result['hypothesis']['testability_score'] == 0.8
    assert result['attempts'] == 1


def test_generate_hypothesis_with_no_gaps_short_circuits(fake_claude):
    resources = {'claude': fake_claude}
    state = {'seed_concept': 'x', 'gaps_found': [], 'context_papers': [],
             'hypothesis': reasoner.empty_hypothesis(), 'hypothesis_id': '', 'attempts': 0,
             'status': '', 'error': ''}

    result = reasoner.generate_hypothesis(state, resources)

    fake_claude.messages.create.assert_not_called()
    assert result['hypothesis']['statement'] == ''


def test_should_retry_hypothesis_stores_on_high_score():
    state = {'hypothesis': {'testability_score': 0.8}, 'attempts': 1}
    assert reasoner.should_retry_hypothesis(state) == 'store'


def test_should_retry_hypothesis_retries_on_low_score():
    state = {'hypothesis': {'testability_score': 0.3}, 'attempts': 1}
    assert reasoner.should_retry_hypothesis(state) == 'retry'


def test_should_retry_hypothesis_gives_up_after_max_attempts():
    state = {'hypothesis': {'testability_score': 0.3}, 'attempts': 3}
    assert reasoner.should_retry_hypothesis(state) == 'store'
