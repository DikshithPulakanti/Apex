# tests/test_reasoner.py
# Unit tests for the Reasoner agent's node logic and routing — DB/Claude mocked.

from tests.conftest import make_claude_json_message
from agents import reasoner


def test_select_seed_returns_gaps(fake_neo4j):
    fake_neo4j.find_research_gaps_for_seed.return_value = [
        {'concept1': 'automatic parallelization', 'concept2': 'large language models', 'gap_score': 0.55},
    ]
    resources = {'neo4j': fake_neo4j}
    state = {'seed_concept': 'large language models', 'gaps_found': [], 'candidates': [],
             'stored': [], 'status': '', 'error': ''}

    result = reasoner.select_seed(state, resources)

    assert len(result['gaps_found']) == 1
    fake_neo4j.find_research_gaps_for_seed.assert_called_once_with(
        'large language models', min_pagerank=0.1, limit=5
    )


def test_select_seed_handles_no_gaps(fake_neo4j):
    fake_neo4j.find_research_gaps_for_seed.return_value = []
    resources = {'neo4j': fake_neo4j}
    state = {'seed_concept': 'x', 'gaps_found': [], 'candidates': [], 'stored': [],
             'status': '', 'error': ''}

    result = reasoner.select_seed(state, resources)

    assert result['gaps_found'] == []
    assert result['error']


def test_gather_context_builds_one_candidate_per_gap(fake_weaviate, fake_embedder):
    resources = {'weaviate': fake_weaviate, 'embedder': fake_embedder}
    gaps = [
        {'concept1': 'a', 'concept2': 'b', 'gap_score': 0.5},
        {'concept1': 'c', 'concept2': 'd', 'gap_score': 0.4},
    ]
    state = {'seed_concept': 'x', 'gaps_found': gaps, 'candidates': [], 'stored': [],
             'status': '', 'error': ''}

    result = reasoner.gather_context(state, resources)

    assert len(result['candidates']) == 2
    assert result['candidates'][0]['gap'] == gaps[0]
    assert result['candidates'][1]['gap'] == gaps[1]
    assert fake_weaviate.hybrid_search.call_count == 2


def test_generate_hypothesis_parses_claude_response(fake_claude):
    fake_claude.messages.create.return_value = make_claude_json_message({
        'statement': 'Automatic parallelization techniques can accelerate LLM training.',
        'rationale': 'Both fields optimize compute graphs but rarely intersect.',
        'supporting_concepts': ['automatic parallelization', 'large language models'],
        'testability_score': 0.8,
        'predicted_impact': 'Faster training pipelines.',
    })
    resources = {'claude': fake_claude}
    gap = {'concept1': 'automatic parallelization', 'concept2': 'large language models',
           'gap_score': 0.55, 'pagerank1': 0.4, 'pagerank2': 0.6,
           'community1': 1, 'community2': 2, 'co_occurrence': 0}
    state = {
        'seed_concept': 'large language models', 'gaps_found': [gap],
        'candidates': [{'gap': gap, 'context_papers': [], 'hypothesis': reasoner.empty_hypothesis(), 'attempts': 0}],
        'stored': [], 'status': '', 'error': '',
    }

    result = reasoner.generate_hypothesis(state, resources)

    assert len(result['candidates']) == 1
    assert result['candidates'][0]['hypothesis']['testability_score'] == 0.8
    assert result['candidates'][0]['attempts'] == 1


def test_generate_hypothesis_with_no_candidates_short_circuits(fake_claude):
    resources = {'claude': fake_claude}
    state = {'seed_concept': 'x', 'gaps_found': [], 'candidates': [], 'stored': [],
             'status': '', 'error': ''}

    result = reasoner.generate_hypothesis(state, resources)

    fake_claude.messages.create.assert_not_called()
    assert result['candidates'] == []


def test_generate_hypothesis_one_bad_candidate_does_not_abort_others(fake_claude):
    good_response = make_claude_json_message({
        'statement': 'Good hypothesis.', 'rationale': 'r', 'supporting_concepts': [],
        'testability_score': 0.9, 'predicted_impact': 'i',
    })
    fake_claude.messages.create.side_effect = [Exception('boom'), good_response]
    resources = {'claude': fake_claude}
    gap1 = {'concept1': 'a', 'concept2': 'b', 'gap_score': 0.5}
    gap2 = {'concept1': 'c', 'concept2': 'd', 'gap_score': 0.5}
    state = {
        'seed_concept': 'x', 'gaps_found': [gap1, gap2],
        'candidates': [
            {'gap': gap1, 'context_papers': [], 'hypothesis': reasoner.empty_hypothesis(), 'attempts': 0},
            {'gap': gap2, 'context_papers': [], 'hypothesis': reasoner.empty_hypothesis(), 'attempts': 0},
        ],
        'stored': [], 'status': '', 'error': '',
    }

    result = reasoner.generate_hypothesis(state, resources)

    assert result['candidates'][0]['hypothesis']['statement'] == ''
    assert result['candidates'][1]['hypothesis']['statement'] == 'Good hypothesis.'


def test_store_hypothesis_stores_multiple_candidates_and_links_papers(fake_neo4j):
    resources = {'neo4j': fake_neo4j}
    candidates = [
        {
            'gap': {'concept1': 'a', 'concept2': 'b'},
            'context_papers': [{'paper_id': 'arxiv:1', 'score': 0.9}],
            'hypothesis': {
                'statement': 'First hypothesis.', 'rationale': 'r',
                'supporting_concepts': ['a', 'b'], 'testability_score': 0.8, 'predicted_impact': 'i',
            },
            'attempts': 1,
        },
        {
            'gap': {'concept1': 'c', 'concept2': 'd'},
            'context_papers': [],
            'hypothesis': reasoner.empty_hypothesis(),  # empty statement -> should be skipped
            'attempts': 3,
        },
    ]
    state = {'seed_concept': 'x', 'gaps_found': [], 'candidates': candidates, 'stored': [],
             'status': '', 'error': ''}

    result = reasoner.store_hypothesis(state, resources)

    assert len(result['stored']) == 1
    assert result['stored'][0]['hypothesis']['statement'] == 'First hypothesis.'
    # MERGE hypothesis + 2 DERIVED_FROM links + 1 CITES link = 4 writes
    assert fake_neo4j.fake_session.run.call_count == 4


def test_should_retry_hypothesis_stores_on_high_score():
    state = {'hypothesis': {'testability_score': 0.8}, 'attempts': 1}
    assert reasoner.should_retry_hypothesis(state) == 'store'


def test_should_retry_hypothesis_retries_on_low_score():
    state = {'hypothesis': {'testability_score': 0.3}, 'attempts': 1}
    assert reasoner.should_retry_hypothesis(state) == 'retry'


def test_should_retry_hypothesis_gives_up_after_max_attempts():
    state = {'hypothesis': {'testability_score': 0.3}, 'attempts': 3}
    assert reasoner.should_retry_hypothesis(state) == 'store'
