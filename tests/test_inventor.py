# tests/test_inventor.py
# Unit tests for the Inventor agent's routing logic and simulation node.

from tests.conftest import FakeResult
from agents import inventor


def test_check_novelty_uses_real_semantic_search_not_self_reference(fake_neo4j, fake_weaviate, fake_embedder):
    fake_neo4j.fake_session.run.return_value = FakeResult(single={
        'statement': 'LLMs can predict protein folding.', 'rationale': 'r',
        'testability_score': 0.7, 'predicted_impact': 'i', 'debate_score': 0.8,
    })
    fake_weaviate.vector_search.return_value = [
        {'paper_id': 'arxiv:1', 'title': 'Close prior work', 'year': 2023, 'distance': 0.1},
        {'paper_id': 'arxiv:2', 'title': 'Distant prior work', 'year': 2021, 'distance': 0.6},
    ]
    resources = {'neo4j': fake_neo4j, 'weaviate': fake_weaviate, 'embedder': fake_embedder}
    state = {'hypothesis_id': 'hyp_1'}

    result = inventor.check_novelty(state, resources)

    assert len(result['prior_art']) == 2
    assert result['prior_art'][0]['similarity'] == 0.9
    assert result['novelty_score'] == 0.1  # 1 - top similarity
    fake_weaviate.vector_search.assert_called_once()
    # The old self-referential check queried other Hypothesis nodes sharing a
    # concept — confirm that Cypher shape is gone, not just that the score changed.
    for call in fake_neo4j.fake_session.run.call_args_list:
        assert 'h.id <> $hyp_id' not in call.args[0]


def test_check_novelty_defaults_to_fully_novel_when_no_prior_art(fake_neo4j, fake_weaviate, fake_embedder):
    fake_neo4j.fake_session.run.return_value = FakeResult(single={
        'statement': 'A totally new idea.', 'rationale': 'r',
        'testability_score': 0.7, 'predicted_impact': 'i', 'debate_score': 0.8,
    })
    fake_weaviate.vector_search.return_value = []
    resources = {'neo4j': fake_neo4j, 'weaviate': fake_weaviate, 'embedder': fake_embedder}
    state = {'hypothesis_id': 'hyp_1'}

    result = inventor.check_novelty(state, resources)

    assert result['prior_art'] == []
    assert result['novelty_score'] == 1.0


def test_should_draft_plan_when_novel_and_simulated_well():
    state = {'novelty_score': 0.7, 'sim_result': {'success_rate': 0.8}}
    assert inventor.should_draft_plan(state) == 'draft'


def test_should_draft_plan_skips_when_not_novel():
    state = {'novelty_score': 0.2, 'sim_result': {'success_rate': 0.8}}
    assert inventor.should_draft_plan(state) == 'skip'


def test_should_draft_plan_skips_when_simulation_weak():
    state = {'novelty_score': 0.9, 'sim_result': {'success_rate': 0.3}}
    assert inventor.should_draft_plan(state) == 'skip'


def test_run_simulation_is_deterministic_given_fixed_seed():
    resources = {}
    state = {'hypothesis_id': 'hyp_1', 'hypothesis': {'testability_score': 0.9},
             'novelty_score': 0.0, 'sim_result': {}, 'plan_draft': {}, 'plan_id': '',
             'status': '', 'error': ''}

    result_a = inventor.run_simulation(state, resources)
    result_b = inventor.run_simulation(state, resources)

    assert result_a['sim_result'] == result_b['sim_result']
    assert 0.0 <= result_a['sim_result']['success_rate'] <= 1.0


def test_skip_plan_sets_status():
    result = inventor.skip_plan({}, {})
    assert 'skipped' in result['status']


def test_draft_research_plan_persists_all_six_fields(fake_claude):
    from tests.conftest import make_claude_json_message

    fake_claude.messages.create.return_value = make_claude_json_message({
        'methodology_sketch': 'Run a controlled comparison.',
        'resources_needed': ['GPU', 'protein dataset'],
        'first_experiment': 'Fine-tune on a small subset first.',
        'key_related_papers': ['Paper A is closest but uses a different architecture.'],
        'open_risks': ['Data may be too sparse.'],
        'novelty_assessment': 'Meaningfully different in scope.',
    })
    resources = {'claude': fake_claude}
    state = {
        'hypothesis': {
            'statement': 'x', 'rationale': 'r', 'predicted_impact': 'i',
            'testability_score': 0.7, 'debate_score': 0.8, 'verdict': 'validated',
            'rebuttal': 'rb', 'counterarguments': ['c1', 'c2', 'c3'],
        },
        'prior_art': [{'title': 'Paper A', 'year': 2023, 'similarity': 0.4}],
    }

    result = inventor.draft_research_plan(state, resources)

    draft = result['plan_draft']
    for field in ('methodology_sketch', 'resources_needed', 'first_experiment',
                  'key_related_papers', 'open_risks', 'novelty_assessment'):
        assert field in draft and draft[field]


def test_store_research_plan_persists_all_fields_including_the_ones_patent_used_to_drop(fake_neo4j):
    resources = {'neo4j': fake_neo4j}
    state = {
        'hypothesis_id': 'hyp_1',
        'novelty_score': 0.6,
        'sim_result': {'success_rate': 0.7},
        'plan_draft': {
            'methodology_sketch': 'm', 'resources_needed': ['r1'], 'first_experiment': 'e',
            'key_related_papers': ['p1'], 'open_risks': ['risk1'], 'novelty_assessment': 'n',
        },
    }

    result = inventor.store_research_plan(state, resources)

    assert result['plan_id']
    merge_call = fake_neo4j.fake_session.run.call_args_list[0]
    for field in ('methodology_sketch', 'resources_needed', 'first_experiment',
                  'key_related_papers', 'open_risks', 'novelty_assessment'):
        assert field in merge_call.kwargs
