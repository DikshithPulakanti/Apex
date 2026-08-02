# tests/test_inventor.py
# Unit tests for the Inventor agent's routing logic and simulation node.

from agents import inventor


def test_should_patent_when_novel_and_simulated_well():
    state = {'novelty_score': 0.7, 'sim_result': {'success_rate': 0.8}}
    assert inventor.should_patent(state) == 'draft'


def test_should_patent_skips_when_not_novel():
    state = {'novelty_score': 0.2, 'sim_result': {'success_rate': 0.8}}
    assert inventor.should_patent(state) == 'skip'


def test_should_patent_skips_when_simulation_weak():
    state = {'novelty_score': 0.9, 'sim_result': {'success_rate': 0.3}}
    assert inventor.should_patent(state) == 'skip'


def test_run_simulation_is_deterministic_given_fixed_seed():
    resources = {}
    state = {'hypothesis_id': 'hyp_1', 'hypothesis': {'testability_score': 0.9},
             'novelty_score': 0.0, 'sim_result': {}, 'patent_draft': {}, 'patent_id': '',
             'status': '', 'error': ''}

    result_a = inventor.run_simulation(state, resources)
    result_b = inventor.run_simulation(state, resources)

    assert result_a['sim_result'] == result_b['sim_result']
    assert 0.0 <= result_a['sim_result']['success_rate'] <= 1.0


def test_skip_patent_sets_status():
    result = inventor.skip_patent({}, {})
    assert 'skipped' in result['status']
