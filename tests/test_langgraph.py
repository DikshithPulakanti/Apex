# tests/test_langgraph.py
# LangGraph fundamentals demo — a toy graph exercising nodes, conditional
# edges, and loops with zero external dependencies. Doubles as a smoke
# test that a langgraph upgrade hasn't changed core StateGraph semantics.

from typing import TypedDict

from langgraph.graph import END, StateGraph


class ResearchState(TypedDict):
    topic: str
    papers_found: int
    hypothesis: str
    score: float
    attempts: int
    status: str


def search_papers(state: ResearchState) -> dict:
    papers = len(state['topic']) * 3
    return {'papers_found': papers, 'status': f'Found {papers} papers'}


def generate_hypothesis(state: ResearchState) -> dict:
    attempts = state.get('attempts', 0) + 1
    hypothesis = f'Hypothesis #{attempts}: {state["topic"]} leads to novel discoveries'
    score = min(0.4 + (attempts * 0.3), 1.0)
    return {
        'hypothesis': hypothesis, 'score': score, 'attempts': attempts,
        'status': f'Generated hypothesis with score {score:.2f}',
    }


def store_hypothesis(state: ResearchState) -> dict:
    return {'status': 'Hypothesis stored successfully'}


def should_retry(state: ResearchState) -> str:
    if state['score'] >= 0.7:
        return 'store'
    elif state.get('attempts', 0) >= 3:
        return 'store'
    else:
        return 'retry'


def build_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node('search_papers', search_papers)
    graph.add_node('generate_hypothesis', generate_hypothesis)
    graph.add_node('store_hypothesis', store_hypothesis)
    graph.add_edge('search_papers', 'generate_hypothesis')
    graph.add_conditional_edges(
        'generate_hypothesis', should_retry,
        {'store': 'store_hypothesis', 'retry': 'generate_hypothesis'},
    )
    graph.add_edge('store_hypothesis', END)
    graph.set_entry_point('search_papers')
    return graph.compile()


def test_langgraph_demo_reaches_acceptable_score():
    app = build_research_graph()
    final_state = app.invoke({
        'topic': 'graph neural networks for drug discovery',
        'papers_found': 0, 'hypothesis': '', 'score': 0.0, 'attempts': 0, 'status': 'starting',
    })

    assert final_state['score'] >= 0.7
    assert final_state['attempts'] > 0
