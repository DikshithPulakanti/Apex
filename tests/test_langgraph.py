# tests/test_langgraph.py
# LangGraph fundamentals demo

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langgraph.graph import StateGraph, END


# ── Step 1: Define State ──────────────────────────────────────────────────
# State is a dictionary that flows through every node.
# TypedDict tells Python exactly what keys and types the state has.

class ResearchState(TypedDict):
    topic: str           # what we're researching
    papers_found: int    # how many papers found so far
    hypothesis: str      # the hypothesis generated
    score: float         # quality score 0-1
    attempts: int        # how many times we've tried
    status: str          # current status message


# ── Step 2: Define Nodes ──────────────────────────────────────────────────
# Each node is a function that takes state and returns updated state.
# It only needs to return the keys it changed.

def search_papers(state: ResearchState) -> dict:
    """
    Simulates searching for papers on a topic.
    In the real Harvester agent, this calls arXiv + Weaviate.
    """
    print(f'  [search_papers] Searching for: {state["topic"]}')
    # Simulate finding papers
    papers = len(state['topic']) * 3  # fake number based on topic length
    print(f'  [search_papers] Found {papers} papers')
    return {
        'papers_found': papers,
        'status': f'Found {papers} papers'
    }


def generate_hypothesis(state: ResearchState) -> dict:
    """
    Simulates generating a hypothesis from papers.
    In the real Reasoner agent, this calls Claude API.
    """
    print(f'  [generate_hypothesis] Generating from {state["papers_found"]} papers...')
    attempts = state.get('attempts', 0) + 1
    # Simulate a hypothesis — quality improves with attempts
    hypothesis = f'Hypothesis #{attempts}: {state["topic"]} leads to novel discoveries'
    score      = min(0.4 + (attempts * 0.3), 1.0)  # gets better each attempt
    print(f'  [generate_hypothesis] Score: {score:.2f} (attempt {attempts})')
    return {
        'hypothesis': hypothesis,
        'score':      score,
        'attempts':   attempts,
        'status':     f'Generated hypothesis with score {score:.2f}'
    }


def store_hypothesis(state: ResearchState) -> dict:
    """
    Stores a validated hypothesis.
    In the real agent, this writes to Neo4j.
    """
    print(f'  [store_hypothesis] Storing: {state["hypothesis"]}')
    return {'status': 'Hypothesis stored successfully'}


# ── Step 3: Define Routing Logic ──────────────────────────────────────────
# This function decides which node to go to next.
# It's called a "conditional edge."

def should_retry(state: ResearchState) -> str:
    """
    If score is too low, retry hypothesis generation.
    If score is good enough, proceed to store.
    Max 3 attempts to avoid infinite loops.
    """
    if state['score'] >= 0.7:
        print(f'  [router] Score {state["score"]:.2f} >= 0.7 → storing hypothesis')
        return 'store'
    elif state.get('attempts', 0) >= 3:
        print(f'  [router] Max attempts reached → storing anyway')
        return 'store'
    else:
        print(f'  [router] Score {state["score"]:.2f} < 0.7 → retrying')
        return 'retry'


# ── Step 4: Build the Graph ───────────────────────────────────────────────

def build_research_graph():
    """
    Assembles the nodes and edges into a runnable graph.
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node('search_papers',       search_papers)
    graph.add_node('generate_hypothesis', generate_hypothesis)
    graph.add_node('store_hypothesis',    store_hypothesis)

    # Add edges — fixed connections
    graph.add_edge('search_papers', 'generate_hypothesis')

    # Add conditional edge — branches based on score
    graph.add_conditional_edges(
        'generate_hypothesis',  # from this node
        should_retry,           # call this function to decide
        {
            'store': 'store_hypothesis',    # if returns 'store' → go here
            'retry': 'generate_hypothesis', # if returns 'retry' → go here (loop!)
        }
    )

    # Add final edge
    graph.add_edge('store_hypothesis', END)

    # Set entry point
    graph.set_entry_point('search_papers')

    return graph.compile()


# ── Step 5: Run it ────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== LangGraph Demo ===\n')

    app = build_research_graph()

    # Initial state — only set what we know at the start
    initial_state = {
        'topic':         'graph neural networks for drug discovery',
        'papers_found':  0,
        'hypothesis':    '',
        'score':         0.0,
        'attempts':      0,
        'status':        'starting'
    }

    print('Running research graph...\n')
    final_state = app.invoke(initial_state)

    print('\n--- Final State ---')
    print(f'Topic:      {final_state["topic"]}')
    print(f'Papers:     {final_state["papers_found"]}')
    print(f'Hypothesis: {final_state["hypothesis"]}')
    print(f'Score:      {final_state["score"]}')
    print(f'Attempts:   {final_state["attempts"]}')
    print(f'Status:     {final_state["status"]}')

    assert final_state['score'] >= 0.7, 'Final score should be acceptable'
    assert final_state['attempts'] > 0, 'Should have made at least one attempt'
    print('\n✅ LangGraph demo working correctly.')