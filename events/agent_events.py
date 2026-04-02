# events/agent_events.py
# Convenience functions for agents to publish APEX events

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events.kafka_manager import EventPublisher

_publisher = None

def get_publisher():
    """Lazy singleton — one publisher shared across all agents."""
    global _publisher
    if _publisher is None:
        _publisher = EventPublisher()
    return _publisher


def emit_papers_ingested(count: int, topics: list, run_id: str = ''):
    """Harvester finished ingesting papers."""
    get_publisher().publish(
        topic='papers.ingested',
        agent='harvester',
        data={
            'papers_count': count,
            'topics': topics,
            'run_id': run_id,
        }
    )


def emit_hypothesis_created(hypothesis_id: str, statement: str, testability: float):
    """Reasoner generated a new hypothesis."""
    get_publisher().publish(
        topic='hypothesis.created',
        agent='reasoner',
        data={
            'hypothesis_id': hypothesis_id,
            'statement': statement[:200],
            'testability': testability,
        },
        key=hypothesis_id,
    )


def emit_hypothesis_validated(hypothesis_id: str, score: float, method: str = 'bert'):
    """Skeptic approved a hypothesis."""
    get_publisher().publish(
        topic='hypothesis.validated',
        agent='skeptic',
        data={
            'hypothesis_id': hypothesis_id,
            'debate_score': score,
            'scoring_method': method,
        },
        key=hypothesis_id,
    )


def emit_hypothesis_rejected(hypothesis_id: str, score: float, reason: str = ''):
    """Skeptic rejected a hypothesis."""
    get_publisher().publish(
        topic='hypothesis.rejected',
        agent='skeptic',
        data={
            'hypothesis_id': hypothesis_id,
            'debate_score': score,
            'reason': reason[:200],
        },
        key=hypothesis_id,
    )


def emit_patent_drafted(patent_id: str, hypothesis_id: str, title: str, novelty: float):
    """Inventor drafted a patent."""
    get_publisher().publish(
        topic='patent.drafted',
        agent='inventor',
        data={
            'patent_id': patent_id,
            'hypothesis_id': hypothesis_id,
            'title': title[:200],
            'novelty_score': novelty,
        },
        key=patent_id,
    )


def emit_agent_status(agent: str, status: str, details: dict = None):
    """Any agent reporting its status."""
    get_publisher().publish(
        topic='agent.status',
        agent=agent,
        data={
            'status': status,
            'details': details or {},
        }
    )


# ── Quick test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Testing Agent Events ===\n')

    emit_agent_status('harvester', 'starting', {'target': '20 topics'})
    emit_papers_ingested(1527, ['cs.AI', 'cs.LG', 'cs.CL'], run_id='test-001')
    emit_hypothesis_created('hyp_test123', 'Neural networks can predict X', 0.85)
    emit_hypothesis_validated('hyp_test123', 0.9993, method='bert')
    emit_patent_drafted('pat_test456', 'hyp_test123', 'Method for predicting X', 0.95)

    print('\n✅ All 5 event types published successfully.')
    get_publisher().close()