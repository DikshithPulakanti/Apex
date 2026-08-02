# tests/conftest.py
# Shared pytest fixtures — mocked infrastructure so unit tests run with
# no live Neo4j/Weaviate/Postgres/Redis/Kafka and no real Claude API calls.

import json
from unittest.mock import MagicMock

import pytest


class FakeResult:
    """Stand-in for a neo4j.Result: supports .single(), iteration, and .data()."""

    def __init__(self, records=None, single=None):
        self._records = records if records is not None else []
        self._single = single

    def single(self):
        return self._single

    def __iter__(self):
        return iter(self._records)

    def data(self):
        return list(self._records)


def make_claude_message(text: str):
    """
    Builds a MagicMock shaped like an anthropic Message. `block.type = 'text'`
    matters — real code filters for it rather than assuming `content[0]` is
    text, since newer models can put a ThinkingBlock at index 0.
    """
    block = MagicMock()
    block.type = 'text'
    block.text = text
    message = MagicMock()
    message.content = [block]
    return message


def make_claude_json_message(payload: dict):
    """Same as make_claude_message, but JSON-encodes a dict payload first."""
    return make_claude_message(json.dumps(payload))


@pytest.fixture
def fake_neo4j():
    """A MagicMock Neo4jClient with a working `.driver.session()` context manager.

    Clean methods (find_research_gaps, get_stats, ...) are plain auto-mocked
    attributes — configure `.return_value` per test. Raw Cypher access via
    `.driver.session()` is wired to `client.fake_session`, whose `.run.return_value`
    you can set to a FakeResult.
    """
    client = MagicMock()
    session = MagicMock()
    session_cm = MagicMock()
    session_cm.__enter__.return_value = session
    session_cm.__exit__.return_value = False
    client.driver.session.return_value = session_cm
    session.run.return_value = FakeResult()
    client.fake_session = session
    return client


@pytest.fixture
def fake_weaviate():
    client = MagicMock()
    client.hybrid_search.return_value = []
    client.vector_search.return_value = []
    client.get_paper_count.return_value = 0
    return client


@pytest.fixture
def fake_embedder():
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 384
    embedder.embed_batch.return_value = [[0.1] * 384]
    return embedder


@pytest.fixture
def fake_claude():
    return MagicMock()


@pytest.fixture
def fake_predictor():
    predictor = MagicMock()
    predictor.predict.return_value = {'label': 1, 'confidence': 0.99, 'verdict': 'valid'}
    return predictor
