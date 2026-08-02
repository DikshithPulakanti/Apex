# tests/test_claude.py
# Tests for the claude_structured() JSON-parsing helper. The Anthropic
# client is mocked (see conftest.fake_claude) — these tests never call
# the real API, so they run for free and deterministically in CI.

import json
from typing import List

from pydantic import BaseModel

from tests.conftest import make_claude_json_message, make_claude_message


class PaperSummary(BaseModel):
    main_contribution: str
    key_concepts: List[str]
    research_domain: str
    novelty_score: float


class Hypothesis(BaseModel):
    statement: str
    rationale: str
    supporting_concepts: List[str]
    testability_score: float
    predicted_impact: str


def claude_structured(client, system: str, user: str, model_class):
    """Calls Claude and parses its JSON response into the given Pydantic model."""
    message = client.messages.create(
        model='claude-sonnet-5',
        max_tokens=1000,
        system=system,
        messages=[{'role': 'user', 'content': user}],
    )

    text = next((b.text for b in message.content if b.type == 'text'), '')
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    data = json.loads(text)
    return model_class(**data)


def test_basic_claude_call_returns_text(fake_claude):
    fake_claude.messages.create.return_value = make_claude_json_message({'text': 'Claude API working'})

    message = fake_claude.messages.create(
        model='claude-sonnet-5', max_tokens=50,
        messages=[{'role': 'user', 'content': 'Say exactly: Claude API working'}],
    )

    assert json.loads(message.content[0].text)['text'] == 'Claude API working'


def test_structured_output_parses_into_pydantic_model(fake_claude):
    fake_claude.messages.create.return_value = make_claude_json_message({
        'main_contribution': 'Bidirectional pre-training for language representations',
        'key_concepts': ['bidirectional encoding', 'transformers', 'pre-training'],
        'research_domain': 'NLP',
        'novelty_score': 0.85,
    })

    summary = claude_structured(fake_claude, 'system prompt', 'user prompt', PaperSummary)

    assert isinstance(summary, PaperSummary)
    assert len(summary.key_concepts) == 3
    assert 0.0 <= summary.novelty_score <= 1.0


def test_structured_output_handles_code_fenced_json(fake_claude):
    fenced = '```json\n' + json.dumps({
        'main_contribution': 'x', 'key_concepts': ['a'],
        'research_domain': 'AI', 'novelty_score': 0.5,
    }) + '\n```'
    fake_claude.messages.create.return_value = make_claude_message(fenced)

    summary = claude_structured(fake_claude, 'system', 'user', PaperSummary)

    assert summary.research_domain == 'AI'


def test_hypothesis_generation_parses_into_pydantic_model(fake_claude):
    fake_claude.messages.create.return_value = make_claude_json_message({
        'statement': 'Automatic parallelization techniques can accelerate LLM training loops.',
        'rationale': 'Both fields optimize computation graphs but rarely intersect.',
        'supporting_concepts': ['automatic parallelization', 'large language models'],
        'testability_score': 0.75,
        'predicted_impact': 'Faster training pipelines for large models.',
    })

    hypothesis = claude_structured(fake_claude, 'system', 'user', Hypothesis)

    assert isinstance(hypothesis, Hypothesis)
    assert len(hypothesis.statement) > 20
    assert 0.0 <= hypothesis.testability_score <= 1.0
