# tests/test_claude.py
# Test Claude API + structured outputs

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import json

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


# ── Define structured output models ──────────────────────────────────────

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


# ── Helper: call Claude and parse JSON response ───────────────────────────

def claude_structured(client, system: str, user: str, model_class):
    """
    Calls Claude and forces it to return JSON matching the Pydantic model.
    We do this manually since instructor 0.4.8 works differently.
    """
    schema = model_class.model_json_schema()

    message = client.messages.create(
        model      = 'claude-sonnet-4-20250514',
        max_tokens = 1000,
        system     = system,
        messages   = [
            {'role': 'user', 'content': user}
        ]
    )

    text = message.content[0].text

    # Extract JSON from response
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    data = json.loads(text)
    return model_class(**data)


# ── Test functions ────────────────────────────────────────────────────────

def test_basic_claude_call(client):
    print('\n--- Test 1: Basic Claude API call ---')
    message = client.messages.create(
        model      = 'claude-sonnet-4-20250514',
        max_tokens = 50,
        messages   = [
            {'role': 'user', 'content': 'Say exactly: Claude API working'}
        ]
    )
    response = message.content[0].text
    print(f'  Response: {response}')
    assert len(response) > 0
    print('  ✓ Basic Claude API call working')


def test_structured_output(client):
    print('\n--- Test 2: Structured output ---')

    abstract = """
    We present BERT, a new language representation model which stands for
    Bidirectional Encoder Representations from Transformers. BERT is designed
    to pre-train deep bidirectional representations from unlabeled text.
    The pre-trained BERT model can be fine-tuned for a wide range of NLP tasks.
    """

    system = f"""You are a research analyst. Analyze the paper abstract and return 
a JSON object matching this exact schema:
{json.dumps(PaperSummary.model_json_schema(), indent=2)}

IMPORTANT: novelty_score must be a float between 0.0 and 1.0 (not 0-10).
Return ONLY valid JSON, no other text."""

    summary = claude_structured(
        client, system,
        f'Analyze this abstract:\n\n{abstract}',
        PaperSummary
    )

    print(f'  Main contribution: {summary.main_contribution[:80]}')
    print(f'  Key concepts:      {summary.key_concepts}')
    print(f'  Domain:            {summary.research_domain}')
    print(f'  Novelty score:     {summary.novelty_score}')

    assert isinstance(summary, PaperSummary)
    assert len(summary.key_concepts) > 0
    assert 0.0 <= summary.novelty_score <= 1.0
    print('  ✓ Structured output working')


def test_hypothesis_generation(client):
    print('\n--- Test 3: Hypothesis generation ---')

    gap_context = """
    Research gap:
    Concept 1: "automatic parallelization" (compiler optimization domain)
    Concept 2: "large language models" (NLP domain)
    These concepts come from different communities and rarely co-occur.
    """

    system = f"""You are a scientific research assistant. Generate a novel hypothesis 
based on the research gap. Return a JSON object matching this schema:
{json.dumps(Hypothesis.model_json_schema(), indent=2)}

IMPORTANT: testability_score must be a float between 0.0 and 1.0 (not 0-10).
Return ONLY valid JSON, no other text."""

    hypothesis = claude_structured(
        client, system,
        f'Generate a hypothesis for this gap:\n\n{gap_context}',
        Hypothesis
    )

    print(f'  Statement:   {hypothesis.statement}')
    print(f'  Testability: {hypothesis.testability_score}')
    print(f'  Impact:      {hypothesis.predicted_impact[:80]}')

    assert isinstance(hypothesis, Hypothesis)
    assert len(hypothesis.statement) > 20
    assert 0.0 <= hypothesis.testability_score <= 1.0
    print('  ✓ Hypothesis generation working')


if __name__ == '__main__':
    print('=== Claude API Tests ===')

    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    test_basic_claude_call(client)
    test_structured_output(client)
    test_hypothesis_generation(client)

    print('\n✅ All Claude API tests passed.')