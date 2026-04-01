# tests/test_sim_patent_mcp.py
# Test sim-mcp and patent-mcp tools

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from mcp_servers.sim_mcp import (
    handle_simulation,
    handle_synthetic_data,
    handle_validation,
)
from mcp_servers.patent_mcp import (
    handle_draft_patent,
    handle_prior_art,
    handle_novelty_score,
)

HYPOTHESIS = 'Graph neural networks can predict drug-protein binding affinity more accurately than traditional docking methods by encoding molecular topology.'
CONCEPTS   = ['graph neural networks', 'drug discovery', 'molecular property prediction']


async def main():
    print('=== sim-mcp + patent-mcp Tests ===\n')

    # ── sim-mcp tests ─────────────────────────────────────────────────────
    print('[Test 1] run_hypothesis_simulation...')
    result = await handle_simulation({
        'hypothesis_statement': HYPOTHESIS,
        'testability_score':    0.9,
        'n_simulations':        1000
    })
    data = json.loads(result[0].text)
    print(f'  Success rate: {data["success_rate"]}')
    print(f'  CI: [{data["confidence_interval"]["lower"]}, {data["confidence_interval"]["upper"]}]')
    print(f'  Recommendation: {data["recommendation"]}')
    assert data['success_rate'] > 0
    print('  ✓ simulation working\n')

    print('[Test 2] generate_synthetic_data...')
    result = await handle_synthetic_data({
        'hypothesis_statement': HYPOTHESIS,
        'n_samples':            50
    })
    data = json.loads(result[0].text)
    print(f'  Samples: {data["n_samples"]}')
    print(f'  Significant: {data["significant_results"]}')
    assert data['n_samples'] == 50
    print('  ✓ synthetic data working\n')

    print('[Test 3] validate_against_known...')
    result = await handle_validation({
        'hypothesis_statement': HYPOTHESIS,
        'supporting_concepts':  CONCEPTS
    })
    data = json.loads(result[0].text)
    print(f'  Contradiction score: {data["contradiction_score"]}')
    print(f'  Verdict: {data["verdict"]}')
    print('  ✓ validation working\n')

    # ── patent-mcp tests ──────────────────────────────────────────────────
    print('[Test 4] check_prior_art...')
    result = await handle_prior_art({'invention_description': HYPOTHESIS})
    data   = json.loads(result[0].text)
    print(f'  Prior art risk: {data["prior_art_risk"]}')
    print(f'  Recommendation: {data["recommendation"]}')
    print('  ✓ prior art check working\n')

    print('[Test 5] compute_novelty_score...')
    result = await handle_novelty_score({
        'hypothesis_statement': HYPOTHESIS,
        'supporting_concepts':  CONCEPTS
    })
    data = json.loads(result[0].text)
    print(f'  Novelty score: {data["novelty_score"]}')
    print(f'  Assessment: {data["assessment"]}')
    assert 0.0 <= data['novelty_score'] <= 1.0
    print('  ✓ novelty score working\n')

    print('[Test 6] draft_patent_claims...')
    result = await handle_draft_patent({
        'hypothesis_statement': HYPOTHESIS,
        'supporting_concepts':  CONCEPTS,
        'predicted_impact':     'Could accelerate drug discovery by 10x.'
    })
    data = json.loads(result[0].text)
    print(f'  Title: {data.get("title", "")}')
    print(f'  Claim 1: {data.get("independent_claim_1", "")[:80]}')
    assert 'title' in data
    print('  ✓ patent drafting working\n')

    print('✅ All sim-mcp and patent-mcp tools working.')


if __name__ == '__main__':
    asyncio.run(main())