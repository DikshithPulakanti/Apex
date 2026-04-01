# tests/test_graph_mcp.py
# Test graph-mcp tools directly

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from mcp_servers.graph_mcp import (
    handle_find_gaps,
    handle_create_hypothesis,
    handle_get_stats,
    handle_get_top_concepts,
    handle_get_hypotheses,
)


async def main():
    print('=== graph-mcp Tool Tests ===\n')

    # Test 1: get_graph_stats
    print('[Test 1] get_graph_stats...')
    result = await handle_get_stats({})
    data   = json.loads(result[0].text)
    print(f'  Papers:      {data["papers"]}')
    print(f'  Authors:     {data["authors"]}')
    print(f'  Hypotheses:  {data["hypotheses"]}')
    assert data['papers'] > 0
    print('  ✓ get_graph_stats working\n')

    # Test 2: find_research_gaps
    print('[Test 2] find_research_gaps...')
    result = await handle_find_gaps({'min_pagerank': 0.3, 'limit': 5})
    data   = json.loads(result[0].text)
    print(f'  Found {data["count"]} gaps')
    for g in data['gaps'][:3]:
        print(f'  → {g["concept1"]} ↔ {g["concept2"]}')
    assert data['count'] > 0
    print('  ✓ find_research_gaps working\n')

    # Test 3: get_top_concepts
    print('[Test 3] get_top_concepts...')
    result = await handle_get_top_concepts({'limit': 5})
    data   = json.loads(result[0].text)
    print(f'  Top concepts:')
    for c in data['concepts'][:3]:
        print(f'  → {c["name"]} (pagerank: {c["pagerank"]:.2f})')
    print('  ✓ get_top_concepts working\n')

    # Test 4: create_hypothesis
    print('[Test 4] create_hypothesis...')
    result = await handle_create_hypothesis({
        'statement':           'Graph neural networks can predict drug-protein binding affinity more accurately than traditional docking methods by encoding molecular topology.',
        'rationale':           'GNNs capture structural relationships in molecular graphs that traditional methods miss.',
        'supporting_concepts': ['graph neural networks', 'drug discovery'],
        'testability_score':   0.9,
        'predicted_impact':    'Could accelerate drug discovery by 10x reducing wet lab screening.'
    })
    data = json.loads(result[0].text)
    print(f'  Created: {data["hypothesis_id"]}')
    assert data['status'] == 'created'
    print('  ✓ create_hypothesis working\n')

    # Test 5: get_hypotheses
    print('[Test 5] get_hypotheses...')
    result = await handle_get_hypotheses({'limit': 5})
    data   = json.loads(result[0].text)
    print(f'  Total hypotheses: {len(data["hypotheses"])}')
    for h in data['hypotheses'][:2]:
        print(f'  → [{h["testability_score"]}] {h["statement"][:60]}')
    assert len(data['hypotheses']) > 0
    print('  ✓ get_hypotheses working\n')

    print('✅ All graph-mcp tools working.')


if __name__ == '__main__':
    asyncio.run(main())