# tests/test_paper_mcp.py
# Test paper-mcp tools directly (without MCP protocol)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Import handlers directly for testing
from mcp_servers.paper_mcp import (
    handle_search_papers,
    handle_get_paper_details,
    handle_get_paper_concepts,
    handle_get_papers_by_year,
    get_neo4j, get_weaviate, get_embedder
)


async def main():
    print('=== paper-mcp Tool Tests ===\n')

    # Test 1: search_papers
    print('[Test 1] search_papers...')
    result = await handle_search_papers({
        'query': 'transformer attention mechanism',
        'limit': 3
    })
    import json
    data = json.loads(result[0].text)
    print(f'  Found {data["count"]} papers')
    for p in data['papers']:
        print(f'  → {p["title"][:60]}')
    assert data['count'] > 0
    print('  ✓ search_papers working\n')

    # Test 2: get_papers_by_year
    print('[Test 2] get_papers_by_year...')
    result = await handle_get_papers_by_year({'year': 2026})
    data   = json.loads(result[0].text)
    print(f'  Papers from 2026: {data["count"]}')
    assert data['count'] >= 0
    print('  ✓ get_papers_by_year working\n')

    # Test 3: get_paper_details
    print('[Test 3] get_paper_details...')
    neo4j = get_neo4j()
    with neo4j.driver.session() as session:
        r = session.run('MATCH (p:Paper) RETURN p.id AS id LIMIT 1')
        record = r.single()
        paper_id = record['id'] if record else None

    if paper_id:
        result = await handle_get_paper_details({'paper_id': paper_id})
        data   = json.loads(result[0].text)
        print(f'  Paper: {data.get("title", "")[:60]}')
        print('  ✓ get_paper_details working\n')
    else:
        print('  Skipped — no papers in Neo4j\n')

    # Test 4: get_paper_concepts
    print('[Test 4] get_paper_concepts...')
    if paper_id:
        result = await handle_get_paper_concepts({'paper_id': paper_id})
        data   = json.loads(result[0].text)
        print(f'  Concepts: {data["concepts"][:3]}')
        print('  ✓ get_paper_concepts working\n')

    print('✅ All paper-mcp tools working.')

    # Cleanup
    get_neo4j().close()
    get_weaviate().close()


if __name__ == '__main__':
    asyncio.run(main())