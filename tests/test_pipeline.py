# tests/test_pipeline.py
# Integration test — full pipeline from arXiv to Neo4j

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.arxiv_scraper import ArxivScraper
from database.neo4j_client import Neo4jClient
from database.redis_client import RedisClient


async def test_full_pipeline():
    print('\n=== APEX Integration Test ===\n')

    # ── Test 1: arXiv scraper ─────────────────────────────────────────────
    print('[Test 1] arXiv scraper...')
    scraper = ArxivScraper(requests_per_second=0.5)
    papers  = await scraper.scrape_query('cat:cs.RO', total=10)
    assert len(papers) > 0, 'Should fetch at least one paper'
    assert papers[0].title != '', 'Paper should have a title'
    assert papers[0].year  > 0,  'Paper should have a year'
    print(f'  ✓ Fetched {len(papers)} papers from arXiv')

    # ── Test 2: Neo4j insert and query back ───────────────────────────────
    print('[Test 2] Neo4j insert and query...')
    neo4j = Neo4jClient()
    neo4j.batch_upsert_papers([p.to_dict() for p in papers])
    count = neo4j.get_paper_count()
    assert count > 0, 'Neo4j should have papers'
    print(f'  ✓ Papers in Neo4j: {count}')

    # ── Test 3: Author linking ────────────────────────────────────────────
    print('[Test 3] Author linking...')
    paper = papers[0]
    if paper.authors:
        neo4j.upsert_author(paper.authors[0])
        neo4j.link_author_to_paper(paper.authors[0], paper.id)
        authors = neo4j.get_authors_of_paper(paper.id)
        assert paper.authors[0] in authors, 'Author should be linked'
        print(f'  ✓ Author linked: {paper.authors[0]}')

    # ── Test 4: Redis caching ─────────────────────────────────────────────
    print('[Test 4] Redis caching...')
    redis = RedisClient()
    redis.mark_processed(paper.id)
    assert redis.is_processed(paper.id) == True
    unprocessed = redis.filter_unprocessed([paper.id, 'fake:id:999'])
    assert unprocessed == ['fake:id:999']
    print(f'  ✓ Redis caching working correctly')

    # ── Test 5: Stats ─────────────────────────────────────────────────────
    print('[Test 5] Graph stats...')
    stats = neo4j.get_stats()
    assert stats['papers'] > 0
    assert stats['authors'] > 0
    print(f'  ✓ Stats: {stats}')

    neo4j.close()
    redis.close()

    print('\n✅ All integration tests passed.')


if __name__ == '__main__':
    asyncio.run(test_full_pipeline())