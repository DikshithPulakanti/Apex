# pipeline/ingest.py
# Full ingestion pipeline with Redis caching — Day 7

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.arxiv_scraper import ArxivScraper
from scrapers.queries import APEX_QUERIES
from database.neo4j_client import Neo4jClient
from database.postgres_client import PostgresClient
from database.redis_client import RedisClient


async def ingest_papers(queries: list, per_query: int = 100) -> dict:
    """
    Full pipeline:
    1. Scrape papers from arXiv concurrently
    2. Filter out already-processed papers using Redis
    3. Insert new papers into Neo4j
    4. Build author graph
    5. Mark papers as processed in Redis
    6. Log run to PostgreSQL
    7. Print stats
    """
    scraper = ArxivScraper(requests_per_second=0.5)

    # ── Step 1: Scrape all queries ────────────────────────────────────────
    print(f'\n[1/5] Scraping {len(queries)} queries from arXiv...')
    papers = await scraper.scrape_all(queries, per_query=per_query)
    print(f'      Found {len(papers)} unique papers')

    if not papers:
        print('      No papers found. Exiting.')
        return {'papers': 0, 'authors': 0}

    # ── Step 2: Filter already-processed papers using Redis ───────────────
    print(f'\n[2/5] Checking Redis cache...')
    try:
        redis_client = RedisClient()
        paper_ids    = [p.id for p in papers]
        new_ids      = set(redis_client.filter_unprocessed(paper_ids))
        new_papers   = [p for p in papers if p.id in new_ids]
        print(f'      {len(papers)} total → {len(new_papers)} new (not yet processed)')
    except Exception as e:
        print(f'      Redis unavailable: {e}. Processing all papers.')
        new_papers   = papers
        redis_client = None

    if not new_papers:
        print('      All papers already processed. Nothing to insert.')
        return {'papers': 0, 'authors': 0}

    # ── Step 3: Insert papers into Neo4j ──────────────────────────────────
    print(f'\n[3/5] Inserting {len(new_papers)} papers into Neo4j...')
    try:
        neo4j = Neo4jClient()
    except Exception as e:
        print(f'      Neo4j unavailable: {e}')
        return {'papers': 0, 'authors': 0}

    try:
        batch_size = 500
        inserted   = 0
        for i in range(0, len(new_papers), batch_size):
            batch = new_papers[i : i + batch_size]
            neo4j.batch_upsert_papers([p.to_dict() for p in batch])
            inserted += len(batch)
            print(f'      Progress: {inserted}/{len(new_papers)} papers')

        # ── Step 4: Build author graph ────────────────────────────────────
        print(f'\n[4/5] Building author graph...')
        author_count = 0
        for paper in new_papers:
            for author_name in paper.authors[:5]:
                neo4j.upsert_author(author_name)
                neo4j.link_author_to_paper(author_name, paper.id)
                author_count += 1
        print(f'      Linked {author_count} author-paper relationships')

        # ── Step 5: Mark papers as processed in Redis ─────────────────────
        if redis_client:
            print(f'\n[5/5] Marking {len(new_papers)} papers as processed in Redis...')
            redis_client.mark_processed_batch([p.id for p in new_papers])
            print(f'      Done.')

        # ── Log to PostgreSQL ─────────────────────────────────────────────
        print(f'\nLogging to PostgreSQL...')
        try:
            postgres = PostgresClient()
            topic    = ', '.join(queries[:3]) + ('...' if len(queries) > 3 else '')
            postgres.log_pipeline_run(topic, len(new_papers))
            postgres.close()
            print(f'      Run logged to PostgreSQL.')
        except Exception as e:
            print(f'      PostgreSQL logging failed: {e}')

        # ── Stats ─────────────────────────────────────────────────────────
        print(f'\n✅ Ingestion complete!')
        neo4j.get_stats()
        return {'papers': len(new_papers), 'authors': author_count}

    finally:
        neo4j.close()
        if redis_client:
            redis_client.close()


if __name__ == '__main__':
    test_queries = APEX_QUERIES
    asyncio.run(ingest_papers(test_queries, per_query=100))