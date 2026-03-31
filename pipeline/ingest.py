# pipeline/ingest.py
# Full ingestion pipeline — Day 5

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.arxiv_scraper import ArxivScraper
from scrapers.queries import APEX_QUERIES
from database.neo4j_client import Neo4jClient
from database.postgres_client import PostgresClient


async def ingest_papers(queries: list[str], per_query: int = 100) -> dict:
    """
    Full pipeline:
    1. Scrape papers from arXiv concurrently
    2. Insert papers into Neo4j
    3. Build author graph
    4. Log run to PostgreSQL
    5. Print stats
    """
    scraper = ArxivScraper(requests_per_second=0.5)

    # ── Step 1: Scrape all queries ────────────────────────────────────────
    print(f'\n[1/4] Scraping {len(queries)} queries from arXiv...')
    papers = await scraper.scrape_all(queries, per_query=per_query)
    print(f'      Found {len(papers)} unique papers')

    if not papers:
        print('      No papers found. Exiting.')
        return {'papers': 0, 'authors': 0}

    # ── Step 2: Insert papers into Neo4j ──────────────────────────────────
    print(f'\n[2/4] Inserting papers into Neo4j...')
    try:
        neo4j = Neo4jClient()
    except Exception as e:
        print(f'      Neo4j unavailable: {e}')
        return {'papers': 0, 'authors': 0}

    try:
        batch_size = 500
        inserted   = 0
        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            neo4j.batch_upsert_papers([p.to_dict() for p in batch])
            inserted += len(batch)
            print(f'      Progress: {inserted}/{len(papers)} papers')

        # ── Step 3: Build author graph ────────────────────────────────────
        print(f'\n[3/4] Building author graph...')
        author_count = 0
        for paper in papers:
            for author_name in paper.authors[:5]:
                neo4j.upsert_author(author_name)
                neo4j.link_author_to_paper(author_name, paper.id)
                author_count += 1
        print(f'      Linked {author_count} author-paper relationships')

        # ── Step 4: Log to PostgreSQL ─────────────────────────────────────
        print(f'\n[4/4] Logging to PostgreSQL...')
        try:
            postgres = PostgresClient()
            topic    = ', '.join(queries[:3]) + ('...' if len(queries) > 3 else '')
            postgres.log_pipeline_run(topic, len(papers))
            postgres.close()
            print(f'      Run logged to PostgreSQL.')
        except Exception as e:
            print(f'      PostgreSQL logging failed: {e}')

        # ── Stats ─────────────────────────────────────────────────────────
        print(f'\n✅ Ingestion complete!')
        stats = neo4j.get_stats()
        return {'papers': len(papers), 'authors': author_count}

    finally:
        neo4j.close()


if __name__ == '__main__':
    # Start with 2 queries and 100 papers each to verify everything works
    test_queries = APEX_QUERIES[:10]
    asyncio.run(ingest_papers(test_queries, per_query=100))