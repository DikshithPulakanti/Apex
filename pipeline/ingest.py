# pipeline/ingest.py
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.arxiv_scraper import ArxivScraper
from database.neo4j_client import Neo4jClient


async def ingest_topic(topic_query: str, count: int):
    print(f'\n[Ingest] Starting: {topic_query}, target={count} papers')

    scraper = ArxivScraper(requests_per_second=0.5)

    # Step 1: Scrape papers — happens regardless of Neo4j status
    print('[Ingest] Starting fetch...')
    papers = await scraper.scrape_query(topic_query, total=count)
    print(f'[Ingest] Received batch — {len(papers)} papers')

    if not papers:
        print('[Ingest] No papers fetched.')
        return

    # Step 2: Try to connect to Neo4j and insert
    try:
        neo4j = Neo4jClient()
        try:
            neo4j.batch_upsert_papers([p.to_dict() for p in papers])
            print(f'[Ingest] Inserted {len(papers)} papers successfully.')
            total = neo4j.get_paper_count()
            print(f'[Ingest] Total papers in Neo4j: {total}')
        finally:
            neo4j.close()
    except Exception as e:
        print(f'[Ingest] Neo4j unavailable — {len(papers)} papers NOT saved. Error: {e}')
        print('[Ingest] Scrape completed but data was lost. Restart Neo4j and re-run.')


async def ingest_multiple(queries: list[str], count: int):
    """
    Runs multiple ingestion tasks concurrently.
    Notice the order batches come back is unpredictable.
    """
    print(f'\n[Ingest] Running {len(queries)} queries concurrently...')
    print('[Ingest] Starting fetch for all queries simultaneously...\n')

    await asyncio.gather(*[
        ingest_topic(query, count)
        for query in queries
    ])

    print('\n[Ingest] All queries complete.')


if __name__ == '__main__':
    # Run 3 queries concurrently — watch the order they print
    queries = ['cat:cs.AI', 'cat:cs.LG', 'cat:cs.CL']
    asyncio.run(ingest_multiple(queries, count=20))