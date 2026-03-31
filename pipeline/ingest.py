# pipeline/ingest.py
# Connects the arXiv scraper to Neo4j

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapers.arxiv_scraper import ArxivScraper
from database.neo4j_client import Neo4jClient


async def ingest_topic(topic_query: str, count: int):
    """
    Scrapes papers for a topic and inserts them into Neo4j.
    """
    print(f'\n[Ingest] Starting: {topic_query}, target={count} papers')

    scraper = ArxivScraper(requests_per_second=0.5)
    neo4j   = Neo4jClient()

    try:
        # Step 1: Scrape papers
        print('[Ingest] Fetching from arXiv...')
        papers = await scraper.scrape_query(topic_query, total=count)
        print(f'[Ingest] Got {len(papers)} papers')

        if not papers:
            print('[Ingest] No papers fetched. Skipping Neo4j insert.')
            return

        # Step 2: Insert into Neo4j
        print('[Ingest] Inserting into Neo4j...')
        try:
            neo4j.batch_upsert_papers([p.to_dict() for p in papers])
            print(f'[Ingest] Inserted {len(papers)} papers successfully.')
        except Exception as e:
            print(f'[Ingest] Neo4j insert failed: {e}')

        # Step 3: Verify
        total = neo4j.get_paper_count()
        print(f'[Ingest] Total papers in Neo4j: {total}')

    finally:
        neo4j.close()


if __name__ == '__main__':
    asyncio.run(ingest_topic('cat:cs.AI', 20))