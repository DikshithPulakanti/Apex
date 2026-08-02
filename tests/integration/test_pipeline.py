# tests/integration/test_pipeline.py
# Integration test — full pipeline from arXiv to Neo4j. Needs live
# Neo4j + Redis (docker-compose up) and network access to arXiv.
# Run explicitly with: pytest -m integration

import pytest

from database.neo4j_client import Neo4jClient
from database.redis_client import RedisClient
from scrapers.arxiv_scraper import ArxivScraper


@pytest.mark.integration
async def test_full_pipeline():
    scraper = ArxivScraper(requests_per_second=0.5)
    papers = await scraper.scrape_query('cat:cs.RO', total=10)
    assert len(papers) > 0, 'Should fetch at least one paper'
    assert papers[0].title != '', 'Paper should have a title'
    assert papers[0].year > 0, 'Paper should have a year'

    neo4j = Neo4jClient()
    neo4j.batch_upsert_papers([p.to_dict() for p in papers])
    count = neo4j.get_paper_count()
    assert count > 0, 'Neo4j should have papers'

    paper = papers[0]
    if paper.authors:
        neo4j.upsert_author(paper.authors[0])
        neo4j.link_author_to_paper(paper.authors[0], paper.id)
        authors = neo4j.get_authors_of_paper(paper.id)
        assert paper.authors[0] in authors, 'Author should be linked'

    redis = RedisClient()
    redis.mark_processed(paper.id)
    assert redis.is_processed(paper.id) is True
    unprocessed = redis.filter_unprocessed([paper.id, 'fake:id:999'])
    assert unprocessed == ['fake:id:999']

    stats = neo4j.get_stats()
    assert stats['papers'] > 0
    assert stats['authors'] > 0

    neo4j.close()
    redis.close()
