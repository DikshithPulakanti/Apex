# tests/test_ingest.py
# Unit tests for pipeline/ingest.py's branching logic — scraper/DB clients
# mocked so these tests don't hit arXiv or need live infra.

from unittest.mock import AsyncMock, MagicMock

from pipeline import ingest


class FakePaper:
    def __init__(self, id):
        self.id = id
        self.authors = ['A. Researcher']

    def to_dict(self):
        return {'id': self.id}


async def test_ingest_papers_returns_early_when_arxiv_finds_nothing(monkeypatch):
    fake_scraper = MagicMock()
    fake_scraper.scrape_all = AsyncMock(return_value=[])
    monkeypatch.setattr(ingest, 'ArxivScraper', lambda **kw: fake_scraper)

    result = await ingest.ingest_papers(['cat:cs.AI'], per_query=10)

    assert result == {'papers': 0, 'authors': 0}


async def test_ingest_papers_falls_back_when_redis_unavailable(monkeypatch):
    papers = [FakePaper('p1'), FakePaper('p2')]
    fake_scraper = MagicMock()
    fake_scraper.scrape_all = AsyncMock(return_value=papers)
    monkeypatch.setattr(ingest, 'ArxivScraper', lambda **kw: fake_scraper)
    monkeypatch.setattr(ingest, 'RedisClient', MagicMock(side_effect=RuntimeError('redis down')))

    fake_neo4j = MagicMock()
    monkeypatch.setattr(ingest, 'Neo4jClient', lambda: fake_neo4j)
    monkeypatch.setattr(ingest, 'PostgresClient', MagicMock(side_effect=RuntimeError('postgres down')))

    result = await ingest.ingest_papers(['cat:cs.AI'], per_query=10)

    assert result['papers'] == 2
    fake_neo4j.batch_upsert_papers.assert_called_once()


async def test_ingest_papers_skips_already_processed(monkeypatch):
    papers = [FakePaper('p1'), FakePaper('p2')]
    fake_scraper = MagicMock()
    fake_scraper.scrape_all = AsyncMock(return_value=papers)
    monkeypatch.setattr(ingest, 'ArxivScraper', lambda **kw: fake_scraper)

    fake_redis = MagicMock()
    fake_redis.filter_unprocessed.return_value = []  # everything already processed
    monkeypatch.setattr(ingest, 'RedisClient', lambda: fake_redis)

    result = await ingest.ingest_papers(['cat:cs.AI'], per_query=10)

    assert result == {'papers': 0, 'authors': 0}
