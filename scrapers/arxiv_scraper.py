import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    year: int
    url: str
    citations: int = 0

    def to_dict(self) -> dict:
        return {
            'id':         self.id,
            'title':      self.title,
            'abstract':   self.abstract,
            'authors':    self.authors,
            'categories': self.categories,
            'year':       self.year,
            'url':        self.url,
            'citations':  self.citations,
        }


class ArxivScraper:
    BASE_URL = 'https://export.arxiv.org/api/query'
    NS = {'atom': 'http://www.w3.org/2005/Atom'}  # XML namespace

    def __init__(self, requests_per_second: float = 3.0):
        # arXiv rate limit: max 3 requests/second
        self.delay = 1.0 / requests_per_second

    def _parse_xml(self, xml_text: str) -> list[Paper]:
        """Parse arXiv API XML response into Paper objects."""
        if not xml_text:
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        papers = []
        for entry in root.findall('atom:entry', self.NS):
            # Extract paper id (strip the URL prefix)
            id_elem = entry.find('atom:id', self.NS)
            if id_elem is None: continue
            paper_id = id_elem.text.split('/abs/')[-1]  # e.g. '2301.00001'

            title_elem = entry.find('atom:title', self.NS)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ''

            summary_elem = entry.find('atom:summary', self.NS)
            abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ''

            # Authors
            authors = []
            for author_elem in entry.findall('atom:author', self.NS):
                name_elem = author_elem.find('atom:name', self.NS)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())

            # Published year
            published_elem = entry.find('atom:published', self.NS)
            year = 0
            if published_elem is not None:
                try:
                    year = datetime.fromisoformat(published_elem.text[:10]).year
                except ValueError:
                    pass

            # Categories
            categories = [
                tag.get('term', '')
                for tag in entry.findall('{http://arxiv.org/schemas/atom}primary_category', {})
            ]
            # Also get additional categories
            for cat in entry.findall('{http://www.w3.org/2005/Atom}category'):
                term = cat.get('term', '')
                if term and term not in categories:
                    categories.append(term)

            papers.append(Paper(
                id=paper_id, title=title, abstract=abstract,
                authors=authors, categories=categories, year=year,
                url=f'https://arxiv.org/abs/{paper_id}'
            ))

        return papers

    async def fetch_batch(self,
                          session: aiohttp.ClientSession,
                          query: str,
                          start: int,
                          max_results: int = 100) -> list[Paper]:
        """Fetch one batch of papers from arXiv."""
        params = {
            'search_query': query,
            'start':        start,
            'max_results':  max_results,
            'sortBy':       'submittedDate',
            'sortOrder':    'descending',
        }
        try:
            async with session.get(self.BASE_URL, params=params,
                                   timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    await asyncio.sleep(self.delay)  # respect rate limit
                    return self._parse_xml(text)
                else:
                    print(f'  arXiv returned HTTP {resp.status} for query: {query}')
                    return []
        except Exception as e:
            print(f'  Error fetching {query} start={start}: {e}')
            return []

    async def scrape_query(self, query: str, total: int = 500) -> list[Paper]:
        """
        Scrape all papers for a single query.
        Paginates automatically to fetch 'total' papers.
        """
        all_papers = []
        batch_size = 100

        async with aiohttp.ClientSession() as session:
            for start in range(0, total, batch_size):
                print(f'  Fetching {query}: papers {start}-{start+batch_size}')
                batch = await self.fetch_batch(session, query, start, batch_size)
                if not batch:  # arXiv returned nothing — stop paginating
                    break
                all_papers.extend(batch)

        return all_papers

    async def scrape_all(self, queries: list[str], per_query: int = 500) -> list[Paper]:
        """
        Scrape multiple queries concurrently.
        Returns deduplicated list of papers.
        """
        print(f'Starting scrape: {len(queries)} queries × {per_query} papers each')

        # Run all queries concurrently
        results = await asyncio.gather(*[
            self.scrape_query(query, per_query)
            for query in queries
        ])

        # Flatten and deduplicate by paper id
        seen = set()
        papers = []
        for batch in results:
            for paper in batch:
                if paper.id not in seen:
                    seen.add(paper.id)
                    papers.append(paper)

        print(f'Scraped {len(papers)} unique papers')
        return papers


# Test run
if __name__ == '__main__':
    scraper = ArxivScraper()

    # Small test — just 100 papers to verify it works
    papers = asyncio.run(scraper.scrape_query('cat:cs.AI', total=100))

    print(f'\nFetched {len(papers)} papers')
    if papers:
        p = papers[0]
        print(f'First paper: {p.title}')
        print(f'Authors:     {p.authors[:3]}')
        print(f'Year:        {p.year}')
        print(f'Categories:  {p.categories}')
