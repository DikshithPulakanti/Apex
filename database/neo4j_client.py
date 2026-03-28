import os
from typing import Optional
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv

load_dotenv()


class Neo4jClient:
    """
    Manages all interactions with the Neo4j graph database.
    Create one instance of this class and reuse it throughout the app.
    """

    def __init__(self):
        uri      = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        user     = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'apexpassword')

        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_connection()

    def _verify_connection(self):
        """Confirm we can actually reach Neo4j. Fail loudly if not."""
        try:
            self.driver.verify_connectivity()
            print('✅ Neo4j connection established')
        except Exception as e:
            raise RuntimeError(f'Cannot connect to Neo4j: {e}')

    def close(self):
        """Always close the driver when your program finishes."""
        self.driver.close()

    # ── Paper operations ──────────────────────────────────────────────────

    def upsert_paper(self, paper: dict) -> dict:
        """
        Insert or update a paper node.
        Uses MERGE so running twice won't create duplicates.

        Args:
            paper: dict with keys: id, title, abstract, year, categories, authors

        Returns:
            The created/updated paper properties
        """
        query = """
            MERGE (p:Paper {id: $id})
            SET p.title      = $title,
                p.abstract   = $abstract,
                p.year       = $year,
                p.categories = $categories,
                p.citations  = $citations
            RETURN p
        """
        with self.driver.session() as session:
            result = session.run(query,
                id         = paper['id'],
                title      = paper.get('title', ''),
                abstract   = paper.get('abstract', ''),
                year       = paper.get('year', 0),
                categories = paper.get('categories', []),
                citations  = paper.get('citations', 0),
            )
            record = result.single()
            return dict(record['p']) if record else {}

    def upsert_author(self, name: str, institution: str = '') -> None:
        """Insert or update an Author node."""
        query = """
            MERGE (a:Author {name: $name})
            SET a.institution = $institution
        """
        with self.driver.session() as session:
            session.run(query, name=name, institution=institution)

    def link_author_to_paper(self, author_name: str, paper_id: str) -> None:
        """Create AUTHORED_BY relationship between existing Author and Paper."""
        query = """
            MATCH (a:Author {name: $author_name})
            MATCH (p:Paper  {id:   $paper_id})
            MERGE (p)-[:AUTHORED_BY]->(a)
        """
        with self.driver.session() as session:
            session.run(query, author_name=author_name, paper_id=paper_id)

    def get_paper(self, paper_id: str) -> Optional[dict]:
        """Fetch a paper by id. Returns None if not found."""
        query = 'MATCH (p:Paper {id: $id}) RETURN p'
        with self.driver.session() as session:
            result = session.run(query, id=paper_id)
            record = result.single()
            return dict(record['p']) if record else None

    def get_paper_count(self) -> int:
        """Return total number of Paper nodes in the database."""
        with self.driver.session() as session:
            result = session.run('MATCH (p:Paper) RETURN count(p) AS n')
            return result.single()['n']

    def batch_upsert_papers(self, papers: list[dict]) -> int:
        """
        Efficiently insert many papers using UNWIND.
        Much faster than calling upsert_paper() in a loop.

        Args:
            papers: list of paper dicts

        Returns:
            Number of papers processed
        """
        query = """
            UNWIND $papers AS paper
            MERGE (p:Paper {id: paper.id})
            SET p.title      = paper.title,
                p.abstract   = paper.abstract,
                p.year       = paper.year,
                p.categories = paper.categories
        """
        with self.driver.session() as session:
            session.run(query, papers=papers)
        return len(papers)
    
    # ── Read methods ──────────────────────────────────────────────────────

    def get_papers_by_year(self, year: int) -> list[dict]:
        """
        Return all papers published in a given year.

        HOW IT WORKS:
        We send a MATCH query asking for every Paper node
        whose 'year' property equals the number we pass in.
        Neo4j sends back one record per matching paper.
        We loop through them, convert each Neo4j node into
        a plain Python dictionary, and collect them in a list.

        PARAMETERS:
            year: the 4-digit year to search for, e.g. 2021

        RETURNS:
            list of paper dictionaries. Empty list if none found.
        """
        query = """
            MATCH (p:Paper {year: $year})
            RETURN p
            ORDER BY p.id
        """
        papers = []

        with self.driver.session() as session:
            result = session.run(query, year=year)
            for record in result:
                papers.append(dict(record['p']))

        print(f'[Neo4jClient] get_papers_by_year({year}) → {len(papers)} papers found.')
        return papers

    def get_authors_of_paper(self, paper_id: str) -> list[str]:
        """
        Return the names of all authors who wrote a given paper.

        HOW IT WORKS:
        We follow the AUTHORED_BY relationship outward from the
        Paper node to find connected Author nodes.
        The arrow direction matters — Paper-[:AUTHORED_BY]->Author
        is how the relationship was stored in link_author_to_paper().

        PARAMETERS:
            paper_id: the unique id of the paper, e.g. 'test:001'

        RETURNS:
            list of author name strings. Empty list if none found.
        """
        query = """
            MATCH (p:Paper {id: $paper_id})-[:AUTHORED_BY]->(a:Author)
            RETURN a.name AS name
            ORDER BY a.name
        """
        authors = []

        with self.driver.session() as session:
            result = session.run(query, paper_id=paper_id)
            for record in result:
                authors.append(record['name'])

        print(f'[Neo4jClient] get_authors_of_paper("{paper_id}") → {len(authors)} author(s) found.')
        return authors

    def get_paper_neighbors(self, paper_id: str) -> list[dict]:
        """
        Return all papers that share at least one author with this paper.

        HOW IT WORKS:
        This traverses TWO relationships in one query:
        Start at our paper → follow AUTHORED_BY to an Author →
        follow AUTHORED_BY backwards to any other Paper that
        same Author has written.

        This is graph traversal — hopping across two edges to
        find connected nodes. This exact pattern is what APEX's
        Reasoner agent will use to discover related research.

        PARAMETERS:
            paper_id: the unique id of the paper

        RETURNS:
            list of neighboring paper dictionaries. Empty list if none found.
        """
        query = """
            MATCH (p:Paper {id: $paper_id})-[:AUTHORED_BY]->(a:Author)<-[:AUTHORED_BY]-(neighbor:Paper)
            WHERE neighbor <> p
            RETURN DISTINCT neighbor
            ORDER BY neighbor.year DESC
        """
        neighbors = []

        with self.driver.session() as session:
            result = session.run(query, paper_id=paper_id)
            for record in result:
                neighbors.append(dict(record['neighbor']))

        print(f'[Neo4jClient] get_paper_neighbors("{paper_id}") → {len(neighbors)} neighbor(s) found.')
        return neighbors


# ── Test it directly ───────────────────────────────────────────────────────
if __name__ == '__main__':
    client = Neo4jClient()

    # ── Setup: create test data we can query against ──────────────────────

    # Create two papers
    client.upsert_paper({
        'id': 'test:paper_A',
        'title': 'Paper A — Graph Neural Networks',
        'abstract': 'Abstract A',
        'year': 2022,
        'categories': ['cs.AI'],
        'citations': 10
    })
    client.upsert_paper({
        'id': 'test:paper_B',
        'title': 'Paper B — Attention Mechanisms',
        'abstract': 'Abstract B',
        'year': 2022,
        'categories': ['cs.LG'],
        'citations': 5
    })

    # Create one author and link them to BOTH papers
    # This means paper_A and paper_B are neighbors of each other
    client.upsert_author('Alice Researcher', 'MIT')
    client.link_author_to_paper('Alice Researcher', 'test:paper_A')
    client.link_author_to_paper('Alice Researcher', 'test:paper_B')

    print('\n--- Testing get_papers_by_year ---')
    papers_2022 = client.get_papers_by_year(2022)
    print(f'Papers from 2022: {len(papers_2022)}')
    for p in papers_2022[:3]:
        print(f'  → {p["id"]} | {p["title"]}')

    papers_1800 = client.get_papers_by_year(1800)
    print(f'Papers from 1800: {papers_1800}')
    assert papers_1800 == [], 'Should be empty list for year with no papers'
    print('  ✓ Empty year returns empty list correctly')

    print('\n--- Testing get_authors_of_paper ---')
    authors = client.get_authors_of_paper('test:paper_A')
    print(f'Authors of paper_A: {authors}')
    assert 'Alice Researcher' in authors, 'Alice should be an author of paper_A'
    print('  ✓ Correct author returned')

    no_authors = client.get_authors_of_paper('this:does:not:exist')
    assert no_authors == [], 'Should be empty list for non-existent paper'
    print('  ✓ Non-existent paper returns empty list correctly')

    print('\n--- Testing get_paper_neighbors ---')
    neighbors = client.get_paper_neighbors('test:paper_A')
    print(f'Neighbors of paper_A: {len(neighbors)}')
    for n in neighbors:
        print(f'  → {n["id"]} | {n["title"]}')
    neighbor_ids = [n['id'] for n in neighbors]
    assert 'test:paper_B' in neighbor_ids, 'paper_B should be a neighbor of paper_A'
    print('  ✓ Correct neighbor returned')

    no_neighbors = client.get_paper_neighbors('this:does:not:exist')
    assert no_neighbors == [], 'Should be empty list for non-existent paper'
    print('  ✓ Non-existent paper returns empty list correctly')

    print('\n✅ All three methods working correctly.')
    print(f'Total papers in Neo4j: {client.get_paper_count()}')

    client.close()