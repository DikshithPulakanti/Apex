# database/neo4j_client.py
# APEX Neo4j Client

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
        uri      = os.getenv('NEO4J_URI',      'bolt://localhost:7687')
        user     = os.getenv('NEO4J_USER',     'neo4j')
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
        """Insert or update a paper node."""
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
        """Create AUTHORED_BY relationship between Author and Paper."""
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

    def batch_upsert_papers(self, papers: list) -> int:
        """Efficiently insert many papers using UNWIND."""
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

    def get_papers_by_year(self, year: int) -> list:
        """Return all papers published in a given year."""
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

    def get_authors_of_paper(self, paper_id: str) -> list:
        """Return the names of all authors who wrote a given paper."""
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

    def get_paper_neighbors(self, paper_id: str) -> list:
        """Return all papers that share at least one author with this paper."""
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

    # ── Concept operations ────────────────────────────────────────────────

    def upsert_concept(self, name: str, domain: str = '') -> None:
        """Insert or update a Concept node."""
        query = """
            MERGE (c:Concept {name: $name})
            SET c.domain = $domain
        """
        with self.driver.session() as session:
            session.run(query, name=name, domain=domain)

    def link_paper_to_concept(self, paper_id: str, concept_name: str) -> None:
        """Create MENTIONS relationship between Paper and Concept."""
        query = """
            MATCH (p:Paper   {id:   $paper_id})
            MATCH (c:Concept {name: $concept_name})
            MERGE (p)-[:MENTIONS]->(c)
        """
        with self.driver.session() as session:
            session.run(query, paper_id=paper_id, concept_name=concept_name)

    def get_concepts_for_paper(self, paper_id: str) -> list:
        """Return all concept names mentioned by a given paper."""
        query = """
            MATCH (p:Paper {id: $paper_id})-[:MENTIONS]->(c:Concept)
            RETURN c.name AS name
            ORDER BY c.name
        """
        concepts = []
        with self.driver.session() as session:
            result = session.run(query, paper_id=paper_id)
            for record in result:
                concepts.append(record['name'])
        return concepts

    # ── Embedding operations ──────────────────────────────────────────────

    def get_papers_without_embeddings(self, limit: int = 100) -> list:
        """Returns papers that don't have an embedding yet."""
        query = """
            MATCH (p:Paper)
            WHERE p.embedding IS NULL
            RETURN p
            LIMIT $limit
        """
        papers = []
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            for record in result:
                papers.append(dict(record['p']))
        print(f'[Neo4jClient] Found {len(papers)} papers without embeddings.')
        return papers

    def set_paper_embedding(self, paper_id: str, embedding: list) -> None:
        """Stores an embedding vector on a Paper node."""
        query = """
            MATCH (p:Paper {id: $paper_id})
            SET p.embedding = $embedding
        """
        with self.driver.session() as session:
            session.run(query, paper_id=paper_id, embedding=embedding)

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns counts of all node types and relationships in the graph."""
        with self.driver.session() as session:
            papers        = session.run('MATCH (p:Paper) RETURN count(p) AS n').single()['n']
            authors       = session.run('MATCH (a:Author) RETURN count(a) AS n').single()['n']
            relationships = session.run('MATCH ()-[r]->() RETURN count(r) AS n').single()['n']

        stats = {
            'papers':        papers,
            'authors':       authors,
            'relationships': relationships
        }
        print(f'[Neo4jClient] Stats: {stats}')
        return stats


# ── Test it directly ───────────────────────────────────────────────────────
if __name__ == '__main__':
    client = Neo4jClient()

    client.upsert_paper({
        'id': 'test:paper_A', 'title': 'Paper A — Graph Neural Networks',
        'abstract': 'Abstract A', 'year': 2022, 'categories': ['cs.AI'], 'citations': 10
    })
    client.upsert_paper({
        'id': 'test:paper_B', 'title': 'Paper B — Attention Mechanisms',
        'abstract': 'Abstract B', 'year': 2022, 'categories': ['cs.LG'], 'citations': 5
    })
    client.upsert_author('Alice Researcher', 'MIT')
    client.link_author_to_paper('Alice Researcher', 'test:paper_A')
    client.link_author_to_paper('Alice Researcher', 'test:paper_B')

    print('\n--- Testing get_papers_by_year ---')
    papers_2022 = client.get_papers_by_year(2022)
    print(f'Papers from 2022: {len(papers_2022)}')

    print('\n--- Testing get_authors_of_paper ---')
    authors = client.get_authors_of_paper('test:paper_A')
    assert 'Alice Researcher' in authors
    print('  ✓ Correct author returned')

    print('\n--- Testing get_paper_neighbors ---')
    neighbors = client.get_paper_neighbors('test:paper_A')
    assert any(n['id'] == 'test:paper_B' for n in neighbors)
    print('  ✓ Correct neighbor returned')

    print('\n--- Testing get_stats ---')
    stats = client.get_stats()
    assert stats['papers'] > 0
    print('  ✓ Stats returned correctly')

    print('\n✅ All methods working correctly.')
    client.close()