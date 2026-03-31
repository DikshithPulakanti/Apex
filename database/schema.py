# database/schema.py
# Run ONCE to set up the complete APEX graph schema

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.neo4j_client import Neo4jClient

SCHEMA_QUERIES = [
    # ── Uniqueness constraints (these create indexes automatically) ───────
    'CREATE CONSTRAINT paper_id_unique   IF NOT EXISTS FOR (p:Paper)      REQUIRE p.id   IS UNIQUE',
    'CREATE CONSTRAINT concept_name_uniq IF NOT EXISTS FOR (c:Concept)    REQUIRE c.name IS UNIQUE',
    'CREATE CONSTRAINT hyp_id_unique     IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id   IS UNIQUE',

    # ── Extra indexes for nodes without constraints ────────────────────────
    'CREATE INDEX author_name  IF NOT EXISTS FOR (a:Author) ON (a.name)',
    'CREATE INDEX agent_name   IF NOT EXISTS FOR (a:Agent)  ON (a.name)',
    'CREATE INDEX patent_id    IF NOT EXISTS FOR (p:Patent) ON (p.id)',

    # ── Seed the 4 Agent nodes ────────────────────────────────────────────
    """
    MERGE (a:Agent {name: 'Harvester'})
    SET a.role   = 'Ingests and processes scientific papers',
        a.model  = 'claude-sonnet-4-20250514',
        a.status = 'idle'
    """,
    """
    MERGE (a:Agent {name: 'Reasoner'})
    SET a.role   = 'Traverses knowledge graph and generates hypotheses',
        a.model  = 'claude-sonnet-4-20250514',
        a.status = 'idle'
    """,
    """
    MERGE (a:Agent {name: 'Skeptic'})
    SET a.role   = 'Adversarially challenges hypotheses',
        a.model  = 'claude-sonnet-4-20250514',
        a.status = 'idle'
    """,
    """
    MERGE (a:Agent {name: 'Inventor'})
    SET a.role   = 'Drafts patent claims from validated hypotheses',
        a.model  = 'claude-sonnet-4-20250514',
        a.status = 'idle'
    """,
]


def setup_schema():
    client = Neo4jClient()
    print('Setting up APEX graph schema...')

    for query in SCHEMA_QUERIES:
        with client.driver.session() as session:
            session.run(query.strip())

    print('✅ Schema setup complete')

    # Verify
    with client.driver.session() as session:
        agents = session.run('MATCH (a:Agent) RETURN a.name ORDER BY a.name').data()
        print(f'Agents created: {[x["a.name"] for x in agents]}')

        constraints = session.run('SHOW CONSTRAINTS').data()
        print(f'Constraints: {len(constraints)}')

        indexes = session.run('SHOW INDEXES').data()
        print(f'Indexes: {len(indexes)}')

    client.close()


if __name__ == '__main__':
    setup_schema()