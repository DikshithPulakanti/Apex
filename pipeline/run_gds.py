# pipeline/run_gds.py
# Runs Neo4j GDS algorithms on the APEX knowledge graph

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.neo4j_client import Neo4jClient


def run_gds_algorithms():
    """
    Runs three GDS algorithms on the concept graph:
    1. PageRank        — finds most influential concepts
    2. Louvain         — finds research community clusters
    3. Betweenness     — finds bridge concepts between communities
    """
    neo4j = Neo4jClient()

    # ── Step 1: Create GDS graph projection ──────────────────────────────
    # Before running algorithms, GDS needs an in-memory copy of your graph
    # called a "projection". It's like a snapshot GDS works on.
    print('\n[GDS] Creating graph projection...')

    # Drop projection if it already exists
    with neo4j.driver.session() as session:
        try:
            session.run("CALL gds.graph.drop('concept-graph', false)")
        except Exception:
            pass

        # Create projection — only Concept nodes and CO_OCCURS_WITH edges
        session.run("""
            CALL gds.graph.project(
                'concept-graph',
                'Concept',
                {
                    CO_OCCURS_WITH: {
                        orientation: 'UNDIRECTED',
                        properties: 'weight'
                    }
                }
            )
        """)
    print('[GDS] Projection created.')

    # ── Step 2: PageRank ──────────────────────────────────────────────────
    print('\n[GDS] Running PageRank...')
    with neo4j.driver.session() as session:
        session.run("""
            CALL gds.pageRank.write('concept-graph', {
                writeProperty: 'pagerank',
                maxIterations: 20,
                dampingFactor: 0.85
            })
        """)
    print('[GDS] PageRank scores written to Concept nodes.')

    # ── Step 3: Louvain Community Detection ───────────────────────────────
    print('\n[GDS] Running Louvain community detection...')
    with neo4j.driver.session() as session:
        session.run("""
            CALL gds.louvain.write('concept-graph', {
                writeProperty: 'community'
            })
        """)
    print('[GDS] Community labels written to Concept nodes.')

    # ── Step 4: Betweenness Centrality ────────────────────────────────────
    print('\n[GDS] Running Betweenness Centrality...')
    with neo4j.driver.session() as session:
        session.run("""
            CALL gds.betweenness.write('concept-graph', {
                writeProperty: 'betweenness'
            })
        """)
    print('[GDS] Betweenness scores written to Concept nodes.')

    # ── Step 5: Drop projection to free memory ────────────────────────────
    with neo4j.driver.session() as session:
        session.run("CALL gds.graph.drop('concept-graph')")
    print('\n[GDS] Projection dropped.')

    # ── Step 6: Show results ──────────────────────────────────────────────
    print('\n--- Top 10 concepts by PageRank ---')
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (c:Concept)
            WHERE c.pagerank IS NOT NULL
            RETURN c.name AS concept, c.pagerank AS pagerank,
                   c.community AS community
            ORDER BY c.pagerank DESC
            LIMIT 10
        """)
        for record in result:
            print(f'  [{record["community"]}] {record["concept"]} '
                  f'(pagerank: {record["pagerank"]:.4f})')

    print('\n--- Top 10 bridge concepts by Betweenness ---')
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (c:Concept)
            WHERE c.betweenness IS NOT NULL
            RETURN c.name AS concept, c.betweenness AS betweenness,
                   c.community AS community
            ORDER BY c.betweenness DESC
            LIMIT 10
        """)
        for record in result:
            print(f'  [{record["community"]}] {record["concept"]} '
                  f'(betweenness: {record["betweenness"]:.2f})')

    print('\n--- Community distribution ---')
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (c:Concept)
            WHERE c.community IS NOT NULL
            RETURN c.community AS community, count(c) AS size
            ORDER BY size DESC
            LIMIT 10
        """)
        for record in result:
            print(f'  Community {record["community"]}: '
                  f'{record["size"]} concepts')

    print('\n✅ GDS algorithms complete.')
    neo4j.close()


if __name__ == '__main__':
    run_gds_algorithms()