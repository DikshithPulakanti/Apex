# pipeline/extract_concepts.py
# Extract concepts from paper abstracts using Claude, store in Neo4j

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import anthropic
from dotenv import load_dotenv
from database.neo4j_client import Neo4jClient

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))


def extract_concepts_from_abstract(abstract: str) -> list:
    """Ask Claude to extract 3-5 key concepts from a paper abstract."""
    if not abstract or len(abstract) < 50:
        return []

    prompt = f"""Extract 3-5 key technical concepts from this research paper abstract.
Return ONLY specific, meaningful research concepts (not generic words like "method" or "approach").
Each concept should be 1-4 words, lowercase.

Abstract: {abstract[:500]}

Return ONLY a JSON array of strings, no markdown:
["concept1", "concept2", "concept3"]"""

    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = next((b.text for b in response.content if b.type == 'text'), '').strip()
        text = text.replace("```json", "").replace("```", "").strip()
        concepts = json.loads(text)
        return [c.lower().strip() for c in concepts if len(c) > 2]
    except Exception as e:
        return []


def run_extraction(batch_size: int = 10, delay: float = 1.0):
    """Extract concepts from all papers that don't have concepts yet."""
    neo4j = Neo4jClient()

    # Find papers without concepts
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (p:Paper)
            WHERE NOT (p)-[:MENTIONS]->(:Concept)
            AND p.abstract IS NOT NULL
            AND size(p.abstract) > 50
            RETURN p.id AS id, p.abstract AS abstract
        """)
        papers = [dict(r) for r in result]

    total = len(papers)
    print(f"=== Concept Extraction ===")
    print(f"Papers without concepts: {total}\n")

    if total == 0:
        print("All papers already have concepts.")
        neo4j.close()
        return

    extracted = 0
    concept_count = 0

    for i, paper in enumerate(papers):
        concepts = extract_concepts_from_abstract(paper['abstract'])

        if concepts:
            with neo4j.driver.session() as session:
                for concept in concepts:
                    # Create concept node
                    session.run("""
                        MERGE (c:Concept {name: $name})
                        ON CREATE SET c.created_at = datetime()
                    """, name=concept)

                    # Link paper → concept
                    session.run("""
                        MATCH (p:Paper {id: $paper_id})
                        MATCH (c:Concept {name: $concept_name})
                        MERGE (p)-[:MENTIONS]->(c)
                    """, paper_id=paper['id'], concept_name=concept)

            extracted += 1
            concept_count += len(concepts)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{total} papers | {concept_count} concepts extracted")

        time.sleep(delay)

    # Rebuild co-occurrence relationships
    print(f"\nRebuilding co-occurrence relationships...")
    with neo4j.driver.session() as session:
        session.run("""
            MATCH (c1:Concept)<-[:MENTIONS]-(p:Paper)-[:MENTIONS]->(c2:Concept)
            WHERE c1.name < c2.name
            WITH c1, c2, count(p) AS weight
            MERGE (c1)-[r:CO_OCCURS_WITH]-(c2)
            SET r.weight = weight
        """)

    # Rebuild PageRank and community detection
    print("Running PageRank...")
    with neo4j.driver.session() as session:
        try:
            session.run("CALL gds.graph.drop('concept-graph', false)")
        except:
            pass
        session.run("""
            CALL gds.graph.project(
                'concept-graph',
                'Concept',
                {CO_OCCURS_WITH: {orientation: 'UNDIRECTED'}}
            )
        """)
        session.run("""
            CALL gds.pageRank.write('concept-graph', {
                writeProperty: 'pagerank',
                maxIterations: 20,
                dampingFactor: 0.85
            })
        """)
        print("Running community detection...")
        session.run("""
            CALL gds.louvain.write('concept-graph', {
                writeProperty: 'community'
            })
        """)
        session.run("CALL gds.graph.drop('concept-graph', false)")

    # Final stats
    with neo4j.driver.session() as session:
        result = session.run("MATCH (c:Concept) RETURN count(c) AS count")
        total_concepts = result.single()['count']
        result = session.run("MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) AS count")
        total_cooccur = result.single()['count']

    print(f"\n=== Extraction Complete ===")
    print(f"Papers processed: {extracted}/{total}")
    print(f"Concepts extracted: {concept_count}")
    print(f"Total concepts in graph: {total_concepts}")
    print(f"Co-occurrence relationships: {total_cooccur}")

    neo4j.close()


if __name__ == '__main__':
    run_extraction(delay=0.5)