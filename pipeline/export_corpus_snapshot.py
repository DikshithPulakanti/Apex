# pipeline/export_corpus_snapshot.py
# Exports the current Neo4j paper corpus to a JSONL snapshot so it can be
# versioned with DVC. Neo4j itself is a live database, not a flat file, so
# this is the only way to pin "what corpus a given run/model was built on."

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from database.neo4j_client import Neo4jClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, 'data', 'exports', 'papers_snapshot.jsonl')


def export_snapshot(output_path: str = OUTPUT_PATH) -> int:
    """Streams every Paper node to a JSONL file, one paper per line."""
    neo4j = Neo4jClient()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with neo4j.driver.session() as session, open(output_path, 'w') as f:
        result = session.run("""
            MATCH (p:Paper)
            RETURN p.id AS id, p.title AS title, p.abstract AS abstract,
                   p.year AS year, p.categories AS categories, p.citations AS citations
            ORDER BY p.id
        """)
        for record in result:
            f.write(json.dumps(dict(record)) + '\n')
            count += 1

    neo4j.close()
    print(f'[export_corpus_snapshot] Wrote {count} papers to {output_path}')
    return count


if __name__ == '__main__':
    export_snapshot()
