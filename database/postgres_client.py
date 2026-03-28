# database/postgres_client.py
# APEX PostgreSQL Client — Day 3 Task 4

import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


class PostgresClient:
    """
    Handles all communication between APEX and PostgreSQL.

    PostgreSQL is APEX's operational logbook — it tracks pipeline
    runs, agent activity, and other simple relational metadata.

    Think of it this way:
        Neo4j  = the knowledge brain  (graph data, relationships)
        Postgres = the logbook        (what happened, when, how many)
    """

    def __init__(self):
        """
        Opens a single connection to PostgreSQL.

        We read credentials from the .env file, exactly like
        we did in Neo4jClient.
        """
        self.conn = psycopg2.connect(
            host     = os.getenv('POSTGRES_HOST',     'localhost'),
            port     = os.getenv('POSTGRES_PORT',     '5432'),
            dbname   = os.getenv('POSTGRES_DB',       'apex_db'),
            user     = os.getenv('POSTGRES_USER',     'apex'),
            password = os.getenv('POSTGRES_PASSWORD', 'apexpassword')
        )

        # autocommit=False means WE control when changes are saved.
        # Nothing is permanent until we call self.conn.commit().
        self.conn.autocommit = False

        print('[PostgresClient] Connected to PostgreSQL.')
        self._create_tables()

    def _create_tables(self):
        """
        Creates the pipeline_runs table if it doesn't exist yet.

        The 'IF NOT EXISTS' means running this twice won't crash
        or create duplicates — same safe pattern as MERGE in Neo4j.

        Column breakdown:
            id           — auto-incrementing number, primary key
            topic        — what was searched, e.g. 'cat:cs.AI'
            started_at   — when the pipeline run began
            papers_found — how many papers were ingested
        """
        query = """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id           SERIAL PRIMARY KEY,
                topic        TEXT,
                started_at   TIMESTAMP,
                papers_found INTEGER
            )
        """

        # A cursor is like a session in Neo4j — it's your active
        # channel for executing SQL commands.
        cursor = self.conn.cursor()
        cursor.execute(query)

        # THIS IS CRITICAL — without commit(), the table creation
        # is not permanently saved. It exists only in this session.
        self.conn.commit()

        cursor.close()
        print('[PostgresClient] pipeline_runs table ready.')

    def log_pipeline_run(self, topic: str, papers_found: int) -> int:
        """
        Inserts one row into pipeline_runs recording a pipeline execution.

        PARAMETERS:
            topic        : what was searched, e.g. 'cat:cs.AI'
            papers_found : how many papers were found and inserted

        RETURNS:
            the id of the newly inserted row
        """
        query = """
            INSERT INTO pipeline_runs (topic, started_at, papers_found)
            VALUES (%s, %s, %s)
            RETURNING id
        """

        cursor = self.conn.cursor()

        # %s is PostgreSQL's placeholder — same concept as $param in Cypher.
        # NEVER build SQL strings with f-strings — same injection risk.
        cursor.execute(query, (topic, datetime.now(), papers_found))

        # RETURNING id means PostgreSQL sends back the id it just created.
        # fetchone() grabs that one row back.
        row = cursor.fetchone()
        new_id = row[0]

        # Save permanently. Without this, the row disappears
        # when the connection closes.
        self.conn.commit()
        cursor.close()

        print(f'[PostgresClient] Logged pipeline run #{new_id}: '
              f'topic="{topic}", papers_found={papers_found}')
        return new_id

    def get_all_runs(self) -> list[tuple]:
        """
        Returns all rows from pipeline_runs, newest first.

        RETURNS:
            list of tuples, each tuple is one row:
            (id, topic, started_at, papers_found)
        """
        query = """
            SELECT id, topic, started_at, papers_found
            FROM pipeline_runs
            ORDER BY started_at DESC
        """

        cursor = self.conn.cursor()
        cursor.execute(query)

        # fetchall() grabs every row at once as a list of tuples.
        rows = cursor.fetchall()
        cursor.close()

        return rows

    def close(self):
        """
        Closes the PostgreSQL connection cleanly.
        Always call this when your program finishes.
        """
        self.conn.close()
        print('[PostgresClient] Connection closed.')


# ── Test it directly ───────────────────────────────────────────────────────
if __name__ == '__main__':
    client = PostgresClient()

    print('\n--- Inserting 3 pipeline run logs ---')
    client.log_pipeline_run('cat:cs.AI',  847)
    client.log_pipeline_run('cat:cs.LG',  612)
    client.log_pipeline_run('cat:cs.CL',  731)

    print('\n--- Reading all rows back ---')
    rows = client.get_all_runs()
    print(f'Total rows in pipeline_runs: {len(rows)}')
    for row in rows:
        print(f'  id={row[0]} | topic={row[1]} | '
              f'started_at={row[2]} | papers_found={row[3]}')

    assert len(rows) >= 3, 'Should have at least 3 rows'
    print('\n✅ PostgreSQL client working correctly.')

    client.close()