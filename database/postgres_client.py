# database/postgres_client.py
# APEX PostgreSQL Client — Day 3 Task 4

import psycopg2
from datetime import datetime
import os
from typing import Optional
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
        Creates the pipeline_runs and hypothesis_reviews tables if they
        don't exist yet.

        The 'IF NOT EXISTS' means running this twice won't crash
        or create duplicates — same safe pattern as MERGE in Neo4j.

        Column breakdown (pipeline_runs):
            id           — auto-incrementing number, primary key
            topic        — what was searched, e.g. 'cat:cs.AI'
            started_at   — when the pipeline run began
            papers_found — how many papers were ingested

        Column breakdown (hypothesis_reviews):
            This is APEX's human-review queue — hypotheses the Skeptic
            couldn't confidently auto-decide land here with status
            'pending' until a human approves/rejects them.
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

        review_query = """
            CREATE TABLE IF NOT EXISTS hypothesis_reviews (
                id                 SERIAL PRIMARY KEY,
                hypothesis_id      TEXT UNIQUE NOT NULL,
                statement_snapshot TEXT,
                bert_confidence    FLOAT,
                bert_verdict       TEXT,
                claude_score       FLOAT,
                claude_verdict     TEXT,
                claude_reasoning   TEXT,
                mlflow_run_id      TEXT,
                status             TEXT NOT NULL DEFAULT 'pending',
                created_at         TIMESTAMP NOT NULL DEFAULT now(),
                decided_at         TIMESTAMP,
                decided_by         TEXT,
                decision_notes     TEXT
            )
        """
        cursor = self.conn.cursor()
        cursor.execute(review_query)
        self.conn.commit()
        cursor.close()
        print('[PostgresClient] hypothesis_reviews table ready.')

    def enqueue_review(self, hypothesis_id: str, statement_snapshot: str,
                        bert_confidence: float, bert_verdict: str,
                        claude_score: float, claude_verdict: str,
                        claude_reasoning: str, mlflow_run_id: str) -> int:
        """
        Flags a hypothesis for human review. Idempotent: re-running the
        Skeptic on the same hypothesis just refreshes the existing pending
        row instead of creating a duplicate.
        """
        query = """
            INSERT INTO hypothesis_reviews
                (hypothesis_id, statement_snapshot, bert_confidence, bert_verdict,
                 claude_score, claude_verdict, claude_reasoning, mlflow_run_id, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending')
            ON CONFLICT (hypothesis_id) DO UPDATE SET
                statement_snapshot = EXCLUDED.statement_snapshot,
                bert_confidence    = EXCLUDED.bert_confidence,
                bert_verdict       = EXCLUDED.bert_verdict,
                claude_score       = EXCLUDED.claude_score,
                claude_verdict     = EXCLUDED.claude_verdict,
                claude_reasoning   = EXCLUDED.claude_reasoning,
                mlflow_run_id      = EXCLUDED.mlflow_run_id,
                status             = 'pending',
                decided_at         = NULL,
                decided_by         = NULL,
                decision_notes     = NULL
            RETURNING id
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (
            hypothesis_id, statement_snapshot, bert_confidence, bert_verdict,
            claude_score, claude_verdict, claude_reasoning, mlflow_run_id,
        ))
        review_id = cursor.fetchone()[0]
        self.conn.commit()
        cursor.close()
        print(f'[PostgresClient] Flagged {hypothesis_id} for human review (review #{review_id})')
        return review_id

    def get_pending_reviews(self) -> list:
        """Returns every hypothesis currently awaiting human review, newest first."""
        query = """
            SELECT id, hypothesis_id, statement_snapshot, bert_confidence, bert_verdict,
                   claude_score, claude_verdict, claude_reasoning, mlflow_run_id,
                   status, created_at
            FROM hypothesis_reviews
            WHERE status = 'pending'
            ORDER BY created_at DESC
        """
        cursor = self.conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return rows

    def get_review(self, hypothesis_id: str) -> Optional[dict]:
        """Fetches one review row by hypothesis id, or None if it doesn't exist."""
        query = """
            SELECT id, hypothesis_id, statement_snapshot, bert_confidence, bert_verdict,
                   claude_score, claude_verdict, claude_reasoning, mlflow_run_id,
                   status, created_at, decided_at, decided_by, decision_notes
            FROM hypothesis_reviews
            WHERE hypothesis_id = %s
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (hypothesis_id,))
        row = cursor.fetchone()
        if not row:
            cursor.close()
            return None
        columns = [desc[0] for desc in cursor.description]
        result = dict(zip(columns, row))
        cursor.close()
        return result

    def decide_review(self, hypothesis_id: str, decision: str,
                       decided_by: str = '', decision_notes: str = '') -> bool:
        """
        Records a human decision ('approved' or 'rejected') for a pending
        review. Guarded by `WHERE status = 'pending'` so a double-click or
        race between two reviewers is a harmless no-op, not a double-decide.

        Returns True if a pending row was actually updated, False if there
        was nothing pending to decide (already decided, or unknown id).
        """
        query = """
            UPDATE hypothesis_reviews
            SET status = %s, decided_at = now(), decided_by = %s, decision_notes = %s
            WHERE hypothesis_id = %s AND status = 'pending'
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (decision, decided_by, decision_notes, hypothesis_id))
        updated = cursor.rowcount > 0
        self.conn.commit()
        cursor.close()
        return updated

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