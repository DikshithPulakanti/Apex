# bridge.py
# APEX Bridge Service — the one thing that actually consumes Kafka and
# exposes it, plus the human-review workflow, over HTTP for the frontend.
#
# Why this exists: a browser can't speak the Kafka wire protocol, and a
# hypothesis flagged `pending_review` needs somewhere a human can act on
# it. This is that somewhere — a thin FastAPI service, not a new agent.

import os
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from events.kafka_manager import TOPICS, EventSubscriber
from events.agent_events import emit_hypothesis_validated, emit_hypothesis_rejected
from database.postgres_client import PostgresClient
from database.neo4j_client import Neo4jClient

_recent_events = deque(maxlen=500)
_events_lock = threading.Lock()


def _consume_forever():
    """
    Background thread: pulls every APEX event off Kafka into a bounded
    in-memory buffer the frontend can poll.

    One EventSubscriber per process lifetime, with a fresh random group_id
    so a bridge restart replays recent topic history into an otherwise
    empty feed. The inner `for` loop naturally exits after ~5s of no new
    messages (see EventSubscriber's consumer_timeout_ms) — re-entering it
    on the *same* consumer is a harmless no-op, not a reconnect, so this
    doesn't replay history again or hammer the broker while idle.
    """
    while True:
        try:
            subscriber = EventSubscriber(TOPICS, group_id=f'bridge-dashboard-{uuid.uuid4().hex[:8]}')
            while True:
                for message in subscriber.consumer:
                    with _events_lock:
                        _recent_events.appendleft(message.value)
        except Exception as e:
            print(f'[bridge] Kafka consumer error, retrying in 5s: {e}')
            time.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_consume_forever, daemon=True).start()
    yield


app = FastAPI(title='APEX Bridge', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # local dev dashboard only
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/healthz')
def healthz():
    return {'status': 'ok'}


@app.get('/events/recent')
def get_recent_events(limit: int = 50):
    with _events_lock:
        return list(_recent_events)[:limit]


@app.get('/reviews')
def list_pending_reviews():
    postgres = PostgresClient()
    try:
        return postgres.get_pending_reviews()
    finally:
        postgres.close()


class ReviewDecision(BaseModel):
    decision: str  # 'approved' or 'rejected'
    decided_by: str = ''
    decision_notes: str = ''


@app.post('/reviews/{hypothesis_id}/decide')
def decide_review(hypothesis_id: str, body: ReviewDecision):
    if body.decision not in ('approved', 'rejected'):
        raise HTTPException(400, "decision must be 'approved' or 'rejected'")

    postgres = PostgresClient()
    try:
        updated = postgres.decide_review(hypothesis_id, body.decision, body.decided_by, body.decision_notes)
    finally:
        postgres.close()

    if not updated:
        raise HTTPException(404, f'No pending review found for {hypothesis_id}')

    status = 'validated' if body.decision == 'approved' else 'rejected'
    neo4j = Neo4jClient()
    try:
        with neo4j.driver.session() as session:
            session.run(
                'MATCH (h:Hypothesis {id: $id}) SET h.status = $status',
                id=hypothesis_id, status=status,
            )
    finally:
        neo4j.close()

    if body.decision == 'approved':
        emit_hypothesis_validated(hypothesis_id, 1.0, method='human')
    else:
        emit_hypothesis_rejected(hypothesis_id, 0.0, reason='rejected by human reviewer')

    patent_id = None
    if body.decision == 'approved':
        from orchestrator import run_inventor_phase
        patent_id, _, _ = run_inventor_phase(hypothesis_id)

    return {'hypothesis_id': hypothesis_id, 'status': status, 'patent_id': patent_id}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('BRIDGE_PORT', 8000)))
