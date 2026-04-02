# events/kafka_manager.py
# APEX Kafka event manager — publish and subscribe to agent events

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import uuid
from datetime import datetime, timezone
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError


# ── APEX Event Topics ────────────────────────────────────────────────────

TOPICS = [
    'papers.ingested',        # Harvester → Reasoner
    'hypothesis.created',     # Reasoner → Skeptic
    'hypothesis.validated',   # Skeptic → Inventor
    'hypothesis.rejected',    # Skeptic → logs
    'patent.drafted',         # Inventor → Frontend
    'agent.status',           # All agents → Frontend (heartbeat/progress)
]


# ── Topic Setup ──────────────────────────────────────────────────────────

def create_topics(bootstrap_servers='localhost:9092'):
    """Create all APEX Kafka topics if they don't exist."""
    admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)

    new_topics = [
        NewTopic(name=t, num_partitions=1, replication_factor=1)
        for t in TOPICS
    ]

    for topic in new_topics:
        try:
            admin.create_topics([topic])
            print(f'  ✅ Created topic: {topic.name}')
        except TopicAlreadyExistsError:
            print(f'  ⏭️  Topic exists: {topic.name}')

    admin.close()
    print(f'\nAll {len(TOPICS)} topics ready.')


# ── Event Publisher ──────────────────────────────────────────────────────

class EventPublisher:
    """
    Publishes structured events to Kafka topics.
    
    Every event has:
    - event_id: unique UUID
    - timestamp: ISO format
    - agent: which agent published it
    - type: event type (matches topic name)
    - data: the actual payload
    """

    def __init__(self, bootstrap_servers='localhost:9092'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
        )
        print('[EventPublisher] Connected to Kafka.')

    def publish(self, topic: str, agent: str, data: dict, key: str = None):
        """Publish an event to a Kafka topic."""
        event = {
            'event_id':  str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'agent':     agent,
            'type':      topic,
            'data':      data,
        }

        self.producer.send(topic, value=event, key=key)
        self.producer.flush()

        print(f'[EventPublisher] {agent} → {topic} (id: {event["event_id"][:8]})')
        return event

    def close(self):
        self.producer.close()


# ── Event Subscriber ─────────────────────────────────────────────────────

class EventSubscriber:
    """
    Subscribes to Kafka topics and processes events with a callback.
    """

    def __init__(self, topics: list, group_id: str, bootstrap_servers='localhost:9092'):
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            consumer_timeout_ms=5000,  # return after 5s of no messages
        )
        self.topics = topics
        print(f'[EventSubscriber:{group_id}] Listening on: {", ".join(topics)}')

    def consume(self, callback):
        """Process each event with the callback function."""
        for message in self.consumer:
            event = message.value
            print(f'[EventSubscriber] Received: {event["type"]} from {event["agent"]}')
            callback(event)

    def consume_one(self):
        """Consume a single event and return it (for testing)."""
        for message in self.consumer:
            return message.value
        return None

    def close(self):
        self.consumer.close()


# ── Quick test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== APEX Kafka Event System ===\n')

    # Step 1: Create topics
    print('Creating topics...')
    create_topics()

    # Step 2: Publish a test event
    print('\nPublishing test event...')
    pub = EventPublisher()
    event = pub.publish(
        topic='agent.status',
        agent='test',
        data={'message': 'Kafka event system initialized', 'status': 'healthy'}
    )
    pub.close()

    # Step 3: Consume the test event
    print('\nConsuming test event...')
    sub = EventSubscriber(topics=['agent.status'], group_id='test-group')
    received = sub.consume_one()
    sub.close()

    if received:
        print(f'\n✅ Event round-trip successful!')
        print(f'   Agent: {received["agent"]}')
        print(f'   Data:  {received["data"]}')
    else:
        print('\n⚠️  No event received (may need to retry)')

    print('\n=== Kafka Ready ===')