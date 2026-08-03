# tests/test_kafka_manager.py
# Unit tests for the Kafka event publisher — kafka-python is mocked so
# these tests don't need a live broker.

from unittest.mock import MagicMock

from events import kafka_manager


def test_topics_list_has_exactly_seven_entries():
    assert len(kafka_manager.TOPICS) == 7
    assert set(kafka_manager.TOPICS) == {
        'papers.ingested', 'hypothesis.created', 'hypothesis.validated',
        'hypothesis.rejected', 'patent.drafted', 'research_plan.created', 'agent.status',
    }


def test_publish_builds_correct_event_envelope(monkeypatch):
    fake_producer = MagicMock()
    monkeypatch.setattr(kafka_manager, 'KafkaProducer', lambda **kwargs: fake_producer)

    publisher = kafka_manager.EventPublisher()
    event = publisher.publish('agent.status', 'harvester', {'status': 'starting'}, key='k1')

    assert event['agent'] == 'harvester'
    assert event['type'] == 'agent.status'
    assert event['data'] == {'status': 'starting'}
    assert 'event_id' in event and 'timestamp' in event
    fake_producer.send.assert_called_once_with('agent.status', value=event, key='k1')
    fake_producer.flush.assert_called_once()
