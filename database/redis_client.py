# database/redis_client.py
# APEX Redis Client — caching layer

import redis
import json
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class RedisClient:
    """
    Handles caching for APEX.
    
    Two main uses:
    1. Track which papers have already been processed
       so we don't re-insert them on every pipeline run
    2. Cache expensive computations like embeddings
    """

    def __init__(self):
        self.client = redis.Redis(
            host             = os.getenv('REDIS_HOST', 'localhost'),
            port             = int(os.getenv('REDIS_PORT', '6379')),
            db               = 0,
            decode_responses = True
        )
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.client.ping()
            print('[RedisClient] Connected to Redis.')
        except Exception as e:
            raise RuntimeError(f'Cannot connect to Redis: {e}')

    def is_processed(self, paper_id: str) -> bool:
        """Returns True if this paper has already been processed."""
        return bool(self.client.exists(f'processed:{paper_id}'))

    def mark_processed(self, paper_id: str) -> None:
        """Marks a paper as processed. Expires after 24 hours."""
        self.client.setex(f'processed:{paper_id}', 86400, '1')

    def mark_processed_batch(self, paper_ids: list) -> None:
        """Marks many papers as processed at once."""
        pipe = self.client.pipeline()
        for paper_id in paper_ids:
            pipe.setex(f'processed:{paper_id}', 86400, '1')
        pipe.execute()

    def filter_unprocessed(self, paper_ids: list) -> list:
        """Returns only the paper ids that haven't been processed yet."""
        return [pid for pid in paper_ids if not self.is_processed(pid)]

    def cache_set(self, key: str, value: dict, expiry_seconds: int = 3600) -> None:
        """Store any dictionary in cache as JSON."""
        self.client.setex(key, expiry_seconds, json.dumps(value))

    def cache_get(self, key: str) -> Optional[dict]:
        """Retrieve a cached dictionary. Returns None if not found or expired."""
        data = self.client.get(key)
        return json.loads(data) if data else None

    def close(self):
        self.client.close()
        print('[RedisClient] Connection closed.')


# ── Test it directly ───────────────────────────────────────────────────────
if __name__ == '__main__':
    client = RedisClient()

    print('\n--- Testing paper processing cache ---')
    client.mark_processed('paper:001')
    client.mark_processed('paper:002')

    assert client.is_processed('paper:001') == True
    assert client.is_processed('paper:999') == False
    print('  ✓ is_processed working correctly')

    unprocessed = client.filter_unprocessed(['paper:001', 'paper:002', 'paper:003'])
    assert unprocessed == ['paper:003']
    print(f'  ✓ filter_unprocessed: {unprocessed}')

    print('\n--- Testing general cache ---')
    client.cache_set('test:key', {'title': 'Test Paper', 'year': 2024})
    result = client.cache_get('test:key')
    assert result['title'] == 'Test Paper'
    print(f'  ✓ cache_set and cache_get working: {result}')

    missing = client.cache_get('this:does:not:exist')
    assert missing is None
    print('  ✓ Missing key returns None correctly')

    print('\n✅ Redis client working correctly.')
    client.close()
