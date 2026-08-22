"""
Smart response cache.

Repetitive queries (extremely common for public/AI APIs — the same prompt,
the same product description, the same FAQ) never need to touch the model
at all. We hash the normalized request payload, check Redis first, and only
fall through to the batching pipeline on a miss.
"""
import hashlib
import json
from typing import Any, Optional

import redis.asyncio as redis


def make_cache_key(model_name: str, payload: dict) -> str:
    # sort_keys makes the hash independent of field ordering
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"cache:{model_name}:{digest}"


class ResultCache:
    def __init__(self, r: redis.Redis, ttl_seconds: int):
        self.r = r
        self.ttl = ttl_seconds

    async def get(self, key: str) -> Optional[Any]:
        val = await self.r.get(key)
        if val is None:
            return None
        return json.loads(val)

    async def set(self, key: str, value: Any) -> None:
        await self.r.set(key, json.dumps(value), ex=self.ttl)
