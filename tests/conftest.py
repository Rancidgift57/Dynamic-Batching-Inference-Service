"""
These tests talk to a real Redis instance (fast, in-memory, no mocking of
XADD/XREADGROUP/Lua-script semantics needed). Point REDIS_URL at any Redis
you have running, e.g.:

    redis-server --daemonize yes
    pytest tests/ -v

If Redis isn't reachable, the whole suite is skipped rather than failing,
so `pytest` still exits cleanly in environments without it.
"""
import os

import pytest
import pytest_asyncio
import redis.asyncio as redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


@pytest_asyncio.fixture
async def redis_client():
    r = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        await r.ping()
    except Exception:
        pytest.skip(f"No Redis reachable at {REDIS_URL} — start one to run this suite")
    await r.flushall()
    yield r
    await r.flushall()
    await r.aclose()
