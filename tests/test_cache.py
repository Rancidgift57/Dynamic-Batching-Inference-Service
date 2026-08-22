import pytest

from gateway.cache import ResultCache, make_cache_key


def test_cache_key_deterministic_and_order_independent():
    k1 = make_cache_key("model-x", {"text": "hello", "n": 1})
    k2 = make_cache_key("model-x", {"n": 1, "text": "hello"})
    assert k1 == k2


def test_cache_key_differs_by_payload_and_model():
    base = make_cache_key("model-x", {"text": "hello"})
    assert base != make_cache_key("model-x", {"text": "world"})
    assert base != make_cache_key("model-y", {"text": "hello"})


@pytest.mark.asyncio
async def test_cache_roundtrip_and_miss(redis_client):
    cache = ResultCache(redis_client, ttl_seconds=5)
    key = make_cache_key("model-x", {"text": "hello"})

    assert await cache.get(key) is None  # miss before set

    await cache.set(key, [1.0, 2.0, 3.0])
    assert await cache.get(key) == [1.0, 2.0, 3.0]

    other_key = make_cache_key("model-x", {"text": "different"})
    assert await cache.get(other_key) is None
