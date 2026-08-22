import asyncio

import pytest

from gateway.rate_limiter import TokenBucketRateLimiter


@pytest.mark.asyncio
async def test_allows_up_to_capacity_then_blocks(redis_client):
    limiter = TokenBucketRateLimiter(redis_client, capacity=3, refill_per_sec=0.001)

    outcomes = [(await limiter.allow("client-a"))[0] for _ in range(5)]
    assert outcomes == [True, True, True, False, False]


@pytest.mark.asyncio
async def test_clients_are_isolated(redis_client):
    limiter = TokenBucketRateLimiter(redis_client, capacity=1, refill_per_sec=0.001)

    allowed_a, _ = await limiter.allow("client-a")
    allowed_a2, _ = await limiter.allow("client-a")
    allowed_b, _ = await limiter.allow("client-b")

    assert allowed_a is True
    assert allowed_a2 is False   # client-a's bucket is empty
    assert allowed_b is True     # client-b has its own independent bucket


@pytest.mark.asyncio
async def test_tokens_refill_over_time(redis_client):
    limiter = TokenBucketRateLimiter(redis_client, capacity=1, refill_per_sec=20)  # 1 token/50ms

    allowed_1, _ = await limiter.allow("client-a")
    allowed_2, _ = await limiter.allow("client-a")
    assert allowed_1 is True
    assert allowed_2 is False

    await asyncio.sleep(0.1)  # enough time for the bucket to refill past 1 token

    allowed_3, _ = await limiter.allow("client-a")
    assert allowed_3 is True
