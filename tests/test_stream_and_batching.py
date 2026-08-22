import asyncio
import json

import pytest
import redis.asyncio as redis

from gateway.redis_stream import StreamGateway
from worker import config as worker_config
from worker.batch_worker import _collect_batch
from worker.model_backend import EchoBackend


async def _run_one_batch_pass(r: redis.Redis, max_size: int = 32):
    """Minimal stand-in for worker.batch_worker.run_worker's single loop
    iteration — claims whatever is pending, runs the echo model, replies."""
    model = EchoBackend()
    try:
        await r.xgroup_create(worker_config.STREAM_KEY, worker_config.GROUP_NAME, id="0", mkstream=True)
    except redis.ResponseError:
        pass

    resp = await r.xreadgroup(
        worker_config.GROUP_NAME, "test-consumer",
        {worker_config.STREAM_KEY: ">"}, count=max_size, block=500,
    )
    if not resp:
        return 0

    items = []
    for _, messages in resp:
        for msg_id, fields in messages:
            items.append((msg_id, fields["request_id"], fields["text"]))

    texts = [text for *_, text in items]
    results = model.predict_batch(texts)
    for (msg_id, request_id, _), vec in zip(items, results):
        await r.rpush(f"result:{request_id}", json.dumps({"embedding": vec}))
        await r.expire(f"result:{request_id}", 30)
    await r.xack(worker_config.STREAM_KEY, worker_config.GROUP_NAME, *[m for m, _, _ in items])
    return len(items)


@pytest.mark.asyncio
async def test_concurrent_requests_get_batched_and_routed_to_correct_result(redis_client):
    gateway = StreamGateway(redis_client)
    await gateway.ensure_group()

    texts = [f"request-{i}" for i in range(6)]
    client_tasks = [
        asyncio.create_task(gateway.submit_and_wait(t, timeout_s=5.0)) for t in texts
    ]
    await asyncio.sleep(0.05)  # let every XADD land before the "worker" reads

    batch_size = await _run_one_batch_pass(redis_client)
    assert batch_size == 6

    results = await asyncio.gather(*client_tasks)

    # Each client's result must match what EchoBackend deterministically
    # produces for *its own* text — proving no cross-talk between requests
    # sharing one batch.
    expected = EchoBackend().predict_batch(texts)
    assert results == expected


@pytest.mark.asyncio
async def test_submit_and_wait_times_out_if_no_worker_responds(redis_client):
    gateway = StreamGateway(redis_client)
    await gateway.ensure_group()

    with pytest.raises(TimeoutError):
        await gateway.submit_and_wait("nobody is listening", timeout_s=0.2)


@pytest.mark.asyncio
async def test_worker_error_propagates_to_waiting_client(redis_client):
    gateway = StreamGateway(redis_client)
    await gateway.ensure_group()

    task = asyncio.create_task(gateway.submit_and_wait("boom", timeout_s=5.0))
    await asyncio.sleep(0.05)

    resp = await redis_client.xreadgroup(
        worker_config.GROUP_NAME, "test-consumer",
        {worker_config.STREAM_KEY: ">"}, count=1, block=500,
    )
    msg_id, fields = resp[0][1][0]
    request_id = fields["request_id"]

    await redis_client.rpush(f"result:{request_id}", json.dumps({"error": "model crashed"}))
    await redis_client.expire(f"result:{request_id}", 30)
    await redis_client.xack(worker_config.STREAM_KEY, worker_config.GROUP_NAME, msg_id)

    with pytest.raises(RuntimeError, match="model crashed"):
        await task


@pytest.mark.asyncio
async def test_collect_batch_accumulates_trickling_requests_within_window(redis_client):
    """Regression test for the batching-window bug: a single-shot
    `xreadgroup(count=N, block=T)` returns as soon as ONE entry exists,
    without waiting out the rest of `T` for more to arrive — so requests
    that trickle in a few ms apart (well inside the batch window) would
    each get processed as their own batch of 1. `_collect_batch` must
    keep re-polling until MAX_BATCH_SIZE or the window elapses, so
    trickling-but-within-window requests land in the same batch."""
    gateway = StreamGateway(redis_client)
    await gateway.ensure_group()

    original_timeout = worker_config.BATCH_TIMEOUT_MS
    worker_config.BATCH_TIMEOUT_MS = 200  # generous window for a stable CI assertion
    try:
        n = 5

        async def trickle():
            for i in range(n):
                await redis_client.xadd(
                    worker_config.STREAM_KEY,
                    {"request_id": f"req-{i}", "text": f"t{i}", "ts": "0"},
                )
                await asyncio.sleep(0.02)  # well within the 200ms window

        producer = asyncio.create_task(trickle())
        items = await _collect_batch(redis_client)
        await producer

        assert len(items) == n, (
            f"expected all {n} trickling requests in one batch, got {len(items)} — "
            "the batch loop is returning as soon as the first item arrives "
            "instead of accumulating for the full BATCH_TIMEOUT_MS window"
        )
    finally:
        worker_config.BATCH_TIMEOUT_MS = original_timeout
