"""
Dynamic Batching Engine.

Every worker replica runs this same loop against the same Redis Stream +
consumer group. `XREADGROUP` guarantees each queued request is claimed by
exactly one worker, but because the stream is shared, *all* replicas draw
from one global backlog — so scaling out the worker pool packs bigger,
denser batches under load instead of just running more small ones.

Two independent triggers decide when a batch is "full enough" to run:
  - MAX_BATCH_SIZE requests collected  -> flush immediately
  - BATCH_TIMEOUT_MS elapsed since the FIRST item in this batch arrived
    -> flush whatever we have

IMPORTANT: `XREADGROUP ... BLOCK ms` only blocks when *zero* entries are
currently available for the group — the instant one entry exists it
returns immediately with whatever's there, it does NOT keep waiting for
more to accumulate. Calling it once per iteration (the naive version of
this loop) therefore returns batches of size 1 under any moderate,
steadily-arriving traffic and only forms real batches when a synchronized
burst happens to land between two polls. `_collect_batch` below explicitly
re-polls in a tight, non-blocking-after-the-first-item loop until either
MAX_BATCH_SIZE is reached or the remaining slice of BATCH_TIMEOUT_MS
elapses, so batches actually form under sustained load, not just bursts.

A second correctness gap this loop closes: if a worker crashes after
`XREADGROUP` claims a message but before `XACK`, that message sits
unacknowledged in its consumer's Pending Entries List forever — nobody
ever retries it, the caller just times out. `_reclaim_stale_messages`
periodically claims (`XAUTOCLAIM`) any message that's been pending longer
than STALE_CLAIM_MS from *any* consumer (including dead ones) and feeds it
back through the same processing path.
"""
import asyncio
import json
import logging
import os
import socket
import time

import redis.asyncio as redis
from prometheus_client import start_http_server

from . import config
from .metrics import BATCH_EXEC_SECONDS, BATCH_SIZE, QUEUE_WAIT_SECONDS, RECLAIMED_TOTAL
from .model_backend import get_model_backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("batch_worker")

CONSUMER_NAME = f"{socket.gethostname()}-{os.getpid()}"


async def _ensure_group(r: redis.Redis):
    try:
        await r.xgroup_create(config.STREAM_KEY, config.GROUP_NAME, id="0", mkstream=True)
    except redis.ResponseError:
        pass  # already exists — expected on every replica after the first


def _parse_messages(resp) -> list[tuple[str, str, str, float]]:
    items = []
    for _, messages in resp:
        for msg_id, fields in messages:
            items.append((msg_id, fields["request_id"], fields["text"], float(fields["ts"])))
    return items


async def _collect_batch(r: redis.Redis) -> list[tuple[str, str, str, float]]:
    """Block (cheaply, no busy-loop) until at least one request is
    available, then keep pulling more — without ever blocking again once
    the batch window has started — until MAX_BATCH_SIZE is hit or
    BATCH_TIMEOUT_MS has elapsed since the first item landed."""
    resp = await r.xreadgroup(
        config.GROUP_NAME, CONSUMER_NAME, {config.STREAM_KEY: ">"},
        count=config.MAX_BATCH_SIZE, block=config.IDLE_BLOCK_MS,
    )
    if not resp:
        return []

    items = _parse_messages(resp)
    deadline = time.monotonic() + (config.BATCH_TIMEOUT_MS / 1000.0)

    while len(items) < config.MAX_BATCH_SIZE:
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            break
        # block=remaining_ms: returns instantly if more are already queued
        # (dense-traffic case), or waits out the rest of the window for
        # the next one to arrive (sparse-traffic case) — either way we
        # never wait past the original BATCH_TIMEOUT_MS budget.
        more = await r.xreadgroup(
            config.GROUP_NAME, CONSUMER_NAME, {config.STREAM_KEY: ">"},
            count=config.MAX_BATCH_SIZE - len(items), block=remaining_ms,
        )
        if not more:
            break
        items.extend(_parse_messages(more))

    return items


async def _process_items(r: redis.Redis, model, items: list[tuple[str, str, str, float]]):
    """Run one batch through the model and fan results back out. Shared by
    the main loop and the stale-message reclaim path so a reclaimed
    message gets exactly the same handling as a freshly-read one."""
    if not items:
        return

    now = time.time()
    for _, _, _, ts in items:
        QUEUE_WAIT_SECONDS.observe(max(0.0, now - ts))
    BATCH_SIZE.observe(len(items))

    texts = [text for *_, text, _ in items]
    t0 = time.perf_counter()
    try:
        results = await asyncio.to_thread(model.predict_batch, texts)
        BATCH_EXEC_SECONDS.observe(time.perf_counter() - t0)
        async with r.pipeline(transaction=False) as pipe:
            for (_, request_id, _, _), vec in zip(items, results):
                result_key = f"result:{request_id}"
                pipe.rpush(result_key, json.dumps({"embedding": vec}))
                pipe.expire(result_key, config.RESULT_TTL_SECONDS)
            await pipe.execute()
    except Exception as e:
        logger.exception("Batch of %d failed", len(items))
        async with r.pipeline(transaction=False) as pipe:
            for _, request_id, _, _ in items:
                result_key = f"result:{request_id}"
                pipe.rpush(result_key, json.dumps({"error": str(e)}))
                pipe.expire(result_key, config.RESULT_TTL_SECONDS)
            await pipe.execute()

    await r.xack(config.STREAM_KEY, config.GROUP_NAME, *[msg_id for msg_id, *_ in items])


async def _reclaim_stale_messages(r: redis.Redis, model):
    """Runs forever as a background task: every RECLAIM_INTERVAL_S,
    XAUTOCLAIM any message that's been pending (claimed but never XACK'd)
    for longer than STALE_CLAIM_MS — almost always because the consumer
    that claimed it died mid-batch — and reprocess it here instead of
    leaving it stuck in that dead consumer's PEL forever."""
    cursor = "0-0"
    while True:
        await asyncio.sleep(config.RECLAIM_INTERVAL_S)
        try:
            while True:
                cursor, messages, _deleted = await r.xautoclaim(
                    config.STREAM_KEY, config.GROUP_NAME, CONSUMER_NAME,
                    min_idle_time=config.STALE_CLAIM_MS, start_id=cursor, count=config.MAX_BATCH_SIZE,
                )
                if not messages:
                    break
                RECLAIMED_TOTAL.inc(len(messages))
                logger.warning("Reclaimed %d stale pending message(s) from a dead/stuck consumer", len(messages))
                items = [(msg_id, fields["request_id"], fields["text"], float(fields["ts"])) for msg_id, fields in messages]
                await _process_items(r, model, items)
                if cursor == "0-0":
                    break
        except Exception:
            logger.exception("Stale-message reclaim pass failed, will retry next interval")


async def run_worker():
    r = redis.from_url(config.REDIS_URL, decode_responses=True)
    await _ensure_group(r)
    model = get_model_backend()
    logger.info(
        "Worker %s ready — backend=%s max_batch=%d timeout_ms=%d",
        CONSUMER_NAME, config.MODEL_BACKEND, config.MAX_BATCH_SIZE, config.BATCH_TIMEOUT_MS,
    )

    reclaim_task = asyncio.create_task(_reclaim_stale_messages(r, model))

    try:
        while True:
            try:
                items = await _collect_batch(r)
            except Exception:
                logger.exception("xreadgroup failed, retrying in 1s")
                await asyncio.sleep(1)
                continue

            await _process_items(r, model, items)
    finally:
        reclaim_task.cancel()


def main():
    start_http_server(config.METRICS_PORT)
    logger.info("Metrics exposed on :%d/metrics", config.METRICS_PORT)
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
