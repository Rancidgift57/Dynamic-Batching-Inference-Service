# app/distributed_queue.py
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import List

import redis.asyncio as redis

STREAM_KEY = "inference:requests"
RESULT_KEY_PREFIX = "inference:result:"
GROUP_NAME = "batchers"


@dataclass
class DistributedQueueItem:
    request_id: str
    text: str
    enqueued_at: float


class RedisCoordinatedQueue:
    """
    Every uvicorn worker process connects to the same Redis instance.
    Requests go into a shared Stream; ONE consumer group means only
    one worker's batch loop claims a given message, but any worker
    can pull the next batch — this restores global batching across
    multiple processes.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.r: redis.Redis | None = None

    async def connect(self):
        self.r = redis.from_url(self.redis_url, decode_responses=True)
        try:
            await self.r.xgroup_create(STREAM_KEY, GROUP_NAME, id="0", mkstream=True)
        except redis.ResponseError:
            pass  # group already exists

    async def submit(self, text: str) -> List[float]:
        request_id = str(uuid.uuid4())
        await self.r.xadd(STREAM_KEY, {"request_id": request_id, "text": text,
                                        "ts": str(time.time())})
        # Poll for the result key (a pub/sub or BLPOP-based version avoids
        # polling entirely — worth building as a v2 once this works)
        result_key = f"{RESULT_KEY_PREFIX}{request_id}"
        while True:
            val = await self.r.get(result_key)
            if val is not None:
                await self.r.delete(result_key)
                payload = json.loads(val)
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                return payload["embedding"]
            await asyncio.sleep(0.002)

    async def collect_batch(self, consumer_name: str, max_size: int, timeout_ms: int):
        resp = await self.r.xreadgroup(
            GROUP_NAME, consumer_name, {STREAM_KEY: ">"},
            count=max_size, block=timeout_ms,
        )
        if not resp:
            return []
        items = []
        for _, messages in resp:
            for msg_id, fields in messages:
                items.append((msg_id, DistributedQueueItem(
                    request_id=fields["request_id"],
                    text=fields["text"],
                    enqueued_at=float(fields["ts"]),
                )))
        return items

    async def publish_result(self, request_id: str, embedding=None, error=None):
        payload = {"error": error} if error else {"embedding": embedding}
        await self.r.set(f"{RESULT_KEY_PREFIX}{request_id}", json.dumps(payload), ex=30)

    async def ack(self, msg_ids: List[str]):
        if msg_ids:
            await self.r.xack(STREAM_KEY, GROUP_NAME, *msg_ids)

