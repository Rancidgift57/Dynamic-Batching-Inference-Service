"""
Ingestion side of the Redis Stream shock absorber.

Zero-blocking ingestion: `submit_and_wait` does one XADD (O(1), stays in
Redis's in-memory stream) and then blocks on a per-request result key with
BRPOP. Unlike a polling loop, BRPOP wakes up the instant a batch worker
pushes the result — no wasted round trips, no polling interval to tune.
"""
import json
import time
import uuid

import redis.asyncio as redis

from . import config


class StreamGateway:
    def __init__(self, r: redis.Redis):
        self.r = r

    async def ensure_group(self):
        try:
            await self.r.xgroup_create(config.STREAM_KEY, config.GROUP_NAME, id="0", mkstream=True)
        except redis.ResponseError:
            pass  # group already exists — fine on every restart/replica

    async def submit_and_wait(self, text: str, timeout_s: float) -> list[float]:
        request_id = str(uuid.uuid4())
        await self.r.xadd(
            config.STREAM_KEY,
            {"request_id": request_id, "text": text, "ts": str(time.time())},
            # Approximate trim caps the stream (and Redis's RAM) if the
            # worker pool is ever *permanently* — not just momentarily —
            # behind ingest, rather than letting it grow without bound.
            # This is a last-resort backpressure valve, not a substitute
            # for fixing capacity: entries trimmed off are lost (their
            # caller has usually already hit the 504 timeout anyway).
            # `approximate=True` keeps XADD O(1) instead of forcing an
            # exact trim on every call.
            maxlen=config.STREAM_MAXLEN,
            approximate=True,
        )

        result_key = f"result:{request_id}"
        popped = await self.r.brpop(result_key, timeout=timeout_s)
        if popped is None:
            raise TimeoutError(f"No batch worker responded within {timeout_s}s")

        _, raw = popped
        payload = json.loads(raw)
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return payload["embedding"]
