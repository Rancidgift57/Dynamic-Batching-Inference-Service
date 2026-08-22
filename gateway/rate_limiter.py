"""
Distributed token-bucket rate limiter.

Every gateway replica (behind Nginx) shares the same Redis instance, so the
limit is enforced globally per client_id, not per-process. The check-and-
decrement happens inside a single Lua script so it's atomic even with many
concurrent gateway workers hitting the same key.
"""
import time
from typing import Tuple

import redis.asyncio as redis

_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])   -- tokens added per second
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local ts = tonumber(bucket[2])

if tokens == nil then
  tokens = capacity
  ts = now
end

local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(capacity, tokens + elapsed * refill_rate)

local allowed = 0
if tokens >= requested then
  tokens = tokens - requested
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, 3600)

return {allowed, tokens}
"""


class TokenBucketRateLimiter:
    def __init__(self, r: redis.Redis, capacity: float, refill_per_sec: float):
        self.r = r
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self._script = self.r.register_script(_TOKEN_BUCKET_LUA)

    async def allow(self, client_id: str, cost: float = 1.0) -> Tuple[bool, float]:
        """Returns (allowed, tokens_remaining)."""
        key = f"ratelimit:{client_id}"
        now = time.time()
        allowed, remaining = await self._script(
            keys=[key], args=[self.capacity, self.refill_per_sec, now, cost]
        )
        return bool(int(allowed)), float(remaining)
