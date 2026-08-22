"""
AI Proxy & Dynamic Batching Gateway — ingestion layer.

This process does no model inference. Its only jobs, in order, are:
  1. Rate-limit the client (Redis token bucket — instant, in-memory).
  2. Check the response cache (Redis GET — instant, in-memory).
  3. On a cache miss, push the request onto a Redis Stream and block on the
     result key until a batch worker replies (or time out).

Because steps 1-2 are sub-millisecond Redis ops and step 3 never spins a
CPU (it's an async wait on the socket), a single gateway replica can hold
open thousands of concurrent in-flight requests without ever touching the
model — that's what lets N of these behind Nginx sustain very high ingest
rates while a much smaller pool of GPU/CPU batch workers does the actual
math.
"""
import logging
import time
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import config
from .cache import ResultCache, make_cache_key
from .metrics import (
    CACHE_HITS_TOTAL,
    RATE_LIMITED_TOTAL,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    TIMEOUTS_TOTAL,
)
from .rate_limiter import TokenBucketRateLimiter
from .redis_stream import StreamGateway
from .schemas import InferRequest, InferResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    r = redis.from_url(config.REDIS_URL, decode_responses=True)
    state["redis"] = r
    state["cache"] = ResultCache(r, config.CACHE_TTL_SECONDS)
    state["limiter"] = TokenBucketRateLimiter(
        r, config.RATE_LIMIT_CAPACITY, config.RATE_LIMIT_REFILL_PER_SEC
    )
    state["stream"] = StreamGateway(r)
    await state["stream"].ensure_group()
    logger.info("Gateway ready — Redis=%s, model=%s", config.REDIS_URL, config.MODEL_NAME)
    yield
    await r.close()


app = FastAPI(title="AI Proxy & Dynamic Batching Gateway", lifespan=lifespan)


def _client_id(request: Request) -> str:
    # Prefer an explicit API key/client header; fall back to peer IP.
    return request.headers.get("X-Client-Id") or (request.client.host if request.client else "unknown")


@app.post("/v1/infer", response_model=InferResponse)
async def infer(req: InferRequest, request: Request):
    t0 = time.perf_counter()
    REQUESTS_TOTAL.inc()

    client_id = _client_id(request)

    if config.RATE_LIMIT_ENABLED:
        allowed, remaining = await state["limiter"].allow(client_id)
        if not allowed:
            RATE_LIMITED_TOTAL.inc()
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded — slow down.",
                headers={"Retry-After": "1"},
            )

    cache_key = make_cache_key(config.MODEL_NAME, {"text": req.text}) if config.CACHE_ENABLED else None
    if cache_key:
        cached = await state["cache"].get(cache_key)
        if cached is not None:
            CACHE_HITS_TOTAL.inc()
            REQUEST_LATENCY_SECONDS.observe(time.perf_counter() - t0)
            return InferResponse(embedding=cached, cached=True)

    try:
        embedding = await state["stream"].submit_and_wait(req.text, config.RESULT_WAIT_TIMEOUT_S)
    except TimeoutError as e:
        TIMEOUTS_TOTAL.inc()
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.exception("infer failed")
        raise HTTPException(status_code=500, detail=str(e))

    if cache_key:
        await state["cache"].set(cache_key, embedding)

    REQUEST_LATENCY_SECONDS.observe(time.perf_counter() - t0)
    return InferResponse(embedding=embedding, cached=False)


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    # A gateway replica is only actually useful if it can reach Redis —
    # every request path (rate limit, cache, enqueue) depends on it. Without
    # this check, a replica with a dead Redis connection still reports 200,
    # so Nginx's max_fails/fail_timeout can never route traffic around it.
    try:
        await state["redis"].ping()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"redis unavailable: {e}")
    return {"status": "ok"}
