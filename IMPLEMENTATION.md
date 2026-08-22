# AI Proxy & Dynamic Batching Gateway

A high-throughput proxy that sits between client applications and a heavy
ML model (embeddings, a reranker, or an LLM/RAG call), built around one
question: **how do you keep ingesting requests at wire speed while a much
slower, much more expensive model does the actual work behind it?**

It is a **stateless proxy tier** (Nginx + many FastAPI replicas) in front
of a **shared Redis Stream** and a **separate pool of batch workers**, so
ingestion and inference can each scale independently — add gateway
replicas to absorb more concurrent connections, add worker replicas (or
GPUs) to absorb more compute, without either tier having to know about the
other's scale.

```
        thousands of clients
                │
                ▼
        ┌───────────────┐
        │     Nginx      │   load balances across N gateway replicas
        └───────┬────────┘
                │
   ┌────────────┼────────────┐
   ▼             ▼            ▼
gateway-1    gateway-2    gateway-3      ← thin, non-blocking FastAPI workers
   │             │            │            (rate limit → cache check → enqueue)
   └────────────┬┴────────────┘
                ▼
      ┌───────────────────┐
      │   Redis (shock     │   Stream (queue) + cache + token-bucket
      │    absorber)        │   counters, all in RAM
      └─────────┬───────────┘
                ▼
   ┌────────────┼────────────┐
   ▼                          ▼
worker-1                  worker-2       ← dynamic batching engine,
   │                          │            packs requests into dense
   ▼                          ▼            batches before calling the model
model                      model
```

---

## Table of Contents

- [How This Hits 1M-Request Scale](#how-this-hits-1m-request-scale)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Capacity Math](#capacity-math)
- [How Can I Use It / Integration with Other Systems](#how-can-i-use-it--integration-with-other-systems)
- [Benchmarking](#benchmarking)
- [Monitoring (Prometheus + Grafana)](#monitoring-prometheus--grafana)
- [Testing](#testing)
- [Operating in Production](#operating-in-production)
- [Relationship to the Dynamic Inference Project](#relationship-to-the-dynamic-inference-project)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [FAQ](#faq)

---

## How This Hits 1M-Request Scale

### 1. Zero-blocking ingestion — Nginx + FastAPI

`nginx/nginx.conf` load-balances (`least_conn`) across `gateway-1..3`
(`gateway/main.py`). Each gateway replica does **no model work** — its
entire job per request is:

1. Check a Redis token bucket (`gateway/rate_limiter.py`) — one Lua script call.
2. Check the Redis cache (`gateway/cache.py`) — one `GET`.
3. On a miss, `XADD` the request onto a Redis Stream and `await` a result key.

None of that touches a CPU-bound code path — it's all async I/O against
Redis over the local network. That's what lets a handful of gateway
replicas hold open thousands of concurrent connections without ever
blocking on the model, so Nginx can keep accepting new connections instead
of queuing them at the TCP layer.

### 2. Distributed Redis Streams — the shock absorber

`inference:requests` (see `gateway/redis_stream.py` /
`worker/batch_worker.py`) is a Redis Stream with a consumer group
(`batchers`). Streams are an append-only, in-RAM log:

- `XADD` is O(1) — ingestion never slows down as the backlog grows.
- A traffic spike just makes the stream longer; nothing is dropped as long
  as Redis has memory for it. `STREAM_MAXLEN` caps this as a last-resort
  backpressure valve (see [Configuration](#configuration)) so a *sustained*
  imbalance between ingest and processing degrades gracefully into
  trimming the oldest entries instead of growing Redis without bound.
- `XREADGROUP` + the `batchers` consumer group guarantees each request is
  claimed by exactly **one** worker, even though every worker reads from
  the same stream — so the queue is a single global backlog, not N
  disconnected per-process queues.
- Result handoff is **not** polling: `submit_and_wait` does `BRPOP` on a
  per-request list key, so the gateway wakes up the instant a worker
  publishes a result instead of re-checking on an interval.
- If a worker crashes after claiming a message but before acknowledging
  it, a background `XAUTOCLAIM` pass (`_reclaim_stale_messages` in
  `worker/batch_worker.py`) reclaims it from the dead consumer and
  reprocesses it, instead of it sitting orphaned forever.

### 3. Dynamic Batching Engine — dense tensors, not one-at-a-time

`worker/batch_worker.py` runs the same loop on every worker replica:
collect up to `MAX_BATCH_SIZE` requests — accumulating for up to
`BATCH_TIMEOUT_MS` measured from the *first* item in the batch, not just a
single non-accumulating read — then run **one** forward pass over the
whole batch (`worker/model_backend.py`), then fan the results back out to
each waiting client via Redis. This is the same core batching algorithm as
the standalone Dynamic Inference service, just reading its queue from
Redis instead of an in-process `asyncio.Queue` — see that project's README
for the throughput numbers this buys you (roughly 2.7–3× at `Batch=16`
over unbatched `Batch=1` in that benchmark).

The model itself is pluggable (`ModelBackend` protocol) — ship with a
sentence-embedding model by default, or point `MODEL_BACKEND` /
`model_backend.py` at an LLM call or a RAG retrieval + generation pipeline
without touching the queueing or batching logic at all.

### 4. Smart Caching & Rate Limiting

- **Cache** (`gateway/cache.py`): the request payload is normalized and
  SHA-256 hashed into a Redis key (`cache:<model>:<hash>`). Repeated
  queries — extremely common for public/AI APIs (the same FAQ, the same
  product description, the same prompt) — are served straight out of
  Redis in sub-millisecond time, **never touching the model or the batch
  queue at all**.
- **Rate limiting** (`gateway/rate_limiter.py`): a Redis-backed token
  bucket, enforced atomically via a Lua script so it's correct even with
  many gateway replicas hitting the same client's key concurrently.
  Exceeding it returns `429` with `Retry-After` instead of piling more
  load onto the batch workers — protecting the expensive part of the
  system from being overwhelmed by any single client.

---

## Project Structure

```
ai-proxy-gateway/
├── gateway/                    # Ingestion tier — no ML dependencies
│   ├── main.py                 # FastAPI app: rate limit → cache → enqueue+wait
│   ├── rate_limiter.py         # Redis Lua-script token bucket
│   ├── cache.py                # Request hashing + Redis GET/SET cache
│   ├── redis_stream.py         # XADD + BRPOP-based result handoff
│   ├── metrics.py               # Prometheus counters/histograms
│   ├── schemas.py               # Pydantic request/response models
│   └── config.py                 # All settings, env-var driven
│
├── worker/                     # Dynamic Batching Engine — owns the model
│   ├── batch_worker.py          # XREADGROUP loop: collect → batch → infer → reply
│   ├── model_backend.py         # Pluggable model interface (embedding / echo / your LLM)
│   ├── metrics.py
│   └── config.py
│
├── nginx/
│   └── nginx.conf               # least_conn load balancer across gateway replicas
│
├── monitoring/
│   └── prometheus.yml           # Scrapes every gateway + worker replica
│
├── benchmarks/
│   ├── locustfile.py            # Unique / repeated (cache-hit) / burst traffic shapes
│   └── run_benchmark.sh
│
├── tests/                       # Fast, real-Redis-backed unit + integration tests
│   ├── conftest.py
│   ├── test_cache.py
│   ├── test_rate_limiter.py
│   └── test_stream_and_batching.py
│
├── docker-compose.yml           # nginx + 3× gateway + redis + 2× worker + prometheus + grafana
├── Dockerfile.gateway / requirements-gateway.txt   # lightweight image, no torch
├── Dockerfile.worker  / requirements-worker.txt    # ML image, torch + sentence-transformers
└── requirements.txt              # combined deps, for local dev of both tiers at once
```

---

## Requirements

- Docker + Docker Compose (recommended path)
- Or locally: Python 3.10+, a running Redis 6+ instance (Redis 7 recommended)

---

## Quickstart

### Docker Compose (recommended)

```bash
docker compose up --build
```

This starts: `redis`, three gateway replicas, `nginx` (public entrypoint on
`:8000`), two batch workers, `prometheus` (`:9090`), and `grafana`
(`:3000`, `admin`/`admin`).

```bash
curl -X POST http://localhost:8000/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"text": "dynamic batching is fun"}'
```

Send the same payload again and check `"cached": true` in the response —
the second call never touches the batch queue.

### Local (no Docker)

```bash
pip install -r requirements.txt
redis-server --daemonize yes

# Terminal 1 — a batch worker (echo backend needs no model download)
MODEL_BACKEND=echo REDIS_URL=redis://localhost:6379 python -m worker.batch_worker

# Terminal 2 — a gateway replica
REDIS_URL=redis://localhost:6379 uvicorn gateway.main:app --host 0.0.0.0 --port 8000
```

Set `MODEL_BACKEND=embedding` (the default) once you want real embeddings
instead of the dependency-free `echo` backend used above for quick local
testing of the plumbing.

---

## Configuration

### Gateway (`gateway/config.py`)

| Variable                     | Default                         | Description                                              |
|-------------------------------|----------------------------------|------------------------------------------------------------|
| `REDIS_URL`                  | `redis://localhost:6379`        | Shared Redis instance                                     |
| `MODEL_NAME`                 | `sentence-transformers/all-MiniLM-L6-v2` | Used only to namespace cache keys                |
| `RATE_LIMIT_ENABLED`         | `true`                           | Toggle the token bucket entirely                           |
| `RATE_LIMIT_CAPACITY`        | `100`                            | Max burst size (tokens) per client                        |
| `RATE_LIMIT_REFILL_PER_SEC`  | `20`                             | Steady-state requests/sec allowed per client               |
| `CACHE_ENABLED`               | `true`                           | Toggle the response cache                                  |
| `CACHE_TTL_SECONDS`          | `300`                            | How long a cached result stays valid                        |
| `RESULT_WAIT_TIMEOUT_S`      | `5.0`                            | How long the gateway waits on a worker before `504`         |
| `STREAM_MAXLEN`               | `200000`                        | Approx. cap on stream length (`XADD ... MAXLEN ~`) — backpressure valve if workers fall permanently behind, not just spike-behind |

### Worker (`worker/config.py`)

| Variable            | Default                         | Description                                                |
|-----------------------|-----------------------------------|----------------------------------------------------------------|
| `REDIS_URL`          | `redis://localhost:6379`        | Shared Redis instance                                       |
| `MODEL_NAME`         | `sentence-transformers/all-MiniLM-L6-v2` | Passed to the embedding backend               |
| `MODEL_BACKEND`      | `embedding`                      | `embedding` (real model) or `echo` (zero-dependency dev/test) |
| `MAX_BATCH_SIZE`     | `32`                             | Max requests per batch before forcing a flush                |
| `BATCH_TIMEOUT_MS`   | `10`                             | Max time to accumulate a batch, measured from the first item |
| `IDLE_BLOCK_MS`      | `1000`                          | How long to block waiting for the *next* batch's first item when the stream is idle |
| `RESULT_TTL_SECONDS` | `30`                             | TTL on the per-request result key                             |
| `STALE_CLAIM_MS`     | `30000`                         | How long a claimed-but-unacked message must be idle before it's treated as abandoned and reprocessed |
| `RECLAIM_INTERVAL_S` | `5.0`                            | How often the stale-message reclaim pass runs                  |
| `METRICS_PORT`       | `9100`                          | Where this worker exposes `/metrics`                          |

Every value above is read from the environment, so tuning it in
Docker/Kubernetes/systemd is just setting env vars — no code changes.

---

## API Reference

| Method | Path        | Description                                                                 |
|--------|-------------|-------------------------------------------------------------------------------|
| `POST` | `/v1/infer` | Body: `{"text": "..."}` → `{"embedding": [...], "cached": bool}`. `429` if rate-limited, `504` if no worker responds in time. |
| `GET`  | `/health`   | Liveness/readiness probe — pings Redis; `200 {"status": "ok"}` or `503` if Redis is unreachable |
| `GET`  | `/metrics`  | Prometheus metrics for that gateway replica                                  |

Optional header: `X-Client-Id` — identifies the caller for rate limiting;
falls back to peer IP if omitted. (See
[Known Limitations](#known-limitations--future-work) — this header is
currently self-reported and unauthenticated.)

**Request**
```json
{ "text": "dynamic batching is fun" }
```

**Response — cache miss (went through the batch queue)**
```json
{ "embedding": [0.0123, -0.0456, "... 384 floats for the default model"], "cached": false }
```

**Response — cache hit**
```json
{ "embedding": [0.0123, -0.0456, "..."], "cached": true }
```

**Error responses**
```json
// 429 — rate limited, Retry-After: 1 header set
{ "detail": "Rate limit exceeded — slow down." }

// 504 — no batch worker replied within RESULT_WAIT_TIMEOUT_S
{ "detail": "No batch worker responded within 5.0s" }
```

---

## Capacity Math

Rough, order-of-magnitude reasoning for why this shape of system, not a
promise of a specific number on specific hardware — always verify with the
benchmarks below on your own infrastructure.

- **Ingestion (gateway tier):** each gateway request is ~2–3 Redis round
  trips (rate limit, cache check, `XADD`) plus an async wait — no CPU-bound
  work. A single async FastAPI worker can hold thousands of concurrent
  in-flight connections; a handful of replicas behind Nginx comfortably
  ingest **tens of thousands of requests/sec**, well past what's needed to
  *accept* a burst toward 1M total requests over a realistic window (e.g.
  1M requests over 60s ≈ ~17K req/s ingest rate — inside a small gateway
  tier's headroom).
- **Buffering (Redis Stream):** a stream entry is a small hash of strings
  (`request_id`, `text`, `ts`). Redis can hold hundreds of thousands of
  these in RAM without breaking a sweat, which is what decouples "requests
  accepted" from "requests processed" during a spike.
- **Processing (worker tier):** this is the actual bottleneck, and it's
  where dynamic batching pays for itself — a batch of 32 costs far less
  than 32× a batch of 1 (see the Dynamic Inference project's benchmark
  tables for measured numbers: ~3× throughput at `Batch=16` vs `Batch=1`
  on CPU). Reaching sustained 1M-request throughput is a function of
  **how many worker replicas × how much each batch amortizes** — scale
  `worker-N` horizontally (and prefer GPU workers, and/or a higher
  `MAX_BATCH_SIZE`) until worker throughput matches your target ingest rate.
- **Caching** removes a fraction of traffic from the worker tier entirely,
  which is why hit rate matters — a workload with even a modest amount of
  repetition (a public-facing endpoint, common prompts/queries) meaningfully
  reduces how much worker capacity you need to provision in the first place.

---

## How Can I Use It / Integration with Other Systems

The gateway only speaks one contract at its edge — `POST /v1/infer` with
`{"text": "..."}` in, `{"embedding": [...], "cached": bool}` out — so
integrating it is mostly "point an HTTP client at it." Below are the
common integration shapes.

### 1. As a drop-in embedding service for any app (any language)

Anything that can make an HTTP POST can use this. It doesn't need to be
Python.

**cURL**
```bash
curl -X POST http://your-gateway-host:8000/v1/infer \
  -H "Content-Type: application/json" \
  -H "X-Client-Id: my-app" \
  -d '{"text": "Explain quantum entanglement simply"}'
```

**Python**
```python
import httpx

resp = httpx.post(
    "http://your-gateway-host:8000/v1/infer",
    json={"text": "Explain quantum entanglement simply"},
    headers={"X-Client-Id": "my-app"},
    timeout=10.0,  # a little above RESULT_WAIT_TIMEOUT_S so you see the 504, not a client-side timeout first
)
resp.raise_for_status()
embedding = resp.json()["embedding"]
```

**Node.js**
```js
const resp = await fetch("http://your-gateway-host:8000/v1/infer", {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-Client-Id": "my-app" },
  body: JSON.stringify({ text: "Explain quantum entanglement simply" }),
});
if (!resp.ok) throw new Error(`gateway error ${resp.status}: ${await resp.text()}`);
const { embedding, cached } = await resp.json();
```

**Go**
```go
body, _ := json.Marshal(map[string]string{"text": "Explain quantum entanglement simply"})
req, _ := http.NewRequest("POST", "http://your-gateway-host:8000/v1/infer", bytes.NewReader(body))
req.Header.Set("Content-Type", "application/json")
req.Header.Set("X-Client-Id", "my-app")
resp, err := http.DefaultClient.Do(req)
```

Practical notes for any client:
- Set your HTTP client timeout **above** `RESULT_WAIT_TIMEOUT_S` (default
  `5.0`s) so a slow batch surfaces as the gateway's own `504` (with a
  useful message) instead of an opaque client-side timeout.
- Always send `X-Client-Id` with a stable value per calling
  application/tenant — it's what the rate limiter and metrics key on. If
  you omit it, every caller behind the same NAT/proxy shares one bucket
  (peer IP fallback), which under-counts distinct clients.
- Retry `429` and `504` with backoff; both are safe to retry (`/v1/infer`
  has no side effects beyond cache writes).

### 2. As a RAG pipeline's embedding layer

This is the shape it was built for: your RAG service calls the gateway
instead of loading a `sentence-transformers` model in-process.

```python
# retrieval-service (separate deployment, no torch/GPU needed itself)
async def embed_query(text: str) -> list[float]:
    resp = await http_client.post(f"{GATEWAY_URL}/v1/infer", json={"text": text})
    resp.raise_for_status()
    return resp.json()["embedding"]

async def retrieve(query: str, top_k: int = 5):
    query_vec = await embed_query(query)
    return vector_db.search(query_vec, top_k=top_k)  # pgvector, Pinecone, Qdrant, Weaviate, etc.
```

For **document ingestion** (embedding thousands of chunks at index-build
time, not per-user-request), don't call `/v1/infer` once per chunk in a
loop from your ingestion script — that pays a full HTTP round trip per
chunk. Instead, fan the calls out concurrently (`asyncio.gather` /
`httpx.AsyncClient` with a connection pool, or a thread pool for sync
code) so many requests are in flight at once; the gateway's whole design
is to absorb exactly that kind of concurrent burst and let the batching
engine pack it into dense batches on the worker side.

### 3. As the front door for an LLM or RAG *generation* call, not just embeddings

`worker/model_backend.py` defines the only contract the batching engine
cares about:

```python
class ModelBackend(Protocol):
    def predict_batch(self, texts: List[str]) -> List:
        ...
```

Swap in an LLM call, a reranker, or a full RAG (retrieve + generate)
pipeline by adding a new backend class and returning it from
`get_model_backend()` based on `MODEL_BACKEND`:

```python
# worker/model_backend.py
class OpenAICompatibleBackend:
    """Batches requests into one call to a hosted LLM API. Note: many
    hosted APIs don't support server-side batching of independent prompts,
    so 'batch' here often means 'fire N concurrent calls and gather them' —
    still valuable, because it's still one flush per BATCH_TIMEOUT_MS window
    instead of the caller managing concurrency itself."""

    def __init__(self):
        import openai
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def predict_batch(self, texts: List[str]) -> List[str]:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(texts)) as pool:
            futures = [
                pool.submit(
                    self.client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": t}],
                )
                for t in texts
            ]
            return [f.result().choices[0].message.content for f in futures]


def get_model_backend() -> ModelBackend:
    if config.MODEL_BACKEND == "echo":
        return EchoBackend()
    if config.MODEL_BACKEND == "openai":
        return OpenAICompatibleBackend()
    return EmbeddingModelBackend()
```

Then `MODEL_BACKEND=openai docker compose up` — the gateway, Nginx, Redis
Stream, rate limiter, cache, and batching/reclaim loop are all unchanged.
If you do this, also widen `gateway/schemas.py`'s `InferResponse` (it
currently assumes `embedding: list[float]`) to whatever shape your backend
returns — the gateway is intentionally thin here, so it's meant to be
edited for your payload shape rather than treated as fixed.

### 4. Behind an existing API gateway / ingress (Kong, Envoy, AWS API Gateway, Apigee)

This project's own `nginx.conf` handles load balancing across gateway
replicas, but nothing stops you from putting it *behind* your
organization's existing edge gateway instead of (or in addition to)
exposing Nginx directly:

```
Client → Your org's API Gateway (auth, TLS, org-wide rate limits, WAF)
       → nginx (this project — least_conn LB across gateway replicas)
       → gateway-1..N → Redis Streams → worker-1..N
```

In that topology, let the outer gateway own authentication/API-key
validation and pass a verified tenant identifier through as `X-Client-Id`
(or have `gateway/main.py`'s `_client_id()` read whatever header your edge
gateway injects after auth) — see the note on `X-Client-Id` trust in
[Known Limitations](#known-limitations--future-work). This gateway's own
token bucket then becomes a *second*, per-tenant line of defense in front
of the batch workers, not the only rate limiter in the system.

### 5. As a Kubernetes Deployment

The compose file maps directly onto k8s primitives:

- `gateway-1..3` → a `Deployment` + `Service` for `gateway`, scaled via
  `HorizontalPodAutoscaler` on CPU or (better) on the
  `gateway_request_latency_seconds` / connection-count metrics exposed at
  `/metrics`.
- `nginx` → either keep it as an in-cluster `Deployment` + `Service`, or
  drop it entirely and let a k8s `Service` (ClusterIP with multiple
  gateway pod endpoints) or an `Ingress` do the load balancing instead.
- `worker-1..2` → a separate `Deployment`, scaled independently — this is
  the whole point of splitting them from the gateway tier. Give GPU
  workers a `nodeSelector`/`resources.limits: nvidia.com/gpu: 1` and a
  separate `HorizontalPodAutoscaler` keyed on `worker_queue_wait_seconds`
  (queue backing up is the real "add more workers" signal, better than
  CPU%, since the model call itself may be GPU-bound).
- `redis` → for anything beyond a demo, use a managed Redis (ElastiCache,
  Memorystore, Redis Cloud) or a `StatefulSet` with persistence, not a bare
  `redis:7-alpine` pod — see [Known Limitations](#known-limitations--future-work)
  on the single-Redis-instance tradeoff.
- Use `/health` as both the liveness *and* readiness probe for `gateway`
  pods — it now verifies Redis connectivity (see the notes in
  [Known Limitations](#known-limitations--future-work) history), so a pod
  that's up but can't reach Redis correctly drops out of the Service's
  endpoint list instead of receiving traffic it can't serve.

### 6. Calling the Redis Stream directly, without going through `/v1/infer`

For an internal service that's already inside the same network/VPC as
Redis and wants to skip the extra HTTP hop, you can enqueue onto
`inference:requests` directly using any Redis client, matching the field
names `gateway/redis_stream.py` writes:

```python
import redis, uuid, time, json

r = redis.Redis(host="redis-host", decode_responses=True)
request_id = str(uuid.uuid4())
r.xadd("inference:requests", {"request_id": request_id, "text": "some text", "ts": str(time.time())})

# block for the result the same way the gateway does
_, raw = r.brpop(f"result:{request_id}", timeout=5)
result = json.loads(raw)
```

This bypasses the gateway's rate limiter and cache entirely, so only do it
for trusted internal callers where that's an acceptable tradeoff — it's
useful for, e.g., a batch/offline job that wants to feed the same worker
pool without going through Nginx, but it's not a substitute for the
`/v1/infer` contract for anything client-facing.

### 7. Feeding it from a message queue / event stream (Kafka, SQS, Pub/Sub)

If your requests originate from an existing queue rather than synchronous
HTTP callers, put a thin consumer in front that reads from your queue and
calls `/v1/infer` (or `XADD`s directly, per #6), fanning out concurrently:

```python
async def consume_kafka_and_infer(consumer, gateway_client):
    async for msg in consumer:
        text = json.loads(msg.value)["text"]
        # fire-and-forget concurrently; don't await each one serially —
        # let the gateway's batching engine absorb the concurrency
        asyncio.create_task(gateway_client.post("/v1/infer", json={"text": text}))
```

The gateway doesn't care whether the caller is a web request handler, a
queue consumer, or a cron job — it's just an HTTP (or Redis Stream)
producer from the gateway's point of view.

### 8. Local development against a mocked backend

For integration-testing *your* service against this gateway without
running a real model, use `MODEL_BACKEND=echo` (see Quickstart) — the
worker returns deterministic fake vectors (`[len(text), hash(text) % 997]`)
so your tests can assert on exact output without any GPU/model download,
while still exercising the real HTTP → Redis → batching → HTTP round trip.

---

## Benchmarking

```bash
docker compose up --build -d
bash benchmarks/run_benchmark.sh
```

`benchmarks/locustfile.py` mixes three traffic shapes so a single run
exercises every part of the system: `UniqueQueryUser` (always a cache miss,
exercises the full batching path), `RepeatedQueryUser` (draws from a small
prompt pool — exercises the cache), and `BurstUser` (near-zero think time —
exercises the rate limiter and 429/503/504 shedding).

For a genuine 1M-request-class load test, run Locust in **distributed
mode** — a single Locust process is itself CPU-bound generating load well
before this gateway becomes the bottleneck:

```bash
locust -f benchmarks/locustfile.py --master --host http://localhost:8000
locust -f benchmarks/locustfile.py --worker --master-host <master-ip>   # × N load-gen machines
```

---

## Monitoring (Prometheus + Grafana)

- **Prometheus** (`:9090`) scrapes all 3 gateway replicas and both worker
  replicas every 2s (`monitoring/prometheus.yml`). Gateway metrics:
  `gateway_requests_total`, `gateway_cache_hits_total`,
  `gateway_rate_limited_total`, `gateway_worker_timeouts_total`,
  `gateway_request_latency_seconds`. Worker metrics: `worker_queue_wait_seconds`,
  `worker_batch_exec_seconds`, `worker_batch_size_distribution`,
  `worker_reclaimed_messages_total` (count of messages recovered from a
  dead/stuck consumer — should normally sit at or near zero; a rising
  count means workers are crashing mid-batch and worth investigating).
- **Grafana** (`:3000`, `admin`/`admin`) — point it at the Prometheus
  datasource to chart cache hit rate, rate-limit rejection rate, batch size
  distribution, and end-to-end latency across all replicas.

Suggested alerts once you're running this for real: `worker_queue_wait_seconds`
p99 climbing (workers can't keep up — scale the worker tier),
`gateway_worker_timeouts_total` rate increasing (workers are down or too
slow relative to `RESULT_WAIT_TIMEOUT_S`), and `worker_reclaimed_messages_total`
increasing (workers are crashing).

---

## Testing

The test suite runs against a real (local) Redis rather than mocking Redis
semantics, and uses the dependency-free `echo` model backend, so it needs
no GPU/torch and runs in well under a second:

```bash
pip install -r requirements.txt
redis-server --daemonize yes
pytest tests/ -v
```

Covers: cache key hashing/round-tripping, token-bucket correctness
(capacity, per-client isolation, refill-over-time), and a full
enqueue → batch → reply cycle through `StreamGateway` + a simulated batch
pass — including that results route back to the *correct* client when
multiple requests share a batch, timeout behavior when no worker responds,
error propagation from a failed batch, and that the batching loop actually
accumulates trickling (not just bursty) traffic within its timeout window.

---

## Operating in Production

A short checklist beyond "it runs in Docker Compose":

- **Auth**: put real authentication in front of `X-Client-Id` (see
  integration pattern #4 and [Known Limitations](#known-limitations--future-work)) —
  as shipped, any caller can self-report any client ID.
- **TLS**: terminate TLS at Nginx (or your outer edge gateway) — this
  project's `nginx.conf` listens on plain `:80` for in-cluster/internal use.
- **Redis durability**: turn on AOF (`appendonly yes`) if losing an
  in-flight backlog on a Redis crash is unacceptable for your workload —
  off by default here for maximum throughput.
- **Redis HA**: a single `redis:7-alpine` container is a single point of
  failure; use a managed Redis or Sentinel/Cluster setup for anything
  beyond development.
- **Sizing**: use `worker_queue_wait_seconds` and `worker_batch_size_distribution`
  from Prometheus to decide whether to add worker replicas, raise
  `MAX_BATCH_SIZE`, or both — don't guess.
- **Timeouts**: keep `RESULT_WAIT_TIMEOUT_S` (gateway), Nginx's
  `proxy_read_timeout`, and your calling client's own HTTP timeout
  consistent with each other (each one should be ≥ the one before it),
  or you'll see confusing timeout errors at the wrong layer.

---

## Relationship to the Dynamic Inference Project

This gateway is architecturally a **horizontal-scale-out evolution** of the
[`Dynamic Inference`](../Dynamic%20Inference) service:

| | Dynamic Inference (Redis-coordinated mode) | AI Proxy & Batching Gateway |
|---|---|---|
| Ingestion | FastAPI workers behind uvicorn `--workers N` | FastAPI gateway replicas behind **Nginx** |
| Queue | Redis Stream (same design) | Redis Stream (same design) |
| Batching | In-process batch loop per worker | Same algorithm, in a **separate worker pool** |
| Result handoff | Polling a Redis key every 2ms | **`BRPOP`** — no polling |
| Caching | None | Redis response cache (hash of payload) |
| Rate limiting | None | Redis token bucket (Lua script, atomic) |
| Model | Fixed embedding model | Pluggable backend (embedding / echo / your LLM or RAG call) |

If you already have the Dynamic Inference service running, `worker/` here
is effectively that project's `distributed_batcher.py` extracted into its
own independently-scalable service, sitting behind a proper proxy tier
instead of being co-located with the request-handling process.

---

## Known Limitations & Future Work

- **Redis persistence is off by default** (`docker-compose.yml` runs a
  vanilla `redis:7-alpine` with no AOF). This maximizes throughput but
  means an in-flight backlog is lost if Redis itself crashes mid-spike.
  Enable AOF (`appendonly yes`) if you need at-least-once delivery through
  a Redis restart, at some cost to write throughput.
- **No request coalescing / thundering-herd protection.** If many
  identical requests miss the cache at the same instant (e.g. right after
  a TTL expiry), each is independently enqueued and computed rather than
  the second-through-Nth waiting on the first's in-flight result. A
  "singleflight"-style lock per cache key would close this gap.
- **Single Redis instance.** Both the stream and the cache/rate-limiter
  live on one Redis node here — for true massive scale, Redis Cluster (or
  separate Redis instances per concern) removes it as a single point of
  contention.
- **One model instance per worker process.** Like the base Dynamic
  Inference project, horizontal worker scaling multiplies memory usage;
  GPU workers change this trade-off and are worth benchmarking separately
  from the CPU numbers referenced above.
- **Fixed-window burst allowance.** The token bucket is a solid general
  limiter, but very bursty legitimate clients (e.g. a batch job) will see
  `429`s once their burst exceeds `RATE_LIMIT_CAPACITY` — tune capacity per
  client tier, or add a per-API-key override table, for production use.
- **`X-Client-Id` is client-supplied and unauthenticated.** `gateway/main.py`'s
  `_client_id()` trusts whatever header the caller sends (falling back to
  peer IP only if it's absent), which is fine for demo/internal use but
  means any external caller can evade the rate limiter by rotating the
  header value. Before exposing this publicly, replace it with a value the
  gateway itself verifies — e.g. an API key looked up against a Redis/DB
  table, or a claim out of a validated JWT — rather than a bare
  self-reported header. See integration pattern #4 above for one way to
  offload this to an existing edge gateway instead.
- **One in-flight batch per worker process.** `run_worker()`'s main loop
  `await`s the model call before starting the next batch collection, so a
  single worker replica never has two batches executing concurrently.
  That's the right call for a GPU worker with one model instance (there's
  only one device to run on anyway), but if a backend can genuinely serve
  overlapping batches (e.g. an async LLM API call), pipelining collection
  and execution would raise throughput per replica.

---

## FAQ

**Does this only work for embeddings?**
No — embeddings are just the default backend. Swap `MODEL_BACKEND` and add
a class implementing `predict_batch(texts) -> list` in
`worker/model_backend.py` to point it at an LLM, reranker, or full RAG
pipeline. See [integration pattern #3](#3-as-the-front-door-for-an-llm-or-rag-generation-call-not-just-embeddings).

**Can I use this from a non-Python service?**
Yes — the public contract is plain HTTP/JSON (`POST /v1/infer`). See
[integration pattern #1](#1-as-a-drop-in-embedding-service-for-any-app-any-language)
for examples in curl, Python, Node.js, and Go.

**What happens if a worker crashes mid-batch?**
The stale-message reclaim loop (`_reclaim_stale_messages`, `XAUTOCLAIM`)
picks up any message that's been claimed-but-unacknowledged for longer
than `STALE_CLAIM_MS` and reprocesses it on another worker — the caller
sees added latency (bounded by `STALE_CLAIM_MS` + `RECLAIM_INTERVAL_S`),
not a lost request, as long as it's within `RESULT_WAIT_TIMEOUT_S`.

**How do I know if I need more workers vs. more gateway replicas?**
Check `worker_queue_wait_seconds` in Prometheus — if it's climbing, requests
are piling up in the Stream faster than workers drain it, so add worker
replicas (or GPUs, or raise `MAX_BATCH_SIZE`). Gateway-tier saturation
shows up as connection errors/timeouts at the Nginx layer well before the
gateway pods themselves become CPU-bound, since they do no model work.

**Is it safe to retry a request that got a 504?**
Yes for the embedding backend (deterministic, side-effect-free beyond a
cache write). If you swap in an LLM backend with real side effects (e.g.
one that calls external tools), make sure `predict_batch` on your custom
backend stays idempotent, or add your own idempotency key on top.
