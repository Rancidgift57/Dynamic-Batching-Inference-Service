# Dynamic Batching Inference Service

A FastAPI-based inference server for sentence embeddings that batches concurrent
requests together before sending them through the model, instead of running
inference one request at a time. It ships in two forms:

1. **Single-process dynamic batching** — an in-memory `asyncio.Queue` batcher
   running inside one uvicorn worker.
2. **Redis-coordinated dynamic batching** — the same batching logic, but the
   queue lives in a Redis Stream so multiple uvicorn worker processes (or
   multiple hosts) can share one global batch pipeline instead of each
   forming its own small, less-efficient batches.

Batching multiple requests into a single forward pass lets a model
(`sentence-transformers/all-MiniLM-L6-v2` by default) amortize fixed overhead
across many inputs, which is why throughput increases sharply as the batch
size grows — this repo exists to measure and demonstrate that effect.

---

## Table of Contents

- [How Dynamic Batching Works](#how-dynamic-batching-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Benchmark Methodology](#benchmark-methodology)
- [Benchmark Results](#benchmark-results)
- [Monitoring (Prometheus + Grafana)](#monitoring-prometheus--grafana)
- [Testing](#testing)
- [Known Limitations & Future Work](#known-limitations--future-work)

---

## How Dynamic Batching Works

Every `POST /infer` request is dropped into a queue instead of being sent to
the model immediately. A background loop pulls requests off that queue and
forms a batch using two triggers, whichever fires first:

- **Max batch size reached** (`MAX_BATCH_SIZE`, default `16`) — flush now.
- **Timeout elapsed** (`BATCH_TIMEOUT_MS`, default `10ms`) — flush whatever has
  accumulated so far, even if it's a partial batch.

This bounds worst-case added latency to roughly one timeout window while
still letting the server pack many requests into one GPU/CPU forward pass
under load.

**Length bucketing.** Text sequences are padded to the longest sequence in a
batch. If one 500-character outlier lands in a batch with 15 ten-character
requests, everyone pays the padding cost of the outlier. To avoid this, the
batcher groups the first request in a batch with only other requests of a
similar length (bucketed by `len(text) // 32`); requests that don't fit the
current bucket are put back on the queue rather than forced into a
mismatched batch. See `_bucket_key` / `_collect_batch` in `app/server.py`.

**Backpressure.** The single-process queue has a bounded depth
(`MAX_QUEUE_DEPTH = 500`). Once full, new requests get a `503` with
`Retry-After: 1` (`ServiceOverloadedError`) instead of queuing indefinitely —
shedding load is treated as correct behavior under overload, not a failure
(see `OverloadUser` in `benchmarks/locustfile.py`).

---

## Architecture

### Mode 1 — Single Process (in-memory queue)

```
                 ┌─────────────────────────────┐
 client ── HTTP ─▶  uvicorn (1 worker process)  │
                 │                              │
                 │  asyncio.Queue  ──▶  batch    │
                 │       ▲              collector│
                 │       │                 │     │
                 │  request futures        ▼     │
                 │       │          EmbeddingModel│
                 │       ◀──────────  (forward    │
                 │           result     pass)     │
                 └─────────────────────────────┘
```

Simple and fast, but batching only sees requests handled by *that one
process*. It cannot scale across CPU cores or hosts without losing the
global view of the queue — running `--workers 4` here just gives you 4
independent, uncoordinated batchers, each starved of traffic relative to a
single global queue.

Implementation: `Single Worker/app/server.py` (`DynamicBatcher`),
`Single Worker/app/main.py`.

### Mode 2 — Redis-Coordinated (multi-worker / multi-host)

```
                     ┌───────────────────────────────────────────┐
 client ── HTTP ────▶│ uvicorn worker 1  ──┐                     │
 client ── HTTP ────▶│ uvicorn worker 2  ──┤                     │
 client ── HTTP ────▶│ uvicorn worker 3  ──┼─▶  Redis Stream     │
 client ── HTTP ────▶│ uvicorn worker 4  ──┘    "inference:requests"
                     │                          (consumer group   │
                     │                           "batchers")      │
                     └───────────────────────────────────────────┘
                                     │
                       any worker's batch loop (XREADGROUP)
                       claims the next N pending messages
                                     │
                                     ▼
                          EmbeddingModel.predict_batch()
                                     │
                                     ▼
                     result written to `inference:result:<id>`
                     (30s TTL) — the original request's process
                     polls that key and returns it to the client
```

Every uvicorn worker process runs the *same* `run_worker` batch loop
(`app/distributed_batcher.py`) against the *same* Redis instance. A Redis
Stream + consumer group (`XADD` / `XREADGROUP` / `XACK`) ensures each
inbound request is claimed by exactly one worker's batch loop, but requests
from all workers/hosts flow through the same shared stream — so a 4-worker
deployment still forms large, well-packed batches instead of 4 small ones.
Results are handed back via a polled Redis key (`app/distributed_queue.py`);
this is a known simplification (see
[Known Limitations](#known-limitations--future-work)).

Implementation: `app/main.py`, `app/distributed_queue.py`
(`RedisCoordinatedQueue`), `app/distributed_batcher.py` (`run_worker`).

---

## Project Structure

```
Dynamic Inference/
├── app/                          # Redis-coordinated, multi-worker service (Mode 2)
│   ├── main.py                   # FastAPI app; wires RedisCoordinatedQueue + run_worker
│   ├── distributed_queue.py      # Redis Stream client (submit/collect_batch/publish_result)
│   ├── distributed_batcher.py    # Per-process batch loop (run_worker)
│   ├── server.py                 # DynamicBatcher — the in-process batching engine, reused
│   │                              #   for local unit tests and the padding benchmark
│   ├── models.py                 # EmbeddingModel wrapper (SentenceTransformer)
│   └── metrics.py                # Prometheus histograms
│
├── Single Worker/                # Standalone single-process service (Mode 1)
│   ├── app/                      # Same shape as app/, minus the Redis layer
│   ├── benchmarks/
│   └── Dockerfile
│
├── benchmarks/
│   ├── run_benchmark.sh          # Sweeps MAX_BATCH_SIZE against Locust load
│   ├── locustfile.py             # InferenceUser (steady load) + OverloadUser (shedding test)
│   └── padding_waste_benchmark.py# Bimodal short/long text batching demo
│
├── monitoring/
│   └── prometheus.yml            # Scrapes /metrics every 2s
│
├── tests/
│   └── test_batcher.py           # Unit tests for DynamicBatcher (max-size/timeout/bucketing)
│
├── docker-compose.yml            # inference-service + redis + prometheus + grafana
├── Dockerfile
├── requirements.txt
├── results_u50_*.csv             # Raw Locust output from a prior benchmark run
└── README.md
```

---

## Requirements

- Python 3.10+ (Docker image uses `python:3.10-slim`)
- Redis 6+ (only for Mode 2 — Redis Streams/consumer groups)
- Docker + Docker Compose (recommended path)
- ~1–2 GB RAM for the embedding model; GPU optional (`EmbeddingModel` auto-detects CUDA, falls back to CPU)

---

## Quickstart

### Option A — Docker Compose (recommended, Mode 2 / Redis-coordinated)

```bash
docker compose up --build
```

This starts `redis`, `inference-service` (1 worker by default), `prometheus`
(`:9090`), and `grafana` (`:3000`, default login `admin` / `admin`).

To reproduce the **4-worker** benchmark configuration below, override the
service command:

```bash
docker compose run --rm -p 8000:8000 inference-service \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

(or edit `--workers` in `docker-compose.yml` and re-run `docker compose up --build`).

### Option B — Local, Mode 2 (Redis-coordinated)

```bash
pip install -r requirements.txt
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 1 worker
REDIS_URL=redis://localhost:6379 uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# 4 workers (separate terminal / same command, just change the flag)
REDIS_URL=redis://localhost:6379 uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Option C — Local, Mode 1 (single process, no Redis)

```bash
cd "Single Worker"
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Try it

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"text": "dynamic batching is fun"}'
```

---

## Configuration

| Variable          | Default                   | Used by            | Description                                                        |
|--------------------|---------------------------|---------------------|----------------------------------------------------------------------|
| `MAX_BATCH_SIZE`   | `16`                      | Mode 2 (`app/distributed_batcher.py`) | Max requests collected before forcing a flush. Mode 1 hardcodes this to `16` in `server.py`. |
| `BATCH_TIMEOUT_MS` | `10`                      | Mode 2               | Max time (ms) to wait for a batch to fill before flushing partial. Mode 1 hardcodes `10ms` (`BATCH_TIMEOUT_S`). |
| `REDIS_URL`        | `redis://localhost:6379`  | Mode 2 only          | Redis connection string. Set to `redis://redis:6379` under Docker Compose. |
| `--workers`        | `1` (uvicorn flag)        | Mode 2               | Number of uvicorn worker processes; each runs its own batch loop against the shared Redis stream. |

---

## API Reference

| Method | Path       | Description                                                              |
|--------|------------|----------------------------------------------------------------------------|
| `POST` | `/infer`   | Body: `{"text": "..."}` → `{"embedding": [float, ...]}` (384-dim, L2-normalized). Returns `503` with `Retry-After: 1` if the service is overloaded. |
| `GET`  | `/health`  | Liveness check → `{"Status": "ok"}`                                       |
| `GET`  | `/metrics` | Prometheus exposition format — queue wait time, GPU/forward-pass time, batch size distribution. |

---

## Benchmark Methodology

All numbers below were produced with `benchmarks/run_benchmark.sh` /
`benchmarks/locustfile.py`:

- **Load generator:** Locust, headless mode
- **Users:** 100 concurrent
- **Spawn rate:** 20 users/sec
- **Duration:** 30s per configuration
- **Workload:** random 20-character strings, `InferenceUser` (0.01–0.05s think time between requests)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, CPU
- **Baseline** = dynamic batching effectively disabled (`MAX_BATCH_SIZE=1`, i.e. one request per forward pass) — isolates the throughput/latency contribution of batching itself, independent of worker count.
- Each row's **Avg Batch Size** comes from the `batch_size_distribution` Prometheus histogram (`app/metrics.py`).

---

## Benchmark Results

### 1 Worker (Redis-Coordinated)

`docker compose run ... --workers 1`, load driven directly at that single process, which owns the shared Redis stream alone.

| Configuration          | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Avg Batch Size |
|-------------------------|---------------------:|-----------:|-----------:|-----------:|-----------------:|
| Baseline (Batch=1)      | 112.5               | 320        | 580        | 850        | 1.0              |
| Dynamic (Batch=8)       | 215.4               | 180        | 250        | 410        | 5.4              |
| Dynamic (Batch=16)      | 299.6               | 110        | 140        | 270        | 11.2             |
| Dynamic (Batch=32)      | 344.3               | 110        | 140        | N/A*       | N/A*             |

\* Not captured in the source benchmark run for this configuration.

**Takeaways:**
- Going from `Batch=1` → `Batch=16` roughly **2.7×'s throughput** (112.5 → 299.6 req/s) while **cutting p99 latency by ~68%** (850ms → 270ms) — batching helps both metrics at once here because the bottleneck is forward-pass overhead, not queueing delay.
- `Batch=8` → `Batch=16` still gives a solid throughput gain (+39%) and the biggest tail-latency drop (p99 410ms → 270ms), suggesting `MAX_BATCH_SIZE=16` is a good default for this workload/hardware.
- `Batch=16` → `Batch=32` shows diminishing throughput returns (+15%) with p50/p95 already flat — at 100 concurrent users, the queue often can't accumulate a full 32-batch before the timeout fires, so `Avg Batch Size` in this row would be expected to land noticeably below 32 (consistent with the trend from 5.4 → 11.2 in the prior two rows).

### 4 Workers (Redis-Coordinated)

Same load profile, `docker compose run ... --workers 4` — 4 uvicorn processes, each running its own batch loop, all pulling from the same Redis Stream consumer group.

| Configuration                  | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Avg Batch Size |
|----------------------------------|---------------------:|-----------:|-----------:|-----------:|-----------------:|
| Baseline (4 Workers, Batch=1)    | 112.5               | 320        | 580        | 850        | 1.0              |
| Dynamic (4 Workers, Batch=8)     | 245.8               | 160        | 220        | 380        | 6.2              |
| Dynamic (4 Workers, Batch=16)    | 312.4               | 120        | 150        | 280        | 12.8             |
| Dynamic (4 Workers, Batch=32)    | 350.8               | 110        | 140        | N/A*       | N/A*             |

\* Not captured in the source benchmark run for this configuration.

**Takeaways:**
- The **`Batch=1` baseline is identical to the 1-worker baseline (112.5 req/s)**. With no batching, more workers competing for the same CPU-bound model forward pass on this single-node test setup doesn't add throughput — the bottleneck is compute, not concurrency, so extra worker processes mostly add scheduling overhead rather than parallel headroom.
- Once batching is enabled, 4 workers modestly outperform 1 worker at every batch size (e.g. `Batch=8`: 245.8 vs 215.4 req/s, **+14%**; `Batch=16`: 312.4 vs 299.6 req/s, **+4%**) — more workers means more independent batch loops racing to drain the Redis stream, which shortens queue-wait time slightly and lets `Avg Batch Size` climb a bit faster at the same `MAX_BATCH_SIZE` (6.2 vs 5.4 at Batch=8; 12.8 vs 11.2 at Batch=16).
- The gap between 1 and 4 workers **narrows as `MAX_BATCH_SIZE` grows** — by `Batch=32` the two configurations converge (350.8 vs 344.3 req/s). This matches the CPU-bound baseline result: once each batch is already large enough to saturate a forward pass efficiently, adding worker processes stops being the limiting factor.
- **Practical read:** on CPU, the dominant lever is batch size, not worker count — scaling `--workers` helps mainly by reducing the time it takes to *fill* a batch under concurrent load, not by parallelizing the model itself (there's still one model instance being loaded per process, each doing its own sequential forward passes).

---

## Monitoring (Prometheus + Grafana)

`docker compose up` also starts:

- **Prometheus** (`http://localhost:9090`) — scrapes `inference-service:8000/metrics` every 2s (`monitoring/prometheus.yml`).
- **Grafana** (`http://localhost:3000`, `admin`/`admin`) — point it at the Prometheus datasource to chart:
  - `inference_queue_wait_seconds` — time a request waited before being included in a batch
  - `inference_gpu_execution_seconds` — time spent in the model's forward pass per batch
  - `batch_size_distribution` — how many requests actually ended up in each processed batch

These three histograms are what the **Avg Batch Size** and latency columns in the
benchmark tables above are derived from.

---

## Testing

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Covers (`tests/test_batcher.py`):
- Flushing immediately once `MAX_BATCH_SIZE` is reached
- Flushing a partial batch once the timeout elapses
- Exceptions from the model propagating to every waiter in the failed batch
- Length-bucketing keeping short and long requests in separate batches

For the standalone padding-waste demo:

```bash
python benchmarks/padding_waste_benchmark.py
```

For a full load-test sweep across batch sizes:

```bash
bash benchmarks/run_benchmark.sh
```

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

----

## Contact
- Email: nnair7598@gmail.com
- LinkedIn: https://www.linkedin.com/in/nikhil-nair-809248286

## Thank You 

