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

## Known Limitations & Future Work

- **Result delivery is poll-based.** `RedisCoordinatedQueue.submit()` polls a
  result key every 2ms rather than blocking on it (see the comment in
  `app/distributed_queue.py`). A pub/sub or `BLPOP`-based handoff would cut
  wasted round trips and slightly reduce tail latency under high fan-out.
- **`p99`/`Avg Batch Size` are missing for both `Batch=32` rows** above — they
  weren't captured in the source benchmark run. Re-run
  `benchmarks/run_benchmark.sh` with a `dynamic_batch32` entry and Prometheus
  scraping enabled to fill these in.
- **Single CPU-bound model instance per worker.** Every uvicorn worker loads
  its own copy of `EmbeddingModel`. On CPU this means N workers cost N× the
  memory and don't parallelize a single forward pass — the throughput gains
  from more workers come entirely from overlapping I/O/queueing, not
  compute parallelism. On GPU this trade-off changes and is worth
  re-benchmarking separately.
- **No horizontal Redis scaling.** A single Redis instance backs the whole
  stream; for very high request volumes it can itself become a bottleneck
  and would need clustering or a dedicated Redis instance.
