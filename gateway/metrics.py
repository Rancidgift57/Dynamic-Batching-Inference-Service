from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "gateway_requests_total", "Total requests received by this gateway replica"
)

CACHE_HITS_TOTAL = Counter(
    "gateway_cache_hits_total", "Requests served directly from the response cache"
)

RATE_LIMITED_TOTAL = Counter(
    "gateway_rate_limited_total", "Requests rejected with 429 by the token bucket"
)

TIMEOUTS_TOTAL = Counter(
    "gateway_worker_timeouts_total", "Requests that timed out waiting for a batch worker"
)

REQUEST_LATENCY_SECONDS = Histogram(
    "gateway_request_latency_seconds",
    "End-to-end latency observed by the gateway, from ingest to response",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
