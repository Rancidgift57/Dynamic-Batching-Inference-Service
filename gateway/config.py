import os

# --- Redis ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

# --- Model identity (used to namespace cache keys; the actual model lives
#     in the worker pool, not in the gateway process) ---
MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Stream / consumer group the worker pool reads from ---
STREAM_KEY = "inference:requests"
GROUP_NAME = "batchers"
# Backpressure valve: caps how large the stream can grow if workers fall
# permanently behind, so a stuck/undersized worker pool degrades into
# dropped-oldest-entries instead of unbounded Redis memory growth.
STREAM_MAXLEN = int(os.environ.get("STREAM_MAXLEN", 200_000))

# --- Rate limiting (token bucket, per client) ---
RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_CAPACITY = float(os.environ.get("RATE_LIMIT_CAPACITY", 100))       # max burst size (tokens)
RATE_LIMIT_REFILL_PER_SEC = float(os.environ.get("RATE_LIMIT_REFILL_PER_SEC", 20))  # steady-state req/s allowed

# --- Response cache ---
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 300))

# --- How long the gateway will wait on a batch worker before returning 504 ---
RESULT_WAIT_TIMEOUT_S = float(os.environ.get("RESULT_WAIT_TIMEOUT_S", 5.0))
