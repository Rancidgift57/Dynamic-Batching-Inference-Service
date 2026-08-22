import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

STREAM_KEY = "inference:requests"
GROUP_NAME = "batchers"

MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "embedding")  # "embedding" | "echo"

MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 32))
BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", 10))

# How long XREADGROUP blocks waiting for the *first* item of the next batch
# when the stream is idle. Separate from BATCH_TIMEOUT_MS (the accumulation
# window for items 2..N of an already-started batch) so an idle worker
# doesn't busy-poll Redis every 10ms with nothing to do.
IDLE_BLOCK_MS = int(os.environ.get("IDLE_BLOCK_MS", 1000))

RESULT_TTL_SECONDS = int(os.environ.get("RESULT_TTL_SECONDS", 30))
METRICS_PORT = int(os.environ.get("METRICS_PORT", 9100))

# --- Stale pending-message reclaim (recovers from a worker crashing after
#     XREADGROUP claims a message but before it XACKs it) ---
STALE_CLAIM_MS = int(os.environ.get("STALE_CLAIM_MS", 30_000))
RECLAIM_INTERVAL_S = float(os.environ.get("RECLAIM_INTERVAL_S", 5.0))
