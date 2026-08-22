from prometheus_client import Counter, Histogram

QUEUE_WAIT_SECONDS = Histogram(
    "worker_queue_wait_seconds",
    "Time a request sat in the Redis Stream before a worker claimed it",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

BATCH_EXEC_SECONDS = Histogram(
    "worker_batch_exec_seconds",
    "Time spent inside the model's forward pass for one batch",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

BATCH_SIZE = Histogram(
    "worker_batch_size_distribution",
    "Number of requests packed into each processed batch",
    buckets=[1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
)

RECLAIMED_TOTAL = Counter(
    "worker_reclaimed_messages_total",
    "Pending stream messages reclaimed from a dead/stuck consumer via XAUTOCLAIM",
)
