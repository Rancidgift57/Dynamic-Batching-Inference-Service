from prometheus_client import Histogram

QUEUE_WAIT_SECONDS = Histogram(
    "inference_queue_wait_seconds",
    "Time a request waited in queue before being included in a batch",
    buckets=[0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0]
)

GPU_EXECUTION_SECONDS = Histogram(
    "inference_gpu_execution_seconds",
    "Time spent inside the forward pass for a batch",
    buckets=[0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0]
)

BATCH_SIZE = Histogram(
    "batch_size_distribution",
    "Number of requests grouped into each processed batch",
    buckets=[1,2,4,8,16,24,32,48,64]
)

