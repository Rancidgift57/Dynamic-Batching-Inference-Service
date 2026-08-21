import asyncio, os, socket, time
from app.models import EmbeddingModel
from app.distributed_queue import RedisCoordinatedQueue
from app.metrics import QUEUE_WAIT_SECONDS, GPU_EXECUTION_SECONDS, BATCH_SIZE

MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 16))
BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", 10))
CONSUMER_NAME = f"{socket.gethostname()}-{os.getpid()}"  # unique per process

async def run_worker(model: EmbeddingModel, q: RedisCoordinatedQueue):
    await q.connect()
    while True:
        items = await q.collect_batch(CONSUMER_NAME, MAX_BATCH_SIZE, BATCH_TIMEOUT_MS)
        if not items:
            continue

        now = time.time()
        for _, item in items:
            QUEUE_WAIT_SECONDS.observe(now - item.enqueued_at)
        BATCH_SIZE.observe(len(items))

        texts = [item.text for _, item in items]
        t0 = time.perf_counter()
        try:
            embeddings = await asyncio.to_thread(model.predict_batch, texts)
            GPU_EXECUTION_SECONDS.observe(time.perf_counter() - t0)
            for (msg_id, item), vec in zip(items, embeddings):
                await q.publish_result(item.request_id, embedding=vec)
        except Exception as e:
            for msg_id, item in items:
                await q.publish_result(item.request_id, error=str(e))

        await q.ack([msg_id for msg_id, _ in items])