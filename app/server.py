import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List
from app.models import EmbeddingModel
from app.metrics import QUEUE_WAIT_SECONDS, GPU_EXECUTION_SECONDS, BATCH_SIZE

logger = logging.getLogger("batcher")

MAX_BATCH_SIZE = 16
BATCH_TIMEOUT_S = 0.010
MAX_QUEUE_DEPTH = 500


class ServiceOverloadedError(Exception):
    """Raised when the queue is at capacity and cannot accept more work."""
    pass


@dataclass
class QueueItem:
    text: str
    future: asyncio.Future
    enqueued_at: float


class DynamicBatcher:
    def __init__(self, model: EmbeddingModel):
        self.model = model
        self.queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=MAX_QUEUE_DEPTH)
        self._task: asyncio.Task | None = None

    async def submit(self, text: str) -> List[float]:
        fut = asyncio.get_event_loop().create_future()
        item = QueueItem(text=text, future=fut, enqueued_at=time.perf_counter())
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            raise ServiceOverloadedError("Queue is full — server is overloaded")
        return await fut

    def start(self):
        self._task = asyncio.create_task(self._batch_processor())
        logger.info("Batch processor started")

    async def stop(self):
        if self._task:
            self._task.cancel()

    @staticmethod
    def _bucket_key(text: str, bucket_size: int = 32) -> int:
        """Groups similar-length texts so a batch isn't dominated by padding
        from one long outlier."""
        return len(text) // bucket_size

    async def _collect_batch(self) -> List[QueueItem]:
        first_item = await self.queue.get()
        bucket = self._bucket_key(first_item.text)
        batch = [first_item]
        deadline = time.perf_counter() + BATCH_TIMEOUT_S
        held_back = []

        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if self._bucket_key(item.text) == bucket:
                batch.append(item)
            else:
                held_back.append(item)

        for item in held_back:
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:
                item.future.set_exception(
                    ServiceOverloadedError("Queue full while re-queuing")
                )
        return batch

    async def _batch_processor(self):
        while True:
            try:
                batch = await self._collect_batch()
                await self._run_batch(batch)
            except asyncio.CancelledError:
                logger.info("Batch processor shutting down")
                break
            except Exception:
                logger.exception("Unexpected error in batch processor loop")

    async def _run_batch(self, batch: List[QueueItem]):
        now = time.perf_counter()
        for item in batch:
            QUEUE_WAIT_SECONDS.observe(now - item.enqueued_at)
        BATCH_SIZE.observe(len(batch))

        texts = [item.text for item in batch]
        t0 = time.perf_counter()
        try:
            embeddings = await asyncio.to_thread(self.model.predict_batch, texts)
            GPU_EXECUTION_SECONDS.observe(time.perf_counter() - t0)
        except Exception as e:
            logger.exception(f"Inference failed for batch of size {len(batch)}")
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)
            return

        for item, vec in zip(batch, embeddings):
            if not item.future.done():
                item.future.set_result(vec)