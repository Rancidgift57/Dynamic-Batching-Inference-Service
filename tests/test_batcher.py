# tests/test_batcher.py
import asyncio
import pytest

from app.server import DynamicBatcher


class FakeModel:
    """Deterministic stand-in: returns [len(text)] as the 'embedding'."""
    def predict_batch(self, texts):
        return [[float(len(t))] for t in texts]


class FailingModel:
    def predict_batch(self, texts):
        raise RuntimeError("simulated inference crash")


@pytest.mark.asyncio
async def test_max_size_trigger_flushes_immediately():
    batcher = DynamicBatcher(FakeModel())
    batcher.start()

    # Fire exactly MAX_BATCH_SIZE (16) requests at once
    results = await asyncio.gather(*[batcher.submit(f"t{i}") for i in range(16)])
    assert len(results) == 16
    await batcher.stop()


@pytest.mark.asyncio
async def test_timeout_trigger_flushes_partial_batch():
    batcher = DynamicBatcher(FakeModel())
    batcher.start()

    # Only 2 requests — must be flushed by the 10ms timeout, not batch size
    results = await asyncio.gather(batcher.submit("a"), batcher.submit("bb"))
    assert results == [[1.0], [2.0]]
    await batcher.stop()


@pytest.mark.asyncio
async def test_exception_propagates_to_every_waiter():
    batcher = DynamicBatcher(FailingModel())
    batcher.start()

    tasks = [batcher.submit(f"t{i}") for i in range(3)]
    with pytest.raises(RuntimeError):
        await asyncio.gather(*tasks)
    await batcher.stop()

# tests/test_batcher.py
@pytest.mark.asyncio
async def test_length_bucketing_separates_short_and_long():
    captured_batches = []

    class RecordingModel:
        def predict_batch(self, texts):
            captured_batches.append(list(texts))
            return [[float(len(t))] for t in texts]

    batcher = DynamicBatcher(RecordingModel())
    batcher.start()

    short_texts = ["a"] * 5
    long_texts = ["b" * 500] * 5
    await asyncio.gather(*[batcher.submit(t) for t in short_texts + long_texts])
    await batcher.stop()

    # Every captured batch should be internally consistent in bucket
    for batch in captured_batches:
        buckets = {DynamicBatcher._bucket_key(t) for t in batch}
        assert len(buckets) == 1