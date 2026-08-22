import asyncio
import time
from app.models import EmbeddingModel
from app.server import DynamicBatcher

async def run(texts, label):
    model = EmbeddingModel()
    batcher = DynamicBatcher(model)
    batcher.start()
    t0 = time.perf_counter()
    await asyncio.gather(*[batcher.submit(t) for t in texts])
    elapsed = time.perf_counter() - t0
    await batcher.stop()
    print(f"{label}: {elapsed:.2f}s total for {len(texts)} requests")

if __name__ == "__main__":
    short = ["short text"] * 100
    long = ["long text " * 100] * 100
    mixed = short + long
    asyncio.run(run(mixed, "bimodal workload"))