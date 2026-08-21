import asyncio
from models import EmbeddingModel
from server import DynamicBatcher

async def main():
    batcher = DynamicBatcher(EmbeddingModel())
    batcher.start()
    results = await asyncio.gather(*[batcher.submit(f"text{i}") for i in range(20)])
    print(len(results),len(results[0]))

asyncio.run(main())