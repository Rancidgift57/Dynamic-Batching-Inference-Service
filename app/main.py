import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse     # <- fixed: added JSONResponse
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.models import EmbeddingModel                       # <- fixed: model, not models
from app.server import DynamicBatcher, ServiceOverloadedError  # now resolvable
from app.distributed_queue import RedisCoordinatedQueue
from app.distributed_batcher import run_worker



logging .basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = EmbeddingModel()
    q = RedisCoordinatedQueue()
    state["queue"] = q
    worker_task = asyncio.create_task(run_worker(model, q))
    yield
    worker_task.cancel()

app = FastAPI(title = "Dynamic Batching inference Service", lifespan=lifespan)

class InferRequest(BaseModel):
    text: str

class InferResponse(BaseModel):
    embedding :list[float]


@app.exception_handler(ServiceOverloadedError)
async def overload_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)},
                         headers={"Retry-After": "1"})


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    q: RedisCoordinatedQueue = state["queue"]
    try:
        embedding = await q.submit(req.text)
    except Exception as e:
        logger.exception("infer failed")
        raise HTTPException(status_code=500, detail=str(e))
    return InferResponse(embedding=embedding)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    return {"Status":"ok"}