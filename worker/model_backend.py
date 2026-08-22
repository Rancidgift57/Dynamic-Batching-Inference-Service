"""
Pluggable inference backend.

The batching engine (`batch_worker.py`) only knows about `predict_batch(texts) -> list`.
What runs behind that call is swappable, which is the point: the same
gateway + Redis Stream + dynamic-batching pipeline works whether the
payload behind it is an embedding model, a reranker, or a call into an LLM
/ RAG pipeline. Add a new backend here and flip MODEL_BACKEND to switch.
"""
import logging
import time
from typing import List, Protocol

from . import config

logger = logging.getLogger("model_backend")


class ModelBackend(Protocol):
    def predict_batch(self, texts: List[str]) -> List:
        ...


class EmbeddingModelBackend:
    """Default backend: sentence-transformers embedding model, batched in one
    forward pass — the same model used in the standalone Dynamic Inference
    service this gateway builds on."""

    def __init__(self, model_name: str = config.MODEL_NAME):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading model '%s' on device=%s", model_name, self.device)
        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model.encode(["warmup"], convert_to_numpy=True)
        logger.info("Model loaded in %.2fs", time.perf_counter() - t0)

    def predict_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


class EchoBackend:
    """Zero-dependency stand-in for local dev, load testing the gateway/Redis
    plumbing, and CI — returns deterministic fake vectors instead of loading
    a real model, so you can push millions of requests through the queueing
    and batching layers without needing a GPU or torch installed."""

    def predict_batch(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t)), float(hash(t) % 997)] for t in texts]


def get_model_backend() -> ModelBackend:
    if config.MODEL_BACKEND == "echo":
        return EchoBackend()
    return EmbeddingModelBackend()
