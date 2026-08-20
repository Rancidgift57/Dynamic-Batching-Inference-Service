import logging
import time
from typing import List
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("model")

class EmbeddingModel:
    """
    Thin wrapper around a SentenceTransformer model.
    Loaded exactly once at process startup.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model '{model_name}' onto device={self.device}")
        t0 = time.perf_counter()
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()
        
        # Optional: Run a dummy warm-up inference to allocate CUDA context ahead of real traffic
        if self.device == "cuda":
            _ = self.model.encode(["warmup query"], convert_to_numpy=True)
            
        logger.info(f"Model loaded and warmed up in {time.perf_counter() - t0:.2f}s")

    @torch.inference_mode()
    def predict_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Tokenizes + embeds a whole batch in a single forward pass.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # Useful default for downstream cosine similarity
        )
        return embeddings.tolist()


if __name__ == "__main__":
    m = EmbeddingModel()
    vecs = m.predict_batch(["hello world", "dynamic batching is fun"])
    print(f"Batch Size: {len(vecs)}, Vector Dimension: {len(vecs[0])}")  # -> Batch Size: 2, Vector Dimension: 384