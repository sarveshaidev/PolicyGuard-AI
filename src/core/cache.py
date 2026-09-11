import faiss
import numpy as np
from src.core.embedder_singleton import embedder
from config.settings import CACHE_SIMILARITY_THRESHOLD
import logging

logger = logging.getLogger(__name__) # Fixed

class SemanticCache:
    def __init__(self): # Fixed
        self.index = faiss.IndexFlatIP(embedder.get_embedding_dimension())
        self.cache = {} 

    def _embed(self, text: str) -> np.ndarray:
        return embedder.encode([text], normalize_embeddings=True).astype('float32')

    def get(self, query: str) -> str | None:
        if self.index.ntotal == 0: return None
        query_vec = self._embed(query)
        scores, indices = self.index.search(query_vec, 1)
        if scores[0][0] >= CACHE_SIMILARITY_THRESHOLD:
            return self.cache[indices[0][0]]["answer"]
        return None

    def put(self, query: str, answer: str):
        query_vec = self._embed(query)
        self.index.add(query_vec)
        self.cache[self.index.ntotal - 1] = {"query": query, "answer": answer}