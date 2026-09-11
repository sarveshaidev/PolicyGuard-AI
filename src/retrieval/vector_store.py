import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from src.core.embedder_singleton import embedder

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_dim: int = None, db_path: Path = None):
        self.embedder = embedder
        self.embedding_dim = embedding_dim or self.embedder.get_embedding_dimension()
        self.db_path = db_path or Path("data/vector_db")
        self.index = None
        self.chunks = []
        self._initialize_index()

    def _initialize_index(self):
        self.index = faiss.IndexFlatL2(self.embedding_dim)

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        if not embeddings: 
            return
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.chunks.extend(chunks)

    def search(self, query_text: str, top_k: int = 5) -> Tuple[List[Dict], List[float]]:
        if self.index.ntotal == 0: 
            return [], []
        
        # Encode query
        query_embedding = self.embedder.encode([query_text], normalize_embeddings=True)
        
        # Perform search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Filter valid results
        results = [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]
        scores = [float(d) for d in distances[0]]
        
        # REMOVED: The logger.info line that was causing the spam in Terminal AND UI
        # The function now returns results silently
        
        return results, scores

    def save(self):
        self.db_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.db_path / "faiss.index"))
        with open(self.db_path / "chunks.pkl", 'wb') as f: 
            pickle.dump(self.chunks, f)

    def load(self) -> bool:
        if not (self.db_path / "faiss.index").exists(): 
            return False
        self.index = faiss.read_index(str(self.db_path / "faiss.index"))
        with open(self.db_path / "chunks.pkl", 'rb') as f: 
            self.chunks = pickle.load(f)
        return True