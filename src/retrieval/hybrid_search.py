import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from rank_bm25 import BM25Okapi
from src.core.embedder_singleton import embedder

logger = logging.getLogger(__name__)

class HybridVectorStore:
    def __init__(self, embedding_dim: int = None, db_path: Path = None):
        self.embedder = embedder
        self.embedding_dim = embedding_dim or self.embedder.get_embedding_dimension()
        self.db_path = db_path or Path("data/vector_db")
        
        # Dense Index (FAISS)
        self.faiss_index = None
        self.chunks = []
        
        # Sparse Index (BM25)
        self.bm25_index = None
        self.tokenized_corpus = []
        
        self._initialize_indices()

    def _initialize_indices(self):
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

    def _tokenize_text(self, text: str) -> List[str]:
        # Simple tokenizer for BM25
        return [w.lower() for w in text.replace('.', ' ').replace(',', ' ').split()]

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        if not embeddings: 
            return
        
        # 1. Add to FAISS (Dense)
        self.faiss_index.add(np.array(embeddings, dtype=np.float32))
        
        # 2. Add to BM25 (Sparse)
        new_tokens = []
        for chunk in chunks:
            tokens = self._tokenize_text(chunk["content"])
            new_tokens.append(tokens)
            self.chunks.append(chunk)
            
        if self.bm25_index is None:
            self.bm25_index = BM25Okapi(new_tokens)
            self.tokenized_corpus = new_tokens
        else:
            # Rebuild BM25 index with new corpus
            self.tokenized_corpus.extend(new_tokens)
            self.bm25_index = BM25Okapi(self.tokenized_corpus)

    def search(self, query_text: str, top_k: int = 5, alpha: float = 0.5) -> Tuple[List[Dict], List[float]]:
        """
        Hybrid Search using Reciprocal Rank Fusion (RRF).
        """
        if self.faiss_index.ntotal == 0:
            return [], []

        # --- A. Dense Search (FAISS) ---
        query_embedding = self.embedder.encode([query_text], normalize_embeddings=True)
        faiss_distances, faiss_indices = self.faiss_index.search(query_embedding, min(top_k * 2, self.faiss_index.ntotal))
        faiss_results = [(idx, 1.0 / (dist + 1e-9)) for idx, dist in zip(faiss_indices[0], faiss_distances[0])]
        
        # --- B. Sparse Search (BM25) ---
        query_tokens = self._tokenize_text(query_text)
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        bm25_results = [(idx, bm25_scores[idx]) for idx in bm25_top_indices]

        # --- C. Reciprocal Rank Fusion (RRF) ---
        rrf_scores = {}
        k_const = 60 
        
        for rank, (idx, score) in enumerate(faiss_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1.0 / (k_const + rank + 1))
            
        for rank, (idx, score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1.0 / (k_const + rank + 1))

        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        final_chunks = [self.chunks[idx] for idx in sorted_indices]
        final_scores = [rrf_scores[idx] for idx in sorted_indices]
        
        return final_chunks, final_scores

    def save(self):
        self.db_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(self.db_path / "faiss.index"))
        with open(self.db_path / "chunks.pkl", 'wb') as f: 
            pickle.dump({"chunks": self.chunks, "tokenized_corpus": self.tokenized_corpus}, f)

    def load(self) -> bool:
        # FIXED INDENTATION STARTS HERE
        if not (self.db_path / "faiss.index").exists(): 
            return False
        
        try:
            # Load FAISS Index
            self.faiss_index = faiss.read_index(str(self.db_path / "faiss.index"))
            
            # Load Chunks and Tokenized Corpus
            with open(self.db_path / "chunks.pkl", 'rb') as f: 
                data = pickle.load(f)
            
            # BACKWARD COMPATIBILITY CHECK
            if isinstance(data, dict):
                # New format: {"chunks": [...], "tokenized_corpus": [...]}
                self.chunks = data.get("chunks", [])
                self.tokenized_corpus = data.get("tokenized_corpus", [])
                logger.info("✅ Loaded Vector DB (New Format)")
            elif isinstance(data, list):
                # Old format: Just a list of chunks
                self.chunks = data
                self.tokenized_corpus = [self._tokenize_text(c["content"]) for c in self.chunks]
                logger.warning("⚠️ Loaded Vector DB (Old Format). Re-tokenizing corpus...")
                self.save() # Save back in new format
            else:
                logger.error(" Invalid data format in chunks.pkl")
                return False
                
            # Initialize BM25 if we have tokens
            if self.tokenized_corpus:
                self.bm25_index = BM25Okapi(self.tokenized_corpus)
                logger.info(f"   BM25 Index initialized with {len(self.tokenized_corpus)} documents.")
            else:
                logger.warning("   No tokenized corpus found. BM25 will be empty.")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Vector DB: {e}")
            return False
        # FIXED INDENTATION ENDS HERE