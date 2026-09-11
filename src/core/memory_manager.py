import sqlite3
import time
from pathlib import Path
from typing import List, Dict
import faiss
import numpy as np
import logging
from src.core.embedder_singleton import embedder

# FIXED: Use __name__
logger = logging.getLogger(__name__)

class MemoryManager:
    # FIXED: Use __init__
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path("data/memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.embedder = embedder
        self.memory_index = None
        self.memory_store = {} 
        
        self._init_db()
        self._load_memory_index()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                user_query TEXT,
                ai_response TEXT,
                summary TEXT,
                embedding_blob BLOB
            )
        ''')
        conn.commit()
        conn.close()

    def _load_memory_index(self):
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("SELECT id, summary, embedding_blob FROM long_term_memory")
        rows = c.fetchall()
        conn.close()

        if not rows:
            return

        embeddings = []
        for row in rows:
            mem_id, summary, blob = row
            if blob:
                emb = np.frombuffer(blob, dtype=np.float32)
                embeddings.append(emb)
                self.memory_store[mem_id] = summary

        if embeddings:
            dim = len(embeddings[0])
            self.memory_index = faiss.IndexFlatIP(dim)
            self.memory_index.add(np.array(embeddings))
            logger.info(f"✅ Loaded {len(embeddings)} memories into RAM.")

    def add_memory(self, user_query: str, ai_response: str, summary: str = None):
        if not summary:
            summary = f"User: {user_query[:40]}... | AI: {ai_response[:40]}..."
        
        emb = self.embedder.encode([summary], normalize_embeddings=True)[0].astype('float32')
        
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute(
            "INSERT INTO long_term_memory (timestamp, user_query, ai_response, summary, embedding_blob) VALUES (?, ?, ?, ?, ?)",
            (time.time(), user_query, ai_response, summary, emb.tobytes())
        )
        new_id = c.lastrowid
        conn.commit()
        conn.close()

        if self.memory_index is None:
            self.memory_index = faiss.IndexFlatIP(len(emb))
        self.memory_index.add(emb.reshape(1, -1))
        self.memory_store[new_id] = summary

    def retrieve_relevant_memories(self, current_query: str, top_k: int = 3) -> List[str]:
        if self.memory_index is None or self.memory_index.ntotal == 0:
            return []

        query_emb = self.embedder.encode([current_query], normalize_embeddings=True)[0].astype('float32').reshape(1, -1)
        scores, indices = self.memory_index.search(query_emb, top_k)
        
        return [f"[Past]: {self.memory_store[idx]}" for idx in indices[0] if idx in self.memory_store]

    def get_short_term_context(self, messages: List[Dict], max_turns: int = 5) -> List[Dict]:
        clean_messages = [m for m in messages if m["role"] != "system"]
        if len(clean_messages) <= max_turns * 2:
            return clean_messages
        return clean_messages[-(max_turns * 2):]