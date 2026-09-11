from src.core.cache import SemanticCache
from src.core.observability import trace_operation
from src.core.exceptions import RetrievalError
from openai import OpenAI
from config.settings import GENERATION_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from src.core.memory_manager import MemoryManager
from typing import List, Dict
import logging
import os

# Silence logs
os.environ["HTTPX_LOG_LEVEL"] = "ERROR"

# FIXED: Use __name__
logger = logging.getLogger(__name__)

class RAGEngine:
    # FIXED: Use __init__
    def __init__(self):
        self.cache = SemanticCache()
        self.vector_store = None
        self.llm_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
        self.memory = MemoryManager() 
        logger.info("🚀 RAG Engine Initialized")

    @trace_operation("RAG_Query")
    def query(self, question: str, chat_history: List[Dict] = None) -> str:
        if chat_history is None:
            chat_history = []

        # 1. Cache Check
        cached_answer = self.cache.get(question)
        if cached_answer:
            return f"[CACHED] {cached_answer}"

        # 2. Memory Retrieval
        relevant_memories = self.memory.retrieve_relevant_memories(question, top_k=3)
        memory_context = "\n".join(relevant_memories) if relevant_memories else ""

        # 3. Document Retrieval
        doc_context = ""
        if self.vector_store:
            try:
                context_chunks, _ = self.vector_store.search(question, top_k=5)
                if context_chunks:
                    doc_context = "\n\n".join([f"[Doc]: {chunk['content']}" for chunk in context_chunks])
            except Exception as e:
                logger.warning(f"Retrieval error: {e}")

        # 4. Build Context
        full_context_parts = []
        if memory_context:
            full_context_parts.append(f"**Past Interactions:**\n{memory_context}")
        if doc_context:
            full_context_parts.append(f"**Documents:**\n{doc_context}")
        
        full_context = "\n\n".join(full_context_parts) if full_context_parts else "No context available."

        # 5. Prepare Messages
        recent_history = self.memory.get_short_term_context(chat_history)
        final_messages = [{"role": "system", "content": f"You are an expert assistant.\nContext:\n{full_context}"}]
        final_messages.extend(recent_history)
        final_messages.append({"role": "user", "content": question})

        # 6. Generate
        try:
            response = self.llm_client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=final_messages,
                temperature=0.1,
                max_tokens=1024
            )
            answer = response.choices[0].message.content.strip()
            
            # 7. Save Memory
            self.memory.add_memory(question, answer)
            self.cache.put(question, answer)
            
            return answer
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {str(e)}"

    def set_vector_store(self, vector_store):
        self.vector_store = vector_store