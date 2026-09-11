class RAGException(Exception):
    """Base exception for RAG pipeline."""
    pass

class RetrievalError(RAGException):
    """Raised when retrieval fails."""
    pass

class GenerationError(RAGException):
    """Raised when LLM generation fails."""
    pass

class CacheMissError(RAGException):
    """Raised when cache miss occurs (optional usage)."""
    pass