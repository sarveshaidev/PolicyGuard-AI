
#!/usr/bin/env python3
"""
PolicyGuard AI - Custom Exception Definitions
==============================================

Structured exception hierarchy for consistent error handling across
the PolicyGuard AI RAG pipeline.

Design goals:
- Keep exception types explicit and easy to catch.
- Preserve backward compatibility with existing callers.
- Provide structured error details for logging/debugging.
- Support safe exception chaining through normal Python semantics.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


class RAGException(Exception):
    """Base exception class for all PolicyGuard AI pipeline errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(message, str):
            message = str(message)

        self.message = message
        self.details = dict(details) if details else {}

        # Keep the normal Exception.args contract intact.
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a readable error message including structured details."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the exception into a serializable dictionary.

        Useful for API responses, structured logging, and monitoring.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""


class GenerationError(RAGException):
    """Raised when LLM answer generation fails."""


class CacheMissError(RAGException):
    """Raised when cache lookup fails to find a matching entry."""


class ValidationError(RAGException):
    """Raised when input or request validation fails."""


class ConfigurationError(RAGException):
    """Raised when application configuration is invalid or missing."""


class SecurityException(RAGException):
    """Raised when security validation or policy enforcement fails."""


class ProcessingError(RAGException):
    """Raised when document or data processing fails."""


__all__ = [
    "RAGException",
    "RetrievalError",
    "GenerationError",
    "CacheMissError",
    "ValidationError",
    "ConfigurationError",
    "SecurityException",
    "ProcessingError",
]

