
#!/usr/bin/env python3
"""
PolicyGuard AI - Production RAG Engine
======================================
Production-oriented Retrieval-Augmented Generation pipeline with:

- Hybrid retrieval (vector + BM25) support
- Optional re-ranking
- Citation-aware answer synthesis
- Semantic/exact caching
- Thread-safe metrics
- Batch query support
- Graceful degradation
- Input validation and bounded resource usage
- Windows/Linux compatibility

Author: PolicyGuard AI Team
Version: 2.0.0
Last Updated: 2026-09-13
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from config.settings import settings
from src.core.exceptions import RAGException, RetrievalError
from src.core.cache import get_cache_instance


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_TOP_K = 3
DEFAULT_RERANK_TOP_K = 3
DEFAULT_MAX_QUERY_LENGTH = 2000
DEFAULT_MAX_BATCH_SIZE = 50
DEFAULT_MAX_WORKERS = 8
DEFAULT_MAX_CONTEXT_CHUNKS = 3
DEFAULT_MAX_CONTENT_LENGTH = 400

ERROR_PREFIX = "Error:"

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


# =============================================================================
# HELPERS
# =============================================================================

def _setting_int(name: str, default: int, minimum: int = 1) -> int:
    """Safely read an integer setting."""
    try:
        value = int(getattr(settings, name, default))
        return max(value, minimum)
    except (TypeError, ValueError):
        return default


def _setting_float(name: str, default: float) -> float:
    """Safely read a float setting."""
    try:
        return float(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default


def _safe_query(query: str) -> str:
    """
    Normalize and validate a query.

    The query is intentionally bounded to prevent oversized requests from
    consuming excessive memory/model/retrieval resources.
    """
    if not isinstance(query, str):
        raise RAGException("Query must be a string")

    query = _CONTROL_CHARS_RE.sub(" ", query)
    query = _WHITESPACE_RE.sub(" ", query).strip()

    max_length = _setting_int(
        "MAX_QUERY_LENGTH",
        DEFAULT_MAX_QUERY_LENGTH,
        minimum=100,
    )

    if not query:
        raise RAGException("Query cannot be empty")

    if len(query) > max_length:
        raise RAGException(
            "Query exceeds the configured maximum length"
        )

    return query


def _safe_top_k(value: Optional[int], default: int) -> int:
    """Validate a top-k value."""
    if value is None:
        return default

    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise RAGException("top_k must be an integer") from exc

    if value <= 0:
        raise RAGException("top_k must be greater than zero")

    # Avoid accidentally requesting an unreasonable number of chunks.
    return min(value, 100)


def _chunk_content(chunk: Any) -> str:
    """Safely extract chunk content."""
    if not isinstance(chunk, dict):
        return ""

    content = chunk.get("content", "")

    if content is None:
        return ""

    if not isinstance(content, str):
        content = str(content)

    return content.strip()


def _chunk_metadata(chunk: Any) -> Dict[str, Any]:
    """Safely extract chunk metadata."""
    if not isinstance(chunk, dict):
        return {}

    metadata = chunk.get("metadata", {})

    return metadata if isinstance(metadata, dict) else {}


def _chunk_source(chunk: Any) -> str:
    """Safely extract source information."""
    metadata = _chunk_metadata(chunk)

    source = metadata.get("source")

    if source is None:
        source = chunk.get("source", "Unknown") if isinstance(chunk, dict) else "Unknown"

    source = str(source).strip()

    return source or "Unknown"


def _chunk_page(chunk: Any) -> Any:
    """Safely extract page information."""
    metadata = _chunk_metadata(chunk)

    page = metadata.get("page")

    if page is None and isinstance(chunk, dict):
        page = chunk.get("page")

    return page


def _chunk_score(chunk: Any) -> float:
    """Safely extract a numeric score."""
    if not isinstance(chunk, dict):
        return 0.0

    try:
        return float(chunk.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _normalize_chunks(chunks: Any) -> List[Dict[str, Any]]:
    """
    Normalize retrieval output.

    Invalid/non-dict chunks are discarded rather than causing the entire
    request to fail.
    """
    if chunks is None:
        return []

    if not isinstance(chunks, (list, tuple)):
        return []

    normalized: List[Dict[str, Any]] = []

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        content = _chunk_content(chunk)

        if not content:
            continue

        normalized_chunk = dict(chunk)

        if not isinstance(normalized_chunk.get("metadata"), dict):
            normalized_chunk["metadata"] = {}

        try:
            normalized_chunk["score"] = float(
                normalized_chunk.get("score", 0.0)
            )
        except (TypeError, ValueError):
            normalized_chunk["score"] = 0.0

        normalized.append(normalized_chunk)

    return normalized


def _embedding_to_list(embedding: Any) -> List[float]:
    """Convert supported embedding objects into a plain Python list."""
    if embedding is None:
        raise RetrievalError("Failed to compute query embedding")

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if not isinstance(embedding, (list, tuple)):
        raise RetrievalError("Query embedding has an invalid format")

    try:
        result = [float(value) for value in embedding]
    except (TypeError, ValueError) as exc:
        raise RetrievalError("Query embedding contains invalid values") from exc

    if not result:
        raise RetrievalError("Query embedding is empty")

    return result


def _safe_error_message(default: str = "Unable to process request") -> str:
    """
    Return a generic user-facing error.

    Internal exception details should remain in logs rather than being exposed
    to users.
    """
    return default


# =============================================================================
# RAG ENGINE
# =============================================================================

def _normalize_scope(
    organization_id: Optional[str],
    namespace: Optional[str],
) -> tuple[str, str]:
    """Normalize and validate the tenant/retrieval scope."""
    org = str(organization_id or "").strip()
    ns = str(namespace or "policy").strip().lower() or "policy"

    if not org:
        # The explicit default tenant is allowed for local/single-tenant
        # operation, but callers must still use an explicit scope internally.
        org = "default"

    if ns not in {"policy", "talent"}:
        raise RAGException(
            f"Unsupported retrieval namespace: {ns}"
        )

    return org, ns


def _filter_chunks_by_scope(
    chunks: Any,
    organization_id: str,
    namespace: str,
) -> List[Dict[str, Any]]:
    """
    Enforce organization and namespace isolation on every chunk.

    This is applied even to pre-retrieved chunks because app/orchestrator
    callers must not be able to bypass the RAG security boundary.
    """
    normalized = _normalize_chunks(chunks)
    authorized: List[Dict[str, Any]] = []

    for chunk in normalized:
        metadata = _chunk_metadata(chunk)

        chunk_org = metadata.get(
            "organization_id",
            chunk.get("organization_id"),
        )
        chunk_namespace = metadata.get(
            "namespace",
            chunk.get("namespace", "policy"),
        )

        if str(chunk_org or "").strip() != organization_id:
            continue

        if str(chunk_namespace or "policy").strip().lower() != namespace:
            continue

        authorized.append(chunk)

    return authorized


class RAGEngine:
    """
    Production RAG pipeline orchestrator.

    The engine is intentionally dependency-injected. The application can
    provide:

        engine.set_vector_store(...)
        engine.set_embedder(...)
        engine.set_hybrid_searcher(...)

    This keeps retrieval infrastructure separate from the orchestration layer.
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        use_hybrid: bool = True,
        enable_cache: bool = True,
        organization_id: Optional[str] = None,
        namespace: str = "policy",
    ):
        default_top_k = _setting_int(
            "TOP_K",
            DEFAULT_TOP_K,
            minimum=1,
        )

        default_rerank_top_k = _setting_int(
            "RERANK_TOP_K",
            DEFAULT_RERANK_TOP_K,
            minimum=1,
        )

        self.top_k = _safe_top_k(top_k, default_top_k)
        self.rerank_top_k = _safe_top_k(
            rerank_top_k,
            default_rerank_top_k,
        )

        self.use_hybrid = bool(
            use_hybrid and getattr(settings, "ENABLE_HYBRID_SEARCH", True)
        )
        self.enable_cache = bool(
            enable_cache and getattr(settings, "ENABLE_SEMANTIC_CACHE", True)
        )
        self.organization_id, self.namespace = _normalize_scope(
            organization_id,
            namespace,
        )

        # Dependency-injected components.
        self.vector_store: Any = None
        self.embedder: Any = None
        self.hybrid_searcher: Any = None
        self.cross_encoder: Any = None

        # Cache.
        self.cache = (
            get_cache_instance(
                organization_id=self.organization_id,
                namespace=self.namespace,
            )
            if self.enable_cache
            else None
        )

        # Metrics.
        self._metrics_lock = threading.Lock()
        self._query_count = 0
        self._successful_queries = 0
        self._failed_queries = 0
        self._total_latency_ms = 0.0
        self._cache_hits = 0

        # Lifecycle.
        self._shutdown = False
        self._shutdown_lock = threading.Lock()

        logger.info(
            "RAGEngine initialized: top_k=%s, rerank_top_k=%s, "
            "hybrid=%s, cache=%s",
            self.top_k,
            self.rerank_top_k,
            self.use_hybrid,
            self.enable_cache,
        )

    # -----------------------------------------------------------------------
    # Dependency injection
    # -----------------------------------------------------------------------

    def set_vector_store(self, vector_store: Any) -> None:
        """Set vector store component."""
        if vector_store is None:
            raise RAGException("Vector store cannot be None")

        self.vector_store = vector_store
        logger.info("Vector store connected to RAG engine")

    def set_embedder(self, embedder: Any) -> None:
        """Set embedder component."""
        if embedder is None:
            raise RAGException("Embedder cannot be None")

        self.embedder = embedder
        logger.info("Embedder connected to RAG engine")

    def set_hybrid_searcher(self, hybrid_searcher: Any) -> None:
        """Set hybrid search component."""
        if hybrid_searcher is None:
            raise RAGException("Hybrid searcher cannot be None")

        self.hybrid_searcher = hybrid_searcher
        logger.info("Hybrid searcher connected to RAG engine")

    def set_cross_encoder(self, cross_encoder: Any) -> None:
        """Set optional cross-encoder/reranker component."""
        if cross_encoder is None:
            raise RAGException("Cross encoder cannot be None")

        self.cross_encoder = cross_encoder
        logger.info("Cross encoder connected to RAG engine")

    # -----------------------------------------------------------------------
    # Cache helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _cache_key(
        query: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        namespace: str = "policy",
    ) -> str:
        """
        Build a deterministic scoped cache key.

        This helper is retained for compatibility/diagnostics. The cache
        implementation itself performs SHA-256 normalization, so callers
        should pass the normalized query rather than a pre-hashed value.
        """
        scope = "|".join(
            [
                str(organization_id or "default"),
                str(namespace or "policy"),
                str(user_id or "global"),
                query,
            ]
        )
        return hashlib.sha256(scope.encode("utf-8")).hexdigest()

    def _get_cached_result(self, query: str) -> Optional[Dict[str, Any]]:
        """Safely retrieve a cache result using the original query text."""
        if not self.cache:
            return None

        try:
            cached = self.cache.get(query)
        except Exception:
            logger.exception("Cache lookup failed")
            return None

        if not isinstance(cached, dict):
            return None

        answer = cached.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return None

        chunks = _normalize_chunks(cached.get("chunks", []))
        return {
            "answer": answer,
            "chunks": chunks,
            "cache_type": cached.get("cache_type", "exact"),
            "similarity": cached.get("similarity"),
        }

    def _cache_result(
        self,
        query: str,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """Safely write a scoped cache result."""
        if not self.cache:
            return
        if not answer or answer.startswith(ERROR_PREFIX):
            return

        try:
            self.cache.set(
                query=query,
                answer=answer,
                chunks=chunks,
                async_mode=True,
            )
        except TypeError:
            try:
                self.cache.set(query=query, answer=answer, chunks=chunks)
            except Exception:
                logger.exception("Cache write failed")
        except Exception:
            logger.exception("Cache write failed")

    # -----------------------------------------------------------------------
    # Query pipeline
    # -----------------------------------------------------------------------

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        user_id: Optional[str] = None,
        use_cache: Optional[bool] = None,
        organization_id: Optional[str] = None,
        namespace: Optional[str] = None,
        conversation_memory: Optional[Sequence[Dict[str, Any]]] = None,
        pre_retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline.

        Pipeline:

            validate
              ↓
            cache lookup
              ↓
            query embedding
              ↓
            retrieval
              ↓
            optional reranking
              ↓
            answer synthesis
              ↓
            cache write
              ↓
            response + metrics
        """
        start_time = time.monotonic()

        with self._metrics_lock:
            self._query_count += 1

        try:
            normalized_query = _safe_query(query)
            k = _safe_top_k(top_k, self.top_k)

            effective_org, effective_namespace = _normalize_scope(
                organization_id
                if organization_id is not None
                else self.organization_id,
                namespace
                if namespace is not None
                else self.namespace,
            )

            # A RAGEngine instance owns a tenant/namespace-scoped cache.
            # Never allow a request to silently switch the engine's scope.
            if (
                effective_org != self.organization_id
                or effective_namespace != self.namespace
            ):
                raise RAGException(
                    "RAG engine scope does not match the requested "
                    "organization/namespace"
                )

            # Cache instances are scoped at construction time.
            cache_allowed = (
                use_cache_flag := (
                    self.enable_cache
                    if use_cache is None
                    else bool(use_cache)
                )
            )
            if (
                self.cache
                and effective_org == self.organization_id
                and effective_namespace == self.namespace
            ):
                cache_allowed = bool(use_cache_flag)
            else:
                cache_allowed = False

            use_cache_flag = cache_allowed

            if pre_retrieved_chunks is not None:
                retrieved_input = _filter_chunks_by_scope(
                    pre_retrieved_chunks,
                    effective_org,
                    effective_namespace,
                )
            else:
                retrieved_input = None

            # ---------------------------------------------------------------
            # STEP 1: Cache
            # ---------------------------------------------------------------

            if use_cache_flag and self.cache:
                cached = self._get_cached_result(normalized_query)

                if cached:
                    latency_ms = self._record_success(
                        start_time,
                        cache_hit=True,
                    )

                    logger.info(
                        "RAG cache hit: latency=%sms",
                        latency_ms,
                    )

                    response = self._build_response(
                        query=normalized_query,
                        answer=cached["answer"],
                        retrieved=cached["chunks"],
                        latency_ms=latency_ms,
                        cache_hit=True,
                        cache_type=cached["cache_type"],
                        retrieval_method="cache",
                    )

                    return response

            # ---------------------------------------------------------------
            # STEP 2: Embed
            # ---------------------------------------------------------------

            if self.embedder is None:
                return self._error_response(
                    normalized_query,
                    "RAG service is not fully configured.",
                    start_time,
                )

            try:
                query_embedding = self.embedder.encode_query(
                    normalized_query
                )
                query_embedding = _embedding_to_list(query_embedding)

            except Exception:
                logger.exception("Query embedding failed")

                return self._error_response(
                    normalized_query,
                    _safe_error_message("Unable to process the query."),
                    start_time,
                )

            # ---------------------------------------------------------------
            # STEP 3: Retrieve
            # ---------------------------------------------------------------

            try:
                if retrieved_input is not None:
                    retrieved = retrieved_input[:k]
                else:
                    retrieved = self._retrieve(
                        normalized_query,
                        query_embedding,
                        k,
                        organization_id=effective_org,
                        namespace=effective_namespace,
                    )

            except Exception:
                logger.exception("RAG retrieval failed")

                return self._error_response(
                    normalized_query,
                    _safe_error_message(
                        "Unable to retrieve relevant policy information."
                    ),
                    start_time,
                )

            # ---------------------------------------------------------------
            # STEP 4: Rerank
            # ---------------------------------------------------------------

            try:
                retrieved = self._maybe_rerank(
                    normalized_query,
                    retrieved,
                    k,
                )

            except Exception:
                logger.exception(
                    "Reranking failed; continuing with retrieved chunks"
                )

            # ---------------------------------------------------------------
            # STEP 5: Synthesize
            # ---------------------------------------------------------------

            try:
                answer = self._synthesize_answer(
                    normalized_query,
                    retrieved,
                )

            except Exception:
                logger.exception("Answer synthesis failed")

                return self._error_response(
                    normalized_query,
                    _safe_error_message(
                        "Unable to generate an answer."
                    ),
                    start_time,
                )

            # ---------------------------------------------------------------
            # STEP 6: Cache
            # ---------------------------------------------------------------

            if use_cache_flag and self.cache:
                self._cache_result(
                    normalized_query,
                    answer,
                    retrieved,
                )

            # ---------------------------------------------------------------
            # STEP 7: Response
            # ---------------------------------------------------------------

            latency_ms = self._record_success(
                start_time,
                cache_hit=False,
            )

            response = self._build_response(
                query=normalized_query,
                answer=answer,
                retrieved=retrieved,
                latency_ms=latency_ms,
                cache_hit=False,
                cache_type=None,
                retrieval_method=(
                    "hybrid"
                    if self.use_hybrid and self.hybrid_searcher
                    else "vector"
                ),
            )

            logger.info(
                "RAG query processed: latency=%sms, chunks=%s",
                latency_ms,
                len(retrieved),
            )

            return response

        except RAGException as exc:
            logger.warning("RAG request rejected: %s", exc)

            return self._error_response(
                query if isinstance(query, str) else "",
                "Invalid query.",
                start_time,
                count_as_failure=True,
            )

        except Exception:
            logger.exception("Unexpected RAG engine failure")

            return self._error_response(
                query if isinstance(query, str) else "",
                "Unable to process the request.",
                start_time,
                count_as_failure=True,
            )

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def _retrieve(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int,
        organization_id: Optional[str] = None,
        namespace: str = "policy",
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using the configured backend."""
        retrieved: Any

        if self.use_hybrid and self.hybrid_searcher is not None:
            alpha = _setting_float(
                "HYBRID_ALPHA",
                0.5,
            )

            # Keep alpha in the expected range.
            alpha = max(0.0, min(1.0, alpha))

            retrieved = self.hybrid_searcher.search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                alpha=alpha,
                organization_id=organization_id,
                namespace=namespace,
            )

        elif self.vector_store is not None:
            retrieved = self.vector_store.search(
                query_embedding,
                top_k=top_k,
                organization_id=organization_id,
                namespace=namespace,
            )

        else:
            raise RetrievalError(
                "No retrieval backend configured"
            )

        normalized = _filter_chunks_by_scope(
            retrieved,
            organization_id or "default",
            namespace,
        )

        return normalized[:top_k]

    # -----------------------------------------------------------------------
    # Reranking
    # -----------------------------------------------------------------------

    def _maybe_rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        requested_top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Optionally rerank retrieved chunks.

        Important behavior:
        - Retrieval is allowed to return up to requested_top_k.
        - Reranking never accidentally discards chunks merely because
          rerank_top_k >= requested_top_k.
        - A reranker failure falls back to original retrieval order.
        """
        if not chunks:
            return []

        if self.rerank_top_k <= 0:
            return chunks[:requested_top_k]

        rerank_count = min(
            self.rerank_top_k,
            len(chunks),
            requested_top_k,
        )

        candidates = chunks[:rerank_count]

        if self.cross_encoder is not None:
            try:
                reranker = self.cross_encoder

                if hasattr(reranker, "rerank"):
                    reranked = reranker.rerank(
                        query,
                        candidates,
                        top_k=rerank_count,
                    )

                    normalized = _filter_chunks_by_scope(
                        reranked,
                        self.organization_id,
                        self.namespace,
                    )

                    if normalized:
                        return normalized[:requested_top_k]

            except Exception:
                logger.exception(
                    "Configured cross-encoder reranking failed"
                )

        # Heuristic fallback.
        query_tokens = {
            token
            for token in re.findall(
                r"\b\w+\b",
                query.lower(),
            )
            if len(token) > 1
        }

        scored: List[Dict[str, Any]] = []

        for chunk in candidates:
            content = _chunk_content(chunk)

            chunk_tokens = {
                token
                for token in re.findall(
                    r"\b\w+\b",
                    content.lower(),
                )
                if len(token) > 1
            }

            if query_tokens and chunk_tokens:
                union = query_tokens | chunk_tokens
                intersection = query_tokens & chunk_tokens

                overlap = (
                    len(intersection) / len(union)
                    if union
                    else 0.0
                )
            else:
                overlap = 0.0

            original_score = _chunk_score(chunk)

            updated = dict(chunk)
            updated["score"] = (
                0.7 * original_score
                + 0.3 * overlap
            )

            scored.append(updated)

        scored.sort(
            key=lambda item: _chunk_score(item),
            reverse=True,
        )

        return scored[:requested_top_k]

    def _rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        requested_top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Backward-compatible alias for older callers/tests."""
        return self._maybe_rerank(
            query,
            chunks,
            requested_top_k or self.top_k,
        )

    # -----------------------------------------------------------------------
    # Synthesis
    # -----------------------------------------------------------------------

    def _synthesize_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Build a conservative answer directly from retrieved policy excerpts.

        This method deliberately does not invent policy information. If
        another LLM synthesis layer is desired, it should be integrated
        separately so retrieval and generation remain independently testable.
        """
        if not chunks:
            return (
                "No relevant policy information was found. "
                "Please rephrase your question or contact HR for assistance."
            )

        answer_parts: List[str] = []

        for index, chunk in enumerate(
            chunks[:DEFAULT_MAX_CONTEXT_CHUNKS],
            start=1,
        ):
            content = _chunk_content(chunk)

            if not content:
                continue

            source = _chunk_source(chunk)
            page = _chunk_page(chunk)

            content = re.sub(
                r"\n{3,}",
                "\n\n",
                content,
            )

            content = re.sub(
                r"(?m)^\s*\*\s+",
                "• ",
                content,
            )

            content = re.sub(
                r"(?m)^\s*-\s+",
                "• ",
                content,
            )

            if len(content) > DEFAULT_MAX_CONTENT_LENGTH:
                content = (
                    content[:DEFAULT_MAX_CONTENT_LENGTH]
                    .rstrip()
                    + "..."
                )

            source_label = source

            if page not in (None, ""):
                source_label += f", Page {page}"

            answer_parts.append(
                f"**Source {index} ({source_label}):**\n"
                f"{content}"
            )

        if not answer_parts:
            return (
                "No usable policy information was found. "
                "Please contact HR for assistance."
            )

        return (
            "Based on your HR documents:\n\n"
            + "\n\n".join(answer_parts)
            + "\n\n"
            "*These excerpts are from your uploaded policy documents. "
            "For complete policy details or clarification, please contact HR.*"
        )

    # -----------------------------------------------------------------------
    # Response/metrics
    # -----------------------------------------------------------------------

    def _build_response(
        self,
        query: str,
        answer: str,
        retrieved: List[Dict[str, Any]],
        latency_ms: int,
        cache_hit: bool,
        cache_type: Optional[str],
        retrieval_method: str,
    ) -> Dict[str, Any]:
        """Build a consistent API response."""
        response: Dict[str, Any] = {
            "answer": answer,
            "retrieved_chunks": retrieved,
            "query": query,
            "chunks_retrieved": len(retrieved),
            "cache_hit": cache_hit,
            "latency_ms": latency_ms,
            "from_cache": cache_hit,
            "retrieval_method": retrieval_method,
            "organization_id": self.organization_id,
            "namespace": self.namespace,
            "security_scope_enforced": True,
        }

        if cache_type:
            response["cache_type"] = cache_type

        if retrieved:
            citations = []

            for chunk in retrieved[:DEFAULT_MAX_CONTEXT_CHUNKS]:
                citations.append(
                    {
                        "source": _chunk_source(chunk),
                        "page": _chunk_page(chunk),
                        "score": _chunk_score(chunk),
                    }
                )

            response["citations"] = citations

        return response

    def _record_success(
        self,
        start_time: float,
        cache_hit: bool,
    ) -> int:
        """Record a successful query and return latency."""
        latency_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        with self._metrics_lock:
            self._successful_queries += 1
            self._total_latency_ms += latency_ms

            if cache_hit:
                self._cache_hits += 1

        return latency_ms

    def _error_response(
        self,
        query: str,
        error_msg: str,
        start_time: float,
        count_as_failure: bool = True,
    ) -> Dict[str, Any]:
        """Build a standardized, user-safe error response."""
        latency_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        with self._metrics_lock:
            self._total_latency_ms += latency_ms

            if count_as_failure:
                self._failed_queries += 1

        return {
            "answer": f"{ERROR_PREFIX} {error_msg}",
            "retrieved_chunks": [],
            "query": query,
            "chunks_retrieved": 0,
            "cache_hit": False,
            "latency_ms": latency_ms,
            "from_cache": False,
            "error": error_msg,
        }

    # -----------------------------------------------------------------------
    # Batch
    # -----------------------------------------------------------------------

    def batch_query(
        self,
        queries: Sequence[str],
        user_id: Optional[str] = None,
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple queries concurrently.

        Results are returned in the same order as the input queries.
        This is important for callers that associate result[i] with
        queries[i].
        """
        if not isinstance(queries, (list, tuple)):
            raise RAGException(
                "queries must be a list or tuple"
            )

        if len(queries) > DEFAULT_MAX_BATCH_SIZE:
            raise RAGException(
                f"Batch size cannot exceed {DEFAULT_MAX_BATCH_SIZE}"
            )

        if not queries:
            return []

        try:
            workers = int(max_workers)
        except (TypeError, ValueError) as exc:
            raise RAGException(
                "max_workers must be an integer"
            ) from exc

        workers = max(1, min(workers, DEFAULT_MAX_WORKERS, len(queries)))

        results: List[Optional[Dict[str, Any]]] = [
            None
        ] * len(queries)

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="rag",
        ) as executor:
            future_to_index = {
                executor.submit(
                    self.query,
                    query,
                    user_id=user_id,
                ): index
                for index, query in enumerate(queries)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                query = queries[index]

                try:
                    results[index] = future.result()

                except Exception:
                    logger.exception(
                        "Batch query failed at index %s",
                        index,
                    )

                    results[index] = self._error_response(
                        query if isinstance(query, str) else "",
                        "Unable to process the request.",
                        time.monotonic(),
                    )

        final_results: List[Dict[str, Any]] = []

        for index, result in enumerate(results):
            if result is None:
                final_results.append(
                    self._error_response(
                        queries[index]
                        if isinstance(queries[index], str)
                        else "",
                        "Unable to process the request.",
                        time.monotonic(),
                    )
                )
            else:
                final_results.append(result)

        logger.info(
            "Batch query complete: %s queries processed",
            len(final_results),
        )

        return final_results

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        with self._metrics_lock:
            query_count = self._query_count
            successful_queries = self._successful_queries
            failed_queries = self._failed_queries
            cache_hits = self._cache_hits
            total_latency = self._total_latency_ms

            avg_latency = (
                total_latency / successful_queries
                if successful_queries > 0
                else 0.0
            )

            cache_hit_rate = (
                cache_hits / query_count * 100
                if query_count > 0
                else 0.0
            )

            return {
                "query_count": query_count,
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "cache_hits": cache_hits,
                "cache_hit_rate_percent": round(
                    cache_hit_rate,
                    1,
                ),
                "avg_latency_ms": round(
                    avg_latency,
                    1,
                ),
                "total_latency_ms": round(
                    total_latency,
                    1,
                ),
                "config": {
                    "top_k": self.top_k,
                    "rerank_top_k": self.rerank_top_k,
                    "use_hybrid": self.use_hybrid,
                    "enable_cache": self.enable_cache,
                    "organization_id": self.organization_id,
                    "namespace": self.namespace,
                },
                "components": {
                    "vector_store": self.vector_store is not None,
                    "embedder": self.embedder is not None,
                    "hybrid_searcher": (
                        self.hybrid_searcher is not None
                    ),
                    "cross_encoder": (
                        self.cross_encoder is not None
                    ),
                    "cache": self.cache is not None,
                },
                "shutdown": self._shutdown,
            }

    # -----------------------------------------------------------------------
    # Cache/lifecycle
    # -----------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear RAG cache if configured."""
        if not self.cache:
            return

        try:
            self.cache.clear()
            logger.info("RAG engine cache cleared")
        except Exception:
            logger.exception("Failed to clear RAG engine cache")

    def shutdown(self) -> None:
        """
        Release engine resources.

        Shutdown is idempotent so callers can safely invoke it more than once.
        """
        with self._shutdown_lock:
            if self._shutdown:
                return

            self._shutdown = True

        if self.cache:
            try:
                self.cache.shutdown()
            except Exception:
                logger.exception("Cache shutdown failed")

        logger.info("RAGEngine shutdown complete")


# =============================================================================
# GLOBAL INSTANCE MANAGEMENT
# =============================================================================

_rag_engine: Optional[RAGEngine] = None
_engine_lock = threading.Lock()


def get_rag_engine(
    top_k: Optional[int] = None,
    rerank_top_k: Optional[int] = None,
    use_hybrid: bool = True,
    enable_cache: bool = True,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> RAGEngine:
    """
    Get or create the global RAG engine.

    The singleton is created lazily and protected by a lock.
    """
    global _rag_engine

    requested_org, requested_namespace = _normalize_scope(
        organization_id,
        namespace,
    )

    with _engine_lock:
        if _rag_engine is None:
            _rag_engine = RAGEngine(
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                use_hybrid=use_hybrid,
                enable_cache=enable_cache,
                organization_id=requested_org,
                namespace=requested_namespace,
            )
        elif (
            _rag_engine.organization_id != requested_org
            or _rag_engine.namespace != requested_namespace
        ):
            # Never reuse a singleton carrying another tenant's cache/scope.
            # A caller should use a request-scoped engine or reset the singleton
            # between tenant contexts rather than crossing scopes.
            raise RAGException(
                "Global RAG engine is scoped to a different "
                "organization/namespace"
            )

        return _rag_engine


def reset_rag_engine() -> None:
    """Reset the global RAG engine, primarily for tests."""
    global _rag_engine

    with _engine_lock:
        engine = _rag_engine
        _rag_engine = None

    if engine is not None:
        engine.shutdown()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def rag_query(
    query: str,
    top_k: Optional[int] = None,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> Dict[str, Any]:
    """Execute a single RAG query."""
    engine = get_rag_engine(
        organization_id=organization_id,
        namespace=namespace,
    )

    return engine.query(
        query=query,
        top_k=top_k,
        user_id=user_id,
        organization_id=organization_id,
        namespace=namespace,
    )


def rag_batch_query(
    queries: List[str],
    user_id: Optional[str] = None,
    max_workers: int = 4,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> List[Dict[str, Any]]:
    """Execute multiple RAG queries within one explicit tenant scope."""
    engine = get_rag_engine(
        organization_id=organization_id,
        namespace=namespace,
    )

    return engine.batch_query(
        queries=queries,
        user_id=user_id,
        max_workers=max_workers,
    )


def get_rag_stats() -> Dict[str, Any]:
    """Get RAG engine statistics."""
    return get_rag_engine().get_stats()


def clear_rag_cache() -> None:
    """Clear the global RAG cache."""
    get_rag_engine().clear_cache()


# =============================================================================
# TEST / DEMO
# =============================================================================

def test_rag_engine() -> None:
    """Basic local smoke test."""
    print("\n🔍 Testing RAG Engine\n")
    print("=" * 70)

    engine = RAGEngine(
        enable_cache=False,
    )

    print("✅ RAG Engine initialized")
    print(
        f"⚙️  Config: top_k={engine.top_k}, "
        f"rerank={engine.rerank_top_k}, "
        f"hybrid={engine.use_hybrid}"
    )

    mock_chunks = [
        {
            "content": (
                "Employees are entitled to 20 days of paid leave "
                "per year. Leave accrues monthly at the rate of "
                "1.67 days per month."
            ),
            "metadata": {
                "source": "PolicyManual.pdf",
                "page": 3,
            },
            "score": 0.92,
        },
        {
            "content": (
                "All leave requests must be submitted at least "
                "2 weeks in advance through the HR portal."
            ),
            "metadata": {
                "source": "PolicyManual.pdf",
                "page": 3,
            },
            "score": 0.88,
        },
    ]

    print("\n📝 Test 1: Answer synthesis")
    print("-" * 70)

    answer = engine._synthesize_answer(
        "What is the leave policy?",
        mock_chunks,
    )

    print(f"   Answer preview: {answer[:200]}...")

    print("\n🔀 Test 2: Heuristic reranking")
    print("-" * 70)

    reranked = engine._maybe_rerank(
        "leave request",
        mock_chunks,
    )

    print(f"   Chunks returned: {len(reranked)}")

    print("\n📊 Test 3: Statistics")
    print("-" * 70)

    stats = engine.get_stats()

    print(f"   Queries: {stats['query_count']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(
        f"   Components: "
        f"vector_store={stats['components']['vector_store']}, "
        f"embedder={stats['components']['embedder']}, "
        f"cache={stats['components']['cache']}"
    )

    print("\n" + "=" * 70)
    print("✅ RAG engine smoke test complete!\n")


if __name__ == "__main__":
    test_rag_engine()

