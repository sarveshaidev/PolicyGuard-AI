
#!/usr/bin/env python3
"""
PolicyGuard AI - Cross-Encoder Re-Ranker
========================================
Production-ready cross-encoder re-ranking with:

- Lazy FastEmbed/ONNX model loading
- Low-memory CPU ONNX inference
- Thread-safe model initialization/inference
- Batch scoring
- SentenceTransformers/PyTorch-free runtime
- Heuristic fallback when the model is unavailable
- Score normalization
- Threshold filtering
- Batch query support
- Safe model unloading
- Statistics and lifecycle management

Author: PolicyGuard AI Team
Version: 2.0.0
Last Updated: 2026-09-13
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# PROJECT PATH
# =============================================================================

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from config.settings import settings
from src.core.exceptions import RAGException


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512
DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.5
MAX_BATCH_SIZE = 100
MAX_TOP_K = 100
MAX_QUERY_LENGTH = 8_000
VALID_NAMESPACES = {"policy", "talent"}
_SCOPE_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")


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


def _setting_float(
    name: str,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Safely read a float setting."""
    try:
        value = float(getattr(settings, name, default))
    except (TypeError, ValueError):
        value = default

    if minimum is not None:
        value = max(value, minimum)

    if maximum is not None:
        value = min(value, maximum)

    return value


def _normalize_query(query: Any) -> str:
    """Validate and normalize a query."""
    if not isinstance(query, str):
        raise RAGException("Query must be a string")

    query = re.sub(r"[\x00-\x1f\x7f]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    if not query:
        raise RAGException("Query cannot be empty")

    if len(query) > MAX_QUERY_LENGTH:
        raise RAGException(
            f"Query exceeds the maximum allowed length of {MAX_QUERY_LENGTH} characters"
        )

    return query


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


def _chunk_score(chunk: Any) -> float:
    """Safely extract the original retrieval score."""
    if not isinstance(chunk, dict):
        return 0.0

    # Support both common names.
    raw_score = chunk.get(
        "score",
        chunk.get("initial_score", 0.0),
    )

    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 0.0


def _normalize_chunks(
    chunks: Any,
) -> List[Dict[str, Any]]:
    """Normalize and filter retrieved chunks."""
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

        if not isinstance(
            normalized_chunk.get("metadata"),
            dict,
        ):
            normalized_chunk["metadata"] = {}

        normalized_chunk["score"] = _chunk_score(
            normalized_chunk
        )

        normalized.append(normalized_chunk)

    return normalized


def _safe_top_k(
    top_k: Optional[int],
    default: int,
    maximum: int = MAX_TOP_K,
) -> int:
    """Validate top-k."""
    if top_k is None:
        top_k = default

    try:
        top_k = int(top_k)
    except (TypeError, ValueError) as exc:
        raise RAGException(
            "top_k must be an integer"
        ) from exc

    if top_k <= 0:
        raise RAGException(
            "top_k must be greater than zero"
        )

    return min(top_k, maximum)


def _safe_batch_size(batch_size: int) -> int:
    """Validate model batch size."""
    try:
        value = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise RAGException(
            "batch_size must be an integer"
        ) from exc

    if value <= 0:
        raise RAGException(
            "batch_size must be greater than zero"
        )

    return min(value, 256)




def _normalize_scope(organization_id: Optional[str], namespace: str) -> Tuple[str, str]:
    """Validate tenant scope for reranking operations."""
    organization = "default" if organization_id is None else str(organization_id).strip()
    resolved_namespace = "policy" if namespace is None else str(namespace).strip().lower()
    if not organization or not _SCOPE_RE.fullmatch(organization):
        raise RAGException("Invalid organization_id")
    if resolved_namespace not in VALID_NAMESPACES:
        raise RAGException(f"Unsupported retrieval namespace: {resolved_namespace}")
    return organization, resolved_namespace


def _filter_scoped_chunks(chunks: List[Dict[str, Any]], organization_id: str, namespace: str) -> List[Dict[str, Any]]:
    """Fail closed when chunk metadata is absent or belongs to another scope."""
    scoped=[]
    for chunk in chunks:
        metadata=chunk.get("metadata") if isinstance(chunk, dict) else None
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("organization_id", "")).strip() != organization_id:
            continue
        if str(metadata.get("namespace", "")).strip().lower() != namespace:
            continue
        scoped.append(chunk)
    return scoped

# =============================================================================
# CROSS-ENCODER RERANKER
# =============================================================================

class CrossEncoderReranker:
    """
    Cross-encoder based document re-ranker.

    The public rerank() API returns chunk dictionaries with the final
    reranker score stored in ``chunk["score"]``.

    This is intentionally compatible with the RAGEngine retrieval pipeline.
    """

    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: Optional[str] = None,
    ):
        self.model_name = (
            str(model_name).strip()
            if model_name
            else self.DEFAULT_MODEL
        )

        if not self.model_name:
            self.model_name = self.DEFAULT_MODEL

        self.batch_size = _safe_batch_size(batch_size)

        try:
            self.max_length = int(max_length)
        except (TypeError, ValueError) as exc:
            raise RAGException(
                "max_length must be an integer"
            ) from exc

        if self.max_length <= 0:
            raise RAGException(
                "max_length must be greater than zero"
            )

        # IMPORTANT:
        # Do NOT define this as a property-backed public attribute.
        # The original implementation had both:
        #
        #     self.device = device
        #
        # and
        #
        #     @property
        #     def device(...)
        #
        # which causes a read-only property conflict.
        self._requested_device = (
            str(device).strip().lower()
            if device
            else None
        )

        self._model: Any = None
        self._is_loaded = False
        self._load_lock = threading.RLock()
        self._inference_lock = threading.RLock()

        self._loaded_device: Optional[str] = None

        # Statistics.
        self._total_reranks = 0
        self._total_items = 0
        self._total_time_ms = 0.0
        self._fallback_count = 0
        self._failed_loads = 0

        self._shutdown = False

        logger.info(
            "CrossEncoderReranker initialized: "
            "model=%s, batch_size=%s, max_length=%s, device=%s",
            self.model_name,
            self.batch_size,
            self.max_length,
            self._requested_device or "auto",
        )

    # -------------------------------------------------------------------------
    # Device management
    # -------------------------------------------------------------------------

    def _detect_device(self) -> str:
        """
        Resolve the execution device without importing PyTorch.

        FastEmbed's CPU ONNX Runtime path is the supported low-memory
        production target. Explicit CUDA/MPS requests therefore fall back
        to CPU rather than importing a heavyweight framework.
        """
        if self._requested_device:
            requested = self._requested_device

            if requested == "cpu":
                return "cpu"

            if requested in {"cuda", "gpu", "mps"}:
                logger.info(
                    "Requested device '%s' is not used by the low-memory "
                    "FastEmbed reranker; selecting CPU",
                    requested,
                )
                return "cpu"

            logger.warning(
                "Unknown device '%s'; using CPU",
                requested,
            )

        return "cpu"

    @property
    def device(self) -> str:
        """Return the active or preferred device."""
        if self._loaded_device:
            return self._loaded_device

        return self._detect_device()

    # -------------------------------------------------------------------------
    # Model lifecycle
    # -------------------------------------------------------------------------

    def _fastembed_model_name(self) -> str:
        """
        Map the legacy SentenceTransformers model identifier to the
        FastEmbed/ONNX equivalent.

        PolicyGuardAI keeps the historical public configuration value
        ``cross-encoder/ms-marco-MiniLM-L-6-v2`` for compatibility, while
        FastEmbed uses the ONNX-converted ``Xenova/...`` identifier.
        """
        model_name = self.model_name.strip()

        if model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2":
            return "Xenova/ms-marco-MiniLM-L-6-v2"

        if model_name == "cross-encoder/ms-marco-MiniLM-L6-v2":
            return "Xenova/ms-marco-MiniLM-L-6-v2"

        return model_name

    def _load_model(self) -> bool:
        """
        Lazily load the FastEmbed ONNX cross-encoder.

        Returns:
            True when the model is ready, otherwise False.
        """
        with self._load_lock:
            if self._shutdown:
                logger.warning(
                    "Cannot load cross-encoder after shutdown"
                )
                return False

            if self._is_loaded and self._model is not None:
                return True

            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

            except ImportError:
                logger.warning(
                    "fastembed is not installed; "
                    "using heuristic reranking"
                )
                self._failed_loads += 1
                return False

            # FastEmbed's ONNX CPU path is intentionally the production
            # default for PolicyGuardAI. It avoids importing PyTorch and
            # SentenceTransformers, which is critical for the 512 MB Render
            # deployment target.
            device = self._detect_device()

            if device != "cpu":
                logger.warning(
                    "FastEmbed reranker currently uses the CPU ONNX path "
                    "for this low-memory deployment; requested device=%s",
                    device,
                )
                device = "cpu"

            fastembed_model_name = self._fastembed_model_name()

            try:
                logger.info(
                    "Loading FastEmbed cross-encoder '%s' "
                    "(configured=%s) on %s",
                    fastembed_model_name,
                    self.model_name,
                    device,
                )

                model = TextCrossEncoder(
                    model_name=fastembed_model_name,
                    lazy_load=False,
                )

                # Warmup is useful but should never prevent the model from
                # being used if a particular ONNX backend has an issue.
                try:
                    list(
                        model.rerank(
                            "test query",
                            ["test document"],
                            batch_size=1,
                        )
                    )
                except Exception:
                    logger.warning(
                        "FastEmbed cross-encoder warmup failed; "
                        "continuing with loaded model",
                        exc_info=True,
                    )

                self._model = model
                self._loaded_device = device
                self._is_loaded = True

                logger.info(
                    "FastEmbed CrossEncoder loaded successfully on %s "
                    "(model=%s)",
                    device,
                    fastembed_model_name,
                )

                return True

            except Exception:
                self._failed_loads += 1

                logger.exception(
                    "Failed to load FastEmbed cross-encoder model '%s'",
                    fastembed_model_name,
                )

                self._model = None
                self._loaded_device = None
                self._is_loaded = False

                return False

    # -------------------------------------------------------------------------
    # Main reranking
    # -------------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        *,
        organization_id: Optional[str] = None,
        namespace: str = "policy",
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using the cross-encoder.

        Returns:
            List of chunk dictionaries sorted by descending relevance.
            Each returned chunk has ``score`` set to the reranker score.
        """
        start_time = time.monotonic()

        query = _normalize_query(query)
        organization_id, namespace = _normalize_scope(organization_id, namespace)

        normalized_chunks = _filter_scoped_chunks(
            _normalize_chunks(chunks),
            organization_id,
            namespace,
        )

        if not normalized_chunks:
            return []

        default_top_k = _setting_int(
            "TOP_K",
            DEFAULT_TOP_K,
            minimum=1,
        )

        top_k = _safe_top_k(
            top_k,
            default_top_k,
        )

        top_k = min(
            top_k,
            len(normalized_chunks),
        )

        # Load model lazily.
        if not self.is_loaded:
            if not self._load_model():
                logger.warning(
                    "Cross-encoder unavailable; "
                    "using heuristic fallback"
                )

                with self._load_lock:
                    self._fallback_count += 1

                return self._heuristic_rerank(
                    query,
                    normalized_chunks,
                    top_k,
                )

        if self._model is None:
            with self._load_lock:
                self._fallback_count += 1

            return self._heuristic_rerank(
                query,
                normalized_chunks,
                top_k,
            )

        try:
            pairs = [
                [
                    query,
                    _chunk_content(chunk),
                ]
                for chunk in normalized_chunks
            ]

            # FastEmbed accepts the query and document list directly and
            # performs ONNX inference without the SentenceTransformers /
            # PyTorch runtime.
            documents = [
                _chunk_content(chunk)
                for chunk in normalized_chunks
            ]

            with self._inference_lock:
                # FastEmbed returns a generator/iterable of scores. Materialize
                # it before converting to a NumPy array; passing the generator
                # directly to np.asarray produces a 0-D object array.
                scores = list(
                    self._model.rerank(
                        query,
                        documents,
                        batch_size=self.batch_size,
                    )
                )

            scores_array = np.asarray(
                scores,
                dtype=np.float32,
            ).reshape(-1)

            if len(scores_array) != len(normalized_chunks):
                raise RAGException(
                    "Cross-encoder returned an unexpected number of scores"
                )

            scored_chunks: List[Dict[str, Any]] = []

            for chunk, score in zip(
                normalized_chunks,
                scores_array.tolist(),
            ):
                try:
                    score_value = float(score)
                except (TypeError, ValueError):
                    score_value = 0.0

                if not np.isfinite(score_value):
                    score_value = 0.0

                updated = dict(chunk)
                updated["score"] = score_value
                updated["reranker_score"] = score_value

                scored_chunks.append(updated)

            scored_chunks.sort(
                key=lambda item: _chunk_score(item),
                reverse=True,
            )

            results = scored_chunks[:top_k]

            elapsed_ms = (
                time.monotonic() - start_time
            ) * 1000

            with self._load_lock:
                self._total_reranks += 1
                self._total_items += len(normalized_chunks)
                self._total_time_ms += elapsed_ms

            best_score = (
                _chunk_score(results[0])
                if results
                else 0.0
            )

            average_score = (
                float(np.mean(scores_array))
                if scores_array.size
                else 0.0
            )

            logger.debug(
                "Cross-encoder reranked %s chunks -> %s; "
                "best=%.4f avg=%.4f latency=%.1fms",
                len(normalized_chunks),
                len(results),
                best_score,
                average_score,
                elapsed_ms,
            )

            return results

        except Exception:
            logger.exception(
                "Cross-encoder inference failed; "
                "using heuristic fallback"
            )

            with self._load_lock:
                self._fallback_count += 1

            return self._heuristic_rerank(
                query,
                normalized_chunks,
                top_k,
            )

    # -------------------------------------------------------------------------
    # Heuristic fallback
    # -------------------------------------------------------------------------

    def _heuristic_rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Lightweight fallback when the ML model cannot be used.

        Combines:
        - token overlap
        - exact phrase presence
        - original retrieval score
        """
        query_lower = query.lower()

        query_tokens = set(
            re.findall(
                r"\b\w{2,}\b",
                query_lower,
            )
        )

        scored: List[Dict[str, Any]] = []

        for position, chunk in enumerate(chunks):
            content = _chunk_content(chunk)
            content_lower = content.lower()

            chunk_tokens = set(
                re.findall(
                    r"\b\w{2,}\b",
                    content_lower,
                )
            )

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

            # Exact query phrase gets a small additional boost.
            phrase_boost = (
                0.15
                if query_lower in content_lower
                else 0.0
            )

            original_score = _chunk_score(chunk)

            # Normalize common retrieval score ranges.
            if original_score < 0:
                original_score = 0.0

            if original_score > 1:
                original_score = 1.0

            combined_score = (
                0.55 * original_score
                + 0.35 * overlap
                + 0.10 * min(phrase_boost / 0.15, 1.0)
            )

            updated = dict(chunk)
            updated["score"] = float(combined_score)
            updated["reranker_score"] = float(combined_score)
            updated["rerank_fallback"] = True

            # Preserve original order as the final tie-breaker.
            scored.append(
                (
                    updated,
                    float(combined_score),
                    position,
                )
            )

        scored.sort(
            key=lambda item: (
                item[1],
                -item[2],
            ),
            reverse=True,
        )

        return [
            item[0]
            for item in scored[:top_k]
        ]

    # -------------------------------------------------------------------------
    # Threshold filtering
    # -------------------------------------------------------------------------

    def rerank_with_threshold(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        threshold: float = DEFAULT_THRESHOLD,
        max_results: Optional[int] = None,
        *,
        organization_id: Optional[str] = None,
        namespace: str = "policy",
    ) -> List[Dict[str, Any]]:
        """
        Re-rank and retain only chunks meeting the score threshold.

        Note:
            Cross-encoder scores are model-dependent. A universal threshold
            such as 0.5 should be treated as configurable rather than a
            mathematically universal relevance boundary.
        """
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise RAGException(
                "threshold must be numeric"
            ) from exc

        if max_results is not None:
            max_results = _safe_top_k(
                max_results,
                DEFAULT_TOP_K,
            )

        organization_id, namespace = _normalize_scope(organization_id, namespace)

        normalized_chunks = _filter_scoped_chunks(
            _normalize_chunks(chunks),
            organization_id,
            namespace,
        )

        if not normalized_chunks:
            return []

        results = self.rerank(
            query,
            normalized_chunks,
            top_k=len(normalized_chunks),
            organization_id=organization_id,
            namespace=namespace,
        )

        filtered = [
            chunk
            for chunk in results
            if _chunk_score(chunk) >= threshold
        ]

        if max_results is not None:
            filtered = filtered[:max_results]

        logger.debug(
            "Threshold filtering: %s -> %s "
            "(threshold=%.4f, max=%s)",
            len(results),
            len(filtered),
            threshold,
            max_results,
        )

        return filtered

    # -------------------------------------------------------------------------
    # Batch reranking
    # -------------------------------------------------------------------------

    def batch_rerank(
        self,
        queries: Sequence[str],
        chunks_list: Sequence[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Re-rank multiple query/chunk sets.

        Results preserve input order.
        """
        if not isinstance(
            queries,
            (list, tuple),
        ):
            raise RAGException(
                "queries must be a list or tuple"
            )

        if not isinstance(
            chunks_list,
            (list, tuple),
        ):
            raise RAGException(
                "chunks_list must be a list or tuple"
            )

        if len(queries) != len(chunks_list):
            raise RAGException(
                "queries and chunks_list must have equal lengths"
            )

        if len(queries) > MAX_BATCH_SIZE:
            raise RAGException(
                f"Batch size cannot exceed {MAX_BATCH_SIZE}"
            )

        results: List[List[Dict[str, Any]]] = []

        for query, chunks in zip(
            queries,
            chunks_list,
        ):
            try:
                results.append(
                    self.rerank(
                        query,
                        chunks,
                        top_k=top_k,
                    )
                )
            except Exception:
                logger.exception(
                    "Batch rerank failed for one query"
                )
                results.append([])

        return results

    # -------------------------------------------------------------------------
    # Properties/statistics
    # -------------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the cross-encoder is currently loaded."""
        return (
            self._is_loaded
            and self._model is not None
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return reranker statistics."""
        with self._load_lock:
            average_time = (
                self._total_time_ms / self._total_reranks
                if self._total_reranks > 0
                else 0.0
            )

            average_items = (
                self._total_items / self._total_reranks
                if self._total_reranks > 0
                else 0.0
            )

            return {
                "model_name": self.model_name,
                "is_loaded": self.is_loaded,
                "device": self.device,
                "requested_device": self._requested_device,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "total_reranks": self._total_reranks,
                "total_items": self._total_items,
                "avg_items_per_rerank": round(
                    average_items,
                    2,
                ),
                "avg_time_ms": round(
                    average_time,
                    2,
                ),
                "fallback_count": self._fallback_count,
                "failed_loads": self._failed_loads,
                "model_type": (
                    type(self._model).__name__
                    if self._model is not None
                    else None
                ),
                "runtime": "fastembed_onnx",
                "fastembed_model": self._fastembed_model_name(),
                "shutdown": self._shutdown,
            }

    # -------------------------------------------------------------------------
    # Unload/shutdown
    # -------------------------------------------------------------------------

    def unload(self) -> None:
        """Unload the model and release model resources."""
        with self._load_lock:
            model = self._model

            self._model = None
            self._is_loaded = False
            self._loaded_device = None

            if model is None:
                return

            try:
                if hasattr(model, "to"):
                    model.to("cpu")
            except Exception:
                logger.debug(
                    "Could not move cross-encoder to CPU before unload",
                    exc_info=True,
                )

            try:
                del model
            except Exception:
                logger.debug(
                    "Cross-encoder object cleanup warning",
                    exc_info=True,
                )

            # FastEmbed/ONNX Runtime owns its inference resources.
            # No PyTorch/CUDA cleanup is required here.
            logger.info(
                "CrossEncoder model unloaded"
            )

    def shutdown(self) -> None:
        """Shutdown reranker resources safely."""
        with self._load_lock:
            if self._shutdown:
                return

            self._shutdown = True

        self.unload()

        logger.info(
            "CrossEncoderReranker shutdown complete"
        )


# =============================================================================
# HYBRID SEARCH + RERANK
# =============================================================================

def hybrid_search_with_rerank(
    query: str,
    vector_store: Any,
    reranker: CrossEncoderReranker,
    initial_top_k: int = 20,
    final_top_k: Optional[int] = None,
    alpha: Optional[float] = None,
    *,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> List[Dict[str, Any]]:
    """
    Perform hybrid retrieval followed by cross-encoder reranking.

    Returns:
        List of chunk dictionaries with reranker scores.
    """
    if vector_store is None:
        raise RAGException(
            "vector_store cannot be None"
        )

    if reranker is None:
        raise RAGException(
            "reranker cannot be None"
        )

    query = _normalize_query(query)
    organization_id, namespace = _normalize_scope(organization_id, namespace)

    initial_top_k = _safe_top_k(
        initial_top_k,
        default=20,
    )

    if final_top_k is None:
        final_top_k = _setting_int(
            "TOP_K",
            DEFAULT_TOP_K,
            minimum=1,
        )

    final_top_k = _safe_top_k(
        final_top_k,
        DEFAULT_TOP_K,
    )

    final_top_k = min(
        final_top_k,
        initial_top_k,
    )

    if alpha is None:
        alpha = _setting_float(
            "HYBRID_ALPHA",
            0.5,
        )
    else:
        try:
            alpha = float(alpha)
        except (TypeError, ValueError) as exc:
            raise RAGException(
                "alpha must be numeric"
            ) from exc

    alpha = max(
        0.0,
        min(1.0, alpha),
    )

    try:
        from src.retrieval.hybrid_search import hybrid_search

        initial_result = hybrid_search(
            query=query,
            vector_store=vector_store,
            top_k=initial_top_k,
            alpha=alpha,
            organization_id=organization_id,
            namespace=namespace,
        )

        # Support either:
        #   (chunks, scores)
        # or simply:
        #   chunks
        if isinstance(initial_result, tuple):
            initial_chunks = initial_result[0]
        else:
            initial_chunks = initial_result

        initial_chunks = _filter_scoped_chunks(
            _normalize_chunks(initial_chunks),
            organization_id,
            namespace,
        )

        if not initial_chunks:
            logger.info(
                "Hybrid retrieval returned no candidates"
            )
            return []

        reranked = reranker.rerank(
            query,
            initial_chunks,
            top_k=final_top_k,
            organization_id=organization_id,
            namespace=namespace,
        )

        logger.info(
            "Hybrid + rerank: %s -> %s chunks; alpha=%.2f",
            len(initial_chunks),
            len(reranked),
            alpha,
        )

        return reranked

    except ImportError:
        logger.exception(
            "Hybrid search module could not be imported"
        )
        return []

    except Exception:
        logger.exception(
            "Hybrid search with reranking failed"
        )
        return []


# =============================================================================
# GLOBAL INSTANCE MANAGEMENT
# =============================================================================

_reranker: Optional[CrossEncoderReranker] = None
_reranker_lock = threading.Lock()


def get_cross_encoder_reranker(
    model_name: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
) -> CrossEncoderReranker:
    """
    Get or create the global cross-encoder reranker.

    Model loading itself remains lazy.
    """
    global _reranker

    with _reranker_lock:
        if _reranker is None:
            _reranker = CrossEncoderReranker(
                model_name=model_name,
                batch_size=batch_size,
                device=device,
            )

        return _reranker


def reset_cross_encoder_reranker() -> None:
    """Reset the global reranker, primarily for testing."""
    global _reranker

    with _reranker_lock:
        reranker = _reranker
        _reranker = None

    if reranker is not None:
        reranker.shutdown()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    *,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> List[Dict[str, Any]]:
    """Quickly rerank chunks using the global reranker."""
    return get_cross_encoder_reranker().rerank(
        query,
        chunks,
        top_k,
        organization_id=organization_id,
        namespace=namespace,
    )


def rerank_with_threshold(
    query: str,
    chunks: List[Dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    max_results: Optional[int] = None,
    *,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> List[Dict[str, Any]]:
    """Quickly rerank with threshold filtering."""
    return get_cross_encoder_reranker().rerank_with_threshold(
        query,
        chunks,
        threshold,
        max_results,
        organization_id=organization_id,
        namespace=namespace,
    )


def get_reranker_stats() -> Dict[str, Any]:
    """Get global reranker statistics."""
    return get_cross_encoder_reranker().get_stats()


# =============================================================================
# TEST / DEMO
# =============================================================================

def test_cross_encoder() -> None:
    """Run a local cross-encoder smoke test."""
    print("\n🔄 Testing CrossEncoder Re-ranker\n")
    print("=" * 70)

    reranker = CrossEncoderReranker()

    print(f"🤖 Model: {reranker.model_name}")
    print(f"🎮 Device: {reranker.device}")
    print(f"⚙️  Batch: {reranker.batch_size}")
    print(f"📏 Max length: {reranker.max_length}")
    print(f"📊 Loaded: {reranker.is_loaded}")
    print("=" * 70)

    query = (
        "What is the leave policy for new employees?"
    )

    mock_chunks = [
        {
            "content": (
                "Employees are entitled to 20 days of paid "
                "leave per year. Leave accrues monthly."
            ),
            "metadata": {
                "source": "employee_handbook.pdf",
                "page": 3,
            },
            "score": 0.85,
        },
        {
            "content": (
                "The company cafeteria serves lunch from "
                "12pm to 2pm on weekdays."
            ),
            "metadata": {
                "source": "office_guide.pdf",
                "page": 12,
            },
            "score": 0.15,
        },
        {
            "content": (
                "New employees must complete a 90-day "
                "probation period before becoming eligible "
                "for full benefits including paid leave."
            ),
            "metadata": {
                "source": "hr_policies.pdf",
                "page": 5,
            },
            "score": 0.75,
        },
        {
            "content": (
                "The office is located downtown. Parking "
                "is available in the basement garage."
            ),
            "metadata": {
                "source": "office_guide.pdf",
                "page": 2,
            },
            "score": 0.10,
        },
        {
            "content": (
                "Leave requests must be submitted at least "
                "2 weeks in advance through the HR portal."
            ),
            "metadata": {
                "source": "leave_policy.pdf",
                "page": 4,
            },
            "score": 0.80,
        },
    ]

    print("\n📝 Re-ranking mock chunks")
    print("-" * 70)

    results = reranker.rerank(
        query,
        mock_chunks,
        top_k=3,
    )

    print(
        f"Results returned: {len(results)}"
    )

    for index, chunk in enumerate(results, 1):
        source = (
            chunk.get("metadata", {})
            .get("source", "Unknown")
        )

        content = _chunk_content(chunk)

        preview = (
            content[:80] + "..."
            if len(content) > 80
            else content
        )

        print(
            f"   {index}. "
            f"score={_chunk_score(chunk):.3f} "
            f"[{source}] {preview}"
        )

    print("\n🎯 Threshold filtering")
    print("-" * 70)

    filtered = reranker.rerank_with_threshold(
        query,
        mock_chunks,
        threshold=0.5,
        max_results=3,
    )

    print(
        f"Results above threshold: "
        f"{len(filtered)}/{len(mock_chunks)}"
    )

    print("\n📊 Statistics")
    print("-" * 70)

    for key, value in reranker.get_stats().items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ CrossEncoder test complete!\n")

    reranker.shutdown()


if __name__ == "__main__":
    test_cross_encoder()
