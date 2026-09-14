
#!/usr/bin/env python3
"""
PolicyGuard AI - Hybrid Search Module
=====================================
Production-ready hybrid retrieval combining BM25 (keyword) + Vector
(semantic) search with:

- Reciprocal Rank Fusion (RRF)
- Configurable BM25/vector weighting
- Vector-only and BM25-only fallbacks
- Safe handling of missing indexes
- Duplicate-safe batch processing
- Input validation
- Thread-safe read-oriented operations
- Windows/Linux compatibility

Author: PolicyGuard AI Team
Version: 2.0.0
Last Updated: 2026-09-13
"""

from __future__ import annotations

import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
from src.core.exceptions import RAGException, RetrievalError


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_TOP_K = 3
DEFAULT_INITIAL_MULTIPLIER = 3
MAX_TOP_K = 100
MAX_INITIAL_K = 100
DEFAULT_RRF_K = 60
MAX_BATCH_SIZE = 100
DEFAULT_MAX_WORKERS = 4
MAX_QUERY_LENGTH = 8_000
VALID_NAMESPACES = {"policy", "talent"}
_SCOPE_PATTERN = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")


# =============================================================================
# HELPERS
# =============================================================================

def _setting_int(
    name: str,
    default: int,
    minimum: int = 1,
) -> int:
    """Safely read an integer setting."""
    try:
        value = int(getattr(settings, name, default))
        return max(value, minimum)
    except (TypeError, ValueError):
        return default


def _setting_float(
    name: str,
    default: float,
) -> float:
    """Safely read a float setting."""
    try:
        return float(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default


def _normalize_query(query: Any) -> str:
    """Validate and normalize a search query."""
    if not isinstance(query, str):
        raise RAGException("Query must be a string")

    query = re.sub(
        r"[\x00-\x1f\x7f]",
        " ",
        query,
    )

    query = re.sub(
        r"\s+",
        " ",
        query,
    ).strip()

    if not query:
        raise RAGException(
            "Query cannot be empty"
        )

    if len(query) > MAX_QUERY_LENGTH:
        raise RAGException(
            f"Query exceeds the maximum allowed length of {MAX_QUERY_LENGTH} characters"
        )

    return query


def _safe_top_k(
    top_k: Optional[int],
    default: int,
) -> int:
    """Validate and bound top_k."""
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

    return min(top_k, MAX_TOP_K)


def _safe_alpha(alpha: Optional[float]) -> float:
    """Validate and clamp hybrid alpha."""
    if alpha is None:
        alpha = _setting_float(
            "HYBRID_ALPHA",
            0.5,
        )

    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise RAGException(
            "alpha must be numeric"
        ) from exc

    if not np.isfinite(alpha):
        raise RAGException(
            "alpha must be finite"
        )

    if not 0.0 <= alpha <= 1.0:
        logger.warning(
            "Alpha %.4f outside [0,1]; clamping",
            alpha,
        )
        alpha = max(
            0.0,
            min(1.0, alpha),
        )

    return alpha


def _safe_chunk(chunk: Any) -> Optional[Dict[str, Any]]:
    """Normalize a chunk into a dictionary."""
    if not isinstance(chunk, dict):
        return None

    content = chunk.get("content", "")

    if content is None:
        return None

    if not isinstance(content, str):
        content = str(content)

    content = content.strip()

    if not content:
        return None

    normalized = dict(chunk)

    normalized["content"] = content

    if not isinstance(
        normalized.get("metadata"),
        dict,
    ):
        normalized["metadata"] = {}

    return normalized


def _chunk_key(
    chunk: Dict[str, Any],
) -> str:
    """
    Generate a stable chunk identity.

    Prefer an explicit chunk ID when available. Fall back to content +
    source/page so duplicate text from different documents is not collapsed.
    """
    metadata = chunk.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    for key in (
        "chunk_id",
        "id",
        "uuid",
    ):
        value = chunk.get(key)

        if value is None:
            value = metadata.get(key)

        if value is not None:
            value = str(value).strip()

            if value:
                return f"id:{value}"

    source = metadata.get(
        "source",
        chunk.get("source", ""),
    )

    page = metadata.get(
        "page",
        chunk.get("page", ""),
    )

    content = chunk.get(
        "content",
        "",
    )

    return (
        f"content:{str(content)}"
        f"|source:{str(source)}"
        f"|page:{str(page)}"
    )


def _normalize_results(
    results: Any,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Normalize retrieval results.

    Supports common forms:
        [(chunk, score), ...]
        [chunk, chunk, ...]
    """
    if results is None:
        return []

    if not isinstance(
        results,
        (list, tuple),
    ):
        return []

    normalized: List[Tuple[Dict[str, Any], float]] = []

    for item in results:
        chunk: Any = None
        score: Any = 0.0

        if (
            isinstance(item, tuple)
            and len(item) >= 2
            and isinstance(item[0], dict)
        ):
            chunk = item[0]
            score = item[1]

        elif (
            isinstance(item, list)
            and len(item) >= 2
            and isinstance(item[0], dict)
        ):
            chunk = item[0]
            score = item[1]

        elif isinstance(item, dict):
            chunk = item
            score = item.get(
                "score",
                item.get("initial_score", 0.0),
            )

        normalized_chunk = _safe_chunk(chunk)

        if normalized_chunk is None:
            continue

        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = 0.0

        if not np.isfinite(numeric_score):
            numeric_score = 0.0

        normalized.append(
            (
                normalized_chunk,
                numeric_score,
            )
        )

    return normalized


def _embedding_to_list(
    embedding: Any,
) -> List[float]:
    """Convert a supported embedding object into a plain list."""
    if embedding is None:
        raise RetrievalError(
            "Query embedding is unavailable"
        )

    if hasattr(
        embedding,
        "tolist",
    ):
        embedding = embedding.tolist()

    if not isinstance(
        embedding,
        (list, tuple),
    ):
        raise RetrievalError(
            "Query embedding has an invalid format"
        )

    try:
        values = [
            float(value)
            for value in embedding
        ]
    except (TypeError, ValueError) as exc:
        raise RetrievalError(
            "Query embedding contains invalid values"
        ) from exc

    if not values:
        raise RetrievalError(
            "Query embedding is empty"
        )

    return values


def _has_usable_index(
    vector_store: Any,
    attribute: str,
) -> bool:
    """
    Safely determine whether an index exists.

    Avoids ``if not index`` because objects such as FAISS indexes may not
    implement meaningful boolean evaluation.
    """
    if not hasattr(
        vector_store,
        attribute,
    ):
        return False

    try:
        value = getattr(
            vector_store,
            attribute,
        )
    except Exception:
        return False

    return value is not None



def _normalize_scope(
    organization_id: Optional[str],
    namespace: str,
) -> Tuple[str, str]:
    """Validate the tenant scope used for every retrieval operation."""
    organization = (
        "default"
        if organization_id is None
        else str(organization_id).strip()
    )
    resolved_namespace = (
        "policy"
        if namespace is None
        else str(namespace).strip().lower()
    )

    if not organization or not _SCOPE_PATTERN.fullmatch(organization):
        raise RAGException("Invalid organization_id")

    if resolved_namespace not in VALID_NAMESPACES:
        raise RAGException(
            f"Unsupported retrieval namespace: {resolved_namespace}"
        )

    return organization, resolved_namespace


def _chunk_in_scope(
    chunk: Dict[str, Any],
    organization_id: str,
    namespace: str,
) -> bool:
    """Fail closed when retrieved metadata is missing or mismatched."""
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        return False

    chunk_org = metadata.get("organization_id")
    chunk_namespace = metadata.get("namespace")

    return (
        str(chunk_org).strip() == organization_id
        and str(chunk_namespace).strip().lower() == namespace
    )


def _filter_scoped_results(
    results: List[Tuple[Dict[str, Any], float]],
    organization_id: str,
    namespace: str,
) -> List[Tuple[Dict[str, Any], float]]:
    """Enforce tenant/namespace isolation after every backend retrieval."""
    return [
        (chunk, score)
        for chunk, score in results
        if _chunk_in_scope(chunk, organization_id, namespace)
    ]


def _scoped_backend_search(
    vector_store: Any,
    method_name: str,
    query_or_embedding: Any,
    top_k: int,
    organization_id: str,
    namespace: str,
) -> Any:
    """
    Call a retrieval backend with its tenant scope.

    Security rule: never retry an incompatible API without the scope
    arguments, because that would turn an integration mismatch into a
    possible cross-tenant data leak.
    """
    method = getattr(vector_store, method_name, None)
    if not callable(method):
        raise RetrievalError(
            f"{method_name} is not available"
        )

    try:
        return method(
            query_or_embedding,
            top_k=top_k,
            organization_id=organization_id,
            namespace=namespace,
        )
    except TypeError as exc:
        raise RetrievalError(
            f"{method_name} does not support required tenant scoping"
        ) from exc

# =============================================================================
# RECIPROCAL RANK FUSION
# =============================================================================

def reciprocal_rank_fusion(
    results_lists: List[
        List[Tuple[Dict[str, Any], float]]
    ],
    weights: Optional[List[float]] = None,
    top_k: Optional[int] = None,
    k: int = DEFAULT_RRF_K,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Combine ranked result lists using Reciprocal Rank Fusion.

    Formula:

        RRF(d) = Σ weight_i / (k + rank_i)

    A critical detail is that a document absent from a result list receives
    NO contribution from that list. The previous implementation assigned it
    a synthetic last rank, which incorrectly boosted absent documents.

    Args:
        results_lists:
            Ranked lists of ``(chunk, score)`` tuples.
        weights:
            Weight for each retrieval source.
        top_k:
            Number of fused results.
        k:
            RRF constant, normally 60.

    Returns:
        ``[(chunk, fused_score), ...]`` sorted descending.
    """
    if not results_lists:
        return []

    if top_k is None:
        top_k = _setting_int(
            "TOP_K",
            DEFAULT_TOP_K,
            minimum=1,
        )

    top_k = _safe_top_k(
        top_k,
        DEFAULT_TOP_K,
    )

    try:
        k = int(k)
    except (TypeError, ValueError) as exc:
        raise RAGException(
            "RRF k must be an integer"
        ) from exc

    if k <= 0:
        raise RAGException(
            "RRF k must be greater than zero"
        )

    # Normalize every input list.
    normalized_lists = [
        _normalize_results(results)
        for results in results_lists
    ]

    if weights is None:
        weights = [
            1.0
            for _ in normalized_lists
        ]
    elif len(weights) != len(normalized_lists):
        logger.warning(
            "RRF weight count (%s) does not match "
            "result-list count (%s); using equal weights",
            len(weights),
            len(normalized_lists),
        )

        weights = [
            1.0
            for _ in normalized_lists
        ]
    else:
        cleaned_weights = []

        for weight in weights:
            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError):
                numeric_weight = 1.0

            if not np.isfinite(
                numeric_weight
            ):
                numeric_weight = 1.0

            # Negative retrieval weights are not meaningful for RRF.
            numeric_weight = max(
                0.0,
                numeric_weight,
            )

            cleaned_weights.append(
                numeric_weight
            )

        weights = cleaned_weights

    # key -> chunk
    all_chunks: Dict[
        str,
        Dict[str, Any]
    ] = {}

    # key -> fused score
    fused_scores: Dict[
        str,
        float
    ] = {}

    for list_index, results in enumerate(
        normalized_lists
    ):
        weight = weights[list_index]

        for rank, (chunk, _) in enumerate(
            results,
            start=1,
        ):
            key = _chunk_key(chunk)

            if key not in all_chunks:
                all_chunks[key] = chunk
                fused_scores[key] = 0.0

            # Only ranked appearances contribute.
            fused_scores[key] += (
                weight / (k + rank)
            )

    sorted_keys = sorted(
        fused_scores,
        key=lambda key: fused_scores[key],
        reverse=True,
    )

    fused_results = [
        (
            all_chunks[key],
            fused_scores[key],
        )
        for key in sorted_keys[:top_k]
    ]

    logger.debug(
        "RRF fusion: %s result lists, %s unique chunks -> %s",
        len(normalized_lists),
        len(all_chunks),
        len(fused_results),
    )

    return fused_results


# =============================================================================
# HYBRID SEARCH
# =============================================================================

def hybrid_search(
    query: str,
    vector_store: Any,
    top_k: Optional[int] = None,
    alpha: Optional[float] = None,
    bm25_boost: float = 1.0,
    *,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> Tuple[
    List[Dict[str, Any]],
    List[float],
]:
    """
    Hybrid BM25 + vector search.

    Alpha semantics:

        alpha = 0.0 -> BM25 only
        alpha = 0.5 -> equal weighting
        alpha = 1.0 -> vector only

    Returns:
        Tuple of:
            - result chunks
            - fused/retrieval scores
    """
    if vector_store is None:
        raise RAGException(
            "vector_store cannot be None"
        )

    query = _normalize_query(query)
    organization_id, namespace = _normalize_scope(
        organization_id,
        namespace,
    )

    default_top_k = _setting_int(
        "TOP_K",
        DEFAULT_TOP_K,
        minimum=1,
    )

    top_k = _safe_top_k(
        top_k,
        default_top_k,
    )

    alpha = _safe_alpha(alpha)

    try:
        bm25_boost = float(
            bm25_boost
        )
    except (TypeError, ValueError) as exc:
        raise RAGException(
            "bm25_boost must be numeric"
        ) from exc

    if not np.isfinite(
        bm25_boost
    ):
        raise RAGException(
            "bm25_boost must be finite"
        )

    bm25_boost = max(
        0.0,
        bm25_boost,
    )

    # Retrieve a larger candidate set before fusion.
    initial_k = min(
        max(
            top_k * DEFAULT_INITIAL_MULTIPLIER,
            top_k,
        ),
        MAX_INITIAL_K,
    )

    bm25_results: List[
        Tuple[Dict[str, Any], float]
    ] = []

    vector_results: List[
        Tuple[Dict[str, Any], float]
    ] = []

    # -------------------------------------------------------------------------
    # BM25
    # -------------------------------------------------------------------------

    bm25_available = (
        hasattr(
            vector_store,
            "bm25_search",
        )
        and _has_usable_index(
            vector_store,
            "bm25_index",
        )
    )

    if bm25_available:
        try:
            raw_bm25 = _scoped_backend_search(
                vector_store,
                "bm25_search",
                query,
                initial_k,
                organization_id,
                namespace,
            )

            bm25_results = _filter_scoped_results(
                _normalize_results(raw_bm25),
                organization_id,
                namespace,
            )

            if bm25_boost != 1.0:
                bm25_results = [
                    (
                        chunk,
                        score * bm25_boost,
                    )
                    for chunk, score in bm25_results
                ]

            logger.debug(
                "BM25 returned %s results",
                len(bm25_results),
            )

        except Exception:
            logger.exception(
                "BM25 search failed; continuing with vector search"
            )

    else:
        logger.debug(
            "BM25 index unavailable"
        )

    # -------------------------------------------------------------------------
    # Vector search
    # -------------------------------------------------------------------------

    vector_available = (
        hasattr(
            vector_store,
            "vector_search",
        )
        and _has_usable_index(
            vector_store,
            "index",
        )
    )

    if vector_available:
        try:
            query_embedding = None

            # Prefer the vector store's embedder if available.
            store_embedder = getattr(
                vector_store,
                "embedder",
                None,
            )

            if store_embedder is not None:
                query_embedding = (
                    store_embedder.encode_query(
                        query
                    )
                )

            if query_embedding is not None:
                query_embedding = _embedding_to_list(
                    query_embedding
                )

                raw_vector = _scoped_backend_search(
                    vector_store,
                    "vector_search",
                    query_embedding,
                    initial_k,
                    organization_id,
                    namespace,
                )

            else:
                # Some vector stores accept a raw query string, but tenant
                # scope is still mandatory.
                raw_vector = _scoped_backend_search(
                    vector_store,
                    "vector_search",
                    query,
                    initial_k,
                    organization_id,
                    namespace,
                )

            vector_results = _filter_scoped_results(
                _normalize_results(raw_vector),
                organization_id,
                namespace,
            )

            logger.debug(
                "Vector search returned %s results",
                len(vector_results),
            )

        except Exception:
            logger.exception(
                "Vector search failed"
            )

    else:
        logger.debug(
            "Vector index unavailable"
        )

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    if not bm25_results and not vector_results:
        logger.warning(
            "No results from BM25 or vector search"
        )
        return [], []

    if not bm25_results:
        logger.info(
            "Using vector-only fallback"
        )

        results = vector_results[:top_k]

        return (
            [chunk for chunk, _ in results],
            [score for _, score in results],
        )

    if not vector_results:
        logger.info(
            "Using BM25-only fallback"
        )

        results = bm25_results[:top_k]

        return (
            [chunk for chunk, _ in results],
            [score for _, score in results],
        )

    # -------------------------------------------------------------------------
    # RRF fusion
    # -------------------------------------------------------------------------

    weights = [
        1.0 - alpha,
        alpha,
    ]

    fused_results = reciprocal_rank_fusion(
        results_lists=[
            bm25_results,
            vector_results,
        ],
        weights=weights,
        top_k=top_k,
        k=DEFAULT_RRF_K,
    )

    chunks = [
        chunk
        for chunk, _ in fused_results
    ]

    scores = [
        score
        for _, score in fused_results
    ]

    logger.info(
        "Hybrid search: BM25=%s + Vector=%s -> %s "
        "results (alpha=%.2f)",
        len(bm25_results),
        len(vector_results),
        len(chunks),
        alpha,
    )

    return chunks, scores


# =============================================================================
# BM25-ONLY SEARCH
# =============================================================================

def bm25_only_search(
    query: str,
    vector_store: Any,
    top_k: Optional[int] = None,
    *,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> Tuple[
    List[Dict[str, Any]],
    List[float],
]:
    """Perform BM25-only search."""
    if vector_store is None:
        raise RAGException(
            "vector_store cannot be None"
        )

    query = _normalize_query(query)
    organization_id, namespace = _normalize_scope(
        organization_id,
        namespace,
    )

    default_top_k = _setting_int(
        "TOP_K",
        DEFAULT_TOP_K,
        minimum=1,
    )

    top_k = _safe_top_k(
        top_k,
        default_top_k,
    )

    if not hasattr(
        vector_store,
        "bm25_search",
    ):
        logger.warning(
            "BM25 search is not available"
        )
        return [], []

    if not _has_usable_index(
        vector_store,
        "bm25_index",
    ):
        logger.warning(
            "BM25 index is not initialized"
        )
        return [], []

    try:
        results = _filter_scoped_results(
            _normalize_results(
                _scoped_backend_search(
                    vector_store,
                    "bm25_search",
                    query,
                    top_k,
                    organization_id,
                    namespace,
                )
            ),
            organization_id,
            namespace,
        )

        return (
            [chunk for chunk, _ in results],
            [score for _, score in results],
        )

    except Exception:
        logger.exception(
            "BM25-only search failed"
        )
        return [], []


# =============================================================================
# VECTOR-ONLY SEARCH
# =============================================================================

def vector_only_search(
    query: Union[
        str,
        List[float],
    ],
    vector_store: Any,
    top_k: Optional[int] = None,
    *,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> Tuple[
    List[Dict[str, Any]],
    List[float],
]:
    """Perform vector-only semantic search."""
    if vector_store is None:
        raise RAGException(
            "vector_store cannot be None"
        )

    organization_id, namespace = _normalize_scope(
        organization_id,
        namespace,
    )

    default_top_k = _setting_int(
        "TOP_K",
        DEFAULT_TOP_K,
        minimum=1,
    )

    top_k = _safe_top_k(
        top_k,
        default_top_k,
    )

    if not hasattr(
        vector_store,
        "vector_search",
    ):
        logger.warning(
            "Vector search is not available"
        )
        return [], []

    if not _has_usable_index(
        vector_store,
        "index",
    ):
        logger.warning(
            "Vector index is not initialized"
        )
        return [], []

    try:
        search_input: Any = query

        if isinstance(
            query,
            str,
        ):
            normalized_query = _normalize_query(
                query
            )

            store_embedder = getattr(
                vector_store,
                "embedder",
                None,
            )

            if store_embedder is not None:
                embedding = (
                    store_embedder.encode_query(
                        normalized_query
                    )
                )

                search_input = _embedding_to_list(
                    embedding
                )

        else:
            search_input = _embedding_to_list(
                query
            )

        results = _filter_scoped_results(
            _normalize_results(
                _scoped_backend_search(
                    vector_store,
                    "vector_search",
                    search_input,
                    top_k,
                    organization_id,
                    namespace,
                )
            ),
            organization_id,
            namespace,
        )

        return (
            [chunk for chunk, _ in results],
            [score for _, score in results],
        )

    except Exception:
        logger.exception(
            "Vector-only search failed"
        )
        return [], []


# =============================================================================
# BATCH HYBRID SEARCH
# =============================================================================

def batch_hybrid_search(
    queries: List[str],
    vector_store: Any,
    top_k: Optional[int] = None,
    alpha: Optional[float] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    *,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> List[
    Tuple[
        List[Dict[str, Any]],
        List[float],
    ]
]:
    """
    Execute multiple hybrid searches concurrently.

    Results always preserve the original query order, including when duplicate
    queries are present.
    """
    if vector_store is None:
        raise RAGException(
            "vector_store cannot be None"
        )

    if not isinstance(
        queries,
        (list, tuple),
    ):
        raise RAGException(
            "queries must be a list or tuple"
        )

    if len(queries) > MAX_BATCH_SIZE:
        raise RAGException(
            f"Batch size cannot exceed {MAX_BATCH_SIZE}"
        )

    if not queries:
        return []

    organization_id, namespace = _normalize_scope(
        organization_id,
        namespace,
    )

    default_top_k = _setting_int(
        "TOP_K",
        DEFAULT_TOP_K,
        minimum=1,
    )

    top_k = _safe_top_k(
        top_k,
        default_top_k,
    )

    alpha = _safe_alpha(alpha)

    try:
        max_workers = int(
            max_workers
        )
    except (TypeError, ValueError) as exc:
        raise RAGException(
            "max_workers must be an integer"
        ) from exc

    max_workers = max(
        1,
        min(
            max_workers,
            len(queries),
            16,
        ),
    )

    normalized_queries = [
        _normalize_query(query)
        for query in queries
    ]

    ordered_results: List[
        Optional[
            Tuple[
                List[Dict[str, Any]],
                List[float],
            ]
        ]
    ] = [None] * len(
        normalized_queries
    )

    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="hybrid-search",
    ) as executor:

        future_to_index = {
            executor.submit(
                hybrid_search,
                query,
                vector_store,
                top_k,
                alpha,
                1.0,
                organization_id=organization_id,
                namespace=namespace,
            ): index
            for index, query in enumerate(
                normalized_queries
            )
        }

        for future in as_completed(
            future_to_index
        ):
            index = future_to_index[
                future
            ]

            try:
                ordered_results[index] = (
                    future.result()
                )

            except Exception:
                logger.exception(
                    "Batch hybrid search failed "
                    "for query index %s",
                    index,
                )

                ordered_results[index] = (
                    [],
                    [],
                )

    results = [
        result
        if result is not None
        else ([], [])
        for result in ordered_results
    ]

    logger.info(
        "Batch hybrid search complete: %s queries",
        len(results),
    )

    return results


# =============================================

