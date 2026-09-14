
#!/usr/bin/env python3
"""
PolicyGuard AI - Production Vector Store
========================================
FAISS-based vector store with BM25 hybrid retrieval.

Features:
- FAISS semantic search
- BM25 keyword search
- Reciprocal-rank-fusion compatible results
- Configurable embedding dimension
- L2 / inner-product metrics
- Thread-safe operations
- Atomic persistence
- Safe loading and validation
- Stable chunk IDs for BM25 mapping
- Batch ingestion and search
- Runtime statistics
- Windows/Linux compatibility

Author: PolicyGuard AI Team
Version: 2.1.0
Last Updated: 2026-09-13
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import shutil
import sys
import tempfile
import threading
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
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

DEFAULT_DIMENSION = 384
DEFAULT_TOP_K = 3
MAX_TOP_K = 100
MAX_BATCH_SIZE = 1000

_SUPPORTED_METRICS = {
    "l2",
    "ip",
}

VECTOR_STORE_SCHEMA_VERSION = 3
DEFAULT_ORGANIZATION_ID = "default"
MAX_ORGANIZATION_ID_LENGTH = 128


def _normalize_organization_id(value: Optional[str]) -> str:
    """Normalize and validate a tenant/organization identifier."""
    if value is None or not str(value).strip():
        return DEFAULT_ORGANIZATION_ID
    value = str(value).strip()
    if len(value) > MAX_ORGANIZATION_ID_LENGTH:
        raise RAGException("organization_id is too long")
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
        raise RAGException("organization_id contains invalid characters")
    return value


def _chunk_organization_id(chunk: Dict[str, Any]) -> str:
    """Return the organization scope stored on a chunk."""
    metadata = chunk.get("metadata", {})
    value = metadata.get("organization_id") if isinstance(metadata, dict) else None
    if value is None:
        value = chunk.get("organization_id", DEFAULT_ORGANIZATION_ID)
    return _normalize_organization_id(value)


def _chunk_namespace(chunk: Dict[str, Any]) -> str:
    """Return an optional logical namespace for a chunk."""
    metadata = chunk.get("metadata", {})
    value = metadata.get("namespace") if isinstance(metadata, dict) else None
    if value is None:
        value = chunk.get("namespace", "policy")
    return str(value).strip() or "policy"


_TOKEN_RE = re.compile(
    r"\b[a-z0-9][a-z0-9\-]*[a-z0-9]\b|\b[a-z0-9]\b",
    re.IGNORECASE,
)


# =============================================================================
# HELPERS
# =============================================================================

def _utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


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
        raise RAGException("top_k must be an integer") from exc

    if top_k <= 0:
        raise RAGException("top_k must be greater than zero")

    return min(top_k, MAX_TOP_K)


def _normalize_query(query: Any) -> str:
    """Validate and normalize a search query."""
    if not isinstance(query, str):
        raise RAGException("Query must be a string")

    query = re.sub(r"[\x00-\x1f\x7f]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    if not query:
        raise RAGException("Query cannot be empty")

    return query


def _safe_chunk(
    chunk: Any,
) -> Optional[Dict[str, Any]]:
    """Normalize a chunk dictionary."""
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

    metadata = normalized.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    normalized["metadata"] = dict(metadata)

    return normalized


def _chunk_id(
    chunk: Dict[str, Any],
    index: int,
) -> str:
    """
    Return a stable ID for a chunk.

    Existing IDs are preserved. Otherwise an ingestion index is used.
    """
    metadata = chunk.get("metadata", {})

    for key in (
        "chunk_id",
        "id",
        "uuid",
    ):
        value = chunk.get(key)

        if value is None and isinstance(metadata, dict):
            value = metadata.get(key)

        if value is not None:
            value = str(value).strip()

            if value:
                return value

    return f"chunk_{index}"


def _chunk_source(
    chunk: Dict[str, Any],
) -> str:
    """Extract source metadata."""
    metadata = chunk.get("metadata", {})

    source = (
        metadata.get("source")
        if isinstance(metadata, dict)
        else None
    )

    if source is None:
        source = chunk.get("source", "Unknown")

    return str(source).strip() or "Unknown"


def _normalize_embeddings(
    embeddings: Any,
    expected_count: int,
    dimension: int,
) -> np.ndarray:
    """Validate and convert embeddings to a FAISS-compatible matrix."""
    if embeddings is None:
        raise RAGException("Embeddings are required")

    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach()

    if hasattr(embeddings, "cpu"):
        embeddings = embeddings.cpu()

    if hasattr(embeddings, "numpy"):
        embeddings = embeddings.numpy()

    try:
        array = np.asarray(
            embeddings,
            dtype=np.float32,
        )
    except Exception as exc:
        raise RAGException(
            "Unable to convert embeddings to numpy array"
        ) from exc

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise RAGException(
            "Embeddings must be a 2D matrix"
        )

    if array.shape[0] != expected_count:
        raise RAGException(
            "Number of embeddings does not match number of chunks"
        )

    if array.shape[1] != dimension:
        raise RAGException(
            f"Embedding dimension mismatch: "
            f"expected {dimension}, got {array.shape[1]}"
        )

    if not np.all(np.isfinite(array)):
        raise RAGException(
            "Embeddings contain NaN or infinite values"
        )

    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


def _normalize_query_embedding(
    embedding: Any,
    dimension: int,
) -> np.ndarray:
    """Validate one query embedding."""
    return _normalize_embeddings(
        embedding,
        expected_count=1,
        dimension=dimension,
    )


def _atomic_replace(
    source: Path,
    destination: Path,
) -> None:
    """Atomically replace destination with source where supported."""
    os.replace(
        str(source),
        str(destination),
    )


def _write_pickle_atomic(
    path: Path,
    data: Any,
) -> None:
    """Write pickle data atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    temp_path = Path(temp_name)

    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(
                data,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            handle.flush()
            os.fsync(handle.fileno())

        _atomic_replace(
            temp_path,
            path,
        )

    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

        raise


def _write_faiss_atomic(
    path: Path,
    index: faiss.Index,
) -> None:
    """Write a FAISS index atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    os.close(fd)

    temp_path = Path(temp_name)

    try:
        faiss.write_index(
            index,
            str(temp_path),
        )

        _atomic_replace(
            temp_path,
            path,
        )

    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

        raise


def _copy_backup(
    source: Path,
    backup_path: Path,
) -> None:
    """Create a backup without moving/removing the live file."""
    if not source.exists():
        return

    backup_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_backup = backup_path.with_name(
        f".{backup_path.name}.tmp"
    )

    try:
        temporary_backup.unlink(missing_ok=True)
    except Exception:
        pass

    try:
        shutil.copy2(
            source,
            temporary_backup,
        )

        os.replace(
            str(temporary_backup),
            str(backup_path),
        )

    except Exception:
        try:
            temporary_backup.unlink(missing_ok=True)
        except Exception:
            pass

        raise


# =============================================================================
# BM25 INDEX
# =============================================================================

class BM25Index:
    """
    BM25 keyword index.

    The index is rebuilt from the complete document collection whenever
    documents change. This avoids inconsistent state caused by trying
    to append documents to an existing rank_bm25 model.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        try:
            self.k1 = float(k1)
            self.b = float(b)
        except (TypeError, ValueError) as exc:
            raise RAGException(
                "Invalid BM25 parameters"
            ) from exc

        if self.k1 <= 0:
            raise RAGException(
                "BM25 k1 must be greater than zero"
            )

        if not 0 <= self.b <= 1:
            raise RAGException(
                "BM25 b must be between 0 and 1"
            )

        self._bm25: Any = None
        self._corpus: List[str] = []
        self._doc_ids: List[str] = []

        self._lock = threading.RLock()

    @property
    def is_available(self) -> bool:
        """Whether a usable BM25 model exists."""
        with self._lock:
            return (
                self._bm25 is not None
                and bool(self._corpus)
                and len(self._corpus) == len(self._doc_ids)
            )

    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Replace the BM25 corpus with the supplied documents.

        The index is rebuilt rather than pretending rank_bm25 supports
        efficient incremental updates.
        """
        if not isinstance(
            documents,
            (list, tuple),
        ):
            raise RAGException(
                "documents must be a list or tuple"
            )

        if not documents:
            self.clear()
            return

        cleaned_documents = [
            str(document)
            if document is not None
            else ""
            for document in documents
        ]

        if doc_ids is None:
            cleaned_ids = [
                f"doc_{index}"
                for index in range(len(cleaned_documents))
            ]
        else:
            if len(doc_ids) != len(cleaned_documents):
                raise RAGException(
                    "documents and doc_ids must have equal lengths"
                )

            cleaned_ids = [
                str(doc_id).strip()
                for doc_id in doc_ids
            ]

        if any(not doc_id for doc_id in cleaned_ids):
            raise RAGException(
                "BM25 document IDs cannot be empty"
            )

        if len(cleaned_ids) != len(set(cleaned_ids)):
            raise RAGException(
                "BM25 document IDs must be unique"
            )

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning(
                "rank_bm25 is not installed; BM25 search disabled"
            )

            with self._lock:
                self._bm25 = None
                self._corpus = cleaned_documents
                self._doc_ids = cleaned_ids

            return

        tokenized_documents = [
            self._tokenize(document)
            for document in cleaned_documents
        ]

        if not any(tokenized_documents):
            with self._lock:
                self._bm25 = None
                self._corpus = cleaned_documents
                self._doc_ids = cleaned_ids

            return

        bm25 = BM25Okapi(
            tokenized_documents,
            k1=self.k1,
            b=self.b,
        )

        with self._lock:
            self._bm25 = bm25
            self._corpus = cleaned_documents
            self._doc_ids = cleaned_ids

        logger.info(
            "BM25 index built: %s documents",
            len(cleaned_documents),
        )

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        """Tokenize text for BM25."""
        return [
            token.lower()
            for token in _TOKEN_RE.findall(str(text))
        ]

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Search BM25 and return document IDs with scores."""
        query = _normalize_query(query)
        top_k = _safe_top_k(top_k, 10)

        with self._lock:
            if not self.is_available:
                return []

            query_tokens = self._tokenize(query)

            if not query_tokens:
                return []

            scores = np.asarray(
                self._bm25.get_scores(query_tokens),
                dtype=np.float64,
            )

            if scores.size == 0:
                return []

            # rank_bm25 can legitimately produce zero/negative scores on
            # small corpora (especially when a query term occurs in every
            # document). A lexical match must not be discarded merely because
            # its BM25 score is non-positive. The old ``scores > 0`` filter
            # caused valid tenant-isolation searches such as ``alpha`` to
            # return no results.
            #
            # Restrict candidates by actual token overlap first. This keeps
            # unrelated zero-score documents out while allowing legitimate
            # zero/negative BM25 matches to be ranked normally.
            query_token_set = set(query_tokens)

            candidate_indices = np.asarray(
                [
                    index
                    for index, document in enumerate(self._corpus)
                    if query_token_set.intersection(
                        self._tokenize(document)
                    )
                ],
                dtype=np.int64,
            )

            if candidate_indices.size == 0:
                return []

            # Stable descending score order. ``mergesort`` preserves the
            # original corpus order for ties, making results deterministic.
            order = np.argsort(
                -scores[candidate_indices],
                kind="mergesort",
            )

            candidate_indices = candidate_indices[order]

            results: List[Tuple[str, float]] = []

            for index in candidate_indices[:top_k]:
                index = int(index)
                score = float(scores[index])

                if not np.isfinite(score):
                    continue

                results.append(
                    (
                        self._doc_ids[index],
                        score,
                    )
                )

            return results

    def get_document(
        self,
        doc_id: str,
    ) -> Optional[str]:
        """Get document text by ID."""
        with self._lock:
            try:
                index = self._doc_ids.index(
                    str(doc_id)
                )
            except ValueError:
                return None

            return self._corpus[index]

    def get_documents(self) -> Dict[str, str]:
        """Return a document-ID -> document-text mapping."""
        with self._lock:
            return dict(
                zip(
                    self._doc_ids,
                    self._corpus,
                )
            )

    def get_persistable_data(self) -> Dict[str, Any]:
        """Return BM25 source data suitable for persistence."""
        with self._lock:
            return {
                "documents": list(self._corpus),
                "doc_ids": list(self._doc_ids),
                "k1": self.k1,
                "b": self.b,
            }

    def clear(self) -> None:
        """Clear BM25 data."""
        with self._lock:
            self._bm25 = None
            self._corpus = []
            self._doc_ids = []


# =============================================================================
# VECTOR STORE
# =============================================================================

class VectorStore:
    """
    FAISS + BM25 vector store.

    FAISS stores embeddings while ``chunks`` stores associated metadata.
    The position of a chunk in ``chunks`` corresponds directly to its
    position in the FAISS index.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "data/vector_db",
        dimension: Optional[int] = None,
        metric_type: str = "l2",
        enable_bm25: bool = True,
        organization_id: Optional[str] = None,
        namespace: str = "policy",
    ):
        self.db_path = Path(db_path)

        self.db_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path = (
            self.db_path / "faiss.index"
        )

        self.chunks_path = (
            self.db_path / "chunks.pkl"
        )

        self.bm25_path = (
            self.db_path / "bm25.pkl"
        )

        self.manifest_path = self.db_path / "manifest.pkl"

        if dimension is None:
            dimension = _setting_int(
                "EMBEDDING_DIMENSION",
                DEFAULT_DIMENSION,
                minimum=1,
            )

        try:
            self.dimension = int(dimension)
        except (TypeError, ValueError) as exc:
            raise RAGException(
                "dimension must be an integer"
            ) from exc

        if self.dimension <= 0:
            raise RAGException(
                "dimension must be greater than zero"
            )

        self.metric_type = str(
            metric_type
        ).lower().strip()

        if self.metric_type not in _SUPPORTED_METRICS:
            raise RAGException(
                "metric_type must be 'l2' or 'ip'"
            )

        self.enable_bm25 = bool(enable_bm25)
        self.organization_id = _normalize_organization_id(organization_id)
        self.namespace = str(namespace).strip() or "policy"
        if len(self.namespace) > 128:
            raise RAGException("namespace is too long")

        self.index: Optional[faiss.Index] = None

        self.chunks: List[
            Dict[str, Any]
        ] = []

        self.bm25_index: Optional[
            BM25Index
        ] = (
            BM25Index()
            if self.enable_bm25
            else None
        )

        self.embedder: Any = None

        self._lock = threading.RLock()
        self._is_loaded = False
        self._shutdown = False

        self._stats = {
            "total_chunks": 0,
            "last_updated": None,
            "load_time_ms": 0.0,
            "search_count": 0,
            "avg_search_time_ms": 0.0,
            "add_count": 0,
        }

        logger.info(
            "VectorStore initialized: "
            "dim=%s, metric=%s, bm25=%s, path=%s",
            self.dimension,
            self.metric_type,
            self.enable_bm25,
            self.db_path,
        )

    # -------------------------------------------------------------------------
    # Dependency injection
    # -------------------------------------------------------------------------

    def set_embedder(
        self,
        embedder: Any,
    ) -> None:
        """Set embedder used for string queries and ingestion."""
        if embedder is None:
            raise RAGException(
                "Embedder cannot be None"
            )

        self.embedder = embedder

        logger.info(
            "Embedder connected to VectorStore"
        )

    # -------------------------------------------------------------------------
    # FAISS index creation
    # -------------------------------------------------------------------------

    def _create_index(self) -> faiss.Index:
        """Create an empty FAISS index."""
        if self.metric_type == "ip":
            return faiss.IndexFlatIP(
                self.dimension
            )

        return faiss.IndexFlatL2(
            self.dimension
        )

    def _validate_index(self) -> None:
        """Validate FAISS/chunk alignment."""
        if self.index is None:
            raise RAGException(
                "FAISS index is not initialized"
            )

        if self.index.d != self.dimension:
            raise RAGException(
                f"FAISS dimension mismatch: "
                f"expected {self.dimension}, "
                f"got {self.index.d}"
            )

        if self.index.ntotal != len(self.chunks):
            raise RAGException(
                "FAISS index and chunk metadata are out of sync"
            )

    def _validate_unique_chunk_ids(
        self,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """Ensure all chunk IDs are unique across the entire store."""
        existing_ids = {
            _chunk_id(chunk, index)
            for index, chunk in enumerate(self.chunks)
        }

        incoming_ids = set()

        for index, chunk in enumerate(chunks):
            chunk_id = _chunk_id(
                chunk,
                len(self.chunks) + index,
            )

            if chunk_id in incoming_ids:
                raise RAGException(
                    f"Duplicate chunk ID in batch: {chunk_id}"
                )

            if chunk_id in existing_ids:
                raise RAGException(
                    f"Chunk ID already exists: {chunk_id}"
                )

            incoming_ids.add(chunk_id)

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load persisted vector store.

        If loading fails, the in-memory store is reset rather than leaving
        a partially loaded index.
        """
        start_time = time.monotonic()

        with self._lock:
            # Never leave old state around when attempting a fresh load.
            self.index = None
            self.chunks = []

            if self.bm25_index:
                self.bm25_index.clear()

            self._is_loaded = False

            try:
                if not self.index_path.exists():
                    logger.info(
                        "No FAISS index at %s",
                        self.db_path,
                    )
                    return False

                if not self.chunks_path.exists():
                    logger.warning(
                        "FAISS index exists but chunks metadata is missing"
                    )
                    return False

                if self.manifest_path.exists():
                    with open(self.manifest_path, "rb") as handle:
                        manifest = pickle.load(handle)
                    if not isinstance(manifest, dict):
                        raise RAGException("Invalid vector-store manifest")
                    persisted_org = _normalize_organization_id(manifest.get("organization_id"))
                    persisted_namespace = str(manifest.get("namespace", "policy")).strip() or "policy"
                    if persisted_org != self.organization_id or persisted_namespace != self.namespace:
                        raise RAGException(
                            "Vector-store tenant/namespace mismatch; refusing to load data "
                            f"for {self.organization_id}/{self.namespace}"
                        )

                loaded_index = faiss.read_index(
                    str(self.index_path)
                )

                with open(
                    self.chunks_path,
                    "rb",
                ) as handle:
                    loaded_chunks = pickle.load(handle)

                if not isinstance(
                    loaded_chunks,
                    list,
                ):
                    raise RAGException(
                        "Persisted chunks must be a list"
                    )

                normalized_chunks: List[
                    Dict[str, Any]
                ] = []

                seen_ids = set()

                for index, chunk in enumerate(
                    loaded_chunks
                ):
                    normalized = _safe_chunk(chunk)

                    if normalized is None:
                        raise RAGException(
                            f"Persisted chunk {index} is invalid"
                        )

                    chunk_id = _chunk_id(
                        normalized,
                        index,
                    )

                    if chunk_id in seen_ids:
                        raise RAGException(
                            f"Duplicate persisted chunk ID: {chunk_id}"
                        )

                    seen_ids.add(chunk_id)

                    metadata = dict(
                        normalized.get(
                            "metadata",
                            {},
                        )
                    )

                    metadata["chunk_id"] = chunk_id
                    normalized["metadata"] = metadata

                    normalized_chunks.append(
                        normalized
                    )

                if loaded_index.d != self.dimension:
                    raise RAGException(
                        "Persisted FAISS index dimension "
                        f"{loaded_index.d} does not match configured "
                        f"dimension {self.dimension}"
                    )

                if loaded_index.ntotal != len(
                    normalized_chunks
                ):
                    raise RAGException(
                        "Persisted FAISS index and chunks are misaligned"
                    )

                self.index = loaded_index
                self.chunks = normalized_chunks

                # -------------------------------------------------------------
                # BM25
                # -------------------------------------------------------------

                if self.enable_bm25 and self.bm25_index:
                    bm25_loaded = False

                    if self.bm25_path.exists():
                        try:
                            with open(
                                self.bm25_path,
                                "rb",
                            ) as handle:
                                bm25_data = pickle.load(
                                    handle
                                )

                            if not isinstance(
                                bm25_data,
                                dict,
                            ):
                                raise RAGException(
                                    "Persisted BM25 data must be a dictionary"
                                )

                            documents = bm25_data.get(
                                "documents",
                                [],
                            )

                            doc_ids = bm25_data.get(
                                "doc_ids",
                                [],
                            )

                            if (
                                isinstance(documents, list)
                                and isinstance(doc_ids, list)
                                and len(documents)
                                == len(self.chunks)
                                and len(doc_ids)
                                == len(self.chunks)
                            ):
                                expected_ids = [
                                    _chunk_id(
                                        chunk,
                                        index,
                                    )
                                    for index, chunk in enumerate(
                                        self.chunks
                                    )
                                ]

                                persisted_ids = [
                                    str(doc_id).strip()
                                    for doc_id in doc_ids
                                ]

                                if (
                                    persisted_ids
                                    == expected_ids
                                    and len(
                                        set(persisted_ids)
                                    )
                                    == len(persisted_ids)
                                ):
                                    k1 = bm25_data.get(
                                        "k1",
                                        1.5,
                                    )

                                    b = bm25_data.get(
                                        "b",
                                        0.75,
                                    )

                                    self.bm25_index.k1 = float(k1)
                                    self.bm25_index.b = float(b)

                                    self.bm25_index.add_documents(
                                        documents,
                                        persisted_ids,
                                    )

                                    bm25_loaded = True

                        except Exception:
                            logger.exception(
                                "Persisted BM25 data could not be loaded; "
                                "rebuilding from chunks"
                            )

                    if not bm25_loaded:
                        self._rebuild_bm25()

                self._is_loaded = True
                self._shutdown = False

                self._stats[
                    "total_chunks"
                ] = len(self.chunks)

                self._stats[
                    "last_updated"
                ] = _utc_now_iso()

                self._stats[
                    "load_time_ms"
                ] = round(
                    (
                        time.monotonic()
                        - start_time
                    )
                    * 1000,
                    2,
                )

                logger.info(
                    "VectorStore loaded: %s chunks, %.2fms",
                    len(self.chunks),
                    self._stats["load_time_ms"],
                )

                return True

            except Exception:
                logger.exception(
                    "VectorStore load failed"
                )

                self.index = None
                self.chunks = []

                if self.bm25_index:
                    self.bm25_index.clear()

                self._is_loaded = False

                self._stats[
                    "total_chunks"
                ] = 0

                return False

    # -------------------------------------------------------------------------
    # BM25 maintenance
    # -------------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 from the current chunk collection."""
        if not self.enable_bm25 or not self.bm25_index:
            return

        documents = [
            chunk.get("content", "")
            for chunk in self.chunks
        ]

        doc_ids = [
            _chunk_id(chunk, index)
            for index, chunk in enumerate(self.chunks)
        ]

        self.bm25_index.add_documents(
            documents,
            doc_ids,
        )

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[Any] = None,
        texts: Optional[List[str]] = None,
        organization_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Add chunks and their embeddings.

        If embeddings are omitted, the configured embedder is used.
        """
        if not chunks:
            return

        if not isinstance(
            chunks,
            (list, tuple),
        ):
            raise RAGException(
                "chunks must be a list or tuple"
            )

        if len(chunks) > MAX_BATCH_SIZE:
            raise RAGException(
                f"Cannot add more than "
                f"{MAX_BATCH_SIZE} chunks in one operation"
            )

        if texts is not None:
            if not isinstance(
                texts,
                (list, tuple),
            ):
                raise RAGException(
                    "texts must be a list or tuple"
                )

            if len(texts) != len(chunks):
                raise RAGException(
                    "texts and chunks must have equal lengths"
                )

        target_org = _normalize_organization_id(organization_id or self.organization_id)
        target_namespace = str(namespace or self.namespace).strip() or "policy"

        normalized_chunks: List[
            Dict[str, Any]
        ] = []

        for index, chunk in enumerate(chunks):
            normalized = _safe_chunk(chunk)

            if normalized is None:
                raise RAGException(
                    f"Invalid chunk at index {index}"
                )

            metadata = dict(normalized.get("metadata", {}))
            existing_org = metadata.get("organization_id", target_org)
            if _normalize_organization_id(existing_org) != target_org:
                raise RAGException(
                    f"Chunk at index {index} belongs to a different organization"
                )
            metadata["organization_id"] = target_org
            metadata["namespace"] = str(metadata.get("namespace", target_namespace)).strip() or target_namespace
            normalized["metadata"] = metadata
            normalized_chunks.append(normalized)

        # ---------------------------------------------------------------------
        # Embeddings
        # ---------------------------------------------------------------------

        if embeddings is None:
            if self.embedder is None:
                raise RetrievalError(
                    "No embeddings supplied and no embedder configured"
                )

            texts_to_embed = [
                chunk["content"]
                for chunk in normalized_chunks
            ]

            try:
                embeddings = self.embedder.encode(
                    texts_to_embed,
                    batch_size=min(
                        32,
                        len(texts_to_embed),
                    ),
                    show_progress_bar=False,
                )
            except Exception as exc:
                logger.exception(
                    "Embedding generation failed"
                )

                raise RetrievalError(
                    "Failed to generate document embeddings"
                ) from exc

        embedding_array = _normalize_embeddings(
            embeddings,
            expected_count=len(normalized_chunks),
            dimension=self.dimension,
        )

        with self._lock:
            if self._shutdown:
                raise RAGException(
                    "VectorStore is shut down"
                )

            start_time = time.monotonic()

            if self.index is None:
                self.index = self._create_index()

            self._validate_index()

            # Validate IDs before changing FAISS or metadata.
            self._validate_unique_chunk_ids(
                normalized_chunks
            )

            starting_index = len(self.chunks)

            prepared_chunks: List[
                Dict[str, Any]
            ] = []

            for offset, chunk in enumerate(
                normalized_chunks
            ):
                chunk_copy = dict(chunk)

                metadata = dict(
                    chunk_copy.get(
                        "metadata",
                        {},
                    )
                )

                chunk_id = _chunk_id(
                    chunk_copy,
                    starting_index + offset,
                )

                metadata["chunk_id"] = chunk_id
                chunk_copy["metadata"] = metadata

                chunk_copy.pop(
                    "embedding",
                    None,
                )

                prepared_chunks.append(
                    chunk_copy
                )

            # -----------------------------------------------------------------
            # Transaction snapshot
            # -----------------------------------------------------------------

            old_index = self.index
            old_chunks = self.chunks
            old_bm25_corpus = (
                list(self.bm25_index._corpus)
                if self.bm25_index is not None
                else []
            )
            old_bm25_ids = (
                list(self.bm25_index._doc_ids)
                if self.bm25_index is not None
                else []
            )

            try:
                # Clone the index so a failed mutation can be rolled back.
                index_snapshot = faiss.clone_index(
                    self.index
                )

                self.index.add(
                    embedding_array
                )

                self.chunks.extend(
                    prepared_chunks
                )

                if self.enable_bm25 and self.bm25_index:
                    self._rebuild_bm25()

                self._validate_index()

            except Exception:
                logger.exception(
                    "VectorStore ingestion failed; rolling back"
                )

                # Restore FAISS.
                self.index = index_snapshot

                # Restore metadata.
                self.chunks = old_chunks

                # Restore BM25.
                if self.bm25_index is not None:
                    self.bm25_index.clear()

                    if old_bm25_corpus:
                        self.bm25_index.add_documents(
                            old_bm25_corpus,
                            old_bm25_ids,
                        )

                # old_index is intentionally retained only as a sanity check.
                if old_index is None:
                    self.index = None

                raise

            self._stats[
                "total_chunks"
            ] = len(self.chunks)

            self._stats[
                "last_updated"
            ] = _utc_now_iso()

            self._stats[
                "add_count"
            ] += 1

            elapsed_ms = (
                time.monotonic()
                - start_time
            ) * 1000

            logger.info(
                "Added %s chunks: %.1fms, total=%s",
                len(prepared_chunks),
                elapsed_ms,
                len(self.chunks),
            )

    # -------------------------------------------------------------------------
    # Vector search
    # -------------------------------------------------------------------------

    def search(
        self,
        query: Union[
            str,
            List[float],
            np.ndarray,
        ],
        top_k: Optional[int] = None,
        organization_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search FAISS within the requested organization/namespace.

        String queries are embedded using the configured embedder.
        """
        default_top_k = _setting_int(
            "TOP_K",
            DEFAULT_TOP_K,
            minimum=1,
        )

        top_k = _safe_top_k(
            top_k,
            default_top_k,
        )

        start_time = time.monotonic()

        with self._lock:
            if (
                self.index is None
                or not self.chunks
            ):
                logger.debug(
                    "VectorStore is empty"
                )
                return []

            self._validate_index()

            if isinstance(query, str):
                query = _normalize_query(query)

                if self.embedder is None:
                    raise RetrievalError(
                        "No embedder configured for string query"
                    )

                try:
                    if hasattr(
                        self.embedder,
                        "encode_query",
                    ):
                        query_embedding = (
                            self.embedder.encode_query(
                                query
                            )
                        )
                    else:
                        query_embedding = (
                            self.embedder.encode(
                                query,
                                show_progress_bar=False,
                            )
                        )

                except Exception as exc:
                    logger.exception(
                        "Query embedding failed"
                    )

                    raise RetrievalError(
                        "Failed to encode search query"
                    ) from exc

            else:
                query_embedding = query

            query_array = _normalize_query_embedding(
                query_embedding,
                self.dimension,
            )

            target_org = _normalize_organization_id(organization_id or self.organization_id)
            target_namespace = str(namespace or self.namespace).strip() or self.namespace
            # Search a larger candidate pool because tenant/namespace filtering
            # happens after FAISS ranking.
            k = min(max(top_k * 10, top_k), len(self.chunks))

            distances, indices = self.index.search(
                query_array,
                k,
            )

            results: List[
                Dict[str, Any]
            ] = []

            for distance, index in zip(
                distances[0],
                indices[0],
            ):
                index = int(index)

                if (
                    index < 0
                    or index >= len(self.chunks)
                ):
                    continue

                source_chunk = self.chunks[index]
                if _chunk_organization_id(source_chunk) != target_org:
                    continue
                if _chunk_namespace(source_chunk) != target_namespace:
                    continue

                chunk = dict(source_chunk)

                try:
                    distance_value = float(distance)
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if not np.isfinite(distance_value):
                    continue

                if self.metric_type == "ip":
                    score = distance_value
                else:
                    score = (
                        1.0
                        / (
                            1.0
                            + max(
                                distance_value,
                                0.0,
                            )
                        )
                    )

                chunk["score"] = float(score)
                chunk["vector_distance"] = distance_value

                results.append(chunk)
                if len(results) >= top_k:
                    break

            elapsed_ms = (
                time.monotonic()
                - start_time
            ) * 1000

            self._stats[
                "search_count"
            ] += 1

            count = self._stats[
                "search_count"
            ]

            previous_avg = self._stats[
                "avg_search_time_ms"
            ]

            self._stats[
                "avg_search_time_ms"
            ] = (
                (
                    previous_avg * (count - 1)
                )
                + elapsed_ms
            ) / count

            return results

    # -------------------------------------------------------------------------
    # BM25 search
    # -------------------------------------------------------------------------

    def bm25_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        organization_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[
        Tuple[
            Dict[str, Any],
            float,
        ]
    ]:
        """Search BM25 within the requested organization/namespace."""
        if not self.enable_bm25:
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

        query = _normalize_query(query)

        with self._lock:
            if (
                self.bm25_index is None
                or not self.bm25_index.is_available
            ):
                return []

            target_org = _normalize_organization_id(organization_id or self.organization_id)
            target_namespace = str(namespace or self.namespace).strip() or self.namespace
            bm25_results = self.bm25_index.search(
                query,
                top_k=min(MAX_TOP_K, max(top_k * 10, top_k)),
            )

            chunk_by_id: Dict[
                str,
                Dict[str, Any],
            ] = {}

            for index, chunk in enumerate(
                self.chunks
            ):
                chunk_by_id[
                    _chunk_id(
                        chunk,
                        index,
                    )
                ] = chunk

            results: List[
                Tuple[
                    Dict[str, Any],
                    float,
                ]
            ] = []

            for doc_id, score in bm25_results:
                chunk = chunk_by_id.get(
                    str(doc_id)
                )

                if chunk is None:
                    continue
                if _chunk_organization_id(chunk) != target_org:
                    continue
                if _chunk_namespace(chunk) != target_namespace:
                    continue

                chunk_copy = dict(chunk)

                chunk_copy["score"] = float(score)
                chunk_copy["bm25_score"] = float(score)

                results.append(
                    (
                        chunk_copy,
                        float(score),
                    )
                )
                if len(results) >= top_k:
                    break

            return results

    # -------------------------------------------------------------------------
    # Vector search compatibility
    # -------------------------------------------------------------------------

    def vector_search(
        self,
        query: Union[
            str,
            List[float],
            np.ndarray,
        ],
        top_k: Optional[int] = None,
        organization_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[
        Tuple[
            Dict[str, Any],
            float,
        ]
    ]:
        """
        Vector search returning ``(chunk, score)`` tuples.

        This is the interface expected by ``hybrid_search.py``.
        """
        chunks = self.search(
            query,
            top_k,
            organization_id=organization_id,
            namespace=namespace,
        )

        return [
            (
                chunk,
                float(
                    chunk.get(
                        "score",
                        0.0,
                    )
                ),
            )
            for chunk in chunks
        ]

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(
        self,
        backup: bool = True,
    ) -> bool:
        """
        Persist FAISS, chunks, and BM25 data.

        Live files are never moved out of place before a successful atomic
        replacement. Backups are copies of the previous known-good files.
        """
        with self._lock:
            if (
                self.index is None
                or not self.chunks
            ):
                logger.warning(
                    "Nothing to save - vector store is empty"
                )
                return False

            self._validate_index()

            start_time = time.monotonic()

            try:
                # -------------------------------------------------------------
                # Prepare chunks
                # -------------------------------------------------------------

                chunks_to_save: List[
                    Dict[str, Any]
                ] = []

                for index, chunk in enumerate(
                    self.chunks
                ):
                    chunk_copy = dict(chunk)

                    metadata = dict(
                        chunk_copy.get(
                            "metadata",
                            {},
                        )
                    )

                    metadata["chunk_id"] = _chunk_id(
                        chunk_copy,
                        index,
                    )

                    chunk_copy["metadata"] = metadata

                    chunk_copy.pop(
                        "embedding",
                        None,
                    )

                    chunks_to_save.append(
                        chunk_copy
                    )

                # Validate uniqueness before writing.
                ids = [
                    _chunk_id(
                        chunk,
                        index,
                    )
                    for index, chunk in enumerate(
                        chunks_to_save
                    )
                ]

                if len(ids) != len(set(ids)):
                    raise RAGException(
                        "Cannot save vector store with duplicate chunk IDs"
                    )

                # -------------------------------------------------------------
                # Backups
                # -------------------------------------------------------------

                if backup:
                    _copy_backup(
                        self.index_path,
                        self.index_path.with_name(
                            f"{self.index_path.name}.bak"
                        ),
                    )

                    _copy_backup(
                        self.chunks_path,
                        self.chunks_path.with_name(
                            f"{self.chunks_path.name}.bak"
                        ),
                    )

                    if self.bm25_path.exists():
                        _copy_backup(
                            self.bm25_path,
                            self.bm25_path.with_name(
                                f"{self.bm25_path.name}.bak"
                            ),
                        )

                # -------------------------------------------------------------
                # FAISS
                # -------------------------------------------------------------

                _write_faiss_atomic(
                    self.index_path,
                    self.index,
                )

                # -------------------------------------------------------------
                # Chunks
                # -------------------------------------------------------------

                _write_pickle_atomic(
                    self.chunks_path,
                    chunks_to_save,
                )

                # -------------------------------------------------------------
                # BM25 source data
                # -------------------------------------------------------------

                if (
                    self.enable_bm25
                    and self.bm25_index is not None
                ):
                    bm25_data = (
                        self.bm25_index.get_persistable_data()
                    )

                    _write_pickle_atomic(
                        self.bm25_path,
                        bm25_data,
                    )

                elif self.bm25_path.exists():
                    self.bm25_path.unlink()

                _write_pickle_atomic(
                    self.manifest_path,
                    {
                        "schema_version": VECTOR_STORE_SCHEMA_VERSION,
                        "organization_id": self.organization_id,
                        "namespace": self.namespace,
                        "dimension": self.dimension,
                        "metric_type": self.metric_type,
                        "chunk_count": len(chunks_to_save),
                        "updated_at": _utc_now_iso(),
                    },
                )

                elapsed_ms = (
                    time.monotonic()
                    - start_time
                ) * 1000

                self._stats[
                    "last_updated"
                ] = _utc_now_iso()

                logger.info(
                    "VectorStore saved: %s chunks, %.1fms",
                    len(self.chunks),
                    elapsed_ms,
                )

                return True

            except Exception:
                logger.exception(
                    "VectorStore save failed"
                )

                # Atomic replacement means previously existing live files
                # remain available if a later write fails. Backups are retained
                # separately for manual recovery.

                return False

    # -------------------------------------------------------------------------
    # Clear
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Clear memory and persisted vector-store data."""
        with self._lock:
            self.index = None
            self.chunks = []

            if self.bm25_index:
                self.bm25_index.clear()

            for path in (
                self.index_path,
                self.chunks_path,
                self.bm25_path,
                self.manifest_path,
            ):
                try:
                    path.unlink(
                        missing_ok=True
                    )
                except Exception:
                    logger.exception(
                        "Failed deleting %s",
                        path,
                    )

            self._is_loaded = False

            self._stats[
                "total_chunks"
            ] = 0

            self._stats[
                "last_updated"
            ] = _utc_now_iso()

        logger.info(
            "VectorStore cleared"
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the vector store is loaded and internally consistent."""
        with self._lock:
            return (
                self._is_loaded
                and self.index is not None
                and self.index.d == self.dimension
                and self.index.ntotal == len(self.chunks)
            )

    @property
    def chunk_count(self) -> int:
        """Return number of stored chunks."""
        with self._lock:
            return len(self.chunks)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return vector-store statistics."""
        with self._lock:
            return {
                **self._stats,
                "db_path": str(self.db_path),
                "organization_id": self.organization_id,
                "namespace": self.namespace,
                "schema_version": VECTOR_STORE_SCHEMA_VERSION,
                "dimension": self.dimension,
                "metric_type": self.metric_type,
                "enable_bm25": self.enable_bm25,
                "bm25_available": (
                    self.bm25_index.is_available
                    if self.bm25_index
                    else False
                ),
                "index_type": (
                    type(self.index).__name__
                    if self.index is not None
                    else None
                ),
                "faiss_vectors": (
                    int(self.index.ntotal)
                    if self.index is not None
                    else 0
                ),
                "embedder_configured": (
                    self.embedder is not None
                ),
                "is_loaded": self.is_loaded,
                "shutdown": self._shutdown,
            }

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release in-memory resources safely."""
        with self._lock:
            if self._shutdown:
                return

            self._shutdown = True

            self.index = None

            if self.bm25_index:
                self.bm25_index.clear()

            self._is_loaded = False

        logger.info(
            "VectorStore shutdown complete"
        )


# =============================================================================
# GLOBAL INSTANCE MANAGEMENT
# =============================================================================

_vector_store: Optional[
    VectorStore
] = None

_vs_lock = threading.Lock()


def get_vector_store(
    db_path: Optional[
        Union[
            str,
            Path,
        ]
    ] = None,
    dimension: Optional[int] = None,
    enable_bm25: bool = True,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> VectorStore:
    """Get or create the global vector-store instance."""
    global _vector_store

    requested_path = Path(
        db_path
        or getattr(
            settings,
            "VECTOR_DB_PATH",
            "data/vector_db",
        )
    )

    requested_dimension = (
        int(dimension)
        if dimension is not None
        else _setting_int(
            "EMBEDDING_DIMENSION",
            DEFAULT_DIMENSION,
            minimum=1,
        )
    )

    with _vs_lock:
        if _vector_store is None:
            _vector_store = VectorStore(
                db_path=requested_path,
                dimension=requested_dimension,
                enable_bm25=enable_bm25,
                organization_id=organization_id,
                namespace=namespace,
            )

            _vector_store.load()

        else:
            if (
                _vector_store.db_path.resolve()
                != requested_path.resolve()
            ):
                raise RAGException(
                    "Global VectorStore is already initialized "
                    f"with path '{_vector_store.db_path}', "
                    f"but path '{requested_path}' was requested"
                )

            if _vector_store.organization_id != _normalize_organization_id(organization_id or _vector_store.organization_id):
                raise RAGException("Global VectorStore organization cannot change after initialization")

            if _vector_store.namespace != (str(namespace).strip() or "policy"):
                raise RAGException("Global VectorStore namespace cannot change after initialization")

            if (
                _vector_store.dimension
                != requested_dimension
            ):
                raise RAGException(
                    "Global VectorStore is already initialized "
                    f"with dimension {_vector_store.dimension}, "
                    f"but dimension {requested_dimension} was requested"
                )

            if (
                _vector_store.enable_bm25
                != bool(enable_bm25)
            ):
                raise RAGException(
                    "Global VectorStore BM25 configuration "
                    "cannot be changed after initialization"
                )

            if _vector_store._shutdown:
                raise RAGException(
                    "Global VectorStore has been shut down; "
                    "call reset_vector_store() before reusing it"
                )

        return _vector_store


def reset_vector_store() -> None:
    """Reset the global vector store, primarily for tests."""
    global _vector_store

    with _vs_lock:
        vector_store = _vector_store
        _vector_store = None

    if vector_store is not None:
        vector_store.shutdown()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def vector_search(
    query: Union[
        str,
        List[float],
        np.ndarray,
    ],
    top_k: Optional[int] = None,
    organization_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Quick vector search."""
    return get_vector_store().search(
        query,
        top_k,
        organization_id=organization_id,
        namespace=namespace,
    )


def bm25_search(
    query: str,
    top_k: Optional[int] = None,
    organization_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> List[
    Tuple[
        Dict[str, Any],
        float,
    ]
]:
    """Quick BM25 search."""
    return get_vector_store().bm25_search(
        query,
        top_k,
        organization_id=organization_id,
        namespace=namespace,
    )


def add_chunks_to_store(
    chunks: List[Dict[str, Any]],
    embeddings: Optional[Any] = None,
    texts: Optional[List[str]] = None,
    organization_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> None:
    """Add chunks to the global vector store."""
    get_vector_store().add_chunks(
        chunks,
        embeddings,
        texts,
        organization_id=organization_id,
        namespace=namespace,
    )


def save_vector_store(
    backup: bool = True,
) -> bool:
    """Save the global vector store."""
    return get_vector_store().save(
        backup
    )


def get_vector_store_stats() -> Dict[str, Any]:
    """Return global vector-store statistics."""
    return get_vector_store().get_stats()


# =============================================================================
# TEST / DEMO
# =============================================================================

def test_vector_store() -> None:
    """Run a local vector-store smoke test."""
    print("\n🗄️ Testing Vector Store\n")
    print("=" * 70)

    with tempfile.TemporaryDirectory(
        prefix="policyguard_vector_test_"
    ) as temp_dir:

        vs = VectorStore(
            db_path=temp_dir,
            dimension=384,
            enable_bm25=True,
        )

        print("✅ VectorStore initialized")
        print(f"📂 Path: {vs.db_path}")
        print(f"📐 Dimension: {vs.dimension}")
        print(f"🔍 BM25 enabled: {vs.enable_bm25}")

        mock_chunks = [
            {
                "content": (
                    "Employees are entitled to 20 days "
                    "of paid leave per year."
                ),
                "metadata": {
                    "source": "policy.pdf",
                    "page": 3,
                },
            },
            {
                "content": (
                    "All leave requests must be submitted "
                    "at least 2 weeks in advance."
                ),
                "metadata": {
                    "source": "policy.pdf",
                    "page": 3,
                },
            },
            {
                "content": (
                    "The company offers health insurance "
                    "and professional development benefits."
                ),
                "metadata": {
                    "source": "benefits.pdf",
                    "page": 1,
                },
            },
        ]

        rng = np.random.default_rng(seed=42)

        embeddings = rng.normal(
            size=(
                len(mock_chunks),
                vs.dimension,
            )
        ).astype(np.float32)

        print("\n📝 Test 1: Add chunks")
        print("-" * 70)

        vs.add_chunks(
            mock_chunks,
            embeddings=embeddings,
        )

        print(f"   Chunks: {vs.chunk_count}")
        print(
            f"   FAISS vectors: "
            f"{vs.index.ntotal if vs.index is not None else 0}"
        )

        print("\n📄 Test 2: BM25 search")
        print("-" * 70)

        bm25_results = vs.bm25_search(
            "leave policy",
            top_k=3,
        )

        print(
            f"   Results: {len(bm25_results)}"
        )

        for index, (
            chunk,
            score,
        ) in enumerate(
            bm25_results,
            start=1,
        ):
            print(
                f"   {index}. "
                f"score={score:.3f} "
                f"[{_chunk_source(chunk)}]"
            )

        print("\n🔍 Test 3: Vector search")
        print("-" * 70)

        vector_results = vs.search(
            embeddings[0],
            top_k=3,
        )

        print(
            f"   Results: {len(vector_results)}"
        )

        for index, chunk in enumerate(
            vector_results,
            start=1,
        ):
            print(
                f"   {index}. "
                f"score={chunk.get('score', 0.0):.4f} "
                f"[{_chunk_source(chunk)}]"
            )

        print("\n💾 Test 4: Persistence")
        print("-" * 70)

        saved = vs.save(
            backup=False
        )

        print(
            f"   Saved: {saved}"
        )

        print("\n📂 Test 5: Reload")
        print("-" * 70)

        reloaded = VectorStore(
            db_path=temp_dir,
            dimension=384,
            enable_bm25=True,
        )

        loaded = reloaded.load()

        print(
            f"   Loaded: {loaded}"
        )
        print(
            f"   Chunks after reload: "
            f"{reloaded.chunk_count}"
        )

        print("\n📊 Test 6: Statistics")
        print("-" * 70)

        for key, value in reloaded.get_stats().items():
            print(
                f"   {key}: {value}"
            )

        print("\n" + "=" * 70)
        print("✅ Vector store test complete!\n")


if __name__ == "__main__":
    test_vector_store()

