#!/usr/bin/env python3
"""
PolicyGuard AI - Lightweight Embedding Model Singleton
=======================================================

Centralized, thread-safe embedding model manager.

Production runtime:
- FastEmbed / ONNX Runtime instead of SentenceTransformers / PyTorch
- BAAI/bge-small-en-v1.5 by default
- Same 384-dimensional embedding space
- Lazy singleton lifecycle
- Conservative CPU batch sizing
- Explicit L2 normalization for cosine-compatible vectors
- Query/document compatibility with the existing public API
- Configurable Hugging Face/FastEmbed cache directory
- Performance statistics
- Safe reset/shutdown

This module intentionally avoids importing PyTorch or sentence-transformers.
That is important for low-memory deployments such as a 512 MB Render instance.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# =============================================================================
# PROJECT PATH
# =============================================================================

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# DEVICE / BATCH CONFIGURATION
# =============================================================================


def detect_optimal_device() -> str:
    """
    Resolve the requested FastEmbed execution device without importing PyTorch.

    FastEmbed's normal CPU runtime uses ONNX Runtime. CUDA is only selected
    when explicitly requested, because CUDA support requires a compatible
    FastEmbed GPU/ONNX Runtime installation.

    Returns:
        "cuda" or "cpu".
    """
    requested = str(
        os.getenv("EMBEDDING_DEVICE", "cpu") or "cpu"
    ).strip().lower()

    if requested in {"cuda", "gpu"}:
        logger.info("CUDA embedding requested through EMBEDDING_DEVICE")
        return "cuda"

    if requested in {"auto"}:
        # Keep low-memory deployments deterministic. FastEmbed CPU is the
        # safe default; GPU can be explicitly enabled with EMBEDDING_DEVICE.
        logger.info("FastEmbed auto device resolved to CPU-safe mode")
        return "cpu"

    return "cpu"


def get_optimal_batch_size(
    device: str,
    model_name: str,
) -> int:
    """
    Select a conservative batch size.

    FastEmbed itself supports larger batches, but PolicyGuardAI is designed
    to coexist with Streamlit, RAG, parsing, and LLM client components inside
    a constrained deployment. Keeping this conservative protects memory.
    """
    model_lower = model_name.lower()

    if "large" in model_lower:
        base_size = 8
    elif "base" in model_lower:
        base_size = 16
    elif "small" in model_lower or "mini" in model_lower:
        base_size = 32
    else:
        base_size = 16

    if device == "cuda":
        return base_size * 2

    return max(4, base_size // 2)


def _l2_normalize(
    embeddings: np.ndarray,
) -> np.ndarray:
    """L2-normalize embedding rows for cosine-compatible similarity."""
    array = np.asarray(embeddings, dtype=np.float32)

    if array.size == 0:
        return array

    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        if norm > 0.0:
            return array / norm
        return array

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, np.finfo(np.float32).eps)

    return array / norms


# =============================================================================
# EMBEDDER SINGLETON
# =============================================================================


class EmbedderSingleton:
    """
    Thread-safe singleton around FastEmbed TextEmbedding.

    The model is initialized when the first global embedder is requested.
    """

    _instance: Optional["EmbedderSingleton"] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "EmbedderSingleton":
        """Create exactly one singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self) -> None:
        """Initialize configuration and load the FastEmbed model once."""
        if getattr(self, "_configured", False):
            return

        with self._instance_lock:
            if getattr(self, "_configured", False):
                return

            self._configured = True

            self.model_name = (
                os.getenv(
                    "EMBEDDING_MODEL",
                    getattr(
                        settings,
                        "EMBEDDING_MODEL",
                        "BAAI/bge-small-en-v1.5",
                    ),
                )
                or "BAAI/bge-small-en-v1.5"
            ).strip()

            self.cache_dir = Path(
                os.getenv(
                    "HF_CACHE_DIR",
                    str(
                        getattr(
                            settings,
                            "HF_CACHE_DIR",
                            project_root / ".cache" / "huggingface",
                        )
                    ),
                )
            )

            self.cache_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Keep the application's cache policy consistent across Hugging
            # Face-compatible components and FastEmbed.
            os.environ.setdefault(
                "HF_HOME",
                str(self.cache_dir),
            )
            os.environ.setdefault(
                "HF_HUB_DISABLE_SYMLINKS_WARNING",
                "1",
            )

            self._device = detect_optimal_device()
            self._batch_size = get_optimal_batch_size(
                self._device,
                self.model_name,
            )

            configured_batch = os.getenv(
                "EMBEDDING_BATCH_SIZE",
            )
            if configured_batch:
                try:
                    self._batch_size = max(
                        1,
                        min(256, int(configured_batch)),
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid EMBEDDING_BATCH_SIZE=%r; using %s",
                        configured_batch,
                        self._batch_size,
                    )

            try:
                self.max_seq_length = int(
                    os.getenv(
                        "EMBEDDING_MAX_SEQ_LENGTH",
                        "512",
                    )
                )
            except (TypeError, ValueError):
                self.max_seq_length = 512

            self.max_seq_length = max(
                1,
                min(512, self.max_seq_length),
            )

            self.normalize_embeddings = True

            self._model = None
            self._dimension: Optional[int] = None
            self._warmup_complete = False
            self._initialized = False
            self._initialization_error: Optional[str] = None

            self._stats: Dict[str, Any] = {
                "total_encodings": 0,
                "total_time_ms": 0.0,
                "last_error": None,
            }

            self._model_lock = threading.RLock()

            logger.info(
                "FastEmbed configuration: model=%s device=%s batch=%s",
                self.model_name,
                self._device,
                self._batch_size,
            )

            self._load_model()

    # -------------------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------------------

    def _import_fastembed(self):
        """Import FastEmbed with a useful error message."""
        try:
            from fastembed import TextEmbedding

            return TextEmbedding

        except ImportError as exc:
            message = (
                "fastembed is not installed. "
                "Install it with: pip install fastembed"
            )

            self._initialization_error = message
            self._stats["last_error"] = message
            logger.error(message)

            raise RuntimeError(message) from exc

    def _create_model(self):
        """Create the FastEmbed model with a CPU-safe fallback."""
        TextEmbedding = self._import_fastembed()

        kwargs: Dict[str, Any] = {
            "model_name": self.model_name,
            "cache_dir": str(self.cache_dir),
            "lazy_load": False,
        }

        if self._device == "cuda":
            kwargs["cuda"] = True

        try:
            return TextEmbedding(**kwargs)
        except Exception:
            if self._device != "cpu":
                logger.exception(
                    "FastEmbed initialization failed on %s; "
                    "retrying on CPU",
                    self._device,
                )
                self._device = "cpu"
                self._batch_size = get_optimal_batch_size(
                    "cpu",
                    self.model_name,
                )
                kwargs.pop("cuda", None)
                return TextEmbedding(**kwargs)

            raise

    def _load_model(self) -> None:
        """Load the FastEmbed model with safe fallback handling."""
        with self._model_lock:
            if self._model is not None:
                return

            start_time = time.perf_counter()

            try:
                logger.info(
                    "Loading FastEmbed model '%s' on %s",
                    self.model_name,
                    self._device,
                )

                self._model = self._create_model()

                self._configure_model()

                load_time = time.perf_counter() - start_time

                logger.info(
                    "FastEmbed model loaded in %.2fs "
                    "(dimension=%s, device=%s)",
                    load_time,
                    self.dimension,
                    self._device,
                )

                self._warmup()

                self._initialized = True
                self._initialization_error = None

            except Exception as exc:
                logger.exception(
                    "FastEmbed model loading failed: %s",
                    exc,
                )

                self._stats["last_error"] = str(exc)
                self._initialization_error = str(exc)
                self._dispose_model()
                self._initialized = False

    def _configure_model(self) -> None:
        """Determine and validate the model's output dimension."""
        if self._model is None:
            return

        dimension = None

        # FastEmbed exposes model metadata through different internal paths
        # across releases. Prefer public metadata when available and fall back
        # to a tiny inference only when necessary.
        try:
            metadata = getattr(self._model, "model", None)
            if metadata is not None:
                dimension = getattr(
                    metadata,
                    "embedding_size",
                    None,
                )
        except Exception:
            dimension = None

        if dimension is None:
            try:
                sample = next(
                    self._model.embed(
                        ["PolicyGuard AI dimension probe"],
                        batch_size=1,
                    )
                )
                dimension = int(np.asarray(sample).shape[-1])
            except Exception as exc:
                logger.error(
                    "Could not determine FastEmbed embedding dimension: %s",
                    exc,
                )
                raise

        self._dimension = int(dimension)

        if self._dimension <= 0:
            raise RuntimeError(
                f"Invalid embedding dimension: {self._dimension}"
            )

    def _warmup(self) -> None:
        """Run a tiny inference to reduce first-request latency."""
        if self._model is None or self._warmup_complete:
            return

        try:
            logger.info("Running FastEmbed embedding warmup")

            vectors = list(
                self._model.embed(
                    [
                        "PolicyGuard AI warmup",
                        "initialization test",
                    ],
                    batch_size=2,
                )
            )

            if len(vectors) != 2:
                raise RuntimeError(
                    "FastEmbed warmup returned an unexpected number "
                    "of vectors"
                )

            self._warmup_complete = True

            logger.info("FastEmbed embedding warmup complete")

        except Exception as exc:
            logger.warning(
                "FastEmbed embedding warmup failed: %s",
                exc,
            )

    # -------------------------------------------------------------------------
    # ENCODING
    # -------------------------------------------------------------------------

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: Optional[bool] = None,
        **kwargs: Any,
    ) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        """
        Encode one or more texts.

        The public API intentionally mirrors the previous SentenceTransformer
        wrapper so callers throughout PolicyGuardAI do not need to change.

        FastEmbed returns generators of NumPy arrays. This method materializes
        them into a NumPy matrix because the vector-store and RAG layers expect
        matrix-style embeddings.
        """
        if self._model is None:
            logger.error("Embedding model is unavailable")
            return None

        single_input = isinstance(texts, str)

        if single_input:
            if not texts.strip():
                logger.warning("Empty query passed to embedder")
                return None

            input_texts = [texts.strip()]

        else:
            if texts is None:
                return None

            input_texts = list(texts)

            if not input_texts:
                if convert_to_numpy:
                    return np.empty(
                        (0, self.dimension),
                        dtype=np.float32,
                    )
                return []

            if any(
                not isinstance(text, str)
                for text in input_texts
            ):
                raise TypeError(
                    "All embedding inputs must be strings"
                )

            input_texts = [
                text.strip()
                for text in input_texts
            ]

            if any(not text for text in input_texts):
                raise ValueError(
                    "Embedding input contains an empty string"
                )

        effective_batch_size = (
            max(1, int(batch_size))
            if batch_size is not None
            else self.batch_size
        )

        effective_normalize = (
            self.normalize_embeddings
            if normalize_embeddings is None
            else bool(normalize_embeddings)
        )

        start = time.perf_counter()

        try:
            with self._model_lock:
                # FastEmbed exposes batch_size and parallel as runtime
                # arguments. Unknown SentenceTransformer-only kwargs are
                # intentionally ignored for compatibility rather than passed
                # into ONNX Runtime.
                embeddings_iter = self._model.embed(
                    input_texts,
                    batch_size=effective_batch_size,
                )
                embeddings = np.asarray(
                    list(embeddings_iter),
                    dtype=np.float32,
                )

            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            if effective_normalize:
                embeddings = _l2_normalize(embeddings)

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0

            self._stats["total_encodings"] += len(input_texts)
            self._stats["total_time_ms"] += elapsed_ms
            self._stats["last_error"] = None

            if elapsed_ms > 1000:
                logger.warning(
                    "Slow embedding operation: %s texts in %.0fms",
                    len(input_texts),
                    elapsed_ms,
                )

            if single_input:
                vector = np.asarray(
                    embeddings[0],
                    dtype=np.float32,
                )

                if convert_to_numpy:
                    return vector

                return vector

            if convert_to_numpy:
                return np.asarray(
                    embeddings,
                    dtype=np.float32,
                )

            return [
                np.asarray(vector, dtype=np.float32)
                for vector in embeddings
            ]

        except Exception as exc:
            self._stats["last_error"] = str(exc)

            logger.exception(
                "FastEmbed encoding failed: %s",
                exc,
            )

            return None

    def encode_query(
        self,
        query: str,
    ) -> Optional[np.ndarray]:
        """Encode a single RAG query with normalized output."""
        if not isinstance(query, str) or not query.strip():
            return None

        result = self.encode(
            query,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        if result is None:
            return None

        return np.asarray(
            result,
            dtype=np.float32,
        )

    def encode_documents(
        self,
        documents: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> Optional[np.ndarray]:
        """Encode document chunks for vector indexing."""
        if not documents:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        result = self.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        if result is None:
            return None

        return np.asarray(
            result,
            dtype=np.float32,
        )

    # -------------------------------------------------------------------------
    # PROPERTIES
    # -------------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Return the actual embedding dimension."""
        if self._dimension is not None:
            return self._dimension

        raise RuntimeError(
            "Embedding model dimension is unavailable"
        )

    @property
    def device(self) -> str:
        """Return the active FastEmbed execution device."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Return the configured default batch size."""
        return self._batch_size

    @property
    def is_available(self) -> bool:
        """Return whether a usable model is currently loaded."""
        return self._model is not None

    @property
    def initialization_error(self) -> Optional[str]:
        """Return the latest model initialization error."""
        return self._initialization_error

    # -------------------------------------------------------------------------
    # STATS
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return embedding performance and configuration statistics."""
        total = int(self._stats["total_encodings"])
        total_time = float(self._stats["total_time_ms"])

        try:
            dimension = self.dimension
        except RuntimeError:
            dimension = None

        return {
            "model": self.model_name,
            "runtime": "fastembed",
            "device": self._device,
            "batch_size": self._batch_size,
            "dimension": dimension,
            "available": self.is_available,
            "initialized": self._initialized,
            "warmup_complete": self._warmup_complete,
            "total_encodings": total,
            "total_time_ms": round(total_time, 2),
            "avg_time_per_encoding_ms": (
                round(total_time / total, 3)
                if total
                else 0.0
            ),
            "last_error": self._stats["last_error"],
            "cache_dir": str(self.cache_dir),
        }

    # -------------------------------------------------------------------------
    # RESET / SHUTDOWN
    # -------------------------------------------------------------------------

    def _dispose_model(self) -> None:
        """Release the current FastEmbed model reference."""
        model = self._model
        self._model = None

        if model is None:
            return

        try:
            del model
        except Exception as exc:
            logger.debug(
                "FastEmbed model cleanup warning: %s",
                exc,
            )

    def reset(self) -> None:
        """Release the loaded model while keeping the singleton object."""
        with self._model_lock:
            self._dispose_model()

            self._warmup_complete = False
            self._initialized = False
            self._dimension = None
            self._initialization_error = None

            logger.info("Embedder model reset")

    def shutdown(self) -> None:
        """Release model resources."""
        with self._model_lock:
            self._dispose_model()

            self._warmup_complete = False
            self._initialized = False
            self._dimension = None

            logger.info("Embedder shutdown complete")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================


embedder: Optional[EmbedderSingleton] = None
_embedder_lock = threading.Lock()


def get_embedder() -> EmbedderSingleton:
    """Return the process-wide lazy embedder singleton."""
    global embedder

    if embedder is None:
        with _embedder_lock:
            if embedder is None:
                embedder = EmbedderSingleton()

    return embedder


def reset_embedder() -> None:
    """Reset the global embedder instance."""
    global embedder

    with _embedder_lock:
        if embedder is not None:
            try:
                embedder.shutdown()
            except Exception as exc:
                logger.warning(
                    "Embedder shutdown during reset failed: %s",
                    exc,
                )

        embedder = None
        EmbedderSingleton._instance = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def encode_texts(
    texts: Union[str, List[str]],
    **kwargs: Any,
) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
    """Encode text using the global embedder."""
    return get_embedder().encode(
        texts,
        **kwargs,
    )


def encode_query(
    query: str,
) -> Optional[np.ndarray]:
    """Encode one query using the global embedder."""
    return get_embedder().encode_query(query)


def encode_documents(
    documents: List[str],
    **kwargs: Any,
) -> Optional[np.ndarray]:
    """Encode document chunks using the global embedder."""
    return get_embedder().encode_documents(
        documents,
        **kwargs,
    )


def get_embedder_stats() -> Dict[str, Any]:
    """Return global embedder statistics."""
    return get_embedder().get_stats()


# =============================================================================
# TEST / DEMO
# =============================================================================


def test_optimized_embedder() -> None:
    """Run a basic embedder smoke test."""
    print("\nPolicyGuard AI - FastEmbed Test\n")
    print("=" * 70)

    try:
        emb = get_embedder()

        print(f"Model: {emb.model_name}")
        print(f"Runtime: FastEmbed / ONNX")
        print(f"Device: {emb.device}")
        print(f"Batch size: {emb.batch_size}")
        print(f"Dimension: {emb.dimension}")
        print(f"Available: {emb.is_available}")

        print("=" * 70)

        query = "What is the company leave policy?"

        start = time.perf_counter()
        query_embedding = encode_query(query)
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        if query_embedding is not None:
            print(f"Query encoded in {elapsed_ms:.1f}ms")
            print(f"Shape: {query_embedding.shape}")
            print(f"Dtype: {query_embedding.dtype}")
            print(f"Norm: {np.linalg.norm(query_embedding):.4f}")
        else:
            print("Query encoding failed")

        documents = [
            "Employees receive annual leave according to company policy.",
            "The code of conduct defines workplace responsibilities.",
            "Managers should approve leave requests through the HR process.",
        ]

        start = time.perf_counter()
        document_embeddings = encode_documents(documents)
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        if document_embeddings is not None:
            print(f"\nBatch encoded in {elapsed_ms:.1f}ms")
            print(f"Shape: {document_embeddings.shape}")
            print(f"Dtype: {document_embeddings.dtype}")
            print(
                "Row norms:",
                np.round(
                    np.linalg.norm(
                        document_embeddings,
                        axis=1,
                    ),
                    4,
                ),
            )
        else:
            print("Document encoding failed")

        print("\nStats:")
        for key, value in emb.get_stats().items():
            print(f"  {key}: {value}")

        print("\nFastEmbed test complete.")

    except Exception as exc:
        logger.exception(
            "FastEmbed test failed: %s",
            exc,
        )
        print(f"Test failed: {exc}")


if __name__ == "__main__":
    test_optimized_embedder()