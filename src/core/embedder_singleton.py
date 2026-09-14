
#!/usr/bin/env python3
"""
PolicyGuard AI - Embedding Model Singleton
===========================================

Centralized, thread-safe embedding model manager.

Features:
- Single shared SentenceTransformer instance
- Lazy initialization
- CUDA / MPS / CPU device detection
- Automatic CPU fallback
- Configurable model and Hugging Face cache directory
- Batch encoding
- Normalized embeddings for cosine similarity
- Model warmup
- Performance statistics
- Safe reset/shutdown
- Consistent embedding dimension

The embedding model is controlled by config.settings so that the cache,
vector store, and RAG pipeline use the same embedding model.
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
# DEVICE DETECTION
# =============================================================================


def detect_optimal_device() -> str:
    """
    Detect the best available device.

    Priority:
        CUDA -> MPS -> CPU

    Returns:
        "cuda", "mps", or "cpu"
    """
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(
                "CUDA GPU detected for embedding acceleration"
            )
            return "cuda"

    except ImportError:
        logger.debug(
            "PyTorch is not installed; CUDA unavailable"
        )
    except Exception as exc:
        logger.warning(
            "CUDA detection failed: %s",
            exc,
        )

    try:
        import torch

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            logger.info(
                "Apple MPS detected for embedding acceleration"
            )
            return "mps"

    except ImportError:
        pass
    except Exception as exc:
        logger.warning(
            "MPS detection failed: %s",
            exc,
        )

    logger.info("Using CPU for embedding computation")
    return "cpu"


def get_optimal_batch_size(
    device: str,
    model_name: str,
) -> int:
    """
    Select a conservative batch size based on device/model size.

    The value is only a default. Individual callers can override it.
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

    if device == "mps":
        return base_size

    return max(4, base_size // 2)


# =============================================================================
# EMBEDDER SINGLETON
# =============================================================================


class EmbedderSingleton:
    """
    Thread-safe singleton around SentenceTransformer.

    The model is loaded only when the first EmbedderSingleton instance
    is requested.
    """

    _instance: Optional["EmbedderSingleton"] = None
    _instance_lock = threading.Lock()

    def __new__(
        cls,
    ) -> "EmbedderSingleton":
        """Create exactly one singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self) -> None:
        """
        Initialize configuration once.

        Actual model loading is performed here because get_embedder()
        itself is already lazy. Initialization is protected by a separate
        lock so concurrent callers cannot load the model twice.
        """
        if getattr(self, "_configured", False):
            return

        with self._instance_lock:
            if getattr(self, "_configured", False):
                return

            self._configured = True

            # -----------------------------------------------------------------
            # Configuration
            # -----------------------------------------------------------------

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
            )

            self.cache_dir = Path(
                os.getenv(
                    "HF_CACHE_DIR",
                    str(
                        getattr(
                            settings,
                            "HF_CACHE_DIR",
                            project_root
                            / ".cache"
                            / "huggingface",
                        )
                    ),
                )
            )

            self.cache_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Make Hugging Face / Sentence Transformers use the
            # application's configured cache location.
            os.environ.setdefault(
                "HF_HOME",
                str(self.cache_dir),
            )
            os.environ.setdefault(
                "SENTENCE_TRANSFORMERS_HOME",
                str(self.cache_dir),
            )

            self._device = detect_optimal_device()
            self._batch_size = get_optimal_batch_size(
                self._device,
                self.model_name,
            )

            self.max_seq_length = int(
                os.getenv(
                    "EMBEDDING_MAX_SEQ_LENGTH",
                    "512",
                )
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
                "Embedder configuration: model=%s device=%s batch=%s",
                self.model_name,
                self._device,
                self._batch_size,
            )

            self._load_model()

    # -------------------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------------------

    def _import_sentence_transformer(self):
        """Import SentenceTransformer with a useful error message."""
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer

        except ImportError as exc:
            message = (
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )

            self._initialization_error = message
            self._stats["last_error"] = message

            logger.error(message)

            raise RuntimeError(message) from exc

    def _load_model(self) -> None:
        """Load the embedding model with automatic CPU fallback."""
        with self._model_lock:
            if self._model is not None:
                return

            SentenceTransformer = self._import_sentence_transformer()

            start_time = time.perf_counter()

            try:
                logger.info(
                    "Loading embedding model '%s' on %s",
                    self.model_name,
                    self._device,
                )

                self._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(self.cache_dir),
                    device=self._device,
                )

                self._configure_model()

                load_time = (
                    time.perf_counter() - start_time
                )

                logger.info(
                    "Embedding model loaded in %.2fs "
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
                    "Embedding model loading failed on %s: %s",
                    self._device,
                    exc,
                )

                self._stats["last_error"] = str(exc)
                self._initialization_error = str(exc)

                # Release partially initialized model before fallback.
                self._dispose_model()

                # GPU/MPS failure should not make the complete application
                # unusable when CPU inference is possible.
                if self._device != "cpu":
                    logger.warning(
                        "Retrying embedding model on CPU"
                    )

                    try:
                        self._device = "cpu"
                        self._batch_size = get_optimal_batch_size(
                            "cpu",
                            self.model_name,
                        )

                        self._model = SentenceTransformer(
                            self.model_name,
                            cache_folder=str(self.cache_dir),
                            device="cpu",
                        )

                        self._configure_model()
                        self._warmup()

                        self._initialized = True
                        self._initialization_error = None

                        logger.info(
                            "Embedding model CPU fallback succeeded"
                        )

                        return

                    except Exception as fallback_exc:
                        logger.exception(
                            "CPU embedding fallback failed: %s",
                            fallback_exc,
                        )

                        self._stats["last_error"] = str(
                            fallback_exc
                        )
                        self._initialization_error = str(
                            fallback_exc
                        )
                        self._dispose_model()

                self._initialized = False

    def _configure_model(self) -> None:
        """Apply model-level configuration after loading."""
        if self._model is None:
            return

        try:
            self._model.max_seq_length = self.max_seq_length
        except Exception as exc:
            logger.debug(
                "Could not set max_seq_length: %s",
                exc,
            )

        try:
            dimension = (
                self._model
                .get_sentence_embedding_dimension()
            )

            if dimension is None:
                raise RuntimeError(
                    "Embedding model did not report a dimension"
                )

            self._dimension = int(dimension)

        except Exception as exc:
            logger.error(
                "Could not determine embedding dimension: %s",
                exc,
            )
            raise

        # Do not blindly call model.half(). Sentence-transformers models
        # can contain components for which manual FP16 conversion is not
        # appropriate. Let PyTorch/model configuration manage precision.
        if self._device == "cuda":
            logger.info(
                "CUDA embedding enabled using model-native precision"
            )

    def _warmup(self) -> None:
        """Run a small inference to reduce first-request latency."""
        if self._model is None or self._warmup_complete:
            return

        try:
            logger.info("Running embedding model warmup")

            self._model.encode(
                [
                    "PolicyGuard AI warmup",
                    "initialization test",
                ],
                batch_size=2,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            self._warmup_complete = True

            logger.info(
                "Embedding model warmup complete"
            )

        except Exception as exc:
            # Warmup failure should not necessarily make the model unusable.
            logger.warning(
                "Embedding model warmup failed: %s",
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

        Returns:
            For a single string:
                one embedding vector.

            For a list:
                an embedding matrix/list.

            None:
                when model encoding fails.
        """
        if self._model is None:
            logger.error(
                "Embedding model is unavailable"
            )
            return None

        # -------------------------------------------------------------
        # Normalize input
        # -------------------------------------------------------------

        single_input = isinstance(texts, str)

        if single_input:
            if not texts.strip():
                logger.warning(
                    "Empty query passed to embedder"
                )
                return None

            input_texts = [texts]

        else:
            if texts is None:
                return None

            input_texts = list(texts)

            if not input_texts:
                return (
                    np.empty(
                        (0, self.dimension),
                        dtype=np.float32,
                    )
                    if convert_to_numpy
                    else []
                )

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
                embeddings = self._model.encode(
                    input_texts,
                    batch_size=effective_batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=convert_to_numpy,
                    normalize_embeddings=effective_normalize,
                    **kwargs,
                )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            self._stats["total_encodings"] += len(
                input_texts
            )
            self._stats["total_time_ms"] += elapsed_ms
            self._stats["last_error"] = None

            if elapsed_ms > 1000:
                logger.warning(
                    "Slow embedding operation: %s texts in %.0fms",
                    len(input_texts),
                    elapsed_ms,
                )

            if single_input:
                if convert_to_numpy:
                    return np.asarray(
                        embeddings[0],
                        dtype=np.float32,
                    )

                return embeddings[0]

            if convert_to_numpy:
                return np.asarray(
                    embeddings,
                    dtype=np.float32,
                )

            return embeddings

        except Exception as exc:
            self._stats["last_error"] = str(exc)

            logger.exception(
                "Embedding encoding failed: %s",
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

        if self._model is not None:
            try:
                dimension = (
                    self._model
                    .get_sentence_embedding_dimension()
                )

                if dimension is not None:
                    self._dimension = int(dimension)
                    return self._dimension

            except Exception:
                pass

        # Do not invent a dimension when the model has failed.
        raise RuntimeError(
            "Embedding model dimension is unavailable"
        )

    @property
    def device(self) -> str:
        """Return the active inference device."""
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
        total = int(
            self._stats["total_encodings"]
        )
        total_time = float(
            self._stats["total_time_ms"]
        )

        try:
            dimension = self.dimension
        except RuntimeError:
            dimension = None

        return {
            "model": self.model_name,
            "device": self._device,
            "batch_size": self._batch_size,
            "dimension": dimension,
            "available": self.is_available,
            "initialized": self._initialized,
            "warmup_complete": self._warmup_complete,
            "total_encodings": total,
            "total_time_ms": round(
                total_time,
                2,
            ),
            "avg_time_per_encoding_ms": round(
                total_time / total,
                3,
            ) if total else 0.0,
            "last_error": self._stats["last_error"],
            "cache_dir": str(self.cache_dir),
        }

    # -------------------------------------------------------------------------
    # RESET / SHUTDOWN
    # -------------------------------------------------------------------------

    def _dispose_model(self) -> None:
        """Release the current model as safely as possible."""
        model = self._model
        self._model = None

        if model is None:
            return

        try:
            if self._device == "cuda":
                try:
                    model.to("cpu")
                except Exception:
                    pass

                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception:
                    pass

        except Exception as exc:
            logger.debug(
                "Model cleanup warning: %s",
                exc,
            )

        del model

    def reset(self) -> None:
        """
        Release the loaded model while keeping the singleton object.

        The next encode attempt will report unavailable unless the global
        singleton is reset and recreated.
        """
        with self._model_lock:
            self._dispose_model()

            self._warmup_complete = False
            self._initialized = False
            self._dimension = None
            self._initialization_error = None

            logger.info(
                "Embedder model reset"
            )

    def shutdown(self) -> None:
        """Release model resources."""
        with self._model_lock:
            self._dispose_model()

            self._warmup_complete = False
            self._initialized = False
            self._dimension = None

            logger.info(
                "Embedder shutdown complete"
            )


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

        # Reset class-level singleton so a completely fresh instance can
        # be constructed after tests/reconfiguration.
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
    print("\nPolicyGuard AI - Embedder Test\n")
    print("=" * 70)

    try:
        emb = get_embedder()

        print(f"Model: {emb.model_name}")
        print(f"Device: {emb.device}")
        print(f"Batch size: {emb.batch_size}")
        print(f"Dimension: {emb.dimension}")
        print(f"Available: {emb.is_available}")

        print("=" * 70)

        # ---------------------------------------------------------------------
        # Test 1: Query
        # ---------------------------------------------------------------------

        print("\nTest 1: Single query")

        query = (
            "What is the company leave policy?"
        )

        start = time.perf_counter()

        query_embedding = encode_query(query)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        if query_embedding is not None:
            print(
                f"Encoded in {elapsed_ms:.1f}ms"
            )
            print(
                f"Shape: {query_embedding.shape}"
            )
            print(
                f"Dtype: {query_embedding.dtype}"
            )
            print(
                f"Norm: {np.linalg.norm(query_embedding):.4f}"
            )
        else:
            print("Encoding failed")

        # ---------------------------------------------------------------------
        # Test 2: Batch
        # ---------------------------------------------------------------------

        print("\nTest 2: Batch documents")

        documents = [
            "Employees are entitled to paid annual leave.",
            "Sick leave requires medical documentation when applicable.",
            "Parental leave is available to qualifying employees.",
            "Unpaid leave must be requested in advance.",
            "Leave balances are tracked in the HR portal.",
        ]

        start = time.perf_counter()

        document_embeddings = encode_documents(
            documents,
            show_progress_bar=False,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        if document_embeddings is not None:
            print(
                f"Encoded {len(documents)} documents "
                f"in {elapsed_ms:.1f}ms"
            )
            print(
                f"Shape: {document_embeddings.shape}"
            )
        else:
            print("Batch encoding failed")

        # ---------------------------------------------------------------------
        # Test 3: Similarity
        # ---------------------------------------------------------------------

        print("\nTest 3: Query/document similarity")

        if (
            query_embedding is not None
            and document_embeddings is not None
            and len(document_embeddings) > 0
        ):
            query_norm = np.linalg.norm(
                query_embedding
            )

            document_norms = np.linalg.norm(
                document_embeddings,
                axis=1,
            )

            if query_norm > 0:
                similarities = (
                    document_embeddings
                    @ query_embedding
                ) / (
                    document_norms * query_norm
                )

                print(
                    "Similarity with first document:",
                    f"{float(similarities[0]):.4f}",
                )

        # ---------------------------------------------------------------------
        # Test 4: Stats
        # ---------------------------------------------------------------------

        print("\nTest 4: Statistics")

        stats = get_embedder_stats()

        for key, value in stats.items():
            if key != "last_error" or value:
                print(f"  {key}: {value}")

        # ---------------------------------------------------------------------
        # Test 5: Empty input
        # ---------------------------------------------------------------------

        print("\nTest 5: Input validation")

        empty_result = encode_query("")

        print(
            "Empty query:",
            "correctly rejected"
            if empty_result is None
            else "unexpectedly encoded",
        )

        print("\n" + "=" * 70)
        print("Embedder test complete.")

    except Exception as exc:
        logger.exception(
            "Embedder test failed: %s",
            exc,
        )
        print(
            f"Embedder test failed: {exc}"
        )


if __name__ == "__main__":
    test_optimized_embedder()

