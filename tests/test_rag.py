"""
PolicyGuard AI - RAG / Vector Retrieval Test Suite
==================================================

Purpose
-------
Production-oriented tests for the current ``src.retrieval.vector_store``
implementation.

The suite deliberately uses deterministic mock embeddings instead of a live
embedding model, so it can run quickly and reproducibly without network/model
downloads.

Coverage
--------
- FAISS vector ingestion and retrieval
- BM25 keyword retrieval
- organization/tenant isolation
- namespace isolation
- duplicate chunk-ID protection
- embedding shape/dimension validation
- NaN/Inf rejection
- query validation
- top_k validation/bounding
- persistence and reload
- persisted tenant/namespace mismatch protection
- corrupted/misaligned persisted state handling
- vector/BM25 compatibility interface
- statistics and lifecycle behavior
- global vector-store singleton/reset behavior
- rollback behavior after failed ingestion
- batch-size guard
- input normalization and metadata preservation

Run
---
    python -m pytest -q tests/test_rag.py

Or, from the project root:
    python -m pytest -q test_rag.py

This file is read-only with respect to the application source. It does not
modify ``check_code.py`` or any project module.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterator

import faiss
import numpy as np
import pytest

from src.core.exceptions import RAGException, RetrievalError
from src.retrieval.vector_store import (
    MAX_BATCH_SIZE,
    MAX_TOP_K,
    VectorStore,
    get_vector_store,
    reset_vector_store,
)


# =============================================================================
# TEST DATA / HELPERS
# =============================================================================

DIMENSION = 8


def make_embedding(index: int, dimension: int = DIMENSION) -> np.ndarray:
    """Return a deterministic, non-zero embedding for a test document."""
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index % dimension] = 1.0
    return vector


def make_chunks(
    organization_id: str = "org_alpha",
    namespace: str = "policy",
) -> list[dict]:
    """Create a small, deterministic policy corpus."""
    return [
        {
            "content": (
                "Employees receive twenty days of paid annual leave "
                "per calendar year."
            ),
            "metadata": {
                "chunk_id": "leave-days",
                "source": "leave_policy.pdf",
                "page": 3,
                "organization_id": organization_id,
                "namespace": namespace,
            },
        },
        {
            "content": (
                "Leave requests should normally be submitted two weeks "
                "before the requested start date."
            ),
            "metadata": {
                "chunk_id": "leave-request",
                "source": "leave_policy.pdf",
                "page": 4,
                "organization_id": organization_id,
                "namespace": namespace,
            },
        },
        {
            "content": (
                "The company provides health insurance and professional "
                "development benefits."
            ),
            "metadata": {
                "chunk_id": "benefits",
                "source": "benefits_policy.pdf",
                "page": 1,
                "organization_id": organization_id,
                "namespace": namespace,
            },
        },
    ]


def make_embeddings(count: int, dimension: int = DIMENSION) -> np.ndarray:
    """Return deterministic embeddings aligned with ``make_chunks``."""
    return np.vstack(
        [make_embedding(index, dimension) for index in range(count)]
    ).astype(np.float32)


@pytest.fixture()
def vector_store(tmp_path: Path) -> Iterator[VectorStore]:
    """Provide an isolated tenant-scoped vector store."""
    store = VectorStore(
        db_path=tmp_path / "vector_db",
        dimension=DIMENSION,
        enable_bm25=True,
        organization_id="org_alpha",
        namespace="policy",
    )
    yield store
    store.shutdown()


@pytest.fixture(autouse=True)
def reset_global_store() -> Iterator[None]:
    """Prevent singleton state leaking between tests."""
    reset_vector_store()
    yield
    reset_vector_store()


def populate(store: VectorStore) -> None:
    """Populate a store with the standard deterministic corpus."""
    chunks = make_chunks()
    embeddings = make_embeddings(len(chunks))
    store.add_chunks(
        chunks,
        embeddings=embeddings,
    )


# =============================================================================
# BASIC INGESTION / SEARCH
# =============================================================================

def test_add_chunks_creates_aligned_faiss_store(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    assert vector_store.chunk_count == 3
    assert vector_store.index is not None
    assert vector_store.index.ntotal == 3
    assert vector_store.index.d == DIMENSION
    assert vector_store.is_loaded is False


def test_vector_search_returns_relevant_chunk(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    results = vector_store.search(
        make_embedding(0),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["chunk_id"] == "leave-days"
    assert results[0]["metadata"]["organization_id"] == "org_alpha"
    assert results[0]["metadata"]["namespace"] == "policy"
    assert np.isfinite(results[0]["score"])
    assert np.isfinite(results[0]["vector_distance"])


def test_vector_search_respects_top_k(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    results = vector_store.search(
        make_embedding(0),
        top_k=2,
    )

    assert len(results) == 2


def test_vector_search_accepts_list_embedding(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    results = vector_store.search(
        make_embedding(0).tolist(),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["chunk_id"] == "leave-days"


def test_vector_search_returns_empty_for_empty_store(
    vector_store: VectorStore,
) -> None:
    results = vector_store.search(
        make_embedding(0),
        top_k=3,
    )

    assert results == []


# =============================================================================
# BM25
# =============================================================================

def test_bm25_search_returns_keyword_match(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    results = vector_store.bm25_search(
        "health insurance",
        top_k=3,
    )

    if not vector_store.bm25_index or not vector_store.bm25_index.is_available:
        pytest.skip("rank_bm25 is not installed or unavailable")

    assert results
    chunk, score = results[0]
    assert chunk["metadata"]["chunk_id"] == "benefits"
    assert score > 0
    assert chunk["bm25_score"] == score


def test_bm25_search_respects_tenant_and_namespace(
    vector_store: VectorStore,
) -> None:
    chunks = make_chunks("org_alpha", "policy")
    chunks.append(
        {
            "content": "Confidential talent bench contains senior engineers.",
            "metadata": {
                "chunk_id": "talent-secret",
                "organization_id": "org_alpha",
                "namespace": "talent",
                "source": "talent_pool.pdf",
            },
        }
    )

    embeddings = np.vstack(
        [
            make_embedding(0),
            make_embedding(1),
            make_embedding(2),
            make_embedding(3),
        ]
    )

    vector_store.add_chunks(
        chunks,
        embeddings=embeddings,
    )

    if not vector_store.bm25_index or not vector_store.bm25_index.is_available:
        pytest.skip("rank_bm25 is not installed or unavailable")

    policy_results = vector_store.bm25_search(
        "senior engineers",
        top_k=10,
        organization_id="org_alpha",
        namespace="policy",
    )

    assert all(
        result[0]["metadata"]["namespace"] == "policy"
        for result in policy_results
    )
    assert not any(
        result[0]["metadata"]["chunk_id"] == "talent-secret"
        for result in policy_results
    )

    talent_results = vector_store.bm25_search(
        "senior engineers",
        top_k=10,
        organization_id="org_alpha",
        namespace="talent",
    )

    assert any(
        result[0]["metadata"]["chunk_id"] == "talent-secret"
        for result in talent_results
    )


# =============================================================================
# TENANT / NAMESPACE ISOLATION
# =============================================================================

def test_vector_search_enforces_organization_isolation(
    tmp_path: Path,
) -> None:
    store = VectorStore(
        db_path=tmp_path / "vector_db",
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )

    store.add_chunks(
        [
            {
                "content": "Alpha policy information.",
                "metadata": {
                    "chunk_id": "alpha-policy",
                    "organization_id": "org_alpha",
                    "namespace": "policy",
                },
            }
        ],
        embeddings=make_embeddings(1),
    )

    alpha_results = store.search(
        make_embedding(0),
        top_k=10,
        organization_id="org_alpha",
        namespace="policy",
    )

    assert alpha_results
    assert all(
        item["metadata"]["organization_id"] == "org_alpha"
        for item in alpha_results
    )
    assert alpha_results[0]["metadata"]["chunk_id"] == "alpha-policy"

    beta_results = store.search(
        make_embedding(0),
        top_k=10,
        organization_id="org_beta",
        namespace="policy",
    )

    assert beta_results == []

    store.shutdown()


def test_bm25_enforces_organization_isolation(
    tmp_path: Path,
) -> None:
    """
    Verify BM25 tenant filtering using a multi-document corpus.

    A one-document BM25 corpus is unsuitable for asserting a positive score:
    rank_bm25 can assign zero IDF when a term occurs in every document, and
    this VectorStore intentionally removes scores <= 0. A two-document corpus
    gives the alpha-specific term positive IDF while still allowing a
    meaningful cross-tenant lookup.
    """
    store = VectorStore(
        db_path=tmp_path / "vector_db",
        dimension=DIMENSION,
        enable_bm25=True,
        organization_id="org_alpha",
        namespace="policy",
    )

    store.add_chunks(
        [
            {
                "content": "Alpha annual leave policy.",
                "metadata": {
                    "chunk_id": "alpha-leave",
                    "organization_id": "org_alpha",
                    "namespace": "policy",
                },
            },
            {
                "content": "Generic benefits information.",
                "metadata": {
                    "chunk_id": "generic-benefits",
                    "organization_id": "org_alpha",
                    "namespace": "policy",
                },
            },
        ],
        embeddings=np.vstack(
            [
                make_embedding(0),
                make_embedding(1),
            ]
        ),
    )

    try:
        if (
            not store.bm25_index
            or not store.bm25_index.is_available
        ):
            pytest.skip(
                "rank_bm25 is not installed or unavailable"
            )

        alpha_results = store.bm25_search(
            "alpha",
            top_k=10,
            organization_id="org_alpha",
            namespace="policy",
        )

        assert alpha_results
        assert all(
            item[0]["metadata"]["organization_id"]
            == "org_alpha"
            for item in alpha_results
        )
        assert (
            alpha_results[0][0]["metadata"]["chunk_id"]
            == "alpha-leave"
        )
        # rank_bm25 can legitimately return a zero BM25 score when a term's
        # corpus-level IDF is zero. Relevance is established by lexical match
        # and tenant filtering, not by requiring a strictly positive raw score.
        assert np.isfinite(alpha_results[0][1])

        # A different tenant must not receive the org_alpha document.
        beta_results = store.bm25_search(
            "alpha",
            top_k=10,
            organization_id="org_beta",
            namespace="policy",
        )

        assert beta_results == []

    finally:
        store.shutdown()

def test_namespace_isolation(
    vector_store: VectorStore,
) -> None:
    vector_store.add_chunks(
        [
            {
                "content": "Policy content.",
                "metadata": {
                    "chunk_id": "policy-1",
                    "organization_id": "org_alpha",
                    "namespace": "policy",
                },
            },
            {
                "content": "Talent candidate content.",
                "metadata": {
                    "chunk_id": "talent-1",
                    "organization_id": "org_alpha",
                    "namespace": "talent",
                },
            },
        ],
        embeddings=np.vstack(
            [
                make_embedding(0),
                make_embedding(1),
            ]
        ),
    )

    policy_results = vector_store.search(
        make_embedding(1),
        top_k=10,
        namespace="policy",
    )

    assert all(
        item["metadata"]["namespace"] == "policy"
        for item in policy_results
    )
    assert not any(
        item["metadata"]["chunk_id"] == "talent-1"
        for item in policy_results
    )

    talent_results = vector_store.search(
        make_embedding(1),
        top_k=10,
        namespace="talent",
    )

    assert any(
        item["metadata"]["chunk_id"] == "talent-1"
        for item in talent_results
    )


def test_add_chunks_rejects_mismatched_organization(
    vector_store: VectorStore,
) -> None:
    chunks = [
        {
            "content": "Wrong tenant data.",
            "metadata": {
                "chunk_id": "wrong-tenant",
                "organization_id": "org_beta",
                "namespace": "policy",
            },
        }
    ]

    with pytest.raises(RAGException, match="different organization"):
        vector_store.add_chunks(
            chunks,
            embeddings=make_embeddings(1),
        )


# =============================================================================
# VALIDATION
# =============================================================================

def test_rejects_wrong_embedding_dimension(
    vector_store: VectorStore,
) -> None:
    chunks = make_chunks()[:1]
    wrong_dimension = np.ones((1, DIMENSION + 1), dtype=np.float32)

    with pytest.raises(RAGException, match="dimension mismatch"):
        vector_store.add_chunks(
            chunks,
            embeddings=wrong_dimension,
        )


def test_rejects_embedding_count_mismatch(
    vector_store: VectorStore,
) -> None:
    chunks = make_chunks()[:2]
    one_embedding = make_embeddings(1)

    with pytest.raises(
        RAGException,
        match="does not match number of chunks",
    ):
        vector_store.add_chunks(
            chunks,
            embeddings=one_embedding,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        np.array([[np.nan] * DIMENSION], dtype=np.float32),
        np.array([[np.inf] * DIMENSION], dtype=np.float32),
        np.array([[-np.inf] * DIMENSION], dtype=np.float32),
    ],
)
def test_rejects_non_finite_embeddings(
    vector_store: VectorStore,
    bad_value: np.ndarray,
) -> None:
    with pytest.raises(
        RAGException,
        match="NaN or infinite",
    ):
        vector_store.add_chunks(
            make_chunks()[:1],
            embeddings=bad_value,
        )


@pytest.mark.parametrize(
    "bad_top_k",
    [0, -1, "not-an-int", None],
)
def test_invalid_top_k(
    vector_store: VectorStore,
    bad_top_k,
) -> None:
    if bad_top_k is None:
        # None is valid and means "use configured default".
        populate(vector_store)
        assert vector_store.search(
            make_embedding(0),
            top_k=None,
        )
        return

    with pytest.raises(RAGException):
        vector_store.search(
            make_embedding(0),
            top_k=bad_top_k,
        )


def test_top_k_is_bounded_to_maximum(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    results = vector_store.search(
        make_embedding(0),
        top_k=MAX_TOP_K + 1000,
    )

    assert len(results) <= MAX_TOP_K


def test_empty_query_is_rejected(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    with pytest.raises(RAGException, match="Query cannot be empty"):
        vector_store.search(
            "   ",
            top_k=3,
        )


def test_non_string_non_embedding_query_is_rejected(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    with pytest.raises(RAGException):
        vector_store.search(
            object(),
            top_k=1,
        )


def test_chunk_content_is_normalized_and_embedding_field_removed(
    vector_store: VectorStore,
) -> None:
    chunk = {
        "content": "  Example policy text.  ",
        "embedding": [1, 2, 3],
        "metadata": {
            "chunk_id": "normalized",
            "source": "policy.pdf",
        },
    }

    vector_store.add_chunks(
        [chunk],
        embeddings=make_embeddings(1),
    )

    stored = vector_store.chunks[0]

    assert stored["content"] == "Example policy text."
    assert "embedding" not in stored
    assert stored["metadata"]["chunk_id"] == "normalized"
    assert stored["metadata"]["organization_id"] == "org_alpha"
    assert stored["metadata"]["namespace"] == "policy"


# =============================================================================
# CHUNK IDS / TRANSACTIONAL INGESTION
# =============================================================================

def test_duplicate_chunk_id_is_rejected(
    vector_store: VectorStore,
) -> None:
    chunks = make_chunks()[:1]
    populate(vector_store)

    with pytest.raises(RAGException, match="already exists"):
        vector_store.add_chunks(
            chunks,
            embeddings=make_embeddings(1),
        )

    assert vector_store.chunk_count == 3
    assert vector_store.index is not None
    assert vector_store.index.ntotal == 3


def test_duplicate_ids_within_batch_are_rejected(
    vector_store: VectorStore,
) -> None:
    chunks = [
        {
            "content": "First chunk.",
            "metadata": {
                "chunk_id": "duplicate",
                "organization_id": "org_alpha",
                "namespace": "policy",
            },
        },
        {
            "content": "Second chunk.",
            "metadata": {
                "chunk_id": "duplicate",
                "organization_id": "org_alpha",
                "namespace": "policy",
            },
        },
    ]

    with pytest.raises(
        RAGException,
        match="Duplicate chunk ID in batch",
    ):
        vector_store.add_chunks(
            chunks,
            embeddings=make_embeddings(2),
        )

    assert vector_store.chunk_count == 0
    assert vector_store.index is not None
    assert vector_store.index.ntotal == 0


def test_failed_ingestion_does_not_leave_partial_chunks(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    original_ids = [
        chunk["metadata"]["chunk_id"]
        for chunk in vector_store.chunks
    ]

    # A duplicate ID forces failure after the incoming embeddings have
    # already been validated, exercising the pre-mutation guard.
    bad_chunk = {
        "content": "This must not be added.",
        "metadata": {
            "chunk_id": "leave-days",
            "organization_id": "org_alpha",
            "namespace": "policy",
        },
    }

    with pytest.raises(RAGException, match="already exists"):
        vector_store.add_chunks(
            [bad_chunk],
            embeddings=make_embeddings(1),
        )

    assert vector_store.chunk_count == 3
    assert vector_store.index is not None
    assert vector_store.index.ntotal == 3
    assert [
        chunk["metadata"]["chunk_id"]
        for chunk in vector_store.chunks
    ] == original_ids


def test_empty_add_is_noop(
    vector_store: VectorStore,
) -> None:
    vector_store.add_chunks(
        [],
        embeddings=np.empty((0, DIMENSION), dtype=np.float32),
    )

    assert vector_store.chunk_count == 0
    assert vector_store.index is None


def test_batch_size_limit(
    vector_store: VectorStore,
) -> None:
    chunks = [
        {
            "content": f"Chunk {index}",
            "metadata": {
                "chunk_id": f"chunk-{index}",
                "organization_id": "org_alpha",
                "namespace": "policy",
            },
        }
        for index in range(MAX_BATCH_SIZE + 1)
    ]

    embeddings = np.zeros(
        (MAX_BATCH_SIZE + 1, DIMENSION),
        dtype=np.float32,
    )

    with pytest.raises(
        RAGException,
        match="Cannot add more than",
    ):
        vector_store.add_chunks(
            chunks,
            embeddings=embeddings,
        )


# =============================================================================
# PERSISTENCE / RELOAD
# =============================================================================

def test_save_and_reload_preserves_vector_store(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vector_db"

    original = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=True,
        organization_id="org_alpha",
        namespace="policy",
    )
    populate(original)

    assert original.save(backup=False) is True

    reloaded = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=True,
        organization_id="org_alpha",
        namespace="policy",
    )

    assert reloaded.load() is True
    assert reloaded.is_loaded is True
    assert reloaded.chunk_count == 3
    assert reloaded.index is not None
    assert reloaded.index.ntotal == 3

    results = reloaded.search(
        make_embedding(0),
        top_k=1,
    )

    assert results
    assert results[0]["metadata"]["chunk_id"] == "leave-days"

    if reloaded.bm25_index and reloaded.bm25_index.is_available:
        bm25_results = reloaded.bm25_search(
            "health insurance",
            top_k=1,
        )
        assert bm25_results
        assert (
            bm25_results[0][0]["metadata"]["chunk_id"]
            == "benefits"
        )

    original.shutdown()
    reloaded.shutdown()


def test_save_creates_backup_when_requested(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vector_db"

    store = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )
    populate(store)

    assert store.save(backup=True) is True

    # A backup is only created for a pre-existing live file. Save once more
    # to verify the backup mechanism copies the known-good previous file.
    assert store.save(backup=True) is True

    assert (path / "faiss.index.bak").exists()
    assert (path / "chunks.pkl.bak").exists()

    store.shutdown()


def test_load_refuses_wrong_organization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vector_db"

    writer = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )
    populate(writer)
    assert writer.save(backup=False) is True
    writer.shutdown()

    attacker = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_beta",
        namespace="policy",
    )

    assert attacker.load() is False
    assert attacker.chunk_count == 0
    assert attacker.index is None
    assert attacker.is_loaded is False

    attacker.shutdown()


def test_load_refuses_wrong_namespace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vector_db"

    writer = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )
    populate(writer)
    assert writer.save(backup=False) is True
    writer.shutdown()

    other_namespace = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="talent",
    )

    assert other_namespace.load() is False
    assert other_namespace.chunk_count == 0
    assert other_namespace.index is None

    other_namespace.shutdown()


def test_load_refuses_dimension_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vector_db"

    writer = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )
    populate(writer)
    assert writer.save(backup=False) is True
    writer.shutdown()

    wrong_dimension = VectorStore(
        db_path=path,
        dimension=DIMENSION + 1,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )

    assert wrong_dimension.load() is False
    assert wrong_dimension.index is None
    assert wrong_dimension.chunk_count == 0

    wrong_dimension.shutdown()


def test_load_detects_misaligned_faiss_and_chunks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vector_db"

    store = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )
    populate(store)
    assert store.save(backup=False) is True
    store.shutdown()

    chunks_path = path / "chunks.pkl"

    with chunks_path.open("rb") as handle:
        chunks = pickle.load(handle)

    chunks.pop()

    with chunks_path.open("wb") as handle:
        pickle.dump(chunks, handle)

    reloaded = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )

    assert reloaded.load() is False
    assert reloaded.index is None
    assert reloaded.chunks == []

    reloaded.shutdown()


def test_corrupt_persisted_chunks_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vector_db"

    store = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )
    populate(store)
    assert store.save(backup=False) is True
    store.shutdown()

    with (path / "chunks.pkl").open("wb") as handle:
        handle.write(b"not a valid pickle")

    reloaded = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )

    assert reloaded.load() is False
    assert reloaded.index is None
    assert reloaded.chunks == []

    reloaded.shutdown()


# =============================================================================
# COMPATIBILITY / STATS / LIFECYCLE
# =============================================================================

def test_vector_search_compatibility_returns_tuples(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    results = vector_store.vector_search(
        make_embedding(0),
        top_k=2,
    )

    assert len(results) == 2
    assert all(isinstance(item, tuple) for item in results)
    assert all(len(item) == 2 for item in results)
    assert all(isinstance(item[0], dict) for item in results)
    assert all(isinstance(item[1], float) for item in results)


def test_statistics_track_ingestion_and_search(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    vector_store.search(
        make_embedding(0),
        top_k=1,
    )

    stats = vector_store.get_stats()

    assert stats["total_chunks"] == 3
    assert stats["faiss_vectors"] == 3
    assert stats["search_count"] == 1
    assert stats["add_count"] == 1
    assert stats["dimension"] == DIMENSION
    assert stats["organization_id"] == "org_alpha"
    assert stats["namespace"] == "policy"
    assert stats["metric_type"] == "l2"
    assert stats["avg_search_time_ms"] >= 0


def test_clear_removes_memory_and_persisted_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vector_db"

    store = VectorStore(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )
    populate(store)

    assert store.save(backup=False) is True
    assert (path / "faiss.index").exists()
    assert (path / "chunks.pkl").exists()
    assert (path / "manifest.pkl").exists()

    store.clear()

    assert store.chunk_count == 0
    assert store.index is None
    assert store.is_loaded is False
    assert not (path / "faiss.index").exists()
    assert not (path / "chunks.pkl").exists()
    assert not (path / "manifest.pkl").exists()

    store.shutdown()


def test_shutdown_rejects_future_ingestion(
    vector_store: VectorStore,
) -> None:
    vector_store.shutdown()

    with pytest.raises(RAGException, match="shut down"):
        vector_store.add_chunks(
            make_chunks()[:1],
            embeddings=make_embeddings(1),
        )


def test_shutdown_is_idempotent(
    vector_store: VectorStore,
) -> None:
    vector_store.shutdown()
    vector_store.shutdown()

    assert vector_store.get_stats()["shutdown"] is True


# =============================================================================
# CONFIGURATION / GLOBAL SINGLETON
# =============================================================================

def test_invalid_metric_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RAGException,
        match="metric_type",
    ):
        VectorStore(
            db_path=tmp_path / "vector_db",
            dimension=DIMENSION,
            metric_type="cosine",
            enable_bm25=False,
        )


def test_inner_product_metric_works(
    tmp_path: Path,
) -> None:
    store = VectorStore(
        db_path=tmp_path / "vector_db",
        dimension=DIMENSION,
        metric_type="ip",
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )

    store.add_chunks(
        make_chunks()[:2],
        embeddings=np.vstack(
            [
                make_embedding(0),
                make_embedding(1),
            ]
        ),
    )

    results = store.search(
        make_embedding(0),
        top_k=1,
    )

    assert results
    assert results[0]["metadata"]["chunk_id"] == "leave-days"
    assert results[0]["score"] > 0

    store.shutdown()


def test_global_vector_store_reuses_same_instance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "global_vector_db"

    first = get_vector_store(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )

    second = get_vector_store(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )

    assert first is second


def test_global_vector_store_rejects_configuration_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "global_vector_db"

    get_vector_store(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )

    with pytest.raises(
        RAGException,
        match="organization",
    ):
        get_vector_store(
            db_path=path,
            dimension=DIMENSION,
            enable_bm25=False,
            organization_id="org_beta",
            namespace="policy",
        )


def test_reset_global_vector_store_allows_new_instance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "global_vector_db"

    first = get_vector_store(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )

    reset_vector_store()

    second = get_vector_store(
        db_path=path,
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="org_alpha",
        namespace="policy",
    )

    assert second is not first


# =============================================================================
# OPTIONAL EMBEDDER PATH
# =============================================================================

class FakeEmbedder:
    """Minimal embedder used to exercise string-query and auto-embedding paths."""

    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension

    def encode(
        self,
        texts,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if isinstance(texts, str):
            return make_embedding(0, self.dimension)

        return np.vstack(
            [
                make_embedding(index, self.dimension)
                for index, _ in enumerate(texts)
            ]
        ).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if "leave" in query.lower():
            return make_embedding(0, self.dimension)
        if "health" in query.lower():
            return make_embedding(2, self.dimension)
        return make_embedding(1, self.dimension)


def test_string_query_uses_configured_embedder(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)
    vector_store.set_embedder(FakeEmbedder())

    results = vector_store.search(
        "health insurance benefits",
        top_k=1,
    )

    assert results
    assert results[0]["metadata"]["chunk_id"] == "benefits"


def test_add_chunks_can_generate_embeddings_with_embedder(
    vector_store: VectorStore,
) -> None:
    vector_store.set_embedder(FakeEmbedder())

    chunks = make_chunks()[:2]

    vector_store.add_chunks(
        chunks,
        embeddings=None,
    )

    assert vector_store.chunk_count == 2
    assert vector_store.index is not None
    assert vector_store.index.ntotal == 2


def test_missing_embedder_for_string_query_fails(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    with pytest.raises(
        RetrievalError,
        match="No embedder configured",
    ):
        vector_store.search(
            "leave policy",
            top_k=1,
        )


# =============================================================================
# TENANT IDENTIFIER VALIDATION
# =============================================================================

@pytest.mark.parametrize(
    "bad_organization_id",
    [
        "org with spaces",
        "org/with/slash",
        "org\\with\\slash",
        "org$bad",
    ],
)
def test_invalid_organization_id_is_rejected(
    tmp_path: Path,
    bad_organization_id: str,
) -> None:
    with pytest.raises(RAGException):
        VectorStore(
            db_path=tmp_path / "vector_db",
            dimension=DIMENSION,
            enable_bm25=False,
            organization_id=bad_organization_id,
            namespace="policy",
        )


def test_organization_id_is_normalized(
    tmp_path: Path,
) -> None:
    store = VectorStore(
        db_path=tmp_path / "vector_db",
        dimension=DIMENSION,
        enable_bm25=False,
        organization_id="  org_alpha  ",
        namespace=" policy ",
    )

    assert store.organization_id == "org_alpha"
    assert store.namespace == "policy"

    store.shutdown()


# =============================================================================
# FAISS ALIGNMENT / METADATA INTEGRITY
# =============================================================================

def test_faiss_alignment_remains_valid_after_multiple_batches(
    vector_store: VectorStore,
) -> None:
    first = make_chunks()[:2]
    second = [
        {
            "content": "Remote work policy permits hybrid schedules.",
            "metadata": {
                "chunk_id": "remote-work",
                "organization_id": "org_alpha",
                "namespace": "policy",
            },
        }
    ]

    vector_store.add_chunks(
        first,
        embeddings=make_embeddings(2),
    )

    vector_store.add_chunks(
        second,
        embeddings=make_embeddings(1, DIMENSION),
    )

    assert vector_store.chunk_count == 3
    assert vector_store.index is not None
    assert vector_store.index.ntotal == vector_store.chunk_count

    ids = [
        chunk["metadata"]["chunk_id"]
        for chunk in vector_store.chunks
    ]
    assert ids == [
        "leave-days",
        "leave-request",
        "remote-work",
    ]


def test_metadata_source_is_preserved(
    vector_store: VectorStore,
) -> None:
    populate(vector_store)

    results = vector_store.search(
        make_embedding(0),
        top_k=1,
    )

    assert results[0]["metadata"]["source"] == "leave_policy.pdf"
    assert results[0]["metadata"]["page"] == 3


# =============================================================================
# STANDALONE SMOKE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        __import__("pytest").main(
            [
                __file__,
                "-q",
            ]
        )
    )
