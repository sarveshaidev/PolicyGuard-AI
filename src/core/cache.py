
#!/usr/bin/env python3
"""
PolicyGuard AI - Optimized Semantic Cache Module
=================================================

High-performance RAG query cache with:

- Exact SHA-256 query matching
- Optional semantic similarity matching
- Batch embedding computation
- SQLite persistence
- Thread-safe connection management
- LRU in-memory hot-query cache
- Background expiration cleanup
- Graceful shutdown
- Automatic schema initialization/migration
- Windows/Linux compatibility

Author: PolicyGuard AI Team
Version: 2.0.0
Last Updated: 2026-09-13
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# SQLITE CONNECTION POOL
# =============================================================================


class SQLiteConnectionPool:
    """
    Small thread-safe SQLite connection pool.

    SQLite connections are created with check_same_thread=False because
    connections are deliberately handed between worker threads. Access to
    individual connections is serialized by the pool: a connection belongs
    to only one caller at a time.

    The pool does not create unbounded temporary connections. If the pool is
    exhausted, callers wait until a connection becomes available.
    """

    def __init__(
        self,
        db_path: Path,
        pool_size: int = 4,
        timeout: float = 30.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.pool_size = max(1, int(pool_size))
        self.timeout = max(1.0, float(timeout))

        self._pool: Queue[sqlite3.Connection] = Queue(
            maxsize=self.pool_size
        )
        self._closed = False
        self._close_lock = threading.Lock()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_pool()

    def _create_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout,
            check_same_thread=False,
            detect_types=(
                sqlite3.PARSE_DECLTYPES
                | sqlite3.PARSE_COLNAMES
            ),
        )

        conn.row_factory = sqlite3.Row

        # WAL significantly improves read/write concurrency.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError as exc:
            logger.warning(
                "Could not enable WAL mode for %s: %s",
                self.db_path,
                exc,
            )

        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

        return conn

    def _initialize_pool(self) -> None:
        """Pre-create connections."""
        created = 0

        for _ in range(self.pool_size):
            try:
                self._pool.put(self._create_connection())
                created += 1
            except Exception as exc:
                logger.warning(
                    "Could not create SQLite connection: %s",
                    exc,
                )

        if created == 0:
            raise RuntimeError(
                f"Unable to create any SQLite connections for {self.db_path}"
            )

        logger.info(
            "SQLite connection pool initialized: %s connections",
            created,
        )

    def get_connection(
        self,
        timeout: Optional[float] = None,
    ) -> sqlite3.Connection:
        """Borrow a connection from the pool."""
        with self._close_lock:
            if self._closed:
                raise RuntimeError("SQLite connection pool is closed")

        wait_timeout = (
            self.timeout if timeout is None else max(0.1, timeout)
        )

        try:
            return self._pool.get(timeout=wait_timeout)
        except Empty as exc:
            raise TimeoutError(
                "Timed out waiting for a SQLite connection"
            ) from exc

    def return_connection(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """Return a connection to the pool."""
        if conn is None:
            return

        try:
            # Never return a connection with an open transaction.
            conn.rollback()
        except sqlite3.Error:
            pass

        with self._close_lock:
            if self._closed:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
                return

        try:
            self._pool.put(conn, timeout=self.timeout)
        except Exception as exc:
            logger.warning(
                "Could not return SQLite connection to pool: %s",
                exc,
            )
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def execute_with_retry(
        self,
        query: str,
        params: Tuple[Any, ...] = (),
        max_retries: int = 4,
        fetch: bool = False,
    ) -> Union[List[sqlite3.Row], int, bool]:
        """
        Execute a SQLite query with retry handling for lock contention.

        Returns:
            SELECT: list of sqlite3.Row objects
            INSERT/UPDATE/DELETE: affected row count when available,
            otherwise True
        """
        last_error: Optional[Exception] = None

        for attempt in range(max(1, max_retries)):
            conn: Optional[sqlite3.Connection] = None

            try:
                conn = self.get_connection()
                cursor = conn.cursor()

                cursor.execute(query, params)

                if fetch:
                    result = cursor.fetchall()
                    return result

                affected = cursor.rowcount
                conn.commit()

                # For INSERT callers that depend on an ID, expose lastrowid.
                if affected == 1 and cursor.lastrowid:
                    return cursor.lastrowid

                return affected if affected >= 0 else True

            except sqlite3.OperationalError as exc:
                last_error = exc

                if (
                    "database is locked" in str(exc).lower()
                    or "database is busy" in str(exc).lower()
                ) and attempt < max_retries - 1:
                    time.sleep(0.05 * (2 ** attempt))
                    continue

                raise

            except Exception:
                raise

            finally:
                if conn is not None:
                    self.return_connection(conn)

        raise RuntimeError(
            f"SQLite operation failed after {max_retries} attempts"
        ) from last_error

    def close_all(self) -> None:
        """Close every pooled connection."""
        with self._close_lock:
            if self._closed:
                return

            self._closed = True

            while True:
                try:
                    conn = self._pool.get_nowait()
                except Empty:
                    break

                try:
                    conn.close()
                except sqlite3.Error:
                    pass

        logger.info("SQLite connection pool closed")


# =============================================================================
# LRU MEMORY CACHE
# =============================================================================


class LRUMemoryCache:
    """Thread-safe LRU cache for hot query results."""

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 300,
    ) -> None:
        self.max_size = max(1, int(max_size))
        self.ttl_seconds = max(1, int(ttl_seconds))

        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._timestamps: Dict[str, datetime] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return a non-expired cached item."""
        with self._lock:
            value = self._cache.get(key)

            if value is None:
                return None

            timestamp = self._timestamps.get(key)

            if timestamp is None:
                self._remove(key)
                return None

            if (
                datetime.now() - timestamp
                > timedelta(seconds=self.ttl_seconds)
            ):
                self._remove(key)
                return None

            self._cache.move_to_end(key)

            # Return a shallow copy so callers cannot mutate the stored
            # dictionary accidentally.
            return dict(value)

    def set(
        self,
        key: str,
        value: Dict[str, Any],
    ) -> None:
        """Insert/update an item and enforce LRU capacity."""
        with self._lock:
            # Updating an existing key must not evict another item.
            if key in self._cache:
                self._cache.pop(key, None)
                self._timestamps.pop(key, None)

            while len(self._cache) >= self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._timestamps.pop(oldest_key, None)

            self._cache[key] = dict(value)
            self._timestamps[key] = datetime.now()

    def _remove(self, key: str) -> None:
        """Remove a key while lock is already held."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)

    def remove(self, key: str) -> None:
        """Remove a specific item."""
        with self._lock:
            self._remove(key)

    def clear(self) -> None:
        """Clear all memory-cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def stats(self) -> Dict[str, int]:
        """Return memory cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
            }


# =============================================================================
# RAG CACHE
# =============================================================================


class RAGCache:
    """
    Two-tier semantic RAG cache.

    Tier 1:
        In-memory LRU exact-query cache.

    Tier 2:
        SQLite exact-query and semantic cache.

    Semantic matching is optional and requires an initialized embedder.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        batch_size: int = 16,
        max_workers: int = 4,
        organization_id: str = "default",
        namespace: str = "policy",
        user_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        self.similarity_threshold = float(
            settings.CACHE_SIMILARITY_THRESHOLD
        )
        self.ttl_seconds = max(
            1,
            int(settings.CACHE_TTL_SECONDS),
        )
        self.batch_size = max(1, int(batch_size))
        self.max_workers = max(1, int(max_workers))

        self.db_path = (
            Path(db_path)
            if db_path
            else project_root / "data" / "cache.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._closed = False
        self._state_lock = threading.RLock()

        self.organization_id = self._normalize_scope(organization_id, "organization_id")
        self.namespace = self._normalize_scope(namespace, "namespace")
        self.user_id = self._normalize_optional_scope(user_id, "user_id")
        self.role = self._normalize_role(role)
        if self.namespace == "talent" and not self.user_id:
            raise ValueError("Talent cache requires user_id for isolation")

        self.conn_pool = SQLiteConnectionPool(
            db_path=self.db_path,
            pool_size=self.max_workers,
            timeout=30.0,
        )

        self._initialize_schema()

        self.memory_cache = LRUMemoryCache(
            max_size=100,
            ttl_seconds=self.ttl_seconds,
        )

        self.embedder = None
        self._init_embedder()

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="rag-cache",
        )

        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        self._start_background_cleanup()

        logger.info(
            "RAGCache initialized: batch_size=%s, workers=%s, db=%s",
            self.batch_size,
            self.max_workers,
            self.db_path,
        )

    # -------------------------------------------------------------------------
    # DATABASE
    # -------------------------------------------------------------------------

    def _initialize_schema(self) -> None:
        """
        Create/migrate the cache schema.

        This is intentionally independent from the authentication database.
        The cache historically uses data/memory.db.
        """
        conn = self.conn_pool.get_connection()

        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    organization_id TEXT NOT NULL DEFAULT 'default',
                    namespace TEXT NOT NULL DEFAULT 'policy',
                    user_id TEXT,
                    role TEXT,
                    query_hash TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    query_embedding BLOB,
                    answer TEXT NOT NULL,
                    chunks_used TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    hits INTEGER NOT NULL DEFAULT 0
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_query_cache_scope_hash
                    ON query_cache(organization_id, namespace, query_hash);

                CREATE INDEX IF NOT EXISTS idx_query_cache_created
                    ON query_cache(created_at);

                CREATE INDEX IF NOT EXISTS idx_query_cache_accessed
                    ON query_cache(last_accessed);
                """
            )

            # Migration support for databases created by older versions.
            columns = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(query_cache)"
                ).fetchall()
            }

            migrations = {
                "organization_id": (
                    "ALTER TABLE query_cache "
                    "ADD COLUMN organization_id TEXT NOT NULL DEFAULT 'default'"
                ),
                "namespace": (
                    "ALTER TABLE query_cache "
                    "ADD COLUMN namespace TEXT NOT NULL DEFAULT 'policy'"
                ),
                "user_id": (
                    "ALTER TABLE query_cache "
                    "ADD COLUMN user_id TEXT"
                ),
                "role": (
                    "ALTER TABLE query_cache "
                    "ADD COLUMN role TEXT"
                ),
                "query_embedding": (
                    "ALTER TABLE query_cache "
                    "ADD COLUMN query_embedding BLOB"
                ),
                "last_accessed": (
                    "ALTER TABLE query_cache "
                    "ADD COLUMN last_accessed TIMESTAMP"
                ),
                "hits": (
                    "ALTER TABLE query_cache "
                    "ADD COLUMN hits INTEGER NOT NULL DEFAULT 0"
                ),
            }

            for column, statement in migrations.items():
                if column not in columns:
                    conn.execute(statement)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_cache_scope "
                "ON query_cache(organization_id, namespace, user_id, role, query_hash)"
            )
            conn.commit()

            logger.info("Cache database schema initialized")

        except Exception:
            conn.rollback()
            raise

        finally:
            self.conn_pool.return_connection(conn)

    # -------------------------------------------------------------------------
    # EMBEDDINGS
    # -------------------------------------------------------------------------

    def _init_embedder(self) -> None:
        """Load the project embedder singleton if available."""
        try:
            from src.core.embedder_singleton import get_embedder

            self.embedder = get_embedder()

            if self.embedder is not None:
                logger.info(
                    "Embedder available for semantic cache"
                )
            else:
                logger.warning(
                    "Embedder singleton is None; "
                    "semantic caching disabled"
                )

        except Exception as exc:
            logger.warning(
                "Embedder unavailable; semantic caching disabled: %s",
                exc,
            )
            self.embedder = None

    def _compute_embeddings_batch(
        self,
        queries: List[str],
    ) -> List[Optional[List[float]]]:
        """Compute embeddings in batches."""
        if not queries:
            return []

        if self.embedder is None:
            return [None] * len(queries)

        results: List[Optional[List[float]]] = []

        for start in range(0, len(queries), self.batch_size):
            batch = queries[start:start + self.batch_size]

            try:
                if hasattr(self.embedder, "encode_documents"):
                    embeddings = self.embedder.encode_documents(batch)
                else:
                    embeddings = self.embedder.encode(
                        batch,
                        batch_size=min(
                            self.batch_size,
                            len(batch),
                        ),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )

                # SentenceTransformer normally returns a 2D array for a
                # list of strings. Normalize single-result edge cases.
                if len(batch) == 1:
                    try:
                        if getattr(embeddings, "ndim", 2) == 1:
                            embeddings = [embeddings]
                    except Exception:
                        pass

                if len(embeddings) != len(batch):
                    logger.error(
                        "Embedder returned %s vectors for %s queries",
                        len(embeddings),
                        len(batch),
                    )
                    results.extend([None] * len(batch))
                    continue

                for embedding in embeddings:
                    if embedding is None:
                        results.append(None)
                        continue

                    try:
                        results.append(
                            embedding.tolist()
                            if hasattr(embedding, "tolist")
                            else list(embedding)
                        )
                    except Exception:
                        results.append(None)

            except Exception as exc:
                logger.exception(
                    "Batch embedding error: %s",
                    exc,
                )
                results.extend([None] * len(batch))

        return results

    @staticmethod
    def _compute_similarity_batch(
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
    ) -> List[float]:
        """Compute cosine similarity using vectorized NumPy operations."""
        if not candidate_embeddings:
            return []

        try:
            import numpy as np

            query_vec = np.asarray(
                query_embedding,
                dtype=np.float32,
            )

            if query_vec.ndim != 1:
                query_vec = query_vec.reshape(-1)

            query_norm = float(np.linalg.norm(query_vec))

            if query_norm == 0.0:
                return [0.0] * len(candidate_embeddings)

            valid_candidates: List[List[float]] = []

            for candidate in candidate_embeddings:
                vector = np.asarray(
                    candidate,
                    dtype=np.float32,
                ).reshape(-1)

                if vector.shape != query_vec.shape:
                    valid_candidates.append(
                        [0.0] * len(query_vec)
                    )
                else:
                    valid_candidates.append(vector.tolist())

            matrix = np.asarray(
                valid_candidates,
                dtype=np.float32,
            )

            norms = np.linalg.norm(
                matrix,
                axis=1,
                keepdims=True,
            )

            norms = np.where(norms == 0.0, 1.0, norms)

            similarities = (
                np.dot(matrix, query_vec)
                / (norms.squeeze(axis=1) * query_norm)
            )

            return similarities.astype(float).tolist()

        except Exception as exc:
            logger.warning(
                "Similarity computation failed: %s",
                exc,
            )
            return [0.0] * len(candidate_embeddings)

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_scope(value: str, field_name: str) -> str:
        """Validate a tenant/namespace scope used in cache keys and queries."""
        if value is None:
            raise ValueError(f"{field_name} cannot be None")
        value = str(value).strip()
        if not value:
            raise ValueError(f"{field_name} cannot be empty")
        if len(value) > 128:
            raise ValueError(f"{field_name} is too long")
        if not re.fullmatch(r"[A-Za-z0-9_.:@-]+", value):
            raise ValueError(
                f"{field_name} contains unsupported characters"
            )
        return value

    @staticmethod
    def _normalize_optional_scope(value: Optional[str], field_name: str) -> Optional[str]:
        """Validate an optional user/scope component."""
        if value is None:
            return None
        value = str(value).strip()
        if not value:
            return None
        if len(value) > 200 or not re.fullmatch(r"[A-Za-z0-9_.:@-]+", value):
            raise ValueError(f"{field_name} contains unsupported characters")
        return value

    @staticmethod
    def _normalize_role(role: Optional[str]) -> Optional[str]:
        """Normalize an optional application role."""
        if role is None:
            return None
        role = str(role).strip().lower()
        if role not in {"viewer", "editor", "admin", "user", "assistant", "system"}:
            raise ValueError("Unsupported role")
        return role

    @staticmethod
    def _serialize_embedding(embedding: Any) -> Optional[bytes]:
        """Serialize embeddings using JSON; never unpickle database data."""
        if embedding is None:
            return None
        try:
            values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            if not values or len(values) > 10000:
                return None
            return json.dumps(values, separators=(",", ":")).encode("utf-8")
        except Exception:
            return None

    @staticmethod
    def _normalize_embedding(value: Any) -> Optional[List[float]]:
        """Validate a decoded embedding vector."""
        if not isinstance(value, list) or not value or len(value) > 10000:
            return None
        try:
            result = [float(x) for x in value]
        except (TypeError, ValueError):
            return None
        if not all(__import__("math").isfinite(x) for x in result):
            return None
        return result

    @staticmethod
    def _deserialize_embedding(value: Any) -> Optional[List[float]]:
        """Safely decode a JSON embedding; legacy pickle is intentionally unsupported."""
        if value is None:
            return None
        try:
            raw = value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
            return RAGCache._normalize_embedding(json.loads(raw))
        except Exception:
            return None

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize query text consistently."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        normalized = " ".join(query.strip().lower().split())

        if not normalized:
            raise ValueError("query cannot be empty")

        return normalized

    def _compute_hash(self, query: str) -> str:
        """Compute SHA-256 hash of normalized query text."""
        normalized = self._normalize_query(query)

        scope = (
            f"{self.organization_id}\x1f{self.namespace}\x1f"
            f"{self.user_id or ''}\x1f{self.role or ''}\x1f{normalized}"
        )
        return hashlib.sha256(
            scope.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _decode_chunks(value: Any) -> List[Dict[str, Any]]:
        """Safely decode cached chunk JSON."""
        if not value:
            return []

        if isinstance(value, list):
            return value

        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return []

    def _build_response(
        self,
        row: sqlite3.Row,
        cache_type: str,
        similarity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert a database row into the public cache response."""
        hits = int(row["hits"] or 0)

        response: Dict[str, Any] = {
            "answer": row["answer"],
            "chunks": self._decode_chunks(row["chunks_used"]),
            "cache_type": cache_type,
            "hits": hits,
            "created_at": row["created_at"],
            "last_accessed": row["last_accessed"],
        }

        if similarity is not None:
            response["similarity"] = float(similarity)

        return response

    # -------------------------------------------------------------------------
    # BACKGROUND CLEANUP
    # -------------------------------------------------------------------------

    def _start_background_cleanup(self) -> None:
        """Start background cleanup worker."""
        def cleanup_loop() -> None:
            while not self._stop_cleanup.is_set():
                try:
                    self._cleanup_expired_entries()
                except Exception as exc:
                    logger.exception(
                        "Cache cleanup thread error: %s",
                        exc,
                    )

                self._stop_cleanup.wait(timeout=300)

        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            name="rag-cache-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

        logger.info("Cache cleanup thread started")

    def _cleanup_expired_entries(self) -> int:
        """Delete expired database entries and return deletion count."""
        try:
            result = self.conn_pool.execute_with_retry(
                """
                DELETE FROM query_cache
                WHERE datetime(created_at) <
                      datetime('now', ?)
                """,
                (f"-{self.ttl_seconds} seconds",),
                fetch=False,
            )

            deleted = int(result) if isinstance(result, int) else 0

            if deleted:
                logger.info(
                    "Cleaned up %s expired cache entries",
                    deleted,
                )

            return deleted

        except Exception as exc:
            logger.exception(
                "Cache cleanup error: %s",
                exc,
            )
            return 0

    # -------------------------------------------------------------------------
    # GET
    # -------------------------------------------------------------------------

    def get(
        self,
        query: str,
        threshold: Optional[float] = None,
        use_semantic: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached answer.

        Lookup order:
        1. LRU memory cache
        2. SQLite exact hash
        3. SQLite semantic similarity
        """
        if self._closed:
            raise RuntimeError("RAGCache is shut down")

        normalized_query = self._normalize_query(query)
        query_hash = self._compute_hash(normalized_query)

        if threshold is None:
            threshold = self.similarity_threshold
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("threshold must be a number") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        # ---------------------------------------------------------------------
        # 1. MEMORY CACHE
        # ---------------------------------------------------------------------

        cached = self.memory_cache.get(query_hash)

        if cached is not None:
            logger.debug(
                "Memory cache HIT: %s",
                normalized_query[:80],
            )
            return cached

        # ---------------------------------------------------------------------
        # 2. EXACT SQLITE MATCH
        # ---------------------------------------------------------------------

        try:
            rows = self.conn_pool.execute_with_retry(
                """
                SELECT
                    id,
                    query_hash,
                    query_text,
                    query_embedding,
                    answer,
                    chunks_used,
                    created_at,
                    last_accessed,
                    hits
                FROM query_cache
                WHERE organization_id = ?
                  AND namespace = ?
                  AND user_id IS ?
                  AND role IS ?
                  AND query_hash = ?
                  AND datetime(created_at) >
                      datetime('now', ?)
                LIMIT 1
                """,
                (
                    self.organization_id,
                    self.namespace,
                    self.user_id,
                    self.role,
                    query_hash,
                    f"-{self.ttl_seconds} seconds",
                ),
                fetch=True,
            )

            if rows:
                row = rows[0]

                new_hits = int(row["hits"] or 0) + 1

                self.conn_pool.execute_with_retry(
                    """
                    UPDATE query_cache
                    SET last_accessed = CURRENT_TIMESTAMP,
                        hits = ?
                    WHERE id = ?
                      AND organization_id = ?
                      AND namespace = ?
                      AND user_id IS ?
                      AND role IS ?
                    """,
                    (
                        new_hits,
                        row["id"],
                        self.organization_id,
                        self.namespace,
                        self.user_id,
                        self.role,
                    ),
                    fetch=False,
                )

                response = self._build_response(
                    row,
                    cache_type="exact",
                )
                response["hits"] = new_hits

                self.memory_cache.set(
                    query_hash,
                    response,
                )

                logger.debug(
                    "SQLite exact HIT: %s",
                    normalized_query[:80],
                )

                return response

        except Exception as exc:
            logger.exception(
                "Exact cache lookup error: %s",
                exc,
            )

        # ---------------------------------------------------------------------
        # 3. SEMANTIC MATCH
        # ---------------------------------------------------------------------

        if not use_semantic or self.embedder is None:
            return None

        try:
            query_embedding = self._compute_embeddings_batch(
                [normalized_query]
            )[0]

            if query_embedding is None:
                return None

            candidates = self.conn_pool.execute_with_retry(
                """
                SELECT
                    id,
                    query_text,
                    query_embedding,
                    answer,
                    chunks_used,
                    created_at,
                    last_accessed,
                    hits
                FROM query_cache
                WHERE organization_id = ?
                  AND namespace = ?
                  AND user_id IS ?
                  AND role IS ?
                  AND query_embedding IS NOT NULL
                  AND datetime(created_at) >
                      datetime('now', ?)
                ORDER BY last_accessed DESC,
                         created_at DESC
                LIMIT 200
                """,
                (
                    self.organization_id,
                    self.namespace,
                    self.user_id,
                    self.role,
                    f"-{self.ttl_seconds} seconds",
                ),
                fetch=True,
            )

            if not candidates:
                return None

            candidate_embeddings: List[List[float]] = []
            candidate_rows: List[sqlite3.Row] = []

            for row in candidates:
                embedding = self._deserialize_embedding(
                    row["query_embedding"]
                )

                if embedding is None:
                    continue

                candidate_embeddings.append(embedding)
                candidate_rows.append(row)

            if not candidate_embeddings:
                return None

            similarities = self._compute_similarity_batch(
                query_embedding,
                candidate_embeddings,
            )

            if not similarities:
                return None

            best_idx = max(
                range(len(similarities)),
                key=lambda index: similarities[index],
            )

            best_similarity = float(similarities[best_idx])

            if best_similarity < threshold:
                return None

            best_row = candidate_rows[best_idx]
            new_hits = int(best_row["hits"] or 0) + 1

            self.conn_pool.execute_with_retry(
                """
                UPDATE query_cache
                SET last_accessed = CURRENT_TIMESTAMP,
                    hits = ?
                WHERE id = ?
                  AND organization_id = ?
                  AND namespace = ?
                """,
                (
                    new_hits,
                    best_row["id"],
                    self.organization_id,
                    self.namespace,
                    self.user_id,
                    self.role,
                ),
                fetch=False,
            )

            response = self._build_response(
                best_row,
                cache_type="semantic",
                similarity=best_similarity,
            )
            response["hits"] = new_hits

            self.memory_cache.set(
                query_hash,
                response,
            )

            logger.debug(
                "Semantic cache HIT: similarity=%.4f query=%s",
                best_similarity,
                normalized_query[:80],
            )

            return response

        except Exception as exc:
            logger.exception(
                "Semantic cache lookup error: %s",
                exc,
            )
            return None

    # -------------------------------------------------------------------------
    # SET
    # -------------------------------------------------------------------------

    def set(
        self,
        query: str,
        answer: str,
        chunks: Optional[List[Dict[str, Any]]] = None,
        async_mode: bool = True,
    ) -> None:
        """
        Store a query/answer pair.

        When async_mode=True, the database operation is queued so the RAG
        response path is not blocked by embedding computation.
        """
        if self._closed:
            raise RuntimeError("RAGCache is shut down")

        normalized_query = self._normalize_query(query)

        if not isinstance(answer, str):
            answer = str(answer)
        if len(answer) > 500_000:
            raise ValueError("answer is too large to cache")
        if chunks is not None and len(chunks) > 500:
            raise ValueError("too many chunks to cache")

        query_hash = self._compute_hash(normalized_query)

        chunks_json = (
            json.dumps(
                chunks,
                ensure_ascii=False,
                default=str,
            )
            if chunks
            else None
        )

        def _store() -> None:
            try:
                embedding_blob = None

                if self.embedder is not None:
                    embedding = self._compute_embeddings_batch(
                        [normalized_query]
                    )[0]

                    if embedding is not None:
                        embedding_blob = self._serialize_embedding(embedding)

                # UPDATE existing entry first so hits and row ID are
                # preserved. INSERT OR REPLACE would delete/recreate the row.
                existing = self.conn_pool.execute_with_retry(
                    """
                    SELECT id, hits
                    FROM query_cache
                    WHERE organization_id = ?
                      AND namespace = ?
                      AND user_id IS ?
                      AND role IS ?
                      AND query_hash = ?
                    LIMIT 1
                    """,
                    (
                        self.organization_id,
                        self.namespace,
                        self.user_id,
                        self.role,
                        query_hash,
                    ),
                    fetch=True,
                )

                if existing:
                    row = existing[0]

                    self.conn_pool.execute_with_retry(
                        """
                        UPDATE query_cache
                        SET query_text = ?,
                            query_embedding = ?,
                            answer = ?,
                            chunks_used = ?,
                            created_at = CURRENT_TIMESTAMP,
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE id = ?
                          AND organization_id = ?
                          AND namespace = ?
                        """,
                        (
                            normalized_query,
                            embedding_blob,
                            answer,
                            chunks_json,
                            row["id"],
                            self.organization_id,
                            self.namespace,
                            self.user_id,
                            self.role,
                        ),
                        fetch=False,
                    )

                    current_hits = int(row["hits"] or 0)

                else:
                    self.conn_pool.execute_with_retry(
                        """
                        INSERT INTO query_cache (
                            organization_id,
                            namespace,
                            user_id,
                            role,
                            query_hash,
                            query_text,
                            query_embedding,
                            answer,
                            chunks_used,
                            created_at,
                            last_accessed,
                            hits
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            CURRENT_TIMESTAMP,
                            CURRENT_TIMESTAMP,
                            0
                        )
                        """,
                        (
                            self.organization_id,
                            self.namespace,
                            self.user_id,
                            self.role,
                            query_hash,
                            normalized_query,
                            embedding_blob,
                            answer,
                            chunks_json,
                        ),
                        fetch=False,
                    )

                    current_hits = 0

                response = {
                    "answer": answer,
                    "chunks": chunks or [],
                    "cache_type": "new",
                    "hits": current_hits,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                }

                self.memory_cache.set(
                    query_hash,
                    response,
                )

                logger.debug(
                    "Cache SET: %s",
                    normalized_query[:80],
                )

            except Exception as exc:
                logger.exception(
                    "Cache set error: %s",
                    exc,
                )

        if async_mode:
            try:
                self._executor.submit(_store)
            except RuntimeError:
                # Executor may have been shut down during application exit.
                logger.warning(
                    "Cache executor unavailable; "
                    "falling back to synchronous cache write"
                )
                _store()
        else:
            _store()

    # -------------------------------------------------------------------------
    # BATCH SET
    # -------------------------------------------------------------------------

    def set_batch(
        self,
        items: List[
            Tuple[str, str, Optional[List[Dict[str, Any]]]]
        ],
    ) -> None:
        """Store multiple cache entries efficiently."""
        if self._closed:
            raise RuntimeError("RAGCache is shut down")

        if not items:
            return

        normalized_items = [
            (
                self._normalize_query(query),
                str(answer),
                chunks,
            )
            for query, answer, chunks in items
        ]

        queries = [item[0] for item in normalized_items]

        embeddings = (
            self._compute_embeddings_batch(queries)
            if self.embedder is not None
            else [None] * len(queries)
        )

        conn = self.conn_pool.get_connection()

        try:
            cursor = conn.cursor()

            for (
                query,
                answer,
                chunks,
            ), embedding in zip(
                normalized_items,
                embeddings,
            ):
                query_hash = self._compute_hash(query)

                chunks_json = (
                    json.dumps(
                        chunks,
                        ensure_ascii=False,
                        default=str,
                    )
                    if chunks
                    else None
                )

                embedding_blob = (
                    self._serialize_embedding(embedding)
                    if embedding is not None
                    else None
                )

                # Preserve existing row IDs/hits.
                existing = cursor.execute(
                    """
                    SELECT id
                    FROM query_cache
                    WHERE organization_id = ?
                      AND namespace = ?
                      AND user_id IS ?
                      AND role IS ?
                      AND query_hash = ?
                    LIMIT 1
                    """,
                    (
                        self.organization_id,
                        self.namespace,
                        self.user_id,
                        self.role,
                        query_hash,
                    ),
                ).fetchone()

                if existing:
                    cursor.execute(
                        """
                        UPDATE query_cache
                        SET query_text = ?,
                            query_embedding = ?,
                            answer = ?,
                            chunks_used = ?,
                            created_at = CURRENT_TIMESTAMP,
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE id = ?
                          AND organization_id = ?
                          AND namespace = ?
                          AND user_id IS ?
                          AND role IS ?
                        """,
                        (
                            query,
                            embedding_blob,
                            answer,
                            chunks_json,
                            existing["id"],
                            self.organization_id,
                            self.namespace,
                            self.user_id,
                            self.role,
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO query_cache (
                            organization_id,
                            namespace,
                            user_id,
                            role,
                            query_hash,
                            query_text,
                            query_embedding,
                            answer,
                            chunks_used,
                            created_at,
                            last_accessed,
                            hits
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            CURRENT_TIMESTAMP,
                            CURRENT_TIMESTAMP,
                            0
                        )
                        """,
                        (
                            self.organization_id,
                            self.namespace,
                            self.user_id,
                            self.role,
                            query_hash,
                            query,
                            embedding_blob,
                            answer,
                            chunks_json,
                        ),
                    )

                self.memory_cache.set(
                    query_hash,
                    {
                        "answer": answer,
                        "chunks": chunks or [],
                        "cache_type": "new",
                        "hits": 0,
                        "created_at": datetime.now().isoformat(),
                        "last_accessed": datetime.now().isoformat(),
                    },
                )

            conn.commit()

            logger.info(
                "Batch cached %s items",
                len(normalized_items),
            )

        except Exception as exc:
            conn.rollback()
            logger.exception(
                "Batch cache error: %s",
                exc,
            )

        finally:
            self.conn_pool.return_connection(conn)

    # -------------------------------------------------------------------------
    # CLEAR / STATS
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Clear both memory and persistent cache."""
        if self._closed:
            raise RuntimeError("RAGCache is shut down")

        try:
            self.memory_cache.clear()

            self.conn_pool.execute_with_retry(
                """
                DELETE FROM query_cache
                WHERE organization_id = ? AND namespace = ?
                  AND user_id IS ? AND role IS ?
                """,
                (
                    self.organization_id,
                    self.namespace,
                    self.user_id,
                    self.role,
                ),
                fetch=False,
            )

            logger.info(
                "RAG cache cleared: memory + SQLite"
            )

        except Exception as exc:
            logger.exception(
                "Cache clear error: %s",
                exc,
            )
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        try:
            memory_stats = self.memory_cache.stats()

            rows = self.conn_pool.execute_with_retry(
                """
                SELECT
                    COUNT(*) AS total_entries,
                    COALESCE(SUM(hits), 0) AS total_hits,
                    SUM(
                        CASE
                            WHEN query_embedding IS NOT NULL
                            THEN 1
                            ELSE 0
                        END
                    ) AS embedded_entries
                FROM query_cache
                WHERE organization_id = ?
                  AND namespace = ?
                  AND user_id IS ?
                  AND role IS ?
                """,
                (
                    self.organization_id,
                    self.namespace,
                    self.user_id,
                    self.role,
                ),
                fetch=True,
            )

            row = rows[0]

            total_entries = int(
                row["total_entries"] or 0
            )
            total_hits = int(
                row["total_hits"] or 0
            )
            embedded_entries = int(
                row["embedded_entries"] or 0
            )

            # This is intentionally called "average hits per entry".
            # SUM(hits) / entries is NOT a true request hit rate because
            # misses are not persisted by this cache.
            average_hits = (
                total_hits / total_entries
                if total_entries
                else 0.0
            )

            return {
                "memory_cache": memory_stats,
                "sqlite": {
                    "total_entries": total_entries,
                    "with_embeddings": embedded_entries,
                    "total_hits": total_hits,
                    "average_hits_per_entry": round(
                        average_hits,
                        2,
                    ),
                },
                "config": {
                    "similarity_threshold": self.similarity_threshold,
                    "ttl_seconds": self.ttl_seconds,
                    "batch_size": self.batch_size,
                    "max_workers": self.max_workers,
                    "db_path": str(self.db_path),
                    "organization_id": self.organization_id,
                    "namespace": self.namespace,
                    "user_id": self.user_id,
                    "role": self.role,
                },
                "status": {
                    "closed": self._closed,
                    "embedder_available": self.embedder is not None,
                },
            }

        except Exception as exc:
            logger.exception(
                "Cache statistics error: %s",
                exc,
            )
            return {"error": str(exc)}

    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------

    def shutdown(self) -> None:
        """Gracefully shut down cache workers and database connections."""
        with self._state_lock:
            if self._closed:
                return

            self._closed = True

        logger.info("Shutting down RAGCache...")

        # Stop cleanup thread.
        self._stop_cleanup.set()

        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)

        # Python's ThreadPoolExecutor.shutdown() does not accept a timeout.
        # wait=True is the portable API.
        self._executor.shutdown(wait=True)

        self.memory_cache.clear()
        self.conn_pool.close_all()

        logger.info("RAGCache shutdown complete")

    def __enter__(self) -> "RAGCache":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> bool:
        """Context manager exit."""
        self.shutdown()
        return False


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================


_cache_instances: Dict[Tuple[str, str, str, Optional[str], Optional[str]], RAGCache] = {}
_cache_lock = threading.Lock()


def get_cache_instance(
    db_path: Optional[str] = None,
    batch_size: int = 16,
    max_workers: int = 4,
    organization_id: str = "default",
    namespace: str = "policy",
    user_id: Optional[str] = None,
    role: Optional[str] = None,
) -> RAGCache:
    """Return a process-wide cache singleton scoped by database, tenant, and namespace."""
    resolved_db = str(
        Path(db_path) if db_path else project_root / "data" / "cache.db"
    )
    scope = (
        resolved_db,
        str(organization_id),
        str(namespace),
        str(user_id) if user_id is not None else None,
        str(role).lower() if role is not None else None,
    )

    with _cache_lock:
        instance = _cache_instances.get(scope)

        if instance is None:
            instance = RAGCache(
                db_path=db_path,
                batch_size=batch_size,
                max_workers=max_workers,
                organization_id=organization_id,
                namespace=namespace,
                user_id=user_id,
                role=role,
            )
            _cache_instances[scope] = instance

        return instance


def reset_cache_instance(
    organization_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> None:
    """Shutdown and remove one scoped cache, or all cache instances."""
    with _cache_lock:
        if organization_id is None and namespace is None:
            instances = list(_cache_instances.items())
            _cache_instances.clear()
        else:
            instances = []
            for key in list(_cache_instances):
                _, org, ns, _, _ = key
                if (
                    organization_id is not None
                    and org != str(organization_id)
                ):
                    continue
                if (
                    namespace is not None
                    and ns != str(namespace)
                ):
                    continue
                instances.append((key, _cache_instances.pop(key)))

    for _, instance in instances:
        try:
            instance.shutdown()
        except Exception as exc:
            logger.warning(
                "Error shutting down cache during reset: %s",
                exc,
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_cached_answer(
    query: str,
    threshold: Optional[float] = None,
    use_semantic: bool = True,
    organization_id: str = "default",
    namespace: str = "policy",
    user_id: Optional[str] = None,
    role: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get a cached answer."""
    return get_cache_instance(
        organization_id=organization_id,
        namespace=namespace,
        user_id=user_id,
        role=role,
    ).get(
        query=query,
        threshold=threshold,
        use_semantic=use_semantic,
    )


def cache_answer(
    query: str,
    answer: str,
    chunks: Optional[List[Dict[str, Any]]] = None,
    async_mode: bool = True,
    organization_id: str = "default",
    namespace: str = "policy",
    user_id: Optional[str] = None,
    role: Optional[str] = None) -> None:
    """Cache a generated answer."""
    get_cache_instance(
        organization_id=organization_id,
        namespace=namespace,
        user_id=user_id,
        role=role,
    ).set(
        query=query,
        answer=answer,
        chunks=chunks,
        async_mode=async_mode,
    )


def cache_batch_answers(
    items: List[
        Tuple[str, str, Optional[List[Dict[str, Any]]]]
    ],
    organization_id: str = "default",
    namespace: str = "policy",
    user_id: Optional[str] = None,
    role: Optional[str] = None) -> None:
    """Cache multiple generated answers."""
    get_cache_instance(
        organization_id=organization_id,
        namespace=namespace,
        user_id=user_id,
        role=role,
    ).set_batch(items)


def get_cache_statistics(
    organization_id: str = "default",
    namespace: str = "policy",
    user_id: Optional[str] = None,
    role: Optional[str] = None) -> Dict[str, Any]:
    """Return cache statistics for a tenant/namespace scope."""
    return get_cache_instance(
        organization_id=organization_id,
        namespace=namespace,
        user_id=user_id,
        role=role,
    ).get_stats()


def clear_all_cache(
    organization_id: str = "default",
    namespace: str = "policy",
    user_id: Optional[str] = None,
    role: Optional[str] = None) -> None:
    """Clear all persistent and memory cache entries for a scope."""
    get_cache_instance(
        organization_id=organization_id,
        namespace=namespace,
        user_id=user_id,
        role=role,
    ).clear()


# =============================================================================
# TEST / DEMO
# =============================================================================


def test_optimized_cache() -> None:
    """Basic cache smoke test."""
    print("\nPolicyGuard AI - RAG Cache Test\n")
    print("=" * 70)

    cache = get_cache_instance(
        batch_size=8,
        max_workers=2,
    )

    print(f"Database: {cache.db_path}")
    print(
        f"Config: batch={cache.batch_size}, "
        f"workers={cache.max_workers}"
    )
    print(
        f"Threshold: {cache.similarity_threshold}, "
        f"TTL={cache.ttl_seconds}s"
    )
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Test 1
    # -------------------------------------------------------------------------

    print("\nTest 1: Memory + SQLite cache")

    query = "What is the leave policy?"
    answer = "Employees get 20 days paid leave annually."

    result = cache.get(
        query,
        use_semantic=False,
    )

    print(
        "Initial lookup:",
        "HIT" if result else "MISS",
    )

    cache.set(
        query,
        answer,
        [
            {
                "source": "policy.pdf",
                "score": 0.92,
            }
        ],
        async_mode=False,
    )

    result = cache.get(
        query,
        use_semantic=False,
    )

    print(
        "Second lookup:",
        "MEMORY HIT" if result else "MISS",
    )

    if result:
        print(
            f"Answer: {result['answer'][:60]}..."
        )
        print(
            f"Type: {result['cache_type']}"
        )

    # -------------------------------------------------------------------------
    # Test 2
    # -------------------------------------------------------------------------

    print("\nTest 2: Batch cache")

    batch_items = [
        (
            "What is PTO?",
            "PTO means Paid Time Off.",
            None,
        ),
        (
            "What is the sick leave policy?",
            "Employees receive sick leave according to policy.",
            None,
        ),
        (
            "What are the remote work rules?",
            "Employees may work remotely according to company policy.",
            None,
        ),
    ]

    cache.set_batch(batch_items)

    print(
        f"Batch cached: {len(batch_items)} items"
    )

    # -------------------------------------------------------------------------
    # Test 3
    # -------------------------------------------------------------------------

    print("\nTest 3: Semantic matching")

    semantic_query = (
        "How many vacation days do employees receive?"
    )

    result = cache.get(
        semantic_query,
        use_semantic=True,
    )

    if (
        result
        and result.get("cache_type") == "semantic"
    ):
        print(
            "Semantic HIT:",
            f"{result.get('similarity', 0.0):.3f}",
        )
    elif result:
        print("Exact/cache HIT")
    else:
        print(
            "No semantic match found "
            "(embedder or threshold may prevent a match)"
        )

    # -------------------------------------------------------------------------
    # Test 4
    # -------------------------------------------------------------------------

    print("\nTest 4: Statistics")

    stats = cache.get_stats()

    if "error" not in stats:
        print(
            "Memory:",
            stats["memory_cache"]["size"],
            "/",
            stats["memory_cache"]["max_size"],
        )
        print(
            "SQLite entries:",
            stats["sqlite"]["total_entries"],
        )
        print(
            "Embedded entries:",
            stats["sqlite"]["with_embeddings"],
        )
        print(
            "Total hits:",
            stats["sqlite"]["total_hits"],
        )
        print(
            "Average hits/entry:",
            stats["sqlite"]["average_hits_per_entry"],
        )

    # -------------------------------------------------------------------------
    # Test 5
    # -------------------------------------------------------------------------

    print("\nTest 5: Exact lookup performance")

    start = time.perf_counter()

    for _ in range(10):
        cache.get(
            query,
            use_semantic=False,
        )

    elapsed = (
        time.perf_counter() - start
    ) / 10

    print(
        f"Average lookup: {elapsed * 1000:.2f} ms"
    )

    print("\n" + "=" * 70)
    print("Cache test complete.")


if __name__ == "__main__":
    test_optimized_cache()
