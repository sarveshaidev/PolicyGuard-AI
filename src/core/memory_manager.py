
#!/usr/bin/env python3
"""
PolicyGuard AI - Memory Manager Module
======================================
Thread-safe conversation history management with:

- Sliding-window context retention
- Token-aware context truncation
- Per-user memory isolation
- Safe, deterministic history compression
- JSON export/import for audit and debugging
- Memory statistics
- Global singleton management

Notes:
- This module intentionally keeps conversation memory in process memory.
- It does NOT persist conversation history to the database.
- Compression is extractive/deterministic; it does not invent policy facts.
- A future LLM-based summarizer can be added without changing the public API.

Author: PolicyGuard AI Team
Version: 2.0.0
Last Updated: 2026-09-13
"""

import json
import logging
import sys
import threading
import sqlite3
import os
import secrets
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

VALID_ROLES = {"user", "assistant", "system"}

DEFAULT_MAX_MESSAGES = 10
DEFAULT_MAX_TOKENS = 2000
DEFAULT_SUMMARY_THRESHOLD = 20

# Number of recent messages retained when compression occurs.
COMPRESSION_RECENT_MESSAGES = 6

# Maximum content length accepted for one message.
MAX_MESSAGE_CONTENT_LENGTH = 100_000

# Maximum number of imported messages processed from an external file.
MAX_IMPORT_MESSAGES = 10_000

# Persistent memory settings.
MEMORY_DB_TIMEOUT_SECONDS = 30
MEMORY_DB_MAX_RETRIES = 4
MEMORY_SCHEMA_VERSION = 1
MAX_PERSISTED_MESSAGES_PER_USER = 500
MAX_MEMORY_METADATA_KEYS = 50
MAX_MEMORY_METADATA_VALUE_LENGTH = 2_000


# =============================================================================
# HELPERS
# =============================================================================

def _utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count without requiring a tokenizer.

    This is deliberately conservative enough for context-window selection.
    It is only an approximation and must not be treated as an exact tokenizer.
    """
    if not text:
        return 0

    # Rough approximation:
    # ~4 characters/token for ordinary English text, with a small overhead.
    return max(1, (len(text) + 3) // 4)


def _validate_user_id(user_id: str) -> bool:
    """Validate a user identifier."""
    return isinstance(user_id, str) and bool(user_id.strip()) and len(user_id.strip()) <= 200


def _normalize_organization_id(organization_id: Optional[str]) -> str:
    """Validate a tenant/organization identifier."""
    org = (organization_id or "default").strip()
    if not org:
        org = "default"
    if len(org) > 128 or not re.fullmatch(r"[A-Za-z0-9_.:@-]+", org):
        raise ValueError("Invalid organization_id")
    return org



def _resolve_memory_db_path() -> Path:
    """Resolve the shared SQLite database path used by PolicyGuard AI."""
    database_url = str(getattr(getattr(__import__("config.settings", fromlist=["settings"]), "settings", None), "DATABASE_URL", "") or "")
    if not database_url.startswith("sqlite:///"):
        # Keep the memory layer operational when a non-SQLite deployment is
        # configured. A dedicated persistence adapter can replace this later.
        return project_root / "data" / "policyguard_memory.db"

    raw_path = database_url[len("sqlite:///"):]
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


MEMORY_DB_FILE = _resolve_memory_db_path()


def _safe_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Bound metadata before it can enter memory or persistent storage."""
    if not isinstance(metadata, dict):
        return {}
    items = list(metadata.items())[:MAX_MEMORY_METADATA_KEYS]
    safe: Dict[str, Any] = {}
    for key, value in items:
        key_text = str(key)[:200]
        try:
            encoded = json.dumps(value, ensure_ascii=False, default=str)
            if len(encoded) > MAX_MEMORY_METADATA_VALUE_LENGTH:
                encoded = encoded[:MAX_MEMORY_METADATA_VALUE_LENGTH]
            safe[key_text] = json.loads(encoded)
        except Exception:
            safe[key_text] = str(value)[:MAX_MEMORY_METADATA_VALUE_LENGTH]
    return safe


def _memory_scope(user_id: str, organization_id: Optional[str]) -> str:
    """Build a collision-resistant tenant/user scope key."""
    org = _normalize_organization_id(organization_id)
    user = user_id.strip()
    return f"{org}:{user}"


def _validate_role(role: str) -> bool:
    """Validate a supported conversation role."""
    return isinstance(role, str) and role.strip().lower() in VALID_ROLES


def _normalize_message(message: Any) -> Optional[Dict[str, Any]]:
    """
    Validate and normalize one imported/stored message.

    Returns:
        Normalized message or None if invalid.
    """
    if not isinstance(message, dict):
        return None

    role = str(message.get("role", "")).strip().lower()
    content = message.get("content", "")

    if role not in VALID_ROLES:
        return None

    if not isinstance(content, str):
        return None

    if not content.strip():
        return None

    if len(content) > MAX_MESSAGE_CONTENT_LENGTH:
        return None

    timestamp = message.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        timestamp = _utc_now_iso()

    metadata = message.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "metadata": _safe_metadata(metadata),
    }


# =============================================================================
# THREAD-SAFE MEMORY MANAGER
# =============================================================================

class MemoryManager:
    """
    Thread-safe manager for per-user conversation history.

    Memory is isolated by user_id. Each user has an independent re-entrant
    lock so concurrent users do not unnecessarily block one another.

    The manager keeps a bounded history but also supports token-aware context
    construction for the current query.
    """

    def __init__(
        self,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        summary_threshold: int = DEFAULT_SUMMARY_THRESHOLD,
        *,
        persistent: bool = True,
        organization_id: Optional[str] = None,
    ):
        """
        Initialize memory manager.

        Args:
            max_messages:
                Maximum number of messages retained per user.

            max_tokens:
                Approximate maximum number of tokens used when constructing
                query context.

            summary_threshold:
                Number of messages at which compression becomes eligible.
                The manager will internally use a larger temporary history
                when necessary so this threshold can actually be reached.
        """
        self.max_messages = self._validate_positive_int(
            max_messages,
            "max_messages",
        )
        self.max_tokens = self._validate_positive_int(
            max_tokens,
            "max_tokens",
        )
        self.summary_threshold = self._validate_positive_int(
            summary_threshold,
            "summary_threshold",
        )
        self.persistent = bool(persistent)
        self.default_organization_id = (
            organization_id.strip()
            if isinstance(organization_id, str) and organization_id.strip()
            else None
        )
        self._persistent_initialized_scopes: set[str] = set()

        if self.persistent:
            self._init_persistent_store()

        # Never allow compression threshold to be smaller than the retained
        # history in a way that causes immediate repeated compression.
        self._compression_recent_messages = min(
            COMPRESSION_RECENT_MESSAGES,
            self.max_messages,
        )

        # Per-user memory.
        #
        # Important:
        # The deque maxlen is deliberately larger than max_messages so that
        # compression can happen BEFORE old messages are silently discarded.
        self._memories: Dict[str, deque] = {}

        # Per-user locks.
        self._locks: Dict[str, threading.RLock] = {}

        # Protects creation/access of user locks and global operations.
        self._global_lock = threading.RLock()

        logger.info(
            "MemoryManager initialized: "
            "max_messages=%s, max_tokens=%s, summary_threshold=%s, persistent=%s",
            self.max_messages,
            self.max_tokens,
            self.summary_threshold,
            self.persistent,
        )

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        """Validate a positive integer configuration value."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")

        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")

        return value

    # -------------------------------------------------------------------------
    # PERSISTENT STORAGE
    # -------------------------------------------------------------------------

    def _persistent_connection(self) -> sqlite3.Connection:
        """Open the shared SQLite persistence connection for memory."""
        MEMORY_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(MEMORY_DB_FILE),
            timeout=MEMORY_DB_TIMEOUT_SECONDS,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_persistent_store(self) -> None:
        """Create the persistent memory schema without deleting existing data."""
        try:
            with self._persistent_connection() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS memory_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scope_key TEXT NOT NULL,
                        organization_id TEXT NOT NULL DEFAULT 'default',
                        user_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_memory_scope
                        ON memory_messages(scope_key, id);

                    CREATE INDEX IF NOT EXISTS idx_memory_user
                        ON memory_messages(organization_id, user_id, id);

                    CREATE TABLE IF NOT EXISTS memory_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    "INSERT OR IGNORE INTO memory_metadata(key,value) VALUES(?,?)",
                    ("schema_version", str(MEMORY_SCHEMA_VERSION)),
                )
                conn.commit()
            try:
                os.chmod(MEMORY_DB_FILE, 0o600)
            except OSError:
                pass
        except Exception:
            logger.exception("Persistent memory initialization failed")
            # Memory remains usable in-process if persistence is unavailable.
            self.persistent = False

    def _scope_parts(
        self,
        user_id: str,
        organization_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Resolve and validate the tenant/user scope."""
        if not _validate_user_id(user_id):
            raise ValueError("user_id must be a non-empty string")
        org = organization_id if organization_id is not None else self.default_organization_id
        org = (org or "default").strip()
        if not org:
            org = "default"
        org = _normalize_organization_id(org)
        return org, _memory_scope(user_id, org)

    def _load_persistent_history_locked(
        self,
        user_id: str,
        organization_id: Optional[str] = None,
    ) -> None:
        """Load persisted messages into the process-local buffer once per scope."""
        if not self.persistent:
            return
        org, scope = self._scope_parts(user_id, organization_id)
        if scope in self._persistent_initialized_scopes:
            return
        try:
            with self._persistent_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT role, content, timestamp, metadata_json
                    FROM memory_messages
                    WHERE scope_key = ? AND organization_id = ? AND user_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (scope, org, user_id, MAX_PERSISTED_MESSAGES_PER_USER),
                ).fetchall()
            memory = self._memories.get(scope)
            if memory is None:
                memory = self._create_memory()
                self._memories[scope] = memory
            for row in reversed(rows):
                try:
                    metadata = json.loads(row["metadata_json"] or "{}")
                except json.JSONDecodeError:
                    metadata = {}
                message = _normalize_message({
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "metadata": metadata,
                })
                if message:
                    memory.append(message)
            self._persistent_initialized_scopes.add(scope)
        except Exception:
            logger.exception("Could not load persistent memory for user %s", user_id)

    def _persist_message(
        self,
        user_id: str,
        message: Dict[str, Any],
        organization_id: Optional[str] = None,
    ) -> None:
        """Persist one message using a tenant-scoped key."""
        if not self.persistent:
            return
        org, scope = self._scope_parts(user_id, organization_id)
        payload = json.dumps(
            _safe_metadata(message.get("metadata")),
            ensure_ascii=False,
            default=str,
        )
        for attempt in range(MEMORY_DB_MAX_RETRIES):
            try:
                with self._persistent_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO memory_messages(
                            scope_key, organization_id, user_id,
                            role, content, timestamp, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            scope, org, user_id,
                            message["role"], message["content"],
                            message["timestamp"], payload,
                        ),
                    )
                    # Keep persistent storage bounded per user.
                    conn.execute(
                        """
                        DELETE FROM memory_messages
                        WHERE organization_id = ? AND user_id = ?
                          AND id NOT IN (
                              SELECT id FROM memory_messages
                              WHERE organization_id = ? AND user_id = ?
                              ORDER BY id DESC LIMIT ?
                          )
                        """,
                        (org, user_id, org, user_id, MAX_PERSISTED_MESSAGES_PER_USER),
                    )
                    conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < MEMORY_DB_MAX_RETRIES - 1:
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                raise
            except Exception:
                logger.exception("Persistent memory write failed for user %s", user_id)
                return

    def _delete_persistent_scope(
        self,
        user_id: str,
        organization_id: Optional[str] = None,
    ) -> bool:
        if not self.persistent:
            return False
        org, scope = self._scope_parts(user_id, organization_id)
        try:
            with self._persistent_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM memory_messages WHERE scope_key = ? AND organization_id = ? AND user_id = ?",
                    (scope, org, user_id),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            logger.exception("Persistent memory deletion failed for user %s", user_id)
            return False

    # -------------------------------------------------------------------------
    # LOCKING
    # -------------------------------------------------------------------------

    def _get_user_lock(
        self,
        user_id: str,
        organization_id: Optional[str] = None,
    ) -> threading.RLock:
        """Get or create a lock for one tenant + user scope."""
        if not _validate_user_id(user_id):
            raise ValueError("user_id must be a non-empty string")
        _, scope = self._scope_parts(user_id, organization_id)
        with self._global_lock:
            lock = self._locks.get(scope)
            if lock is None:
                lock = threading.RLock()
                self._locks[scope] = lock
            return lock

    # -------------------------------------------------------------------------
    # MEMORY STORAGE
    # -------------------------------------------------------------------------

    def _create_memory(self) -> deque:
        """
        Create a user memory buffer.

        The buffer has enough room for the compression threshold plus recent
        messages. This fixes the original situation where max_messages=10 and
        summary_threshold=20 made compression impossible.
        """
        buffer_size = max(
            self.max_messages,
            self.summary_threshold,
            self._compression_recent_messages + 1,
        )

        return deque(maxlen=buffer_size)

    def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        organization_id: Optional[str] = None,
    ) -> bool:
        """
        Add a message to a user's conversation history.

        Args:
            user_id: Unique user identifier.
            role: user, assistant, or system.
            content: Message text.
            metadata: Optional metadata dictionary.

        Returns:
            True if successfully added, otherwise False.
        """
        if not _validate_user_id(user_id):
            logger.warning("Rejected message with invalid user_id")
            return False

        if not _validate_role(role):
            logger.warning(
                "Rejected message with invalid role for user %s: %r",
                user_id,
                role,
            )
            return False

        if not isinstance(content, str) or not content.strip():
            logger.warning(
                "Rejected empty/non-string message for user %s",
                user_id,
            )
            return False

        if len(content) > MAX_MESSAGE_CONTENT_LENGTH:
            logger.warning(
                "Rejected oversized message for user %s (%s characters)",
                user_id,
                len(content),
            )
            return False

        if metadata is not None and not isinstance(metadata, dict):
            logger.warning(
                "Rejected message with invalid metadata for user %s",
                user_id,
            )
            return False

        lock = self._get_user_lock(user_id, organization_id)
        _, scope = self._scope_parts(user_id, organization_id)

        with lock:
            self._load_persistent_history_locked(user_id, organization_id)
            if scope not in self._memories:
                self._memories[scope] = self._create_memory()

            message = {
                "role": role.strip().lower(),
                "content": content,
                "timestamp": _utc_now_iso(),
                "metadata": _safe_metadata(metadata),
            }

            self._memories[scope].append(message)
            self._persist_message(user_id, message, organization_id)

            # Compress before the bounded buffer has to discard useful history.
            if len(self._memories[scope]) >= self.summary_threshold:
                self._compress_history_locked(user_id, organization_id)

            logger.debug(
                "Added %s message for user %s (stored=%s)",
                role,
                user_id,
                len(self._memories[scope]),
            )

            return True

    def get_history(
        self,
        user_id: str,
        include_system: bool = True,
        limit: Optional[int] = None,
        *,
        organization_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.

        Returns a copy so callers cannot mutate internal memory accidentally.
        """
        if not _validate_user_id(user_id):
            return []

        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise ValueError("limit must be an integer or None")

            if limit < 0:
                raise ValueError("limit cannot be negative")

        lock = self._get_user_lock(user_id, organization_id)
        _, scope = self._scope_parts(user_id, organization_id)

        with lock:
            self._load_persistent_history_locked(user_id, organization_id)
            if scope not in self._memories:
                return []

            messages = list(self._memories[scope])

            if not include_system:
                messages = [
                    message
                    for message in messages
                    if message.get("role") != "system"
                ]

            if limit is not None and limit > 0:
                messages = messages[-limit:]
            elif limit == 0:
                messages = []

            # Return independent dictionaries.
            return [
                {
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "timestamp": message.get("timestamp"),
                    "metadata": dict(message.get("metadata", {})),
                }
                for message in messages
            ]

    # -------------------------------------------------------------------------
    # CONTEXT BUILDING
    # -------------------------------------------------------------------------

    def get_context_for_query(
        self,
        user_id: str,
        current_query: str,
        max_tokens: Optional[int] = None,
        *,
        organization_id: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build a context window for the current query.

        Messages are selected from newest to oldest until the approximate
        token budget is reached, then returned in chronological order.
        """
        if not _validate_user_id(user_id):
            raise ValueError("user_id must be a non-empty string")

        if not isinstance(current_query, str):
            raise ValueError("current_query must be a string")

        if max_tokens is None:
            max_tokens = self.max_tokens
        else:
            max_tokens = self._validate_positive_int(
                max_tokens,
                "max_tokens",
            )

        messages = self.get_history(user_id, organization_id=organization_id)

        if not messages:
            return f"Current Query: {current_query}", []

        selected_reversed: List[Dict[str, Any]] = []
        total_tokens = 0

        # Reserve a small amount of budget for the current query itself.
        query_tokens = _estimate_tokens(current_query)
        available_tokens = max(1, max_tokens - query_tokens)

        for message in reversed(messages):
            content = str(message.get("content", ""))
            role = str(message.get("role", "unknown"))

            message_tokens = _estimate_tokens(content) + 6

            # If even one message is larger than the available budget, keep
            # the most recent message with truncated content rather than
            # returning no context at all.
            if not selected_reversed and message_tokens > available_tokens:
                max_chars = max(1, (available_tokens - 6) * 4)

                truncated_content = content[:max_chars].rstrip()

                if len(truncated_content) < len(content):
                    truncated_content += "…"

                selected_reversed.append(
                    {
                        **message,
                        "content": truncated_content,
                    }
                )
                total_tokens = available_tokens
                break

            if total_tokens + message_tokens > available_tokens:
                break

            selected_reversed.append(message)
            total_tokens += message_tokens

        included_messages = list(reversed(selected_reversed))

        context_parts = []

        for message in included_messages:
            role_label = str(
                message.get("role", "unknown")
            ).upper()

            context_parts.append(
                f"{role_label}: {message.get('content', '')}"
            )

        if context_parts:
            context = (
                "Conversation History:\n"
                + "\n".join(context_parts)
                + f"\n\nCurrent Query: {current_query}"
            )
        else:
            context = f"Current Query: {current_query}"

        logger.debug(
            "Built context for user %s: messages=%s, estimated_tokens=%s",
            user_id,
            len(included_messages),
            total_tokens,
        )

        return context, included_messages

    # -------------------------------------------------------------------------
    # COMPRESSION
    # -------------------------------------------------------------------------

    def _compress_history_locked(self, user_id: str, organization_id: Optional[str] = None) -> None:
        """
        Compress older history while preserving recent conversation turns.

        This method expects the caller to already hold the user's lock.

        Unlike the previous implementation, this does NOT insert invented
        policy facts such as specific leave balances. It creates a neutral
        extractive summary containing actual earlier messages.
        """
        _, scope = self._scope_parts(user_id, organization_id)
        messages = self._memories.get(scope)

        if not messages:
            return

        if len(messages) < self.summary_threshold:
            return

        # Keep enough recent messages to preserve immediate conversational
        # context.
        recent_count = min(
            self._compression_recent_messages,
            len(messages),
        )

        older_messages = list(messages)[:-recent_count]
        recent_messages = list(messages)[-recent_count:]

        if not older_messages:
            return

        summary_text = self._build_extractive_summary(older_messages)

        summary_message = {
            "role": "system",
            "content": summary_text,
            "timestamp": _utc_now_iso(),
            "metadata": {
                "compressed": True,
                "compression_method": "extractive",
                "original_count": len(older_messages),
            },
        }

        # After compression we keep the summary plus recent messages.
        compressed_messages = [summary_message] + recent_messages

        # The active context history should never exceed max_messages.
        compressed_messages = compressed_messages[-self.max_messages:]

        self._memories[scope] = deque(
            compressed_messages,
            maxlen=max(
                self.max_messages,
                self.summary_threshold,
                self._compression_recent_messages + 1,
            ),
        )

        logger.info(
            "Compressed history for user %s: %s messages -> %s messages",
            user_id,
            len(messages),
            len(self._memories[scope]),
        )

    @staticmethod
    def _build_extractive_summary(
        messages: List[Dict[str, Any]],
    ) -> str:
        """
        Build a safe extractive summary from actual conversation content.

        This deliberately avoids pretending to understand policy facts. The
        content is taken directly from the user's earlier messages and/or
        assistant responses.
        """
        if not messages:
            return "[Earlier conversation history was empty.]"

        # Keep a bounded amount of text in the summary so repeated compression
        # does not create an ever-growing system message.
        max_summary_chars = 4_000
        lines: List[str] = []

        for message in messages:
            role = str(message.get("role", "unknown")).upper()
            content = str(message.get("content", "")).strip()

            if not content:
                continue

            # Preserve actual content, but cap each entry.
            max_entry_chars = 500

            if len(content) > max_entry_chars:
                content = content[:max_entry_chars].rstrip() + "…"

            lines.append(f"- {role}: {content}")

            if sum(len(line) + 1 for line in lines) >= max_summary_chars:
                break

        if not lines:
            return "[Earlier conversation history could not be summarized.]"

        summary = (
            "[Summary of earlier conversation. "
            "This summary is extractive and contains only prior messages:]\n"
            + "\n".join(lines)
        )

        if len(summary) > max_summary_chars:
            summary = summary[:max_summary_chars].rstrip() + "…"

        return summary

    # -------------------------------------------------------------------------
    # CLEARING
    # -------------------------------------------------------------------------

    def clear_user_memory(
        self,
        user_id: str,
        *,
        organization_id: Optional[str] = None,
    ) -> bool:
        """Clear all conversation memory for one user."""
        if not _validate_user_id(user_id):
            return False

        lock = self._get_user_lock(user_id, organization_id)
        _, scope = self._scope_parts(user_id, organization_id)

        with lock:
            self._load_persistent_history_locked(user_id, organization_id)
            existed = scope in self._memories
            persistent_existed = self._delete_persistent_scope(user_id, organization_id)

            if existed:
                del self._memories[scope]

                # Do NOT remove the lock here.
                #
                # Removing the lock while another thread can still obtain a
                # new lock for the same user creates two independent locks and
                # can cause concurrent access to the same memory state.
                logger.info(
                    "Cleared conversation memory for user %s",
                    user_id,
                )

            return existed or persistent_existed

    def clear_all_memory(self) -> None:
        """Clear all user memories safely."""
        with self._global_lock:
            locks = list(self._locks.values())

            # Acquire all existing user locks while holding the global lock.
            # add/get operations only need the global lock to obtain a lock,
            # then operate under the user lock, so this prevents concurrent
            # mutations during the clear operation.
            for lock in locks:
                lock.acquire()

            try:
                self._memories.clear()
                self._persistent_initialized_scopes.clear()
                if self.persistent:
                    try:
                        with self._persistent_connection() as conn:
                            conn.execute("DELETE FROM memory_messages")
                            conn.commit()
                    except Exception:
                        logger.exception("Persistent memory clear-all failed")
            finally:
                for lock in reversed(locks):
                    lock.release()

        logger.info("Cleared all conversation memory")

    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------

    def get_stats(
        self,
        user_id: Optional[str] = None,
        *,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get memory statistics.

        If user_id is supplied, returns user-specific statistics.
        Otherwise returns aggregate statistics.
        """
        if user_id is not None:
            if not _validate_user_id(user_id):
                return {
                    "error": "Invalid user_id",
                    "user_id": user_id,
                }

            lock = self._get_user_lock(user_id, organization_id)
            _, scope = self._scope_parts(user_id, organization_id)

            with lock:
                self._load_persistent_history_locked(user_id, organization_id)
                messages = self._memories.get(scope)

                if messages is None:
                    return {
                        "error": "User not found",
                        "user_id": user_id,
                    }

                message_list = list(messages)

                return {
                    "user_id": user_id,
                    "message_count": len(message_list),
                    "max_messages": self.max_messages,
                    "oldest_timestamp": (
                        message_list[0].get("timestamp")
                        if message_list
                        else None
                    ),
                    "newest_timestamp": (
                        message_list[-1].get("timestamp")
                        if message_list
                        else None
                    ),
                    "compression_applied": any(
                        bool(
                            message.get("metadata", {}).get("compressed")
                        )
                        for message in message_list
                    ),
                }

        with self._global_lock:
            scopes = list(self._memories.keys())
            memories = [
                self._memories[scope]
                for scope in scopes
            ]

            total_messages = sum(
                len(memory)
                for memory in memories
            )

            total_users = len(memories)

            return {
                "total_users": total_users,
                "total_messages": total_messages,
                "avg_messages_per_user": (
                    total_messages / total_users
                    if total_users
                    else 0.0
                ),
                "max_messages_per_conversation": self.max_messages,
                "config": {
                    "max_tokens": self.max_tokens,
                    "summary_threshold": self.summary_threshold,
                    "compression_recent_messages": (
                        self._compression_recent_messages
                    ),
                },
            }

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------

    def export_user_memory(
        self,
        user_id: str,
        filepath: Optional[Union[str, Path]] = None,
        *,
        organization_id: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Export a user's conversation history to JSON.

        Returns:
            Path to exported file, or None on failure.
        """
        if not _validate_user_id(user_id):
            logger.warning("Cannot export memory: invalid user_id")
            return None

        lock = self._get_user_lock(user_id, organization_id)
        _, scope = self._scope_parts(user_id, organization_id)

        with lock:
            self._load_persistent_history_locked(user_id, organization_id)
            if scope not in self._memories:
                logger.warning(
                    "No memory found for user %s",
                    user_id,
                )
                return None

            if filepath is None:
                export_dir = project_root / "data" / "traces"
                export_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                safe_user = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)[:80]
                filepath = export_dir / f"memory_{safe_user}_{timestamp}.json"
            else:
                filepath = Path(filepath).expanduser()
                filepath.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            messages = list(self._memories[scope])

            org, _ = self._scope_parts(user_id, organization_id)
            export_data = {
                "schema_version": 3,
                "organization_id": org,
                "user_id": user_id,
                "exported_at": _utc_now_iso(),
                "messages": messages,
                "stats": {
                    "message_count": len(messages),
                    "max_messages": self.max_messages,
                },
                "config": {
                    "max_messages": self.max_messages,
                    "max_tokens": self.max_tokens,
                    "summary_threshold": self.summary_threshold,
                },
            }

            try:
                with filepath.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        export_data,
                        file,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )

                logger.info(
                    "Exported memory for user %s to %s",
                    user_id,
                    filepath,
                )

                return filepath

            except (OSError, TypeError, ValueError) as exc:
                logger.error(
                    "Memory export failed for user %s: %s",
                    user_id,
                    exc,
                )
                return None

    # -------------------------------------------------------------------------
    # IMPORT
    # -------------------------------------------------------------------------

    def import_user_memory(
        self,
        user_id: str,
        filepath: Union[str, Path],
        *,
        organization_id: Optional[str] = None,
    ) -> bool:
        """
        Import conversation history from a JSON export.

        Imported data is validated and normalized before replacing the user's
        current memory.
        """
        if not _validate_user_id(user_id):
            logger.warning("Cannot import memory: invalid user_id")
            return False

        if not filepath:
            logger.warning("Cannot import memory: empty filepath")
            return False

        filepath = Path(filepath).expanduser()
        lock = self._get_user_lock(user_id, organization_id)
        _, scope = self._scope_parts(user_id, organization_id)

        with lock:
            try:
                with filepath.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    import_data = json.load(file)

            except (OSError, json.JSONDecodeError) as exc:
                logger.error(
                    "Memory import failed for user %s: %s",
                    user_id,
                    exc,
                )
                return False

            if not isinstance(import_data, dict):
                logger.warning(
                    "Invalid import structure: %s",
                    filepath,
                )
                return False

            imported_user_id = import_data.get("user_id")

            if imported_user_id != user_id:
                logger.warning(
                    "User ID mismatch in import file %s: expected=%s actual=%s",
                    filepath,
                    user_id,
                    imported_user_id,
                )
                return False

            expected_org, _ = self._scope_parts(user_id, organization_id)
            imported_org = import_data.get("organization_id")
            if imported_org is not None:
                try:
                    if _normalize_organization_id(str(imported_org)) != expected_org:
                        logger.warning("Organization mismatch in import file %s", filepath)
                        return False
                except ValueError:
                    return False

            messages = import_data.get("messages")

            if not isinstance(messages, list):
                logger.warning(
                    "Import file contains no valid messages list: %s",
                    filepath,
                )
                return False

            if len(messages) > MAX_IMPORT_MESSAGES:
                logger.warning(
                    "Import file contains too many messages: %s",
                    len(messages),
                )
                return False

            normalized_messages: List[Dict[str, Any]] = []

            for raw_message in messages:
                normalized = _normalize_message(raw_message)

                if normalized is not None:
                    normalized_messages.append(normalized)

            if not normalized_messages:
                logger.warning(
                    "No valid messages found in import file: %s",
                    filepath,
                )
                return False

            # Keep the newest messages if the import is larger than the
            # configured retention window.
            normalized_messages = normalized_messages[
                -self.max_messages:
            ]

            new_memory = self._create_memory()

            for message in normalized_messages:
                new_memory.append(message)

            self._memories[scope] = new_memory

            if self.persistent:
                self._delete_persistent_scope(user_id, organization_id)
                for message in normalized_messages:
                    self._persist_message(user_id, message, organization_id)
                try:
                    _, scope = self._scope_parts(user_id, organization_id)
                    self._persistent_initialized_scopes.add(scope)
                except ValueError:
                    pass

            logger.info(
                "Imported %s messages for user %s from %s",
                len(normalized_messages),
                user_id,
                filepath,
            )

            return True


# =============================================================================
# GLOBAL INSTANCE MANAGEMENT
# =============================================================================

_memory_manager: Optional[MemoryManager] = None
_manager_lock = threading.RLock()


def get_memory_manager(
    max_messages: Optional[int] = None,
    max_tokens: Optional[int] = None,
    summary_threshold: Optional[int] = None,
    organization_id: Optional[str] = None,
) -> MemoryManager:
    """
    Get or create the global MemoryManager singleton.

    The first call creates the singleton. Later calls return the same
    instance, even if different configuration arguments are supplied.
    """
    global _memory_manager

    with _manager_lock:
        if _memory_manager is None:
            kwargs: Dict[str, int] = {}

            if max_messages is not None:
                kwargs["max_messages"] = max_messages

            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            if summary_threshold is not None:
                kwargs["summary_threshold"] = summary_threshold
            if organization_id is not None:
                kwargs["organization_id"] = organization_id

            _memory_manager = MemoryManager(**kwargs)

        return _memory_manager


def reset_memory_manager() -> None:
    """Reset the global memory manager singleton, primarily for tests."""
    global _memory_manager

    with _manager_lock:
        manager = _memory_manager

        if manager is not None:
            manager.clear_all_memory()

        _memory_manager = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def add_conversation_message(
    user_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    organization_id: Optional[str] = None,
) -> bool:
    """Add one conversation message using the global manager."""
    return get_memory_manager().add_message(
        user_id=user_id,
        role=role,
        content=content,
        metadata=metadata,
        organization_id=organization_id,
    )


def get_conversation_history(
    user_id: str,
    include_system: bool = True,
    limit: Optional[int] = None,
    *,
    organization_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get conversation history using the global manager."""
    return get_memory_manager().get_history(
        user_id=user_id,
        include_system=include_system,
        limit=limit,
        organization_id=organization_id,
    )


def build_query_context(
    user_id: str,
    query: str,
    max_tokens: Optional[int] = None,
    *,
    organization_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build context for a query using the global manager."""
    return get_memory_manager().get_context_for_query(
        user_id=user_id,
        current_query=query,
        max_tokens=max_tokens,
        organization_id=organization_id,
    )


def clear_user_conversation(
    user_id: str,
    *,
    organization_id: Optional[str] = None,
) -> bool:
    """Clear one user's conversation memory."""
    return get_memory_manager().clear_user_memory(user_id, organization_id=organization_id)


def export_conversation(
    user_id: str,
    filepath: Optional[Union[str, Path]] = None,
    *,
    organization_id: Optional[str] = None,
) -> Optional[Path]:
    """Export one user's conversation memory."""
    return get_memory_manager().export_user_memory(
        user_id=user_id,
        filepath=filepath,
        organization_id=organization_id,
    )


def import_conversation(
    user_id: str,
    filepath: Union[str, Path],
    *,
    organization_id: Optional[str] = None,
) -> bool:
    """Import one user's conversation memory."""
    return get_memory_manager().import_user_memory(
        user_id=user_id,
        filepath=filepath,
        organization_id=organization_id,
    )


# =============================================================================
# TEST / DEMO
# =============================================================================

def test_memory_manager() -> None:
    """Basic self-test for the memory manager."""
    print("\n🧠 Testing Memory Manager")
    print("=" * 70)

    # Use a smaller configuration so compression can be demonstrated.
    manager = MemoryManager(
        max_messages=10,
        max_tokens=2000,
        summary_threshold=8,
    )

    user_id = "test_user_123"

    print(
        "Configuration:",
        f"max_messages={manager.max_messages},",
        f"max_tokens={manager.max_tokens},",
        f"summary_threshold={manager.summary_threshold}",
    )

    # -------------------------------------------------------------------------
    # Test 1: Add messages
    # -------------------------------------------------------------------------
    print("\n1. Adding messages")

    conversation = [
        ("user", "What is the leave policy?"),
        (
            "assistant",
            "Please refer to the approved company leave policy.",
        ),
        ("user", "How do I request leave?"),
        (
            "assistant",
            "Submit a leave request through the approved HR process.",
        ),
        ("user", "What about sick leave?"),
        (
            "assistant",
            "Sick leave is handled according to the applicable policy.",
        ),
        ("user", "Can I work remotely?"),
        (
            "assistant",
            "Remote work depends on the applicable workplace policy.",
        ),
    ]

    for role, content in conversation:
        success = manager.add_message(
            user_id,
            role,
            content,
        )
        print(
            f"   {'✅' if success else '❌'} {role}: {content}"
        )

    # -------------------------------------------------------------------------
    # Test 2: History
    # -------------------------------------------------------------------------
    print("\n2. Retrieving history")

    history = manager.get_history(user_id)

    print(f"   Messages stored: {len(history)}")

    for index, message in enumerate(history, start=1):
        print(
            f"   {index}. "
            f"[{message['role']}] "
            f"{message['content'][:70]}"
        )

    # -------------------------------------------------------------------------
    # Test 3: Context
    # -------------------------------------------------------------------------
    print("\n3. Building query context")

    context, included = manager.get_context_for_query(
        user_id,
        "What should I check before submitting a request?",
        max_tokens=200,
    )

    print(f"   Included messages: {len(included)}")
    print(f"   Context length: {len(context)} characters")

    # -------------------------------------------------------------------------
    # Test 4: Statistics
    # -------------------------------------------------------------------------
    print("\n4. Statistics")

    print("   User stats:")
    print(json.dumps(
        manager.get_stats(user_id),
        indent=2,
    ))

    print("   Global stats:")
    print(json.dumps(
        manager.get_stats(),
        indent=2,
    ))

    # -------------------------------------------------------------------------
    # Test 5: Export/import
    # -------------------------------------------------------------------------
    print("\n5. Export/import")

    export_path = project_root / "data" / "traces" / "memory_test.json"

    exported = manager.export_user_memory(
        user_id,
        export_path,
    )

    if exported:
        print(f"   ✅ Exported to: {exported}")

        manager.clear_user_memory(user_id)

        imported = manager.import_user_memory(
            user_id,
            exported,
        )

        print(
            f"   {'✅' if imported else '❌'} Import completed"
        )

        try:
            exported.unlink(missing_ok=True)
        except OSError:
            pass
    else:
        print("   ❌ Export failed")

    # -------------------------------------------------------------------------
    # Test 6: Clear
    # -------------------------------------------------------------------------
    print("\n6. Clearing memory")

    before_clear = len(manager.get_history(user_id))

    manager.clear_user_memory(user_id)

    after_clear = len(manager.get_history(user_id))

    print(f"   Before clear: {before_clear}")
    print(f"   After clear:  {after_clear}")
    print(
        f"   {'✅' if after_clear == 0 else '❌'} "
        "Clear successful"
    )

    print("\n" + "=" * 70)
    print("✅ Memory manager test complete")


if __name__ == "__main__":
    test_memory_manager()

