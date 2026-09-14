
#!/usr/bin/env python3
"""
PolicyGuard AI - Enterprise HR RAG Platform
============================================

Production-grade Streamlit application.

Core capabilities
-----------------
- Secure authentication with SQLite + bcrypt
- RBAC: viewer / editor / admin
- Enterprise audit logging
- Protected document ingestion
- FAISS + BM25 hybrid retrieval
- Cross-encoder reranking
- RAG engine integration
- LangGraph orchestration integration
- Security guard / prompt-injection protection
- PII-aware audit previews
- Session-safe Streamlit architecture
- Responsive enterprise UI
- Graceful subsystem degradation
- Operational health/status dashboard

Run
---
    streamlit run app.py

Environment
-----------
Configuration is loaded through config.settings when available and
.env/environment variables when configured.

IMPORTANT
---------
Do not rely on the development admin credentials in production.
Create/rotate the administrative credential before exposing the app.
"""

from __future__ import annotations

# =============================================================================
# BOOTSTRAP
# =============================================================================

import hashlib
import inspect
import html
import json
import logging
import os
import pickle
import re
import sqlite3
import sys
import tempfile
import time
import uuid
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Reduce noisy HuggingFace/Tokenizer logs before importing ML modules.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# =============================================================================
# PROJECT PATHS
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
UPLOAD_DIR = DATA_DIR / "uploads"
LOG_DIR = PROJECT_ROOT / "logs"

for _directory in (DATA_DIR, VECTOR_DB_DIR, UPLOAD_DIR, LOG_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================

LOG_FILE = LOG_DIR / "app.log"

logger = logging.getLogger("policyguard.app")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
        mode="a",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger.propagate = False

# =============================================================================
# OPTIONAL ENVIRONMENT LOADER
# =============================================================================

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================

import bcrypt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import faiss
except Exception:
    faiss = None

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================


class _FallbackSettings:
    """Minimal fallback so the UI remains bootable during partial deployment."""

    APP_VERSION = "1.0.0"
    APP_NAME = "PolicyGuard AI"
    SECRET_KEY = os.getenv("SECRET_KEY", "")

    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{PROJECT_ROOT / 'nexus_auth.db'}",
    )

    VECTOR_DB_PATH = str(VECTOR_DB_DIR)

    TOP_K = int(os.getenv("TOP_K", "5"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))

    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))

    ENABLE_TRACING = os.getenv("ENABLE_TRACING", "false").lower() == "true"


try:
    from config.settings import settings as app_settings
except Exception as exc:
    logger.warning("Could not load config.settings: %s", exc)
    app_settings = _FallbackSettings()

settings = app_settings

# =============================================================================
# DATABASE LOCATION
# =============================================================================


def _resolve_sqlite_path() -> Path:
    """
    Resolve the SQLite database used by the application.

    Supports:
        - DATABASE_URL=sqlite:///relative/path.db
        - DATABASE_URL=sqlite:////absolute/path.db
        - fallback nexus_auth.db
    """
    configured = str(getattr(settings, "DATABASE_URL", "") or "").strip()

    if configured.startswith("sqlite:///"):
        raw_path = configured[len("sqlite:///") :]

        # sqlite:////absolute/path => raw starts with / on Unix.
        if raw_path.startswith("/"):
            path = Path(raw_path)
        else:
            path = PROJECT_ROOT / raw_path

        return path.resolve()

    return PROJECT_ROOT / "nexus_auth.db"


DB_FILE = _resolve_sqlite_path()
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONSTANTS
# =============================================================================

VALID_ROLES = ("viewer", "editor", "admin")
ROLE_LEVEL = {
    "viewer": 1,
    "editor": 2,
    "admin": 3,
}

MAX_AUDIT_PREVIEW = 180
MAX_USERNAME_LENGTH = 64
MAX_PASSWORD_LENGTH = 256
MAX_QUERY_LENGTH = 5000
MAX_ORGANIZATION_ID_LENGTH = 100
ORGANIZATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,99}$")

ALLOWED_UPLOAD_TYPES = {
    "pdf",
    "docx",
    "xlsx",
    "txt",
    "md",
}

# =============================================================================
# SAFE TEXT HELPERS
# =============================================================================


def _clean_text(value: Any, max_length: int = 10000) -> str:
    """Normalize arbitrary input into bounded text."""
    if value is None:
        return ""

    text = str(value).replace("\x00", " ").strip()

    if len(text) > max_length:
        text = text[:max_length] + "…"

    return text


def _safe_username(value: Any) -> str:
    """Normalize a username into a safe bounded identifier."""
    username = _clean_text(value, MAX_USERNAME_LENGTH)

    if not re.fullmatch(r"[A-Za-z0-9_.@-]{3,64}", username):
        return ""

    return username


def _safe_role(role: Any) -> str:
    role = _clean_text(role, 20).lower()
    return role if role in VALID_ROLES else "viewer"


def _configured_organization_id() -> str:
    """Return the deployment's default organization identifier."""
    raw = (
        os.getenv("POLICYGUARD_ORGANIZATION_ID")
        or getattr(settings, "ORGANIZATION_ID", None)
        or "default"
    )
    value = _clean_text(raw, MAX_ORGANIZATION_ID_LENGTH)
    if not ORGANIZATION_ID_PATTERN.fullmatch(value):
        logger.critical("Invalid POLICYGUARD_ORGANIZATION_ID; refusing unsafe scope.")
        return "default"
    return value


def _current_organization_id() -> str:
    """Return the organization bound to the authenticated Streamlit session."""
    session_org = _clean_text(
        st.session_state.get("organization_id"),
        MAX_ORGANIZATION_ID_LENGTH,
    )
    if session_org and ORGANIZATION_ID_PATTERN.fullmatch(session_org):
        return session_org
    return _configured_organization_id()


def _safe_namespace(route: Optional[str] = None) -> str:
    """Map application routes to isolated retrieval namespaces."""
    return "talent" if route == "Talent Intelligence" else "policy"


def _authorized_session() -> Tuple[str, str, str]:
    """Return username, role and organization only from the authenticated session."""
    if not st.session_state.get("authenticated"):
        raise PermissionError("Authentication is required.")
    username = _safe_username(st.session_state.get("username", ""))
    role = _safe_role(st.session_state.get("user_role", "viewer"))
    organization_id = _current_organization_id()
    if not username:
        raise PermissionError("Invalid authenticated user.")
    if not ORGANIZATION_ID_PATTERN.fullmatch(organization_id):
        raise PermissionError("Invalid organization scope.")
    return username, role, organization_id


def _safe_filename(filename: str) -> str:
    """Prevent path traversal while preserving a readable filename."""
    name = Path(_clean_text(filename, 255)).name

    # Remove characters problematic for local filesystem/log rendering.
    name = re.sub(r"[^A-Za-z0-9._() \-]+", "_", name)

    return name[:255] or "uploaded_document"


def _escape(value: Any) -> str:
    return html.escape(_clean_text(value, 1000), quote=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# =============================================================================
# DATABASE
# =============================================================================


def _db_connect() -> sqlite3.Connection:
    """Create a configured SQLite connection."""
    conn = sqlite3.connect(
        DB_FILE,
        timeout=30,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row

    # SQLite reliability/performance settings.
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
    except sqlite3.Error:
        pass

    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row["name"] for row in rows}
    except sqlite3.Error:
        return set()


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = _table_columns(conn, table)

    if column not in columns:
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
        )


def _migrate_database_schema(conn: sqlite3.Connection) -> None:
    """
    Safely migrate older PolicyGuard databases.

    Migration is intentionally additive. Existing data is preserved.
    """
    # These migrations only execute after base tables are created.
    _ensure_column(
        conn,
        "users",
        "organization_id",
        "TEXT DEFAULT 'default'",
    )
    _ensure_column(
        conn,
        "users",
        "failed_login_attempts",
        "INTEGER DEFAULT 0",
    )
    _ensure_column(
        conn,
        "users",
        "last_failed_login",
        "TIMESTAMP",
    )

    audit_columns = {
        "tokens_used": "INTEGER DEFAULT 0",
        "cost_usd": "REAL DEFAULT 0.0",
        "model_used": "TEXT",
        "query_preview": "TEXT",
        "threat_type": "TEXT",
        "blocked": "INTEGER DEFAULT 0",
        "ip_address": "TEXT",
    }

    for column, definition in audit_columns.items():
        _ensure_column(
            conn,
            "audit_log",
            column,
            definition,
        )

    document_columns = {
        "filepath": "TEXT",
        "uploaded_by": "TEXT",
        "file_size": "INTEGER DEFAULT 0",
        "chunk_count": "INTEGER DEFAULT 0",
        "status": "TEXT DEFAULT 'active'",
        "ocr_used": "INTEGER DEFAULT 0",
    }

    for column, definition in document_columns.items():
        _ensure_column(
            conn,
            "documents",
            column,
            definition,
        )
    for table in (
        "audit_log",
        "access_requests",
        "talent_candidates",
        "talent_searches",
        "talent_matches",
        "chat_sessions",
        "chat_messages",
    ):
        _ensure_column(conn, table, "organization_id", "TEXT DEFAULT 'default'")

    _ensure_column(conn, "query_cache", "organization_id", "TEXT DEFAULT 'default'")
    _ensure_column(conn, "query_cache", "namespace", "TEXT DEFAULT 'policy'")
    _ensure_column(conn, "query_cache", "user_id", "TEXT")
    _ensure_column(conn, "query_cache", "role", "TEXT DEFAULT 'viewer'")

    # Normalize legacy NULL scopes without changing non-null tenant assignments.
    for table in (
        "users", "audit_log", "documents", "access_requests",
        "talent_candidates", "talent_searches", "talent_matches",
        "chat_sessions", "chat_messages", "query_cache",
    ):
        conn.execute(
            f"UPDATE {table} SET organization_id = COALESCE(NULLIF(organization_id, ''), 'default')"
        )


def _init_database() -> bool:
    """Create all application tables and indexes."""
    try:
        with closing(_db_connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    organization_id TEXT NOT NULL DEFAULT 'default',
                    role TEXT NOT NULL DEFAULT 'viewer'
                        CHECK(role IN ('viewer', 'editor', 'admin')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                    last_failed_login TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    organization_id TEXT NOT NULL DEFAULT 'default',
                    action TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    details TEXT,
                    ip_address TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    model_used TEXT,
                    query_preview TEXT,
                    threat_type TEXT,
                    blocked INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    organization_id TEXT NOT NULL DEFAULT 'default',
                    filepath TEXT,
                    uploaded_by TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    ocr_used INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS access_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    organization_id TEXT NOT NULL DEFAULT 'default',
                    from_role TEXT NOT NULL,
                    to_role TEXT NOT NULL,
                    reason TEXT,
                    status TEXT DEFAULT 'pending'
                        CHECK(status IN ('pending', 'approved', 'rejected', 'expired')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    approved_by TEXT,
                    approved_at TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS talent_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_name TEXT NOT NULL,
                    organization_id TEXT NOT NULL DEFAULT 'default',
                    resume_filename TEXT NOT NULL,
                    resume_path TEXT,
                    resume_text TEXT NOT NULL,
                    uploaded_by TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'bench',
                    source_document_id INTEGER,
                    FOREIGN KEY(source_document_id) REFERENCES documents(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS talent_searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    searched_by TEXT NOT NULL,
                    organization_id TEXT NOT NULL DEFAULT 'default',
                    job_title TEXT NOT NULL,
                    job_description TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS talent_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_id INTEGER NOT NULL,
                    organization_id TEXT NOT NULL DEFAULT 'default',
                    candidate_id INTEGER NOT NULL,
                    semantic_score REAL NOT NULL DEFAULT 0.0,
                    keyword_score REAL NOT NULL DEFAULT 0.0,
                    final_score REAL NOT NULL DEFAULT 0.0,
                    rank_position INTEGER NOT NULL,
                    matched_skills TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(search_id) REFERENCES talent_searches(id) ON DELETE CASCADE,
                    FOREIGN KEY(candidate_id) REFERENCES talent_candidates(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    organization_id TEXT NOT NULL DEFAULT 'default',
                    title TEXT NOT NULL DEFAULT 'New conversation',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    organization_id TEXT NOT NULL DEFAULT 'default',
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_talent_candidates_status
                    ON talent_candidates(status);

                CREATE INDEX IF NOT EXISTS idx_talent_candidates_uploaded_by
                    ON talent_candidates(uploaded_by);

                CREATE INDEX IF NOT EXISTS idx_talent_searches_user
                    ON talent_searches(searched_by);

                CREATE INDEX IF NOT EXISTS idx_talent_matches_search
                    ON talent_matches(search_id);

                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
                    ON chat_sessions(username, updated_at);

                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                    ON chat_messages(session_id, created_at);

                CREATE TABLE IF NOT EXISTS query_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE NOT NULL,
                    organization_id TEXT NOT NULL DEFAULT 'default',
                    namespace TEXT NOT NULL DEFAULT 'policy',
                    user_id TEXT,
                    role TEXT NOT NULL DEFAULT 'viewer',
                    query_text TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    chunks_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hits INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_users_username
                    ON users(username);

                CREATE INDEX IF NOT EXISTS idx_users_role
                    ON users(role);

                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                    ON audit_log(timestamp);

                CREATE INDEX IF NOT EXISTS idx_audit_username
                    ON audit_log(username);

                CREATE INDEX IF NOT EXISTS idx_audit_action
                    ON audit_log(action);

                CREATE INDEX IF NOT EXISTS idx_documents_filename
                    ON documents(filename);

                CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at
                    ON documents(uploaded_at);

                CREATE INDEX IF NOT EXISTS idx_access_status
                    ON access_requests(status);

                CREATE INDEX IF NOT EXISTS idx_cache_last_accessed
                    ON query_cache(last_accessed);

                CREATE INDEX IF NOT EXISTS idx_users_org
                    ON users(organization_id, username);

                CREATE INDEX IF NOT EXISTS idx_audit_org_time
                    ON audit_log(organization_id, timestamp);

                CREATE INDEX IF NOT EXISTS idx_documents_org_time
                    ON documents(organization_id, uploaded_at);

                CREATE INDEX IF NOT EXISTS idx_talent_candidates_org
                    ON talent_candidates(organization_id, updated_at);

                CREATE INDEX IF NOT EXISTS idx_talent_searches_org
                    ON talent_searches(organization_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_talent_matches_org
                    ON talent_matches(organization_id, search_id);

                CREATE INDEX IF NOT EXISTS idx_chat_sessions_org_user
                    ON chat_sessions(organization_id, username, updated_at);

                CREATE INDEX IF NOT EXISTS idx_chat_messages_org_session
                    ON chat_messages(organization_id, session_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_cache_scope
                    ON query_cache(organization_id, namespace, user_id, role, last_accessed);
                """
            )

            _migrate_database_schema(conn)

            # Development convenience only.
            # Production should use an explicit administrator creation process.
            auto_create_admin = (
                os.getenv("POLICYGUARD_CREATE_DEFAULT_ADMIN", "false").lower()
                == "true"
            )

            if auto_create_admin:
                existing = conn.execute(
                    "SELECT id FROM users WHERE username = ? AND organization_id = ?",
                    ("admin", _configured_organization_id()),
                ).fetchone()

                if existing is None:
                    password = os.getenv("POLICYGUARD_ADMIN_PASSWORD", "")

                    if len(password) >= 12:
                        password_hash = _hash_password(password)

                        if password_hash:
                            conn.execute(
                                """
                                INSERT INTO users
                                    (username, password_hash, role)
                                VALUES (?, ?, 'admin')
                                """,
                                ("admin", password_hash),
                            )
                            logger.warning(
                                "Created admin account from environment configuration."
                            )
                    else:
                        logger.warning(
                            "Default admin requested but "
                            "POLICYGUARD_ADMIN_PASSWORD is missing or too weak."
                        )

            conn.commit()

        logger.info("Database initialized: %s", DB_FILE)
        return True

    except Exception:
        logger.exception("Database initialization failed")
        return False



# =============================================================================
# PASSWORD / AUTHENTICATION
# =============================================================================


def _hash_password(password: str) -> Optional[bytes]:
    """Hash password with bcrypt."""
    try:
        if not password:
            return None

        if len(password) > MAX_PASSWORD_LENGTH:
            return None

        return bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=12),
        )
    except Exception:
        logger.exception("Password hashing failed")
        return None

_DB_READY = _init_database()

def _verify_password(
    stored_hash: Any,
    password: str,
) -> bool:
    """Verify password regardless of SQLite BLOB/string representation."""
    try:
        if not stored_hash or not password:
            return False

        if isinstance(stored_hash, str):
            stored_hash = stored_hash.encode("utf-8")

        return bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash,
        )
    except Exception:
        return False


def _audit_log(
    username: str,
    action: str,
    details: str = "",
    *,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
    model_used: Optional[str] = None,
    ip_address: Optional[str] = None,
    query_preview: Optional[str] = None,
    threat_type: Optional[str] = None,
    blocked: bool = False,
) -> None:
    """
    Write a best-effort audit record.

    Audit failures must never crash the user-facing request.
    """
    try:
        preview = _clean_text(query_preview, MAX_AUDIT_PREVIEW)

        with closing(_db_connect()) as conn:
            conn.execute(
                """
                INSERT INTO audit_log (
                    username,
                    organization_id,
                    action,
                    details,
                    ip_address,
                    tokens_used,
                    cost_usd,
                    model_used,
                    query_preview,
                    threat_type,
                    blocked
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _clean_text(username, MAX_USERNAME_LENGTH) or "unknown",
                    _current_organization_id(),
                    _clean_text(action, 100),
                    _clean_text(details, 2000),
                    _clean_text(ip_address, 100),
                    max(0, int(tokens_used or 0)),
                    max(0.0, float(cost_usd or 0.0)),
                    _clean_text(model_used, 200),
                    preview,
                    _clean_text(threat_type, 200),
                    1 if blocked else 0,
                ),
            )
            conn.commit()

    except Exception:
        logger.debug("Audit logging failed", exc_info=True)


def _register_user(
    username: str,
    password: str,
) -> Tuple[bool, str]:
    """Register a viewer account."""
    username = _safe_username(username)

    if not username:
        return (
            False,
            "Username must be 3-64 characters and contain only letters, numbers, _, ., @ or -.",
        )

    if len(password) < 12:
        return False, "Password must be at least 12 characters."

    if len(password) > MAX_PASSWORD_LENGTH:
        return False, "Password is too long."

    password_hash = _hash_password(password)

    if password_hash is None:
        return False, "Unable to securely create the account."

    try:
        with closing(_db_connect()) as conn:
            conn.execute(
                """
                INSERT INTO users (
                    username,
                    organization_id,
                    password_hash,
                    role
                )
                VALUES (?, ?, ?, 'viewer')
                """,
                (username, _configured_organization_id(), password_hash),
            )
            conn.commit()

        _audit_log(
            username,
            "USER_REGISTERED",
            "New viewer account created.",
        )

        return True, "Registration successful. You can now sign in."

    except sqlite3.IntegrityError:
        return False, "That username already exists."

    except Exception:
        logger.exception("Registration failed")
        return False, "Registration failed due to an internal error."


def _login_user(
    username: str,
    password: str,
) -> Tuple[bool, str, Optional[str], Optional[int]]:
    """
    Authenticate a user.

    Returns:
        success, message, role, user_id
    """
    username = _safe_username(username)

    if not username or not password:
        return False, "Invalid username or password.", None, None

    try:
        with closing(_db_connect()) as conn:
            row = conn.execute(
                """
                SELECT
                    id,
                    username,
                    organization_id,
                    password_hash,
                    role,
                    is_active,
                    failed_login_attempts,
                    last_failed_login
                FROM users
                WHERE organization_id = ? AND username = ?
                """,
                (_current_organization_id(), username),
            ).fetchone()

            if row is None:
                _audit_log(
                    username,
                    "LOGIN_FAILED",
                    "User not found.",
                    blocked=True,
                )
                return False, "Invalid username or password.", None, None

            user_id = int(row["id"])
            role = _safe_role(row["role"])
            is_active = bool(row["is_active"])
            failed_attempts = int(row["failed_login_attempts"] or 0)
            last_failed = row["last_failed_login"]

            if not is_active:
                _audit_log(
                    username,
                    "LOGIN_FAILED",
                    "Account inactive.",
                    blocked=True,
                )
                return False, "This account is inactive.", None, None

            # 30-minute lockout after 5 failed attempts.
            if failed_attempts >= 5 and last_failed:
                try:
                    failed_at = datetime.fromisoformat(
                        str(last_failed).replace("Z", "")
                    )
                    if datetime.now() < failed_at + timedelta(minutes=30):
                        _audit_log(
                            username,
                            "LOGIN_FAILED",
                            "Temporary lockout.",
                            blocked=True,
                        )
                        return (
                            False,
                            "Account temporarily locked. Please try again later.",
                            None,
                            None,
                        )
                except ValueError:
                    pass

            if not _verify_password(row["password_hash"], password):
                conn.execute(
                    """
                    UPDATE users
                    SET
                        failed_login_attempts =
                            COALESCE(failed_login_attempts, 0) + 1,
                        last_failed_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (user_id,),
                )
                conn.commit()

                _audit_log(
                    username,
                    "LOGIN_FAILED",
                    "Invalid password.",
                    blocked=True,
                )

                return False, "Invalid username or password.", None, None

            conn.execute(
                """
                UPDATE users
                SET
                    failed_login_attempts = 0,
                    last_failed_login = NULL,
                    last_login = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (user_id,),
            )
            conn.commit()

        _audit_log(
            username,
            "LOGIN_SUCCESS",
            f"Role: {role}",
        )

        return (
            True,
            f"Welcome back, {username}.",
            role,
            user_id,
        )

    except Exception:
        logger.exception("Login failed")
        return (
            False,
            "Unable to complete login due to an internal error.",
            None,
            None,
        )


def _get_all_users() -> List[Dict[str, Any]]:
    """Return user records for administrators."""
    try:
        with closing(_db_connect()) as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    username,
                    role,
                    created_at,
                    last_login,
                    is_active
                FROM users
                WHERE organization_id = ?
                ORDER BY created_at DESC
                """
            , (_current_organization_id(),)).fetchall()

        return [dict(row) for row in rows]

    except Exception:
        logger.exception("Could not load users")
        return []


def _update_user_role(
    user_id: int,
    new_role: str,
    updated_by: str,
) -> bool:
    """Change a user's role through the hardened RBAC service."""
    new_role = _safe_role(new_role)

    if new_role not in VALID_ROLES:
        return False

    if HARDENED_AUTH_DATABASE_AVAILABLE and _auth_database is not None:
        try:
            success, message = _auth_database.update_user_role(
                int(user_id),
                new_role,
                updated_by,
            )
            if success:
                _audit_log(
                    updated_by,
                    "ROLE_CHANGED",
                    f"User ID {int(user_id)} changed to '{new_role}'.",
                )
            else:
                logger.warning(
                    "Hardened role update denied for user %s: %s",
                    user_id,
                    message,
                )
            return bool(success)
        except Exception:
            logger.exception("Hardened role update failed")
            return False

    logger.critical(
        "Hardened auth database is unavailable; refusing privileged role change."
    )
    return False


def _set_user_active(
    user_id: int,
    active: bool,
    changed_by: str,
) -> bool:
    """Enable/disable a user through the hardened RBAC service."""
    if HARDENED_AUTH_DATABASE_AVAILABLE and _auth_database is not None:
        try:
            success, message = _auth_database.update_user_active_status(
                int(user_id),
                bool(active),
                changed_by,
            )
            if success:
                _audit_log(
                    changed_by,
                    "USER_STATUS_CHANGED",
                    f"User ID {int(user_id)} active={bool(active)}.",
                )
            else:
                logger.warning(
                    "Hardened account-status update denied for user %s: %s",
                    user_id,
                    message,
                )
            return bool(success)
        except Exception:
            logger.exception("Hardened account-status update failed")
            return False

    logger.critical(
        "Hardened auth database is unavailable; refusing privileged account-status change."
    )
    return False


def _delete_user(
    user_id: int,
    deleted_by: str,
) -> bool:
    """
    Deactivate an account through the hardened service.

    Physical deletion is intentionally avoided so audit/compliance history
    remains intact.
    """
    if HARDENED_AUTH_DATABASE_AVAILABLE and _auth_database is not None:
        try:
            success, message = _auth_database.delete_user(
                int(user_id),
                deleted_by,
            )
            if success:
                _audit_log(
                    deleted_by,
                    "USER_DEACTIVATED",
                    f"User ID {int(user_id)} deactivated.",
                )
            else:
                logger.warning(
                    "Hardened user deletion/deactivation denied for user %s: %s",
                    user_id,
                    message,
                )
            return bool(success)
        except Exception:
            logger.exception("Hardened user deletion failed")
            return False

    logger.critical(
        "Hardened auth database is unavailable; refusing privileged account deletion."
    )
    return False




# =============================================================================
# AUDIT / DOCUMENT QUERIES
# =============================================================================


def _get_audit_logs(
    limit: int = 200,
    username: Optional[str] = None,
    action_filter: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Fetch audit events using parameterized SQL."""
    limit = max(1, min(int(limit), 5000))

    try:
        query = "SELECT * FROM audit_log WHERE organization_id = ?"
        params: List[Any] = [_current_organization_id()]

        if username:
            query += " AND username = ?"
            params.append(username)

        if action_filter:
            query += " AND action LIKE ?"
            params.append(f"%{action_filter}%")

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with closing(_db_connect()) as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    except Exception:
        logger.exception("Audit query failed")
        return []


def _get_documents() -> List[Dict[str, Any]]:
    """Return indexed documents."""
    try:
        with closing(_db_connect()) as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    filename,
                    filepath,
                    uploaded_by,
                    uploaded_at,
                    file_size,
                    chunk_count,
                    status,
                    ocr_used
                FROM documents
                WHERE organization_id = ?
                ORDER BY uploaded_at DESC
                """
            , (_current_organization_id(),)).fetchall()

        return [dict(row) for row in rows]

    except Exception:
        logger.exception("Document query failed")
        return []


def _get_document_stats() -> Dict[str, Any]:
    """Get aggregate document statistics."""
    defaults = {
        "documents": 0,
        "chunks": 0,
        "bytes": 0,
        "ocr_documents": 0,
    }

    try:
        with closing(_db_connect()) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS documents,
                    COALESCE(SUM(chunk_count), 0) AS chunks,
                    COALESCE(SUM(file_size), 0) AS bytes,
                    COALESCE(SUM(ocr_used), 0) AS ocr_documents
                FROM documents
                WHERE status != 'deleted' AND organization_id = ?
                """,
                (_current_organization_id(),),
            ).fetchone()

        if row:
            return {
                "documents": int(row["documents"] or 0),
                "chunks": int(row["chunks"] or 0),
                "bytes": int(row["bytes"] or 0),
                "ocr_documents": int(row["ocr_documents"] or 0),
            }

    except Exception:
        logger.debug("Document stats unavailable", exc_info=True)

    return defaults


# =============================================================================
# OPTIONAL PROJECT MODULES
# =============================================================================

# Settings are already loaded above.

# Security
try:
    from src.security.guard_model import validate_user_query as _security_validate_query

    SECURITY_AVAILABLE = True
except Exception as exc:
    SECURITY_AVAILABLE = False
    logger.critical(
        "Security guard unavailable; query processing will fail closed: %s",
        exc,
    )

    def _security_validate_query(
        query: str,
        username: str,
        user_role: str,
    ) -> Tuple[bool, str, str]:
        return (
            False,
            "Security validation is unavailable. The request cannot be processed safely.",
            "",
        )


# Embedder + cache + memory
try:
    from src.core.embedder_singleton import get_embedder

    EMBEDDER_AVAILABLE = True
except Exception as exc:
    EMBEDDER_AVAILABLE = False
    get_embedder = None
    logger.warning("Embedder unavailable: %s", exc)

try:
    from src.core.cache import get_cached_answer, cache_answer

    CACHE_AVAILABLE = True
except Exception as exc:
    CACHE_AVAILABLE = False
    logger.warning("Semantic cache unavailable: %s", exc)

    def get_cached_answer(*args: Any, **kwargs: Any) -> Any:
        return None

    def cache_answer(*args: Any, **kwargs: Any) -> Any:
        return None


try:
    from src.core.memory_manager import build_query_context

    MEMORY_AVAILABLE = True
except Exception as exc:
    MEMORY_AVAILABLE = False
    logger.warning("Memory manager unavailable: %s", exc)

    def build_query_context(*args: Any, **kwargs: Any) -> Any:
        return ""


# Vector store
try:
    from src.retrieval.vector_store import get_vector_store

    VECTOR_STORE_AVAILABLE = True
except Exception as exc:
    VECTOR_STORE_AVAILABLE = False
    get_vector_store = None
    logger.warning("Vector store unavailable: %s", exc)


# Hybrid retrieval
try:
    from src.retrieval.hybrid_search import hybrid_search

    HYBRID_SEARCH_AVAILABLE = True
except Exception as exc:
    HYBRID_SEARCH_AVAILABLE = False
    hybrid_search = None
    logger.warning("Hybrid search unavailable: %s", exc)


# Reranker
try:
    from src.retrieval.cross_encoder import (
        get_cross_encoder_reranker,
        hybrid_search_with_rerank,
    )

    RERANKER_AVAILABLE = True
except Exception as exc:
    RERANKER_AVAILABLE = False
    get_cross_encoder_reranker = None
    hybrid_search_with_rerank = None
    logger.warning("Reranker unavailable: %s", exc)


# RAG engine
try:
    from src.pipeline.rag_engine import get_rag_engine, rag_query

    RAG_AVAILABLE = True
except Exception as exc:
    RAG_AVAILABLE = False
    get_rag_engine = None
    rag_query = None
    logger.warning("RAG engine unavailable: %s", exc)


# Graph
try:
    from src.orchestrator.graph import process_query_via_graph as _graph_process_query

    GRAPH_AVAILABLE = True
except Exception as exc:
    GRAPH_AVAILABLE = False
    _graph_process_query = None
    logger.warning("LangGraph orchestrator unavailable: %s", exc)


# Parser
try:
    from src.ingestion.multimodal_parser import get_multimodal_parser

    PARSER_AVAILABLE = True
except Exception as exc:
    PARSER_AVAILABLE = False
    get_multimodal_parser = None
    logger.warning("Multimodal parser unavailable: %s", exc)


# Hardened authentication/RBAC service. The application wrappers below
# delegate privileged account changes to this service so UI-level checks cannot
# bypass tenant/last-admin protections implemented in src.auth.database.
try:
    from src.auth import database as _auth_database

    HARDENED_AUTH_DATABASE_AVAILABLE = True
except Exception as exc:
    _auth_database = None
    HARDENED_AUTH_DATABASE_AVAILABLE = False
    logger.warning(
        "Hardened auth database service unavailable: %s",
        exc,
    )


# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================

APP_VERSION = _clean_text(
    getattr(settings, "APP_VERSION", "1.0.0"),
    50,
) or "1.0.0"

APP_NAME = _clean_text(
    getattr(settings, "APP_NAME", "PolicyGuard AI"),
    100,
) or "PolicyGuard AI"

st.set_page_config(
    page_title=f"{APP_NAME} | Enterprise HR",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/",
        "About": (
            f"**{APP_NAME}**\n\n"
            f"Enterprise HR RAG Platform · v{APP_VERSION}"
        ),
    },
)

# =============================================================================
# PREMIUM UI
# =============================================================================

_CUSTOM_CSS = """
<style>
:root {
    --pg-bg: #070b14;
    --pg-surface: rgba(17, 24, 39, .78);
    --pg-surface-2: rgba(24, 32, 48, .82);
    --pg-border: rgba(148, 163, 184, .15);
    --pg-text: #f8fafc;
    --pg-muted: #94a3b8;
    --pg-accent: #60a5fa;
    --pg-purple: #a78bfa;
    --pg-green: #34d399;
    --pg-yellow: #fbbf24;
    --pg-red: #fb7185;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(96,165,250,.12), transparent 28%),
        radial-gradient(circle at 85% 5%, rgba(167,139,250,.10), transparent 26%),
        linear-gradient(145deg, #070b14 0%, #0d1320 48%, #101827 100%);
    color: var(--pg-text);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(7,11,20,.98), rgba(13,19,32,.98));
    border-right: 1px solid var(--pg-border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

h1, h2, h3, h4 {
    color: #f8fafc !important;
    letter-spacing: -.02em;
}

.hero {
    padding: 2.2rem 2.4rem;
    border: 1px solid var(--pg-border);
    border-radius: 24px;
    background:
        linear-gradient(135deg,
            rgba(17,24,39,.92),
            rgba(30,41,59,.62));
    box-shadow: 0 20px 70px rgba(0,0,0,.24);
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: clamp(2.3rem, 5vw, 4.3rem);
    line-height: 1;
    font-weight: 850;
    margin: 0;
    background: linear-gradient(135deg, #93c5fd, #c4b5fd, #6ee7b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    margin-top: 1rem;
    color: #a8b3c4;
    font-size: 1.08rem;
    line-height: 1.7;
    max-width: 850px;
}

.pg-card {
    min-height: 185px;
    padding: 1.35rem;
    border: 1px solid var(--pg-border);
    border-radius: 20px;
    background: linear-gradient(
        145deg,
        rgba(17,24,39,.88),
        rgba(30,41,59,.58)
    );
    box-shadow: 0 10px 35px rgba(0,0,0,.15);
}

.pg-card:hover {
    border-color: rgba(96,165,250,.35);
}

.pg-card-icon {
    font-size: 2rem;
    margin-bottom: .65rem;
}

.pg-card-title {
    font-size: 1.05rem;
    font-weight: 750;
    color: #f8fafc;
    margin-bottom: .45rem;
}

.pg-card-text {
    color: #94a3b8;
    line-height: 1.55;
    font-size: .92rem;
}

.metric-card {
    padding: 1.25rem;
    border-radius: 18px;
    border: 1px solid var(--pg-border);
    background: rgba(15,23,42,.72);
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
}

.metric-label {
    color: #94a3b8;
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .12em;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .45rem .8rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 700;
    border: 1px solid transparent;
}

.status-online {
    background: rgba(52,211,153,.10);
    color: #6ee7b7;
    border-color: rgba(52,211,153,.20);
}

.status-warning {
    background: rgba(251,191,36,.10);
    color: #fcd34d;
    border-color: rgba(251,191,36,.20);
}

.status-error {
    background: rgba(251,113,133,.10);
    color: #fda4af;
    border-color: rgba(251,113,133,.20);
}

.role-admin {
    color: #fda4af;
}

.role-editor {
    color: #fcd34d;
}

.role-viewer {
    color: #93c5fd;
}

.small-muted {
    color: #94a3b8;
    font-size: .84rem;
}

.footer {
    padding: 1rem 1.25rem;
    margin-top: 2rem;
    border-top: 1px solid var(--pg-border);
    color: #64748b;
    font-size: .78rem;
    text-align: center;
}

.chat-meta {
    color: #64748b;
    font-size: .75rem;
    margin-top: .5rem;
}

.chat-question-card {
    border: 1px solid var(--pg-border);
    border-radius: 12px;
    background: rgba(30, 41, 59, .48);
    padding: .85rem 1rem;
    margin: .1rem 0 .35rem 0;
    color: #f8fafc;
    font-size: .96rem;
    line-height: 1.55;
    box-shadow: 0 4px 16px rgba(0, 0, 0, .12);
}

.chat-question-label {
    color: #94a3b8;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .02em;
    margin-bottom: .28rem;
    text-transform: uppercase;
}
.chat-response-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
    opacity: 0.82;
}


div[data-testid="stMetric"] {
    background: rgba(15,23,42,.58);
    border: 1px solid var(--pg-border);
    padding: 1rem;
    border-radius: 16px;
}

button[kind="primary"] {
    border-radius: 12px;
}

.stButton > button {
    border-radius: 11px;
}

section[data-testid="stFileUploaderDropzone"] {
    border-radius: 14px;
}
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================


def _init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "username": None,
        "user_role": None,
        "user_id": None,
        "organization_id": None,
        "view": "home",
        "messages": [],
        "query_history": [],
        "last_log": {},
        "last_uploaded": [],
        "system_status": "initializing",
        "show_onboarding": True,
        "request_id": None,
        "last_error": None,
        "documents_refresh": 0,
        "vector_store_ready": False,
        "rag_engine": None,
        "active_chat_session_id": None,
        "chat_sessions_loaded": False,
        "route_trace": [],
        "persistent_memory_enabled": True,
        "talent_jd_text": "",
        "talent_jd_title": "",
        "talent_results": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()

# =============================================================================
# AUTHORIZATION
# =============================================================================


def _check_permission(required_role: str) -> bool:
    """Check hierarchical RBAC."""
    if not st.session_state.get("authenticated"):
        return False

    current_role = _safe_role(
        st.session_state.get("user_role", "viewer")
    )
    required_role = _safe_role(required_role)

    return ROLE_LEVEL.get(current_role, 0) >= ROLE_LEVEL.get(
        required_role,
        999,
    )


def _require_login() -> None:
    if not st.session_state.get("authenticated"):
        st.error("Please sign in to access this area.")
        st.stop()


# =============================================================================
# SYSTEM HEALTH
# =============================================================================


def _system_components() -> Dict[str, bool]:
    """Return current subsystem availability."""
    vector_files = (
        (VECTOR_DB_DIR / "faiss.index").exists()
        and (VECTOR_DB_DIR / "chunks.pkl").exists()
    )

    return {
        "Database": _DB_READY,
        "Security Guard": SECURITY_AVAILABLE,
        "Embedder": EMBEDDER_AVAILABLE,
        "Vector Store": VECTOR_STORE_AVAILABLE and vector_files,
        "Hybrid Search": HYBRID_SEARCH_AVAILABLE,
        "Reranker": RERANKER_AVAILABLE,
        "RAG Engine": RAG_AVAILABLE,
        "LangGraph": GRAPH_AVAILABLE,
        "Parser": PARSER_AVAILABLE,
        "Semantic Cache": CACHE_AVAILABLE,
        "Memory": MEMORY_AVAILABLE,
        "Persistent Memory": _DB_READY,
        "Talent Intelligence": _DB_READY and EMBEDDER_AVAILABLE,
    }


def _update_system_status() -> None:
    components = _system_components()

    if not _DB_READY:
        st.session_state.system_status = "error"
    elif components.get("Vector Store"):
        st.session_state.system_status = "online"
    else:
        st.session_state.system_status = "awaiting_data"


def _show_status_indicator() -> None:
    _update_system_status()

    status = st.session_state.system_status

    if status == "online":
        st.markdown(
            '<div class="status-pill status-online">● System Online</div>',
            unsafe_allow_html=True,
        )
    elif status == "awaiting_data":
        st.markdown(
            '<div class="status-pill status-warning">● Awaiting Knowledge Base</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-error">● System Degraded</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# VECTOR STORE / RAG INITIALIZATION
# =============================================================================


@st.cache_resource(show_spinner=False)
def _load_vector_store_cached() -> Tuple[Any, bool]:
    """
    Load the persisted vector store.

    The function deliberately validates both the FAISS index and chunk file.
    """
    if not VECTOR_STORE_AVAILABLE or get_vector_store is None:
        return None, False

    if faiss is None:
        logger.error("FAISS is unavailable.")
        return None, False

    index_file = VECTOR_DB_DIR / "faiss.index"
    chunks_file = VECTOR_DB_DIR / "chunks.pkl"

    if not index_file.exists() or not chunks_file.exists():
        return None, False

    try:
        with chunks_file.open("rb") as handle:
            chunks = pickle.load(handle)

        if not isinstance(chunks, list):
            raise ValueError("Persisted chunks must be a list.")

        # Normalize persisted metadata. Legacy single-tenant artifacts are
        # explicitly assigned to the deployment's configured legacy scope;
        # newly indexed data always carries an explicit organization/namespace.
        normalized_persisted = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            item = dict(chunk)
            meta = item.get("metadata")
            meta = dict(meta) if isinstance(meta, dict) else {}
            meta.setdefault("organization_id", _configured_organization_id())
            meta.setdefault("namespace", "policy")
            item["metadata"] = meta
            normalized_persisted.append(item)
        chunks = normalized_persisted

        index = faiss.read_index(str(index_file))

        vector_store = get_vector_store()

        vector_store.index = index
        vector_store.chunks = chunks

        if EMBEDDER_AVAILABLE and get_embedder is not None:
            try:
                vector_store.set_embedder(
                    get_embedder()
                )
            except Exception:
                logger.warning(
                    "Could not attach embedder to vector store.",
                    exc_info=True,
                )

        if not RAG_AVAILABLE or get_rag_engine is None:
            return None, False

        engine = get_rag_engine()

        if hasattr(engine, "set_vector_store"):
            engine.set_vector_store(vector_store)

        if EMBEDDER_AVAILABLE and get_embedder is not None and hasattr(engine, "set_embedder"):
            try:
                runtime_embedder = get_embedder()
                if runtime_embedder is not None:
                    engine.set_embedder(runtime_embedder)
            except Exception:
                logger.warning(
                    "Could not attach embedder to RAG engine.",
                    exc_info=True,
                )

        return engine, True

    except Exception:
        logger.exception("Vector store loading failed")
        return None, False


def _load_data(force: bool = False) -> None:
    """
    Initialize/reload the persisted RAG runtime.

    Streamlit reruns are session-based, while FAISS/chunks are persisted on disk.
    The chat view can therefore recover the knowledge base automatically without
    requiring the user to press a manual refresh button.
    """
    if force:
        try:
            _load_vector_store_cached.clear()
        except Exception:
            pass
        st.session_state.vector_store_ready = False
        st.session_state.rag_engine = None

    if st.session_state.get("vector_store_ready") and st.session_state.get("rag_engine") is not None:
        return

    try:
        engine, ready = _load_vector_store_cached()

        if ready and engine is not None:
            st.session_state.rag_engine = engine
            st.session_state.vector_store_ready = True
            st.session_state.system_status = "online"
        else:
            st.session_state.rag_engine = None
            st.session_state.vector_store_ready = False
            st.session_state.system_status = "awaiting_data"

    except Exception:
        logger.exception("Data initialization failed")
        st.session_state.rag_engine = None
        st.session_state.vector_store_ready = False
        st.session_state.system_status = "error"


_load_data()



# =============================================================================
# PERSISTENT CONVERSATION MEMORY
# =============================================================================


def _ensure_chat_session(username: str) -> int:
    """Create or recover a durable chat session for the signed-in user."""
    existing = st.session_state.get("active_chat_session_id")
    if existing:
        try:
            with closing(_db_connect()) as conn:
                row = conn.execute(
                    "SELECT id FROM chat_sessions WHERE id = ? AND username = ? AND organization_id = ? AND is_active = 1",
                    (int(existing), username, _current_organization_id()),
                ).fetchone()
            if row:
                return int(existing)
        except Exception:
            logger.debug("Could not validate active chat session", exc_info=True)

    try:
        with closing(_db_connect()) as conn:
            row = conn.execute(
                """
                SELECT id
                FROM chat_sessions
                WHERE organization_id = ? AND username = ? AND is_active = 1
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (_current_organization_id(), username),
            ).fetchone()

            if row:
                session_id = int(row["id"])
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO chat_sessions (username, organization_id, title)
                    VALUES (?, ?, ?)
                    """,
                    (username, _current_organization_id(), "New conversation"),
                )
                session_id = int(cursor.lastrowid)

            conn.commit()

        st.session_state.active_chat_session_id = session_id
        return session_id

    except Exception:
        logger.exception("Could not initialize persistent chat session")
        return 0


def _load_persistent_messages(username: str, session_id: int) -> List[Dict[str, Any]]:
    """Load durable conversation memory into the Streamlit session."""
    if not session_id:
        return []

    try:
        with closing(_db_connect()) as conn:
            rows = conn.execute(
                """
                SELECT role, content, metadata
                FROM chat_messages
                WHERE organization_id = ? AND session_id = ? AND EXISTS (
                    SELECT 1 FROM chat_sessions
                    WHERE id = ? AND username = ? AND organization_id = ?
                )
                ORDER BY id ASC
                LIMIT 200
                """,
                (_current_organization_id(), session_id, session_id, username, _current_organization_id()),
            ).fetchall()

        messages: List[Dict[str, Any]] = []
        for row in rows:
            item: Dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
            }
            try:
                if row["metadata"]:
                    parsed = json.loads(row["metadata"])
                    if isinstance(parsed, dict):
                        item["metadata"] = parsed
            except Exception:
                pass
            messages.append(item)

        return messages

    except Exception:
        logger.exception("Persistent memory load failed")
        return []


def _persist_chat_message(
    username: str,
    session_id: int,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist one conversation message safely."""
    if not session_id or role not in {"user", "assistant", "system"}:
        return

    try:
        metadata_json = json.dumps(metadata or {}, default=str)[:10000]

        with closing(_db_connect()) as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (
                    session_id, organization_id, role, content, metadata
                )
                SELECT ?, ?, ?, ?, ?
                WHERE EXISTS (
                    SELECT 1 FROM chat_sessions
                    WHERE id = ? AND username = ? AND organization_id = ?
                )
                """,
                (
                    session_id,
                    _current_organization_id(),
                    role,
                    _clean_text(content, 20000),
                    metadata_json,
                    session_id,
                    username,
                    _current_organization_id(),
                ),
            )

            conn.execute(
                """
                UPDATE chat_sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND username = ? AND organization_id = ?
                """,
                (session_id, username, _current_organization_id()),
            )

            conn.commit()

    except Exception:
        logger.debug("Persistent chat write failed", exc_info=True)


def _start_new_persistent_chat(username: str) -> int:
    """Create a new durable conversation without deleting previous history."""
    try:
        with closing(_db_connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_sessions (username, organization_id, title)
                VALUES (?, ?, ?)
                """,
                (username, _current_organization_id(), "New conversation"),
            )
            conn.commit()
            session_id = int(cursor.lastrowid)

        st.session_state.active_chat_session_id = session_id
        st.session_state.messages = []
        st.session_state.query_history = []
        st.session_state.route_trace = []
        return session_id

    except Exception:
        logger.exception("Could not create new chat session")
        return 0


def _initialize_persistent_memory(username: str) -> None:
    """Hydrate the current Streamlit session from durable chat storage."""
    if st.session_state.get("chat_sessions_loaded"):
        return

    session_id = _ensure_chat_session(username)

    if session_id:
        messages = _load_persistent_messages(username, session_id)
        st.session_state.messages = messages

    st.session_state.chat_sessions_loaded = True


def _chat_memory_context(
    username: str,
    session_id: int,
    max_messages: int = 12,
) -> str:
    """Return a compact, bounded memory window for the model."""
    messages = _load_persistent_messages(username, session_id)
    recent = messages[-max_messages:]

    if not recent:
        return ""

    lines = []
    for message in recent:
        role = message.get("role", "user")
        content = _clean_text(message.get("content", ""), 1800)
        if content:
            lines.append(f"{role.upper()}: {content}")

    return "\n".join(lines)


# =============================================================================
# QUERY ROUTING
# =============================================================================


def _route_query(query: str) -> Dict[str, Any]:
    """
    Lightweight deterministic router.

    The route is deliberately visible in the UI and metadata so users can see
    which product capability handled the request.
    """
    normalized = _clean_text(query, MAX_QUERY_LENGTH).lower()

    talent_terms = (
        "resume", "cv", "candidate", "candidates", "bench",
        "ai engineer", "software engineer", "developer", "hire",
        "hiring", "job description", "jd", "skills", "shortlist",
        "short list", "who should", "best fit", "talent", "vacancy",
    )

    policy_terms = (
        "policy", "leave", "benefit", "benefits", "attendance",
        "conduct", "harassment", "grievance", "payroll", "holiday",
        "holidays", "employee", "disciplinary", "code of conduct",
        "working hours", "remote work", "maternity", "insurance",
        "compliance", "procedure", "rule",
    )

    if any(term in normalized for term in talent_terms):
        route = "Talent Intelligence"
        reason = "Recruiting, candidate or job-description intent detected."
    elif any(term in normalized for term in policy_terms):
        route = "HR Policy RAG"
        reason = "HR policy or employee-procedure intent detected."
    else:
        route = "General HR Assistant"
        reason = "General HR question; policy retrieval remains the grounding source when available."

    return {
        "route": route,
        "reason": reason,
        "router_version": "deterministic-v1",
    }


def _render_professional_answer(answer: str) -> None:
    """
    Render the assistant response as a bordered message card matching the
    bordered user-question card.

    The response content remains unchanged; only its presentation is wrapped
    in the same native Streamlit border treatment. Routing, retrieval and
    source telemetry is rendered separately in the expandable panel below.
    """
    text = _clean_text(answer, 20000)
    if not text:
        return

    with st.container(border=True):
        st.markdown(
            '<div class="chat-response-label">PolicyGuard AI</div>',
            unsafe_allow_html=True,
        )
        st.markdown(text)


def _format_citation(citation: Any) -> str:
    """Format citation/source metadata consistently for the telemetry panel."""
    if isinstance(citation, dict):
        source = (
            citation.get("source")
            or citation.get("filename")
            or citation.get("file_name")
            or "Indexed HR document"
        )
        page = citation.get("page") or citation.get("page_number")
        score = citation.get("score")
        label = _clean_text(source, 500)
        if page not in (None, ""):
            label += f" — Page {page}"
        if score not in (None, ""):
            try:
                label += f" — relevance {float(score):.2f}"
            except (TypeError, ValueError):
                pass
        return label

    return _clean_text(citation, 700)


def _render_route_trace(metadata: Dict[str, Any]) -> None:
    """Show transparent routing, retrieval, and source telemetry in one panel."""
    route = metadata.get("route") or metadata.get("agent") or "HR Assistant"
    retrieval_count = metadata.get(
        "chunks_retrieved",
        metadata.get("retrieval_count", 0),
    )

    with st.expander("🔎 Routing, Retrieval & Sources", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Route", str(route))
        with c2:
            st.metric("Policy Chunks", int(retrieval_count or 0))
        with c3:
            st.metric(
                "Latency",
                f"{int(metadata.get('latency_ms', 0) or 0)} ms",
            )

        st.caption(
            str(
                metadata.get("route_reason")
                or "Request routed through the HR intelligence pipeline."
            )
        )

        details = []
        if metadata.get("retrieval_method"):
            details.append(("Retrieval", metadata.get("retrieval_method")))
        if metadata.get("retrieval_model"):
            details.append(("Embedding model", metadata.get("retrieval_model")))
        if metadata.get("model_used"):
            details.append(("Generation model", metadata.get("model_used")))
        if metadata.get("cache_hit") is not None:
            details.append(("Cache", "HIT" if metadata.get("cache_hit") else "MISS"))
        if metadata.get("namespace"):
            details.append(("Namespace", metadata.get("namespace")))

        if details:
            for label, value in details:
                st.write(f"**{label}:** {_clean_text(value, 500)}")

        citations = metadata.get("citations") or metadata.get("sources") or []
        if isinstance(citations, (list, tuple)) and citations:
            st.markdown("**Sources & citations**")
            for index, citation in enumerate(citations[:10], start=1):
                st.markdown(f"{index}. {_format_citation(citation)}")
        else:
            st.caption("No separate citation metadata was returned for this response.")

        retrieved = metadata.get("retrieved_chunks")
        if isinstance(retrieved, list) and retrieved:
            with st.expander("Retrieved evidence", expanded=False):
                for index, chunk in enumerate(retrieved[:6], start=1):
                    if not isinstance(chunk, dict):
                        continue
                    content = _clean_text(chunk.get("content", ""), 900)
                    if not content:
                        continue
                    st.markdown(f"**Evidence {index}**")
                    st.caption(_format_citation(chunk))
                    st.write(content)


def _extract_candidate_name(filename: str, text_value: str) -> str:
    """Best-effort candidate name extraction without requiring a structured resume."""
    stem = Path(filename).stem
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()

    if cleaned and not re.search(
        r"\b(resume|cv|curriculum vitae|profile)\b",
        cleaned,
        re.I,
    ):
        return cleaned[:120]

    for pattern in (
        r"(?im)^\s*name\s*[:\-]\s*(.+)$",
        r"(?im)^\s*candidate\s*[:\-]\s*(.+)$",
    ):
        match = re.search(pattern, text_value or "")
        if match:
            return _clean_text(match.group(1), 120)

    return cleaned[:120] or "Unnamed Candidate"


def _extract_skill_terms(text_value: str) -> set[str]:
    """Extract useful searchable skill phrases from HR/JD text."""
    normalized = re.sub(r"[^a-zA-Z0-9+#.\- ]+", " ", text_value.lower())

    known = {
        "python", "java", "javascript", "typescript", "react", "node.js",
        "sql", "postgresql", "mysql", "mongodb", "aws", "azure", "gcp",
        "docker", "kubernetes", "terraform", "git", "linux", "fastapi",
        "django", "flask", "pytorch", "tensorflow", "scikit-learn",
        "machine learning", "deep learning", "generative ai", "genai",
        "llm", "nlp", "computer vision", "rag", "langchain", "langgraph",
        "faiss", "hugging face", "transformers", "spark", "databricks",
        "airflow", "power bi", "tableau", "excel", "communication",
        "leadership", "project management", "agile", "scrum",
    }

    return {
        skill for skill in known
        if skill in normalized
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))

    if denom <= 1e-12:
        return 0.0

    return float(np.dot(a, b) / denom)


def _index_candidate_resume(
    uploaded_file: Any,
    username: str,
) -> Dict[str, Any]:
    """Parse and persist one employee/candidate resume."""
    if not PARSER_AVAILABLE or get_multimodal_parser is None:
        raise RuntimeError("Document parser is unavailable.")

    destination = _persist_uploaded_file(uploaded_file, _current_organization_id())

    try:
        # Resume text is untrusted content and can contain prompt-injection
        # payloads. Validate extracted text before it is used by AI matching.
        parser = get_multimodal_parser()
        result = parser.parse_file(str(destination))

        if not isinstance(result, dict):
            raise RuntimeError("Parser returned an invalid result.")

        chunks = result.get("chunks") or []
        texts = [
            _clean_text(chunk.get("content", ""), 20000)
            for chunk in chunks
            if isinstance(chunk, dict) and _clean_text(chunk.get("content"))
        ]

        resume_text = "\n\n".join(texts).strip()

        if not resume_text:
            raise RuntimeError(
                "No readable resume text was extracted."
            )

        resume_text = _validate_talent_input(
            resume_text,
            username,
            user_role="editor" if _check_permission("editor") else "viewer",
            field_name="Resume Content",
            max_length=50000,
        )

        candidate_name = _extract_candidate_name(
            uploaded_file.name,
            resume_text,
        )

        with closing(_db_connect()) as conn:
            # Same filename + same user is treated as an update, not a duplicate.
            existing = conn.execute(
                """
                SELECT id
                FROM talent_candidates
                WHERE organization_id = ? AND resume_filename = ? AND uploaded_by = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (_current_organization_id(), _safe_filename(uploaded_file.name), username),
            ).fetchone()

            if existing:
                candidate_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE talent_candidates
                    SET candidate_name = ?,
                        resume_text = ?,
                        resume_path = ?,
                        updated_at = CURRENT_TIMESTAMP,
                        status = 'bench'
                    WHERE id = ?
                    """,
                    (
                        candidate_name,
                        resume_text,
                        str(destination),
                        candidate_id,
                    ),
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO talent_candidates (
                        candidate_name,
                        organization_id,
                        resume_filename,
                        resume_path,
                        resume_text,
                        uploaded_by,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, 'bench')
                    """,
                    (
                        candidate_name,
                        _current_organization_id(),
                        _safe_filename(uploaded_file.name),
                        str(destination),
                        resume_text,
                        username,
                    ),
                )
                candidate_id = int(cursor.lastrowid)

            conn.commit()

        _audit_log(
            username,
            "TALENT_RESUME_UPLOADED",
            f"Candidate: {candidate_name}; file: {_safe_filename(uploaded_file.name)}",
        )

        return {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "filename": _safe_filename(uploaded_file.name),
            "chunks": len(texts),
        }

    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _get_candidates(username: str, include_all: bool = False) -> List[Dict[str, Any]]:
    """Return candidate records according to HR permissions."""
    try:
        with closing(_db_connect()) as conn:
            if include_all:
                rows = conn.execute(
                    """
                    SELECT id, candidate_name, resume_filename, uploaded_by,
                           created_at, updated_at, status
                    FROM talent_candidates
                    WHERE organization_id = ?
                    ORDER BY updated_at DESC, id DESC
                    """,
                    (_current_organization_id(),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, candidate_name, resume_filename, uploaded_by,
                           created_at, updated_at, status
                    FROM talent_candidates
                    WHERE organization_id = ? AND uploaded_by = ?
                    ORDER BY updated_at DESC, id DESC
                    """,
                    (_current_organization_id(), username),
                ).fetchall()

        return [dict(row) for row in rows]

    except Exception:
        logger.exception("Could not load talent candidates")
        return []


def _get_candidate_records(
    username: str,
    include_all: bool = False,
) -> List[Dict[str, Any]]:
    """Return full candidate records within the caller's authorized scope."""
    try:
        with closing(_db_connect()) as conn:
            if include_all:
                rows = conn.execute(
                    """
                    SELECT id, candidate_name, resume_filename,
                           resume_path, resume_text, uploaded_by, status
                    FROM talent_candidates
                    WHERE organization_id = ?
                    ORDER BY id ASC
                    """,
                    (_current_organization_id(),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, candidate_name, resume_filename,
                           resume_path, resume_text, uploaded_by, status
                    FROM talent_candidates
                    WHERE organization_id = ? AND uploaded_by = ?
                    ORDER BY id ASC
                    """,
                    (_current_organization_id(), username),
                ).fetchall()
        return [dict(row) for row in rows]
    except Exception:
        logger.exception("Could not load candidate records")
        return []


def _resolve_candidate_resume_path(
    candidate: Dict[str, Any],
    username: str,
    include_all: bool = False,
) -> Path:
    """Resolve a stored resume path only after authorization and containment checks."""
    owner = str(candidate.get("uploaded_by") or "")
    if not include_all and owner != username:
        raise PermissionError("You are not authorized to view this resume.")

    raw_path = candidate.get("resume_path")
    if not raw_path:
        raise FileNotFoundError("The stored resume file is unavailable.")

    resume_path = Path(str(raw_path)).resolve()
    upload_root = (UPLOAD_DIR / _current_organization_id()).resolve()

    try:
        resume_path.relative_to(upload_root)
    except ValueError as exc:
        raise PermissionError("Resume storage location is outside the application upload area.") from exc

    if not resume_path.is_file():
        raise FileNotFoundError("The stored resume file is no longer available.")

    return resume_path


def _render_resume_viewer(
    candidate: Dict[str, Any],
    username: str,
) -> None:
    """Render an authorized resume preview and download action."""
    include_all = _check_permission("admin")

    try:
        resume_path = _resolve_candidate_resume_path(
            candidate,
            username,
            include_all=include_all,
        )
    except PermissionError as exc:
        st.error(str(exc))
        return
    except FileNotFoundError as exc:
        st.warning(str(exc))
        return

    resume_bytes = resume_path.read_bytes()
    filename = _safe_filename(candidate.get("resume_filename") or resume_path.name)

    st.markdown(
        f"### Resume · {_escape(candidate.get('candidate_name', 'Candidate'))}"
    )
    st.caption(
        f"{_escape(filename)} · Uploaded by {_escape(candidate.get('uploaded_by', 'unknown'))}"
    )

    st.download_button(
        "⬇️ Download Resume",
        data=resume_bytes,
        file_name=filename,
        mime="application/pdf" if resume_path.suffix.lower() == ".pdf" else "application/octet-stream",
        key=f"download_resume_{candidate.get('id')}",
        use_container_width=True,
    )

    if resume_path.suffix.lower() == ".pdf":
        import base64
        encoded = base64.b64encode(resume_bytes).decode("ascii")
        st.components.v1.html(
            f'''<iframe src="data:application/pdf;base64,{encoded}"
                width="100%" height="800" style="border:1px solid #333; border-radius:8px;"
                title="Resume preview"></iframe>''',
            height=820,
            scrolling=True,
        )
    else:
        st.info(
            "Inline preview is available for PDF resumes. Use Download Resume "
            "to open this document in its native application."
        )


def _score_candidates(
    job_title: str,
    job_description: str,
    username: str,
) -> List[Dict[str, Any]]:
    """Rank internal candidates against a JD using semantic + skill overlap."""
    if not EMBEDDER_AVAILABLE or get_embedder is None:
        raise RuntimeError("Embedding subsystem is unavailable.")

    candidates = _get_candidate_records(
        username,
        include_all=_check_permission("admin"),
    )

    if not candidates:
        return []

    runtime_embedder = get_embedder()
    if runtime_embedder is None:
        raise RuntimeError("Embedding model is unavailable.")

    safe_job_title = _validate_talent_input(
        job_title,
        username,
        user_role="editor" if _check_permission("editor") else "viewer",
        field_name="Job Title",
        max_length=200,
    )
    safe_job_description = _validate_talent_input(
        job_description,
        username,
        user_role="editor" if _check_permission("editor") else "viewer",
        field_name="Job Description",
        max_length=12000,
    )

    jd_text = _clean_text(
        f"{safe_job_title}\n{safe_job_description}",
        20000,
    )

    jd_embedding = np.asarray(
        runtime_embedder.encode_query(jd_text),
        dtype=np.float32,
    ).reshape(-1)

    jd_skills = _extract_skill_terms(jd_text)

    ranked = []

    for candidate in candidates:
        resume_text = _clean_text(candidate.get("resume_text"), 50000)

        resume_embedding = np.asarray(
            runtime_embedder.encode_query(resume_text),
            dtype=np.float32,
        ).reshape(-1)

        semantic = max(
            0.0,
            min(
                1.0,
                (_cosine_similarity(jd_embedding, resume_embedding) + 1.0) / 2.0,
            ),
        )

        candidate_skills = _extract_skill_terms(resume_text)
        matched = sorted(jd_skills.intersection(candidate_skills))

        keyword = (
            len(matched) / len(jd_skills)
            if jd_skills
            else 0.0
        )

        # Semantic fit is primary; explicit skill overlap is the explainability
        # layer. This is decision support, not an automated hiring decision.
        final_score = (semantic * 0.70) + (keyword * 0.30)

        ranked.append({
            "candidate_id": candidate["id"],
            "candidate": candidate["candidate_name"],
            "resume": candidate["resume_filename"],
            "status": candidate["status"],
            "semantic_score": round(semantic * 100, 1),
            "skill_match": round(keyword * 100, 1),
            "fit_score": round(final_score * 100, 1),
            "matched_skills": ", ".join(matched) if matched else "No known skill overlap detected",
        })

    ranked.sort(
        key=lambda item: (
            float(item["fit_score"]),
            float(item["skill_match"]),
        ),
        reverse=True,
    )

    for position, item in enumerate(ranked, start=1):
        item["rank"] = position

    return ranked


def _save_talent_search(
    username: str,
    job_title: str,
    job_description: str,
    results: List[Dict[str, Any]],
) -> None:
    """Persist transparent talent-ranking results for HR/admin review."""
    try:
        with closing(_db_connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO talent_searches (
                    searched_by, organization_id, job_title, job_description
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    _current_organization_id(),
                    _clean_text(job_title, 200),
                    _clean_text(job_description, 12000),
                ),
            )
            search_id = int(cursor.lastrowid)

            for result in results:
                conn.execute(
                    """
                    INSERT INTO talent_matches (
                        search_id, organization_id, candidate_id,
                        semantic_score, keyword_score,
                        final_score, rank_position, matched_skills
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        search_id,
                        _current_organization_id(),
                        int(result["candidate_id"]),
                        float(result["semantic_score"]),
                        float(result["skill_match"]),
                        float(result["fit_score"]),
                        int(result["rank"]),
                        result["matched_skills"],
                    ),
                )

            conn.commit()

        _audit_log(
            username,
            "TALENT_SEARCH_EXECUTED",
            f"JD: {job_title}; candidates ranked: {len(results)}",
        )

    except Exception:
        logger.exception("Talent search persistence failed")


# =============================================================================
# UNTRUSTED HR/TALENT INPUT VALIDATION
# =============================================================================


def _validate_talent_input(
    text_value: str,
    username: str,
    user_role: str,
    field_name: str,
    max_length: int,
) -> str:
    """
    Validate recruiter-supplied JD/talent text through the same security
    boundary used by Live Chat before it reaches embeddings or AI components.

    Legitimate recruiting content is preserved; the SecurityGuard remains the
    authoritative decision point for injection/unsafe-input detection.
    """
    bounded = _clean_text(text_value, max_length)

    if not bounded:
        raise ValueError(f"{field_name} is required.")

    safe, reason, sanitized = _security_validate_query(
        query=bounded,
        username=username,
        user_role=user_role,
    )

    if not safe:
        _audit_log(
            username,
            "TALENT_INPUT_BLOCKED",
            f"{field_name}: {reason}",
            query_preview=bounded,
            threat_type=reason,
            blocked=True,
        )
        raise PermissionError(
            reason
            or f"The supplied {field_name} was blocked by the security policy."
        )

    sanitized_text = _clean_text(sanitized or bounded, max_length)

    if not sanitized_text:
        raise ValueError(
            f"The supplied {field_name} was empty after security validation."
        )

    return sanitized_text


# =============================================================================
# QUERY PROCESSING
# =============================================================================


def _extract_answer(result: Any) -> str:
    """
    Normalize different RAG/graph response formats into displayable text.
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        for key in (
            "answer",
            "response",
            "final_answer",
            "output",
            "content",
            "text",
        ):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Some graph implementations return messages.
        messages = result.get("messages")

        if isinstance(messages, Sequence):
            for message in reversed(messages):
                if isinstance(message, dict):
                    content = message.get("content")
                else:
                    content = getattr(message, "content", None)

                if isinstance(content, str) and content.strip():
                    return content.strip()

    if hasattr(result, "content"):
        content = getattr(result, "content", None)
        if isinstance(content, str):
            return content.strip()

    return _clean_text(result, 20000)


def _extract_metadata(result: Any) -> Dict[str, Any]:
    """Extract optional metrics/citations metadata."""
    if not isinstance(result, dict):
        return {}

    metadata: Dict[str, Any] = {}

    for key in (
        "metrics",
        "metadata",
        "citations",
        "sources",
        "retrieved_chunks",
        "router_decision",
        "agent",
        "model_used",
        "latency_ms",
        "cache_hit",
        "tokens_used",
        "cost_usd",
    ):
        if key in result:
            metadata[key] = result[key]

    return metadata



def _retrieve_policy_context(
    query: str,
    top_k: int = 6,
    organization_id: Optional[str] = None,
    namespace: str = "policy",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Retrieve policy chunks directly from the persisted vector store."""
    if not VECTOR_STORE_AVAILABLE or get_vector_store is None:
        raise RuntimeError("Vector store is unavailable.")
    if not EMBEDDER_AVAILABLE or get_embedder is None:
        raise RuntimeError("Embedding subsystem is unavailable.")

    runtime_embedder = get_embedder()
    if runtime_embedder is None:
        raise RuntimeError("get_embedder() returned None.")

    query_embedding = runtime_embedder.encode_query(query)
    if query_embedding is None:
        raise RuntimeError("Query embedding generation returned None.")

    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
        query_embedding = query_embedding[0]
    if query_embedding.ndim != 1:
        raise RuntimeError(f"Invalid query embedding shape: {query_embedding.shape}")
    if query_embedding.shape[0] != runtime_embedder.dimension:
        raise RuntimeError(
            "Query embedding dimension mismatch: "
            f"model={runtime_embedder.dimension}, query={query_embedding.shape[0]}"
        )
    if not np.isfinite(query_embedding).all():
        raise RuntimeError("Query embedding contains NaN or infinite values.")

    vector_store = get_vector_store()
    if vector_store is None:
        raise RuntimeError("get_vector_store() returned None.")

    if getattr(vector_store, "embedder", None) is None:
        setter = getattr(vector_store, "set_embedder", None)
        if callable(setter):
            setter(runtime_embedder)
        else:
            vector_store.embedder = runtime_embedder

    organization_id = organization_id or _current_organization_id()
    if not ORGANIZATION_ID_PATTERN.fullmatch(organization_id):
        raise PermissionError("Invalid organization scope.")
    namespace = "talent" if namespace == "talent" else "policy"

    search_kwargs = {
        "query": query,
        "embedding": query_embedding,
        "top_k": max(top_k * 5, top_k),
        "organization_id": organization_id,
        "namespace": namespace,
    }

    try:
        signature = inspect.signature(vector_store.search)
        supported = set(signature.parameters)
        scoped_kwargs = {
            key: value for key, value in search_kwargs.items()
            if key in supported
        }
        if "query" in supported:
            results = vector_store.search(**scoped_kwargs)
        else:
            # Legacy signature: retrieve a wider candidate set, then enforce
            # tenant/namespace filtering locally. Never fall back to an
            # unscoped result being returned to the caller.
            legacy_args = [query_embedding]
            if "top_k" in supported:
                results = vector_store.search(
                    legacy_args[0],
                    top_k=max(top_k * 10, top_k),
                )
            else:
                results = vector_store.search(legacy_args[0])
    except Exception:
        logger.exception("Scoped vector retrieval failed")
        raise

    if not isinstance(results, (list, tuple)):
        results = [results] if results else []

    normalized: List[Dict[str, Any]] = []
    for item in results[:top_k]:
        if not isinstance(item, dict):
            continue
        content = _clean_text(
            item.get("content")
            or item.get("text")
            or item.get("page_content")
            or "",
            12000,
        )
        if not content:
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        item_org = _clean_text(
            metadata.get("organization_id"),
            MAX_ORGANIZATION_ID_LENGTH,
        )
        item_namespace = _clean_text(
            metadata.get("namespace") or "policy",
            50,
        )

        if item_org != organization_id or item_namespace != namespace:
            continue

        normalized.append({
            **item,
            "content": content,
            "metadata": metadata,
        })

        if len(normalized) >= top_k:
            break

    return normalized, {
        "retrieval_count": len(normalized),
        "retrieval_method": "direct-vector-fallback",
        "retrieval_model": getattr(runtime_embedder, "model_name", "unknown"),
    }


def _is_generic_no_context_answer(answer: str) -> bool:
    """Identify generic orchestration responses that are not grounded answers."""
    normalized = _clean_text(answer, 4000).lower()
    if not normalized:
        return True
    markers = (
        "i don't have any policy context",
        "i do not have any policy context",
        "no policy context available",
        "no policy context is available",
        "knowledge base is not ready",
        "please contact hr for more information",
        "i couldn't find relevant policy information",
        "i could not find relevant policy information",
        "error processing query",
        "error generating answer",
        "error:",
    )
    return any(marker in normalized for marker in markers)

def _process_query(
    query: str,
    username: str,
    user_role: str,
) -> Tuple[str, Dict[str, Any]]:
    """Execute the production query pipeline with deterministic retrieval."""
    started = time.perf_counter()
    query = _clean_text(query, MAX_QUERY_LENGTH)
    if not query:
        raise ValueError("Please enter a question.")
    if len(query) < 2:
        raise ValueError("Please provide a more detailed question.")

    if not SECURITY_AVAILABLE:
        _audit_log(
            username,
            "QUERY_BLOCKED",
            "Security guard unavailable; request failed closed.",
            query_preview=query,
            threat_type="SECURITY_GUARD_UNAVAILABLE",
            blocked=True,
        )
        raise PermissionError(
            "Security validation is currently unavailable. "
            "The request cannot be processed safely."
        )

    safe, security_reason, sanitized_query = _security_validate_query(
        query=query,
        username=username,
        user_role=user_role,
    )
    if not safe:
        _audit_log(
            username,
            "QUERY_BLOCKED",
            security_reason,
            query_preview=sanitized_query or query,
            threat_type=security_reason,
            blocked=True,
        )
        raise PermissionError(
            security_reason or "The query was blocked by the security policy."
        )

    sanitized_query = _clean_text(
        sanitized_query or query,
        MAX_QUERY_LENGTH,
    )
    request_id = str(uuid.uuid4())
    st.session_state.request_id = request_id

    route_metadata = _route_query(sanitized_query)
    metadata_route = dict(route_metadata)
    st.session_state.route_trace.append({
        "request_id": request_id,
        **route_metadata,
        "timestamp": _now_iso(),
    })

    # Semantic cache is optional; a disabled cache must never disable RAG.
    if CACHE_AVAILABLE:
        try:
            cache_signature = inspect.signature(get_cached_answer)
            required_scope = {"organization_id", "namespace", "user_id", "role"}
            if required_scope.issubset(cache_signature.parameters):
                cached_answer = get_cached_answer(
                    sanitized_query,
                    organization_id=_current_organization_id(),
                    namespace=_safe_namespace(route_metadata["route"]),
                    user_id=username,
                    role=user_role,
                )
            else:
                # Never fall back to a legacy unscoped cache in a multi-tenant app.
                logger.warning("Cache API lacks tenant scope; cache read disabled.")
                cached_answer = None
        except Exception:
            logger.debug("Cache read failed", exc_info=True)
            cached_answer = None

        if cached_answer:
            answer = _extract_answer(cached_answer)
            if answer and not _is_generic_no_context_answer(answer):
                latency_ms = int((time.perf_counter() - started) * 1000)
                metadata = {
                    "request_id": request_id,
                    "cache_hit": True,
                    "latency_ms": latency_ms,
                    "agent": "semantic-cache",
                    "model_used": "cache",
                }
                _audit_log(
                    username,
                    "QUERY_CACHE_HIT",
                    f"Request: {request_id}",
                    query_preview=sanitized_query,
                    model_used="cache",
                )
                return answer, metadata

    context = ""
    persistent_memory = _chat_memory_context(
        username,
        int(st.session_state.get("active_chat_session_id") or 0),
        max_messages=12,
    )

    if MEMORY_AVAILABLE:
        try:
            context = build_query_context(
                st.session_state.get("messages", []),
                sanitized_query,
            )
        except TypeError:
            try:
                context = build_query_context(
                    st.session_state.get("messages", [])
                )
            except Exception:
                context = ""
        except Exception:
            logger.debug("Memory context unavailable", exc_info=True)

    if persistent_memory:
        context = (
            f"{context}\n\nPersistent conversation memory:\n{persistent_memory}"
            if context
            else f"Persistent conversation memory:\n{persistent_memory}"
        )

    metadata: Dict[str, Any] = {
        **metadata_route,
        "route": route_metadata["route"],
        "route_reason": route_metadata["reason"],
    }
    retrieved_chunks: List[Dict[str, Any]] = []

    # IMPORTANT: retrieve BEFORE Graph. The Graph retriever node expects
    # retrieved_chunks to be supplied by the caller; previously app.py called
    # Graph without them, causing the graph to produce "no policy context".
    try:
        retrieved_chunks, retrieval_metadata = _retrieve_policy_context(
            sanitized_query,
            top_k=max(6, int(getattr(settings, "TOP_K", 5) or 5)),
            organization_id=_current_organization_id(),
            namespace=_safe_namespace(route_metadata["route"]),
        )
        metadata.update(retrieval_metadata)
        logger.info(
            "Retrieved %d policy chunks for request %s",
            len(retrieved_chunks),
            request_id,
        )
    except Exception as retrieval_exc:
        logger.warning(
            "Primary policy retrieval failed; continuing to RAG fallback: %s",
            retrieval_exc,
            exc_info=True,
        )

    result: Any = None

    # Graph is the preferred OpenRouter generation path. Pass the chunks in
    # the exact API shape expected by process_query_via_graph().
    if GRAPH_AVAILABLE and _graph_process_query is not None:
        try:
            result = _graph_process_query(
                sanitized_query,
                retrieved_chunks=retrieved_chunks,
                user_context={
                    "username": username,
                    "user_role": user_role,
                    "retrieval_strategy": {},
                    "conversation_memory": context,
                    "route": route_metadata["route"],
                    "response_style": "professional_detailed",
                    "organization_id": _current_organization_id(),
                    "namespace": (
                        "talent"
                        if route_metadata["route"] == "Talent Intelligence"
                        else "policy"
                    ),
                },
            )
            graph_metadata = _extract_metadata(result)
            metadata.update(graph_metadata)
        except Exception as graph_exc:
            logger.warning(
                "Graph processing failed; attempting RAG engine: %s",
                graph_exc,
                exc_info=True,
            )

    answer = _extract_answer(result)
    if _is_generic_no_context_answer(answer):
        answer = ""

    # If Graph was unavailable or returned no usable answer, use the RAG
    # engine. It has its own vector-store retrieval path and can synthesize
    # directly from the indexed chunks.
    if not answer and RAG_AVAILABLE:
        try:
            engine = st.session_state.get("rag_engine")
            if engine is None and get_rag_engine is not None:
                engine = get_rag_engine()

            if engine is not None and hasattr(engine, "set_vector_store"):
                try:
                    engine.set_vector_store(get_vector_store())
                except Exception:
                    logger.debug("Could not attach vector store to RAG engine", exc_info=True)

            if engine is not None and hasattr(engine, "set_embedder") and get_embedder is not None:
                try:
                    engine.set_embedder(get_embedder())
                except Exception:
                    logger.debug("Could not attach embedder to RAG engine", exc_info=True)

            if engine is not None and hasattr(engine, "query"):
                query_kwargs = {
                    "top_k": max(6, int(getattr(settings, "TOP_K", 5) or 5)),
                    "user_id": username,
                    "use_cache": False,
                    "organization_id": _current_organization_id(),
                    "namespace": _safe_namespace(route_metadata["route"]),
                }
                try:
                    query_signature = inspect.signature(engine.query)
                    query_kwargs = {
                        key: value
                        for key, value in query_kwargs.items()
                        if key in query_signature.parameters
                    }
                    if "organization_id" not in query_signature.parameters:
                        raise RuntimeError(
                            "RAG engine query API lacks explicit organization scope."
                        )
                    result = engine.query(sanitized_query, **query_kwargs)
                except (TypeError, ValueError) as scope_exc:
                    raise RuntimeError(
                        "RAG engine scope validation could not be established."
                    ) from scope_exc
                metadata.update(_extract_metadata(result))
                answer = _extract_answer(result)
                if _is_generic_no_context_answer(answer):
                    answer = ""
            elif rag_query is not None:
                rag_signature = inspect.signature(rag_query)
                if "organization_id" not in rag_signature.parameters:
                    raise RuntimeError(
                        "RAG query API lacks explicit organization scope."
                    )
                result = rag_query(
                    sanitized_query,
                    username=username,
                    user_role=user_role,
                    organization_id=_current_organization_id(),
                    namespace=_safe_namespace(route_metadata["route"]),
                )
                metadata.update(_extract_metadata(result))
                answer = _extract_answer(result)
                if _is_generic_no_context_answer(answer):
                    answer = ""
        except Exception:
            logger.exception("RAG processing failed")

    # Last-resort grounded answer: never invent policy. Return retrieved
    # excerpts with source attribution so the product remains useful even if
    # the OpenRouter call fails.
    if not answer and retrieved_chunks:
        parts = [
            "Based on the indexed HR policy documents, the following information is relevant to your question:",
            "",
        ]
        for index, chunk in enumerate(retrieved_chunks[:5], start=1):
            content = _clean_text(chunk.get("content", ""), 1800)
            meta = chunk.get("metadata") or {}
            source = (
                meta.get("source")
                or meta.get("filename")
                or meta.get("file_name")
                or "Indexed HR policy document"
            )
            page = meta.get("page") or meta.get("page_number")
            label = f"{source}, page {page}" if page else str(source)
            parts.append(f"**Source {index}: {label}**")
            parts.append(content)
            parts.append("")
        parts.append(
            "This response is grounded only in the indexed document content. "
            "For a definitive HR interpretation or policy decision, please refer "
            "to the source document or contact HR."
        )
        answer = "\n".join(parts).strip()
        metadata["agent"] = "grounded-document-fallback"

    if not answer:
        answer = (
            "I could not find sufficient information in the indexed HR policy "
            "documents to answer that question reliably. Please try a more "
            "specific question or contact HR for clarification."
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    metadata.update({
        "request_id": request_id,
        "cache_hit": False,
        "latency_ms": latency_ms,
        "chunks_retrieved": len(retrieved_chunks),
    })

    if CACHE_AVAILABLE and answer and not answer.startswith("I could not find sufficient information"):
        try:
            cache_signature = inspect.signature(cache_answer)
            required_scope = {"organization_id", "namespace", "user_id", "role"}
            if required_scope.issubset(cache_signature.parameters):
                cache_answer(
                    sanitized_query,
                    answer,
                    organization_id=_current_organization_id(),
                    namespace=_safe_namespace(route_metadata["route"]),
                    user_id=username,
                    role=user_role,
                )
            else:
                logger.warning("Cache API lacks tenant scope; cache write disabled.")
        except Exception:
            logger.debug("Cache write failed", exc_info=True)

    _audit_log(
        username,
        "QUERY_EXECUTED",
        f"Request: {request_id}",
        tokens_used=int(metadata.get("tokens_used") or 0),
        cost_usd=float(metadata.get("cost_usd") or 0.0),
        model_used=metadata.get("model_used"),
        query_preview=sanitized_query,
    )

    return answer, metadata


# =============================================================================
# DOCUMENT INGESTION
# =============================================================================


def _persist_uploaded_file(
    uploaded_file: Any,
    organization_id: Optional[str] = None,
) -> Path:
    """
    Persist an uploaded Streamlit file safely.

    Files are stored under data/uploads/<UUID>_<safe_name>.
    """
    original_name = _safe_filename(uploaded_file.name)

    suffix = Path(original_name).suffix.lower()

    if suffix.lstrip(".") not in ALLOWED_UPLOAD_TYPES:
        raise ValueError(
            f"Unsupported file type: {suffix or 'unknown'}"
        )

    max_mb = int(
        getattr(
            settings,
            "MAX_UPLOAD_SIZE_MB",
            os.getenv("MAX_UPLOAD_SIZE_MB", "25"),
        )
        or 25
    )

    max_bytes = max(1, max_mb) * 1024 * 1024

    data = uploaded_file.getvalue()

    if len(data) > max_bytes:
        raise ValueError(
            f"{original_name} exceeds the {max_mb} MB upload limit."
        )

    organization_id = organization_id or _current_organization_id()
    if not ORGANIZATION_ID_PATTERN.fullmatch(organization_id):
        raise ValueError("Invalid organization scope.")

    tenant_upload_dir = UPLOAD_DIR / organization_id
    tenant_upload_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    destination = tenant_upload_dir / unique_name

    destination.write_bytes(data)

    return destination


def _record_document(
    filename: str,
    filepath: Path,
    uploaded_by: str,
    file_size: int,
    chunk_count: int,
    ocr_used: bool,
) -> None:
    with closing(_db_connect()) as conn:
        conn.execute(
            """
            INSERT INTO documents (
                filename,
                organization_id,
                filepath,
                uploaded_by,
                file_size,
                chunk_count,
                status,
                ocr_used
            )
            VALUES (?, ?, ?, ?, ?, 'active', ?)
            """,
            (
                filename,
                _current_organization_id(),
                str(filepath),
                uploaded_by,
                int(file_size),
                int(chunk_count),
                1 if ocr_used else 0,
            ),
        )
        conn.commit()


def _refresh_vector_runtime() -> None:
    """
    Clear Streamlit's cached vector-store resource and rebuild the session state.
    """
    try:
        _load_vector_store_cached.clear()
    except Exception:
        pass

    st.session_state.vector_store_ready = False
    st.session_state.rag_engine = None

    _load_data()


def _index_uploaded_file(
    uploaded_file: Any,
    username: str,
) -> Dict[str, Any]:
    """
    Parse, embed and persist one uploaded document.

    The function is intentionally transactional at the application level:
    a document DB record is created only after parsing/indexing succeeds.
    """
    if not PARSER_AVAILABLE or get_multimodal_parser is None:
        raise RuntimeError("Document parser is unavailable.")

    if not VECTOR_STORE_AVAILABLE or get_vector_store is None:
        raise RuntimeError("Vector store is unavailable.")

    destination = _persist_uploaded_file(uploaded_file, _current_organization_id())

    try:
        parser = get_multimodal_parser()
        result = parser.parse_file(str(destination))

        if not isinstance(result, dict):
            raise RuntimeError("Parser returned an invalid result.")

        chunks = result.get("chunks") or []

        if not isinstance(chunks, list):
            raise RuntimeError("Parser chunks must be a list.")

        valid_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk, dict)
            and _clean_text(chunk.get("content"))
        ]

        if not valid_chunks:
            raise RuntimeError(
                "No usable text was extracted from this document."
            )

        file_bytes = uploaded_file.getvalue()
        document_hash = hashlib.sha256(file_bytes).hexdigest()

        # Parser chunk IDs are document-local (often 0, 1, 2, ...). FAISS
        # metadata is global, so namespace IDs by document content hash.
        # This prevents the exact "Chunk ID already exists: 0" failure when
        # a second document is indexed.
        normalized_chunks = []
        for index, chunk in enumerate(valid_chunks):
            chunk_copy = dict(chunk)
            original_id = chunk_copy.get("chunk_id", index)
            metadata = chunk_copy.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata = dict(metadata)
            metadata.setdefault("source", _safe_filename(uploaded_file.name))
            metadata["organization_id"] = _current_organization_id()
            metadata["namespace"] = "policy"
            metadata["document_hash"] = document_hash
            metadata["document_id"] = document_hash[:16]
            metadata.setdefault("chunk_index", index)
            chunk_copy["metadata"] = metadata
            chunk_copy["chunk_id"] = f"doc_{document_hash[:24]}_chunk_{index}_{str(original_id)}"
            normalized_chunks.append(chunk_copy)

        valid_chunks = normalized_chunks

        texts = [
            _clean_text(chunk.get("content"), 100000)
            for chunk in valid_chunks
        ]

        # -------------------------------------------------------------
        # Embeddings
        # -------------------------------------------------------------

        if not EMBEDDER_AVAILABLE or get_embedder is None:
            raise RuntimeError(
                "Embedding subsystem is unavailable. "
                "Check src/core/embedder_singleton.py and "
                "sentence-transformers installation."
            )

        try:
            logger.info(
                "Initializing embedding model for %s",
                uploaded_file.name,
            )

            runtime_embedder = get_embedder()

            if runtime_embedder is None:
                raise RuntimeError(
                    "get_embedder() returned None."
                )

            logger.info(
                "Embedding model ready: model=%s device=%s dimension=%s",
                runtime_embedder.model_name,
                runtime_embedder.device,
                runtime_embedder.dimension,
            )

            if not texts:
                raise RuntimeError(
                    "No text chunks are available for embedding."
                )

            logger.info(
                "Generating embeddings for %d chunks",
                len(texts),
            )

            embeddings = runtime_embedder.encode_documents(
                texts,
                batch_size=runtime_embedder.batch_size,
                show_progress_bar=False,
            )

            if embeddings is None:
                raise RuntimeError(
                    "The embedding model returned None."
                )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            if embeddings.ndim != 2:
                raise RuntimeError(
                    "Invalid embedding shape: "
                    f"{embeddings.shape}"
                )

            if embeddings.shape[0] != len(texts):
                raise RuntimeError(
                    "Embedding count does not match chunk count: "
                    f"chunks={len(texts)}, "
                    f"embeddings={embeddings.shape[0]}"
                )

            if embeddings.shape[1] != runtime_embedder.dimension:
                raise RuntimeError(
                    "Embedding dimension mismatch: "
                    f"model={runtime_embedder.dimension}, "
                    f"generated={embeddings.shape[1]}"
                )

            if not np.isfinite(embeddings).all():
                raise RuntimeError(
                    "Embedding matrix contains NaN or infinite values."
                )

            logger.info(
                "Embeddings generated successfully: shape=%s",
                embeddings.shape,
            )

        except Exception as embedding_exc:
            logger.exception(
                "Embedding generation failed for %s: %s",
                uploaded_file.name,
                embedding_exc,
            )

            raise RuntimeError(
                f"Could not generate embeddings for "
                f"{uploaded_file.name}: {embedding_exc}"
            ) from embedding_exc

        # -------------------------------------------------------------
        # Vector store
        # -------------------------------------------------------------

        vector_store = get_vector_store()

        if vector_store is None:
            raise RuntimeError("get_vector_store() returned None.")

        if getattr(vector_store, "embedder", None) is None:
            setter = getattr(vector_store, "set_embedder", None)
            if callable(setter):
                setter(runtime_embedder)
            else:
                vector_store.embedder = runtime_embedder

        # Idempotency: if this exact document was already indexed, do not
        # append duplicate FAISS vectors.
        existing_chunks = getattr(vector_store, "chunks", [])
        existing_ids = {
            str(chunk.get("chunk_id"))
            for chunk in existing_chunks
            if isinstance(chunk, dict) and chunk.get("chunk_id") is not None
        }

        new_chunks = []
        new_embeddings = []
        new_texts = []
        skipped = 0

        for chunk, embedding, text in zip(valid_chunks, embeddings, texts):
            if str(chunk.get("chunk_id")) in existing_ids:
                skipped += 1
                continue
            new_chunks.append(chunk)
            new_embeddings.append(embedding)
            new_texts.append(text)

        if new_chunks:
            vector_store.add_chunks(
                new_chunks,
                np.asarray(new_embeddings, dtype=np.float32),
                texts=new_texts,
            )
        elif skipped == len(valid_chunks):
            logger.info(
                "Document already indexed; skipped %d existing chunks for %s.",
                skipped,
                uploaded_file.name,
            )
        else:
            raise RuntimeError("No new chunks were available for indexing.")

        if not hasattr(vector_store, "save"):
            raise RuntimeError(
                "Vector store does not expose save()."
            )

        vector_store.save()

        _record_document(
            filename=_safe_filename(uploaded_file.name),
            filepath=destination,
            uploaded_by=username,
            file_size=int(uploaded_file.size or len(uploaded_file.getvalue())),
            chunk_count=len(valid_chunks),
            ocr_used=bool(result.get("ocr_used", False)),
        )

        _audit_log(
            username,
            "DOCUMENT_UPLOADED",
            (
                f"File: {_safe_filename(uploaded_file.name)}; "
                f"chunks: {len(valid_chunks)}; "
                f"ocr: {bool(result.get('ocr_used', False))}"
            ),
        )

        return {
            "filename": _safe_filename(uploaded_file.name),
            "chunks": len(valid_chunks),
            "ocr_used": bool(result.get("ocr_used", False)),
            "embedding": embeddings is not None,
        }

    except Exception:
        # Do not leave an unusable partially written upload around.
        try:
            destination.unlink(missing_ok=True)
        except Exception:
            pass

        raise


# =============================================================================
# AUTHENTICATION PAGE
# =============================================================================


def _show_auth_page() -> None:
    """Render a polished authentication experience."""
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">PolicyGuard AI</div>
            <div class="hero-sub">
                Enterprise HR intelligence with secure retrieval,
                role-based access control, document grounding,
                auditability and AI-assisted policy discovery.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    cards = [
        (
            "🔐",
            "Enterprise Security",
            "RBAC, security validation, brute-force protection and audit trails.",
        ),
        (
            "🧠",
            "Grounded AI",
            "Hybrid retrieval, reranking and RAG orchestration for document-based answers.",
        ),
        (
            "📈",
            "Operational Visibility",
            "Query performance, document health, system components and audit events.",
        ),
    ]

    for column, (icon, title, description) in zip(
        (c1, c2, c3),
        cards,
    ):
        with column:
            st.markdown(
                f"""
                <div class="pg-card">
                    <div class="pg-card-icon">{icon}</div>
                    <div class="pg-card-title">{_escape(title)}</div>
                    <div class="pg-card-text">{_escape(description)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    login_tab, register_tab = st.tabs(
        ["🔑 Sign In", "✨ Create Account"]
    )

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                key="login_username",
                placeholder="Your username",
            )

            password = st.text_input(
                "Password",
                type="password",
                key="login_password",
                placeholder="Your password",
            )

            submitted = st.form_submit_button(
                "Sign In",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            with st.spinner("Authenticating securely…"):
                success, message, role, user_id = _login_user(
                    username,
                    password,
                )

            if success:
                st.session_state.authenticated = True
                st.session_state.username = _safe_username(username)
                st.session_state.user_role = _safe_role(role)
                st.session_state.user_id = user_id

                organization_id = _configured_organization_id()
                try:
                    with closing(_db_connect()) as conn:
                        user_row = conn.execute(
                            "SELECT organization_id FROM users WHERE id = ? AND username = ?",
                            (int(user_id), _safe_username(username)),
                        ).fetchone()
                    if user_row and user_row["organization_id"]:
                        organization_id = _clean_text(
                            user_row["organization_id"], MAX_ORGANIZATION_ID_LENGTH
                        )
                except Exception:
                    logger.exception("Could not resolve authenticated organization.")

                if not ORGANIZATION_ID_PATTERN.fullmatch(organization_id):
                    st.error("Your account has an invalid organization scope. Contact an administrator.")
                    return

                st.session_state.organization_id = organization_id
                st.session_state.view = "home"
                st.session_state.show_onboarding = True
                st.session_state.first_login = True
                st.session_state.messages = []
                st.session_state.query_history = []

                st.success(message)
                time.sleep(0.4)
                st.rerun()
            else:
                st.error(message)

    with register_tab:
        with st.form("register_form", clear_on_submit=True):
            new_username = st.text_input(
                "Username",
                key="register_username",
                placeholder="3-64 characters",
            )

            new_password = st.text_input(
                "Password",
                type="password",
                key="register_password",
                placeholder="At least 12 characters",
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                key="register_password_confirm",
                placeholder="Repeat your password",
            )

            submitted = st.form_submit_button(
                "Create Account",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                with st.spinner("Creating your account…"):
                    success, message = _register_user(
                        new_username,
                        new_password,
                    )

                if success:
                    st.success(message)
                    st.info(
                        "New accounts start with Viewer permissions. "
                        "An administrator can promote the account when appropriate."
                    )
                else:
                    st.error(message)

    st.caption(
        f"{APP_NAME} v{APP_VERSION} · Enterprise HR Intelligence"
    )


# =============================================================================
# SIDEBAR
# =============================================================================


def _show_user_profile(
    username: str,
    role: str,
) -> None:
    initial = _escape(username[:1].upper() or "U")

    st.markdown(
        f"""
        <div style="
            padding:1rem;
            border:1px solid rgba(148,163,184,.15);
            border-radius:18px;
            background:rgba(15,23,42,.62);
            margin-bottom:1rem;
        ">
            <div style="display:flex;align-items:center;gap:.75rem;">
                <div style="
                    width:46px;
                    height:46px;
                    border-radius:50%;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-weight:800;
                    color:white;
                    background:linear-gradient(135deg,#60a5fa,#a78bfa);
                ">
                    {initial}
                </div>
                <div>
                    <div style="font-weight:750;color:#f8fafc;">
                        {_escape(username)}
                    </div>
                    <div class="small-muted role-{_escape(role)}">
                        {_escape(role.upper())}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_upload_control(username: str) -> None:
    if not _check_permission("editor"):
        return

    st.subheader("Knowledge Base", divider=True)

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=sorted(ALLOWED_UPLOAD_TYPES),
        accept_multiple_files=True,
        key="document_uploader",
        help=(
            "Supported: PDF, DOCX, XLSX, TXT and Markdown. "
            "Maximum file size is controlled by MAX_UPLOAD_SIZE_MB."
        ),
    )

    if not uploaded_files:
        return

    processed = st.session_state.get("last_uploaded", [])

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        fingerprint = (
            f"{uploaded_file.name}:"
            f"{uploaded_file.size}:"
            f"{hashlib.sha256(file_bytes).hexdigest()}"
        )

        if fingerprint in processed:
            continue

        with st.status(
            f"Indexing {uploaded_file.name}…",
            expanded=False,
        ) as status:
            try:
                result = _index_uploaded_file(
                    uploaded_file,
                    username,
                )

                processed.append(fingerprint)
                st.session_state.last_uploaded = processed

                status.update(
                    label=(
                        f"Indexed {result['filename']} "
                        f"({result['chunks']} chunks)"
                    ),
                    state="complete",
                )

            except Exception as exc:
                logger.exception(
                    "Document ingestion failed for %s",
                    uploaded_file.name,
                )
                status.update(
                    label=f"Failed: {uploaded_file.name}",
                    state="error",
                )
                st.error(
                    f"Could not index `{uploaded_file.name}`: "
                    f"{_clean_text(exc, 250)}"
                )

    if st.session_state.get("last_uploaded"):
        if st.button(
            "Refresh Knowledge Base",
            use_container_width=True,
        ):
            _refresh_vector_runtime()
            st.rerun()


def _logout() -> None:
    username = st.session_state.get("username") or "unknown"

    _audit_log(
        username,
        "LOGOUT",
        "User signed out.",
    )

    for key, value in {
        "authenticated": False,
        "username": None,
        "user_role": None,
        "user_id": None,
        "organization_id": None,
        "view": "home",
        "messages": [],
        "query_history": [],
        "last_log": {},
        "last_error": None,
    }.items():
        st.session_state[key] = value

    st.rerun()


def _render_sidebar() -> None:
    username = st.session_state.username or "User"
    role = _safe_role(st.session_state.user_role)

    with st.sidebar:
        _show_user_profile(username, role)
        _show_status_indicator()

        st.divider()

        st.subheader("Navigation", divider=True)

        navigation = [
            ("⌂  Overview", "home"),
            ("💬  Live Chat", "chat"),
            ("📊  Dashboard", "dashboard"),
            ("📚  Documents", "documents"),
        ]

        if _check_permission("editor"):
            navigation.append(("🎯  Talent Intelligence", "talent"))

        navigation.append(("🏗️  Architecture", "architecture"))

        if _check_permission("admin"):
            navigation.extend(
                [
                    ("🛡️  Audit Logs", "audit"),
                    ("👥  User Management", "users"),
                ]
            )

        labels = [label for label, _ in navigation]
        current_view = st.session_state.get("view", "home")

        current_index = next(
            (
                i
                for i, (_, key) in enumerate(navigation)
                if key == current_view
            ),
            0,
        )

        selected = st.radio(
            "Navigation",
            labels,
            index=current_index,
            label_visibility="collapsed",
        )

        selected_key = next(
            key
            for label, key in navigation
            if label == selected
        )

        st.session_state.view = selected_key

        st.divider()

        _show_upload_control(username)

        st.divider()

        with st.expander("System Health"):
            components = _system_components()

            for name, available in components.items():
                if available:
                    st.success(name)
                    # st.success(name)
                else:
                    st.warning(name)

        with st.expander("Help"):
            st.markdown(
                """
                **Viewer**
                - Ask policy questions
                - View permitted analytics

                **Editor (HR)**
                - Everything a Viewer can do
                - Upload and index policy documents
                - Maintain an authorized internal talent pool
                - Run JD-to-resume talent matching

                **Admin (HR Lead)**
                - Everything above
                - Manage users and roles
                - View enterprise audit logs
                - See company-wide talent intelligence

                **Tip:** Answers should be treated as
                document-grounded assistance, not a replacement
                for official HR/legal decisions.
                """
            )

        st.divider()

        if st.button(
            "Sign Out",
            use_container_width=True,
        ):
            _logout()


# =============================================================================
# ONBOARDING
# =============================================================================


def _show_onboarding() -> None:
    if not st.session_state.get("show_onboarding"):
        return

    role = _safe_role(
        st.session_state.get("user_role", "viewer")
    )

    with st.container(border=True):
        st.markdown("### 👋 Welcome to PolicyGuard AI")

        st.write(
            "Your workspace is ready. Start with the Live Chat to ask "
            "questions about indexed HR policies."
        )

        if role in ("editor", "admin"):
            st.info(
                "You can upload policy documents from the sidebar. "
                "Once indexed, they become available to the retrieval pipeline."
            )
        else:
            st.info(
                "You have Viewer access. Ask an administrator to upload "
                "or update the knowledge base."
            )

        if st.button(
            "Got it",
            type="primary",
        ):
            st.session_state.show_onboarding = False
            st.rerun()


# =============================================================================
# OVERVIEW
# =============================================================================


def _render_home_view(user_role: str) -> None:
    stats = _get_document_stats()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">PolicyGuard AI</div>
            <div class="hero-sub">
                A secure enterprise HR knowledge platform for
                policy discovery, document-grounded Q&A,
                retrieval intelligence and operational governance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    features = [
        (
            c1,
            "🧠",
            "Grounded Intelligence",
            "RAG + hybrid retrieval + reranking for evidence-oriented answers.",
        ),
        (
            c2,
            "🛡️",
            "Security First",
            "RBAC, query inspection, PII-aware auditing and login protection.",
        ),
        (
            c3,
            "📚",
            "Knowledge Base",
            f"{stats['documents']} documents · {stats['chunks']} indexed chunks.",
        ),
        (
            c4,
            "📈",
            "Observable",
            "Audit events, latency, cache behavior and system health.",
        ),
    ]

    for column, icon, title, description in features:
        with column:
            st.markdown(
                f"""
                <div class="pg-card">
                    <div class="pg-card-icon">{icon}</div>
                    <div class="pg-card-title">{_escape(title)}</div>
                    <div class="pg-card-text">{_escape(description)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.subheader("Workspace Status")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Documents", stats["documents"])

    with m2:
        st.metric("Indexed Chunks", stats["chunks"])

    with m3:
        st.metric(
            "OCR Documents",
            stats["ocr_documents"],
        )

    with m4:
        st.metric(
            "Your Role",
            _safe_role(user_role).upper(),
        )

    st.divider()

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("How the platform works")

        st.markdown(
            """
            **01 · Ingest**  
            HR policies and reference documents are parsed and chunked.

            **02 · Protect**  
            Queries pass through security and access-control checks.

            **03 · Retrieve**  
            Semantic and lexical retrieval identify relevant evidence.

            **04 · Rerank**  
            A cross-encoder can refine the most relevant passages.

            **05 · Answer**  
            The RAG/orchestration layer produces a grounded response.

            **06 · Audit**  
            Important operations are recorded for operational governance.
            """
        )

    with right:
        st.subheader("Recommended workflow")

        if stats["documents"] == 0:
            st.warning(
                "Your knowledge base is empty.",
            )

            if _check_permission("editor"):
                st.write(
                    "Upload your HR policy documents using the sidebar."
                )
            else:
                st.write(
                    "Ask an Editor or Admin to upload the required documents."
                )
        else:
            st.success(
                "Knowledge base is populated.",
            )

            st.write(
                "Open Live Chat and ask a policy question using natural language. "
                "Your indexed knowledge base and previous conversation memory will "
                "be restored automatically."
            )

            if _check_permission("editor"):
                st.info(
                    "HR users can also open Talent Intelligence to compare authorized "
                    "bench profiles against a Job Description."
                )

            if st.button(
                "Open Live Chat",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.view = "chat"
                st.rerun()


# =============================================================================
# LIVE CHAT
# =============================================================================


def _render_chat_view(
    username: str,
    user_role: str,
) -> None:
    _require_login()

    # Hydrate durable conversation memory every time Live Chat is opened.
    # The persisted vector index is loaded independently, so page refreshes do
    # not require a manual "Refresh Knowledge Base" action.
    _load_data()
    _initialize_persistent_memory(username)

    st.title("💬 Live Chat")
    st.caption(
        "Ask questions about your indexed HR policies and reference documents."
    )

    # Session actions
    top1, top2, top3 = st.columns([1, 1, 4])

    with top1:
        if st.button(
            "New Chat",
            use_container_width=True,
        ):
            _start_new_persistent_chat(username)
            st.rerun()

    with top2:
        st.caption(
            f"{len(st.session_state.messages)} messages"
        )

    if not st.session_state.vector_store_ready:
        st.warning(
            "The knowledge base is not currently loaded. "
            "Document-grounded answers may be unavailable."
        )

    # Existing conversation
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        content = _clean_text(
            message.get("content", ""),
            20000,
        )

        with st.chat_message(
            role,
            avatar="🧑" if role == "user" else "🛡️",
        ):
            if role == "user":
                st.markdown(
                    f'<div class="chat-question-card">'
                    f'<div class="chat-question-label">Your question</div>'
                    f'{html.escape(content).replace(chr(10), "<br>")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(content)

            metadata = message.get("metadata") or {}

            if metadata:
                latency = metadata.get("latency_ms")

                if latency is not None:
                    st.markdown(
                        f'<div class="chat-meta">'
                        f"Request {html.escape(str(metadata.get('request_id', '—'))[:12])}"
                        f" · {html.escape(str(latency))} ms"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                if role == "assistant":
                    _render_route_trace(metadata)

    query = st.chat_input(
        "Ask about an HR policy, benefit, procedure or rule…",
    )

    if not query:
        return

    query = _clean_text(query, MAX_QUERY_LENGTH)

    # Immediately render user's message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    _persist_chat_message(
        username,
        int(st.session_state.get("active_chat_session_id") or 0),
        "user",
        query,
    )

    with st.chat_message(
        "user",
        avatar="🧑",
    ):
        st.markdown(
            f'<div class="chat-question-card">'
            f'<div class="chat-question-label">Your question</div>'
            f'{html.escape(query).replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.chat_message(
        "assistant",
        avatar="🛡️",
    ):
        with st.spinner("Searching policy knowledge and generating an answer…"):
            started = time.perf_counter()

            try:
                answer, metadata = _process_query(
                    query,
                    username,
                    user_role,
                )

                elapsed = int(
                    (time.perf_counter() - started) * 1000
                )

                metadata.setdefault(
                    "latency_ms",
                    elapsed,
                )

                _render_professional_answer(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata,
                    }
                )

                _persist_chat_message(
                    username,
                    int(st.session_state.get("active_chat_session_id") or 0),
                    "assistant",
                    answer,
                    metadata,
                )

                _render_route_trace(metadata)

                history_entry = {
                    "timestamp": _now_iso(),
                    "query": query,
                    "agent": metadata.get(
                        "agent",
                        metadata.get(
                            "router_decision",
                            "RAG",
                        ),
                    ),
                    "latency_ms": metadata.get(
                        "latency_ms",
                        elapsed,
                    ),
                    "cache_hit": bool(
                        metadata.get("cache_hit", False)
                    ),
                    "model_used": metadata.get(
                        "model_used",
                        "",
                    ),
                    "request_id": metadata.get(
                        "request_id",
                        "",
                    ),
                }

                st.session_state.query_history.append(
                    history_entry
                )

                st.session_state.last_log = metadata

            except PermissionError as exc:
                message = _clean_text(exc, 1000)

                st.error(
                    f"Security policy blocked this request: {message}"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I can't process that request because it "
                            "violates the current security policy."
                        ),
                    }
                )

            except ValueError as exc:
                message = _clean_text(exc, 1000)
                st.warning(message)

            except Exception:
                logger.exception("Chat request failed")

                st.error(
                    "Something went wrong while processing the request. "
                    "Please try again. If the problem persists, contact an administrator."
                )

                st.session_state.last_error = (
                    "Chat request failed."
                )

    st.caption(
        f"Session queries: {len(st.session_state.query_history)}"
        f" · System: {st.session_state.system_status.upper()}"
    )



# =============================================================================
# TALENT INTELLIGENCE VIEW
# =============================================================================


def _render_talent_view(
    username: str,
    user_role: str,
) -> None:
    """HR talent matching workspace for Editors and Administrators."""
    _require_login()

    if not _check_permission("editor"):
        st.error("Editor or Administrator permissions are required.")
        return

    st.title("🎯 Talent Intelligence")
    st.caption(
        "Internal talent discovery for HR: upload bench resumes, provide a job "
        "description, and rank candidates by explainable semantic + skill fit."
    )

    st.info(
        "This feature is decision support only. It does not make hiring, "
        "promotion, compensation or termination decisions. HR remains responsible "
        "for reviewing candidates against objective, job-relevant criteria."
    )

    tab_rank, tab_people, tab_upload = st.tabs(
        ["🔎 Match Talent", "👥 Bench Pool", "📄 Add Resumes"]
    )

    with tab_upload:
        st.subheader("Build the internal talent pool")
        uploaded_resumes = st.file_uploader(
            "Upload employee / candidate resumes",
            type=sorted(ALLOWED_UPLOAD_TYPES),
            accept_multiple_files=True,
            key="talent_resume_uploader",
            help="Use only resumes you are authorized to process.",
        )

        if uploaded_resumes:
            for resume in uploaded_resumes:
                fingerprint = (
                    f"{resume.name}:{resume.size}:"
                    f"{hashlib.sha256(resume.getvalue()).hexdigest()}"
                )
                processed = st.session_state.setdefault(
                    "processed_talent_uploads", []
                )

                if fingerprint in processed:
                    continue

                with st.status(f"Indexing {resume.name}…", expanded=False) as status:
                    try:
                        result = _index_candidate_resume(resume, username)
                        processed.append(fingerprint)
                        status.update(
                            label=f"Added {result['candidate_name']}",
                            state="complete",
                        )
                    except Exception as exc:
                        logger.exception("Talent resume ingestion failed")
                        status.update(
                            label=f"Failed: {resume.name}",
                            state="error",
                        )
                        st.error(
                            f"Could not process `{resume.name}`: "
                            f"{_clean_text(exc, 500)}"
                        )

    candidates = _get_candidates(
        username,
        include_all=_check_permission("admin"),
    )

    with tab_people:
        st.subheader("Internal / bench pool")

        if not candidates:
            st.info(
                "No resumes are in the talent pool yet. Use “Add Resumes” to "
                "upload authorized employee or candidate profiles."
            )
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Profiles", len(candidates))
            with c2:
                st.metric(
                    "Bench",
                    sum(1 for c in candidates if c.get("status") == "bench"),
                )
            with c3:
                st.metric(
                    "Visible Scope",
                    "Company-wide" if _check_permission("admin") else "Your uploads",
                )

            display = pd.DataFrame(candidates)
            if not display.empty:
                st.dataframe(
                    display[
                        [
                            c for c in (
                                "candidate_name",
                                "resume_filename",
                                "uploaded_by",
                                "status",
                                "updated_at",
                            )
                            if c in display.columns
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_rank:
        st.subheader("Find the best internal fit")

        default_title = st.session_state.get("talent_jd_title", "")
        default_jd = st.session_state.get("talent_jd_text", "")

        jd_file = st.file_uploader(
            "Optional: upload a Job Description",
            type=sorted(ALLOWED_UPLOAD_TYPES),
            key="talent_jd_uploader",
        )

        st.caption(
            "You can either paste the JD below or upload a PDF/DOCX/TXT document. "
            "An uploaded JD is extracted automatically and placed into the same matching workflow."
        )

        if jd_file is not None:
            try:
                parser = get_multimodal_parser()
                jd_destination = _persist_uploaded_file(jd_file)
                try:
                    parsed = parser.parse_file(str(jd_destination))
                finally:
                    try:
                        jd_destination.unlink(missing_ok=True)
                    except Exception:
                        logger.debug("Could not remove temporary JD file", exc_info=True)
                jd_chunks = parsed.get("chunks") if isinstance(parsed, dict) else []
                jd_text = "\n\n".join(
                    _clean_text(c.get("content", ""), 20000)
                    for c in (jd_chunks or [])
                    if isinstance(c, dict)
                ).strip()
                if jd_text:
                    validated_jd = _validate_talent_input(
                        jd_text,
                        username,
                        user_role,
                        field_name="Job Description",
                        max_length=12000,
                    )
                    default_jd = validated_jd
                    if not default_title:
                        default_title = _extract_candidate_name(
                            jd_file.name,
                            validated_jd,
                        )
                    st.session_state.talent_jd_text = validated_jd
                    st.session_state.talent_jd_title = _clean_text(default_title, 200)
            except Exception as exc:
                st.warning(f"Could not read the Job Description: {_clean_text(exc, 500)}")

        job_title = st.text_input(
            "Role / Job Title",
            value=default_title,
            placeholder="e.g. AI Engineer",
            key="talent_job_title",
        )

        job_description = st.text_area(
            "Job Description / Requirements",
            value=default_jd,
            height=220,
            placeholder=(
                "Paste the JD, required skills, experience, responsibilities "
                "and other objective job-related criteria here."
            ),
            key="talent_job_description",
        )

        if st.button(
            "Rank Candidates",
            type="primary",
            use_container_width=True,
        ):
            if not candidates:
                st.warning("Add at least one resume before ranking candidates.")
            elif not job_title.strip() or not job_description.strip():
                st.warning("Provide both a job title and a job description.")
            else:
                with st.spinner("Comparing the talent pool against the JD…"):
                    try:
                        safe_job_title = _validate_talent_input(
                            job_title,
                            username,
                            user_role,
                            field_name="Job Title",
                            max_length=200,
                        )
                        safe_job_description = _validate_talent_input(
                            job_description,
                            username,
                            user_role,
                            field_name="Job Description",
                            max_length=12000,
                        )

                        results = _score_candidates(
                            safe_job_title,
                            safe_job_description,
                            username,
                        )
                        _save_talent_search(
                            username,
                            safe_job_title,
                            safe_job_description,
                            results,
                        )
                        st.session_state.talent_results = results
                    except Exception as exc:
                        logger.exception("Talent ranking failed")
                        st.error(
                            f"Talent ranking failed: {_clean_text(exc, 700)}"
                        )

        results = st.session_state.get("talent_results", [])

        if results:
            st.divider()
            st.subheader("Ranked shortlist")

            st.caption(
                "Scores are relevance indicators based on the supplied JD and "
                "resume content. HR should independently review the underlying resumes."
            )

            candidate_lookup = {
                int(candidate["id"]): candidate
                for candidate in _get_candidate_records(
                    username,
                    include_all=_check_permission("admin"),
                )
                if candidate.get("id") is not None
            }

            for result in results:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2.4, 1, 1])

                    with c1:
                        st.markdown(
                            f"### #{result['rank']} · {_escape(result['candidate'])}"
                        )
                        st.caption(
                            f"{_escape(result['resume'])} · Status: {_escape(result['status'])}"
                        )
                        st.write(
                            f"**Matched skills:** {result['matched_skills']}"
                        )

                    with c2:
                        st.metric(
                            "Fit Score",
                            f"{result['fit_score']:.1f}%",
                        )

                    with c3:
                        st.metric(
                            "Skill Match",
                            f"{result['skill_match']:.1f}%",
                        )

                    candidate = candidate_lookup.get(int(result["candidate_id"]))
                    if candidate:
                        with st.expander("📄 View Resume & Match Details", expanded=False):
                            detail_col1, detail_col2 = st.columns(2)
                            with detail_col1:
                                st.write(f"**Candidate:** {candidate['candidate_name']}")
                                st.write(f"**Resume:** {candidate['resume_filename']}")
                                st.write(f"**Status:** {candidate['status']}")
                            with detail_col2:
                                st.write(f"**Semantic Match:** {result['semantic_score']:.1f}%")
                                st.write(f"**Skill Match:** {result['skill_match']:.1f}%")
                                st.write(f"**Overall Fit:** {result['fit_score']:.1f}%")
                            st.write(f"**Matched skills:** {result['matched_skills']}")
                            st.write("**Resume evidence:**")
                            evidence = _clean_text(candidate.get("resume_text", ""), 1800)
                            st.text_area(
                                "Relevant resume content",
                                value=evidence,
                                height=180,
                                disabled=True,
                                key=f"resume_evidence_{candidate['id']}_{result['rank']}",
                                label_visibility="collapsed",
                            )
                            _render_resume_viewer(candidate, username)

            st.download_button(
                "Export shortlist CSV",
                data=pd.DataFrame(results).to_csv(index=False).encode("utf-8"),
                file_name="policyguard_talent_shortlist.csv",
                mime="text/csv",
                use_container_width=True,
            )


# =============================================================================
# DASHBOARD
# =============================================================================


def _render_dashboard_view(username: str) -> None:
    _require_login()

    st.title("📊 Dashboard")
    st.caption(
        "Operational visibility for the current Streamlit session and knowledge base."
    )

    history = st.session_state.get(
        "query_history",
        [],
    )

    documents = _get_document_stats()
    components = _system_components()

    total_queries = len(history)

    latencies = [
        float(item.get("latency_ms", 0) or 0)
        for item in history
        if item.get("latency_ms") is not None
    ]

    avg_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0
    )

    cache_hits = sum(
        1
        for item in history
        if item.get("cache_hit")
    )

    cache_rate = (
        cache_hits / total_queries * 100
        if total_queries
        else 0
    )

    online_count = sum(
        1 for value in components.values()
        if value
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Session Queries",
            total_queries,
        )

    with m2:
        st.metric(
            "Avg Latency",
            f"{avg_latency:.0f} ms",
        )

    with m3:
        st.metric(
            "Cache Hit Rate",
            f"{cache_rate:.1f}%",
        )

    with m4:
        st.metric(
            "Healthy Components",
            f"{online_count}/{len(components)}",
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Knowledge Base")

        st.metric(
            "Documents",
            documents["documents"],
        )
        st.metric(
            "Indexed Chunks",
            documents["chunks"],
        )
        st.metric(
            "Storage",
            f"{documents['bytes'] / (1024 * 1024):.2f} MB",
        )

    with right:
        st.subheader("Runtime Components")

        for component, available in components.items():
            if available:
                st.success(
                    component,
                )
            else:
                st.warning(
                    component,
                )

    if history:
        st.divider()
        st.subheader("Recent Queries")

        df = pd.DataFrame(history[-50:])

        columns = [
            column
            for column in (
                "timestamp",
                "query",
                "agent",
                "latency_ms",
                "cache_hit",
                "model_used",
            )
            if column in df.columns
        ]

        if columns:
            st.dataframe(
                df[columns],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "No queries have been executed in this session yet."
        )


# =============================================================================
# DOCUMENTS
# =============================================================================


def _render_documents_view(user_role: str) -> None:
    _require_login()

    st.title("📚 Document Management")
    st.caption(
        "Inspect the documents currently registered in the PolicyGuard knowledge base."
    )

    if not _check_permission("editor"):
        st.info(
            "Viewer access does not include document management."
        )
        return

    documents = _get_documents()

    if not documents:
        st.info(
            "No documents have been indexed yet. "
            "Use the sidebar uploader to add the first document."
        )
        return

    df = pd.DataFrame(documents)

    if not df.empty:
        display_columns = [
            column
            for column in (
                "filename",
                "uploaded_by",
                "uploaded_at",
                "file_size",
                "chunk_count",
                "ocr_used",
                "status",
            )
            if column in df.columns
        ]

        display_df = df[display_columns].copy()

        if "file_size" in display_df.columns:
            display_df["file_size"] = (
                display_df["file_size"].fillna(0) / (1024 * 1024)
            ).round(2)
            display_df.rename(
                columns={"file_size": "size_mb"},
                inplace=True,
            )

        if "ocr_used" in display_df.columns:
            display_df["ocr_used"] = display_df[
                "ocr_used"
            ].map(
                {
                    1: "Yes",
                    0: "No",
                }
            )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        f"{len(documents)} document record(s)"
    )


# =============================================================================
# AUDIT LOGS
# =============================================================================


def _render_audit_view(user_role: str) -> None:
    _require_login()

    if not _check_permission("admin"):
        st.error(
            "Administrator permissions are required to view audit logs."
        )
        return

    st.title("🛡️ Audit Logs")
    st.caption(
        "Security, authentication, document and query events."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        username_filter = st.text_input(
            "Username",
            placeholder="All users",
        )

    with c2:
        action_filter = st.text_input(
            "Action contains",
            placeholder="e.g. QUERY",
        )

    with c3:
        limit = st.number_input(
            "Maximum events",
            min_value=10,
            max_value=5000,
            value=250,
            step=10,
        )

    logs = _get_audit_logs(
        limit=int(limit),
        username=username_filter.strip() or None,
        action_filter=action_filter.strip() or None,
    )

    if not logs:
        st.info("No audit events matched the selected filters.")
        return

    df = pd.DataFrame(logs)

    display_columns = [
        column
        for column in (
            "timestamp",
            "username",
            "action",
            "details",
            "model_used",
            "tokens_used",
            "cost_usd",
            "threat_type",
            "blocked",
        )
        if column in df.columns
    ]

    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Showing {len(logs)} event(s)."
    )


# =============================================================================
# USER MANAGEMENT
# =============================================================================


def _render_users_view() -> None:
    _require_login()

    if not _check_permission("admin"):
        st.error(
            "Administrator permissions are required."
        )
        return

    st.title("👥 User Management")
    st.caption(
        "Manage role assignments and account activation without destroying audit history."
    )

    if not HARDENED_AUTH_DATABASE_AVAILABLE or _auth_database is None:
        st.error(
            "The hardened authentication service is unavailable. "
            "User-management operations are disabled for safety."
        )
        return

    users = _get_all_users()

    if not users:
        st.info("No users found.")
        return

    for user in users:
        user_id = int(user["id"])
        username = str(user["username"])
        role = _safe_role(user["role"])
        active = bool(user["is_active"])

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(
                [2.2, 1.3, 1.2, 1]
            )

            with c1:
                st.markdown(
                    f"**{_escape(username)}**"
                )
                st.caption(
                    f"Created: {user.get('created_at', '—')} · "
                    f"Last login: {user.get('last_login', 'Never')}"
                )

            with c2:
                new_role = st.selectbox(
                    "Role",
                    list(VALID_ROLES),
                    index=list(VALID_ROLES).index(role),
                    key=f"user_role_{user_id}",
                    label_visibility="collapsed",
                )

            with c3:
                active_label = (
                    "Active"
                    if active
                    else "Inactive"
                )

                if active:
                    st.success(
                        active_label,
                    )
                else:
                    st.error(
                        active_label,
                    )

            with c4:
                if st.button(
                    "Save",
                    key=f"user_save_{user_id}",
                    use_container_width=True,
                ):
                    # Prevent accidental self-demotion.
                    if username == st.session_state.username:
                        if new_role != "admin":
                            st.error(
                                "You cannot remove your own administrator role."
                            )
                        else:
                            st.success("No changes required.")
                    elif _update_user_role(
                        user_id,
                        new_role,
                        st.session_state.username,
                    ):
                        st.success("Updated.")
                        time.sleep(0.2)
                        st.rerun()
                    else:
                        st.error("Unable to update role.")

            status_label = (
                "Deactivate account"
                if active
                else "Activate account"
            )

            if username != st.session_state.username:
                if st.button(
                    status_label,
                    key=f"user_active_{user_id}",
                ):
                    if _set_user_active(
                        user_id,
                        not active,
                        st.session_state.username,
                    ):
                        st.success("Account status updated.")
                        time.sleep(0.2)
                        st.rerun()
                    else:
                        st.error(
                            "Unable to update account status."
                        )


# =============================================================================
# ARCHITECTURE
# =============================================================================


def _render_architecture_view() -> None:
    st.title("🏗️ Architecture")
    st.caption(
        "PolicyGuard AI's application and intelligence pipeline."
    )

    st.subheader("System Pipeline")

    pipeline = [
        (
            "1",
            "Document Ingestion",
            "PDF / DOCX / XLSX / TXT / Markdown → extraction → chunking.",
        ),
        (
            "2",
            "Embedding",
            "Sentence-transformer embeddings create semantic representations.",
        ),
        (
            "3",
            "Hybrid Retrieval",
            "FAISS semantic retrieval combines with BM25 lexical retrieval.",
        ),
        (
            "4",
            "Reranking",
            "Cross-encoder reranking improves relevance of retrieved passages.",
        ),
        (
            "5",
            "Orchestration",
            "LangGraph/RAG pipeline coordinates query processing.",
        ),
        (
            "6",
            "Routing",
            "Requests are classified into HR Policy RAG, Talent Intelligence or General HR Assistant, with the route exposed to the user.",
        ),
        (
            "7",
            "Persistent Memory",
            "Conversation history is stored in SQLite and restored when Live Chat is reopened.",
        ),
        (
            "8",
            "Talent Intelligence",
            "Authorized HR users can rank internal talent against a job description using semantic and skill-fit signals.",
        ),
        (
            "9",
            "Security & Audit",
            "Queries and privileged operations are security-checked and audited.",
        ),
    ]

    for number, title, description in pipeline:
        with st.container(border=True):
            c1, c2 = st.columns([0.6, 7])

            with c1:
                st.markdown(f"### {number}")

            with c2:
                st.markdown(f"**{title}**")
                st.caption(description)

    st.divider()

    st.subheader("Runtime Configuration")

    config_rows = [
        {
            "Setting": "Application",
            "Value": APP_NAME,
        },
        {
            "Setting": "Version",
            "Value": APP_VERSION,
        },
        {
            "Setting": "Database",
            "Value": str(DB_FILE),
        },
        {
            "Setting": "Vector DB",
            "Value": str(VECTOR_DB_DIR),
        },
        {
            "Setting": "Security Guard",
            "Value": "Available" if SECURITY_AVAILABLE else "Unavailable",
        },
        {
            "Setting": "RAG Engine",
            "Value": "Available" if RAG_AVAILABLE else "Unavailable",
        },
        {
            "Setting": "LangGraph",
            "Value": "Available" if GRAPH_AVAILABLE else "Unavailable",
        },
    ]

    st.dataframe(
        pd.DataFrame(config_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Production deployment should keep secrets outside source control "
        "and should use durable storage for the database and vector index."
    )


# =============================================================================
# FOOTER
# =============================================================================


def _render_footer() -> None:
    username = _escape(
        st.session_state.get("username") or "User"
    )
    role = _escape(
        _safe_role(
            st.session_state.get("user_role", "viewer")
        )
    )

    st.markdown(
        f"""
        <div class="footer">
            <strong>{_escape(APP_NAME)}</strong> v{_escape(APP_VERSION)}
            · Signed in as {username} ({role})
            · Enterprise HR Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN ROUTER
# =============================================================================


def main() -> None:
    """Application entry point."""
    if not _DB_READY:
        st.error(
            "The application database could not be initialized."
        )
        st.info(
            "Check filesystem permissions and database configuration, "
            "then restart Streamlit."
        )
        return

    if not st.session_state.get("authenticated"):
        _show_auth_page()
        return

    # Validate session role on every rerun.
    role = _safe_role(
        st.session_state.get("user_role", "viewer")
    )

    if role not in VALID_ROLES:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.error("Your session is invalid. Please sign in again.")
        return

    username = _safe_username(
        st.session_state.get("username", "")
    )

    organization_id = _clean_text(
        st.session_state.get("organization_id"),
        MAX_ORGANIZATION_ID_LENGTH,
    )
    if not organization_id or not ORGANIZATION_ID_PATTERN.fullmatch(organization_id):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.organization_id = None
        st.error("Your session has no valid organization scope. Please sign in again.")
        return

    if not username:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.error("Your session is invalid. Please sign in again.")
        return

    _render_sidebar()

    _show_onboarding()

    view = st.session_state.get("view", "home")

    if view == "home":
        _render_home_view(role)

    elif view == "chat":
        _render_chat_view(
            username,
            role,
        )

    elif view == "dashboard":
        _render_dashboard_view(username)

    elif view == "documents":
        _render_documents_view(role)

    elif view == "talent":
        _render_talent_view(username, role)

    elif view == "audit":
        _render_audit_view(role)

    elif view == "architecture":
        _render_architecture_view()

    elif view == "users":
        if _check_permission("admin"):
            _render_users_view()
        else:
            st.error(
                "Administrator permissions are required."
            )

    else:
        st.session_state.view = "home"
        st.rerun()

    _render_footer()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

