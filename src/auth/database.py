#!/usr/bin/env python3
"""
PolicyGuard AI - Production Authentication & Database Layer
=============================================================

Centralized SQLite authentication/database service for local development
and single-node deployments.

Responsibilities:
- Database initialization and migrations
- User authentication
- bcrypt password hashing
- Brute-force protection
- RBAC permission checks
- User administration
- Account activation/deactivation
- Access-request workflow
- Audit logging
- Query telemetry
- Database statistics
- SQLite backup support

IMPORTANT:
- No default admin password is created.
- Production secrets must come from environment/configuration.
- app.py should call this module instead of directly manipulating SQLite.
- PostgreSQL migration can be introduced later behind the same service API.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import bcrypt

from config.settings import settings


logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

def _resolve_sqlite_database_path(database_url: str) -> Path:
    """
    Resolve a SQLite DATABASE_URL into an absolute filesystem path.

    Supported:
        sqlite:///./nexus_auth.db
        sqlite:////absolute/path/database.db
    """
    if not database_url.startswith("sqlite:///"):
        raise ValueError(
            "This authentication module currently supports SQLite only. "
            "Use DATABASE_URL=sqlite:///... for local deployment."
        )

    raw_path = database_url[len("sqlite:///"):]

    if not raw_path:
        raise ValueError("SQLite database path cannot be empty")

    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        path = settings.BASE_DIR / path

    return path.resolve()


DB_FILE: Path = _resolve_sqlite_database_path(settings.DATABASE_URL)

DB_TIMEOUT_SECONDS = 30
DB_MAX_RETRIES = 4

MAX_FAILED_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 30

PASSWORD_MIN_LENGTH = 8
USERNAME_MIN_LENGTH = 3
USERNAME_MAX_LENGTH = 50

# Enterprise security limits and tenant defaults.
DEFAULT_ORGANIZATION_ID = "default"
ORGANIZATION_MAX_LENGTH = 100
ACCESS_REASON_MAX_LENGTH = 2000
AUDIT_PREVIEW_MAX_LENGTH = 4000


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

@contextmanager
def get_db_connection() -> Iterator[sqlite3.Connection]:
    """
    Open a configured SQLite connection.

    Every connection:
    - enables foreign keys
    - uses Row objects
    - uses WAL mode for improved concurrent read behavior
    - configures a busy timeout
    """
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    conn: Optional[sqlite3.Connection] = None

    try:
        # Python 3.12+ deprecates sqlite3's legacy automatic timestamp
        # adapters/converters. Keep SQLite timestamps as ISO-compatible strings
        # and parse them explicitly only where datetime objects are required.
        conn = sqlite3.connect(
            str(DB_FILE),
            timeout=DB_TIMEOUT_SECONDS,
        )

        conn.row_factory = sqlite3.Row

        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")

        # WAL is appropriate for the expected Streamlit/local deployment
        # workload and permits readers while a writer is active.
        conn.execute("PRAGMA journal_mode = WAL")

        yield conn

    except sqlite3.Error:
        if conn is not None:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
        raise

    finally:
        if conn is not None:
            conn.close()


def _execute_with_retry(
    query: str,
    params: Tuple[Any, ...] = (),
    *,
    fetch: Optional[str] = None,
    commit: bool = True,
) -> Any:
    """
    Execute a SQLite statement with retry handling for locked databases.

    fetch:
        None   -> return rowcount
        one    -> return one row
        all    -> return all rows
    """
    last_error: Optional[Exception] = None

    for attempt in range(DB_MAX_RETRIES):
        try:
            with get_db_connection() as conn:
                cursor = conn.execute(query, params)

                if commit:
                    conn.commit()

                if fetch == "one":
                    return cursor.fetchone()

                if fetch == "all":
                    return cursor.fetchall()

                return cursor.rowcount

        except sqlite3.OperationalError as exc:
            last_error = exc

            if "locked" in str(exc).lower() and attempt < DB_MAX_RETRIES - 1:
                delay = 0.1 * (2 ** attempt)
                time.sleep(delay)
                continue

            raise

    if last_error:
        raise last_error

    raise RuntimeError("Database operation failed unexpectedly")


# =============================================================================
# SCHEMA
# =============================================================================

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash BLOB NOT NULL,
    role TEXT NOT NULL DEFAULT 'viewer'
        CHECK(role IN ('viewer', 'editor', 'admin')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active INTEGER NOT NULL DEFAULT 1
        CHECK(is_active IN (0, 1)),
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    last_failed_login TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    action TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    details TEXT,
    ip_address TEXT,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    model_used TEXT,
    query_preview TEXT,
    threat_type TEXT,
    blocked INTEGER NOT NULL DEFAULT 0
        CHECK(blocked IN (0, 1))
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    filepath TEXT,
    uploaded_by TEXT,
    uploaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    file_size INTEGER,
    chunk_count INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    ocr_used INTEGER NOT NULL DEFAULT 0
        CHECK(ocr_used IN (0, 1))
);

CREATE TABLE IF NOT EXISTS access_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    from_role TEXT NOT NULL
        CHECK(from_role IN ('viewer', 'editor', 'admin')),
    to_role TEXT NOT NULL
        CHECK(to_role IN ('viewer', 'editor', 'admin')),
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'approved', 'rejected', 'expired')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    approved_by TEXT,
    approved_at TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS query_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT UNIQUE NOT NULL,
    query_text TEXT NOT NULL,
    query_embedding TEXT,
    answer TEXT NOT NULL,
    chunks_used TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    hits INTEGER NOT NULL DEFAULT 0
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

CREATE INDEX IF NOT EXISTS idx_audit_blocked
    ON audit_log(blocked);

CREATE INDEX IF NOT EXISTS idx_documents_filename
    ON documents(filename);

CREATE INDEX IF NOT EXISTS idx_documents_uploaded_by
    ON documents(uploaded_by);

CREATE INDEX IF NOT EXISTS idx_requests_user
    ON access_requests(user_id);

CREATE INDEX IF NOT EXISTS idx_requests_status
    ON access_requests(status);

CREATE INDEX IF NOT EXISTS idx_cache_hash
    ON query_cache(query_hash);
"""


# =============================================================================
# MIGRATION HELPERS
# =============================================================================

def _get_table_columns(
    conn: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    """Return the current column names for a table."""
    rows = conn.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    return {row["name"] for row in rows}


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    """Add a column to an existing table when it doesn't exist."""
    columns = _get_table_columns(conn, table_name)

    if column_name not in columns:
        conn.execute(
            f"ALTER TABLE {table_name} "
            f"ADD COLUMN {column_name} {column_definition}"
        )

        logger.info(
            "Database migration: added %s.%s",
            table_name,
            column_name,
        )


def _migrate_database(conn: sqlite3.Connection) -> None:
    """
    Migrate legacy PolicyGuard databases.

    IMPORTANT:
    Tables are created BEFORE migrations are attempted.
    This fixes the original ordering problem where PRAGMA/migrations
    could run before required tables existed.
    """
    # Legacy users columns
    _add_column_if_missing(
        conn,
        "users",
        "is_active",
        "INTEGER NOT NULL DEFAULT 1",
    )

    _add_column_if_missing(
        conn,
        "users",
        "failed_login_attempts",
        "INTEGER NOT NULL DEFAULT 0",
    )

    _add_column_if_missing(
        conn,
        "users",
        "last_failed_login",
        "TIMESTAMP",
    )

    # Legacy audit columns
    _add_column_if_missing(
        conn,
        "audit_log",
        "tokens_used",
        "INTEGER NOT NULL DEFAULT 0",
    )

    _add_column_if_missing(
        conn,
        "audit_log",
        "cost_usd",
        "REAL NOT NULL DEFAULT 0.0",
    )

    _add_column_if_missing(
        conn,
        "audit_log",
        "model_used",
        "TEXT",
    )

    _add_column_if_missing(
        conn,
        "audit_log",
        "query_preview",
        "TEXT",
    )

    _add_column_if_missing(
        conn,
        "audit_log",
        "threat_type",
        "TEXT",
    )

    _add_column_if_missing(
        conn,
        "audit_log",
        "blocked",
        "INTEGER NOT NULL DEFAULT 0",
    )

    # Cache compatibility
    _add_column_if_missing(
        conn,
        "query_cache",
        "query_embedding",
        "TEXT",
    )

    _add_column_if_missing(
        conn,
        "query_cache",
        "last_accessed",
        "TIMESTAMP",
    )

    # Backfill legacy NULL cache access timestamps.
    conn.execute("""
        UPDATE query_cache
        SET last_accessed = COALESCE(last_accessed, created_at)
        WHERE last_accessed IS NULL
    """)

    conn.commit()


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def init_database() -> bool:
    """
    Initialize and migrate the application database.

    Returns:
        True when initialization succeeds.
    """
    try:
        DB_FILE.parent.mkdir(parents=True, exist_ok=True)

        with get_db_connection() as conn:
            # FIRST create missing tables.
            conn.executescript(_SCHEMA_SQL)

            # THEN migrate legacy installations.
            _migrate_database(conn)

            conn.commit()

        logger.info("Database initialized successfully: %s", DB_FILE)

        return True

    except Exception:
        logger.exception("Database initialization failed")
        return False


# =============================================================================
# PASSWORD SECURITY
# =============================================================================

def _validate_password(password: str) -> Optional[str]:
    """Return validation error or None."""
    if not password:
        return "Password is required"

    if len(password) < PASSWORD_MIN_LENGTH:
        return (
            f"Password must be at least "
            f"{PASSWORD_MIN_LENGTH} characters"
        )

    if not any(char.islower() for char in password):
        return "Password must contain a lowercase letter"

    if not any(char.isupper() for char in password):
        return "Password must contain an uppercase letter"

    if not any(char.isdigit() for char in password):
        return "Password must contain a number"

    return None


def hash_password(
    password: str,
    rounds: int = 12,
) -> Optional[bytes]:
    """Hash a password with bcrypt."""
    if _validate_password(password):
        return None

    try:
        return bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=rounds),
        )
    except Exception:
        logger.exception("Password hashing failed")
        return None


def _normalize_stored_hash(
    stored_hash: Union[bytes, str],
) -> Optional[bytes]:
    """
    Normalize legacy and current bcrypt storage formats.

    Current format:
        bcrypt ASCII bytes/string

    Legacy compatibility:
        base64 encoded bcrypt hash
    """
    if not stored_hash:
        return None

    if isinstance(stored_hash, bytes):
        value = stored_hash
    else:
        value = stored_hash.encode("utf-8")

    # Standard bcrypt hashes begin with $2...
    if value.startswith(b"$2"):
        return value

    # Try legacy base64 format.
    try:
        decoded = base64.b64decode(value, validate=True)

        if decoded.startswith(b"$2"):
            return decoded
    except Exception:
        pass

    return None


def verify_password(
    stored_hash: Union[bytes, str],
    password: str,
) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        normalized = _normalize_stored_hash(stored_hash)

        if normalized is None or not password:
            return False

        return bcrypt.checkpw(
            password.encode("utf-8"),
            normalized,
        )

    except Exception:
        logger.exception("Password verification failed")
        return False


# =============================================================================
# USER VALIDATION
# =============================================================================

def _validate_username(username: str) -> Optional[str]:
    """Return username validation error or None."""
    if not username:
        return "Username is required"

    if not (
        USERNAME_MIN_LENGTH
        <= len(username)
        <= USERNAME_MAX_LENGTH
    ):
        return (
            f"Username must be between "
            f"{USERNAME_MIN_LENGTH} and "
            f"{USERNAME_MAX_LENGTH} characters"
        )

    if not username.replace("_", "").isalnum():
        return (
            "Username can contain only letters, "
            "numbers, and underscores"
        )

    return None


def _validate_role(role: str) -> bool:
    """Validate application role."""
    return role in settings.ROLES


# =============================================================================
# AUDIT LOGGING
# =============================================================================

def _log_audit(
    username: str,
    action: str,
    details: str = "",
    *,
    ip_address: Optional[str] = None,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
    model_used: Optional[str] = None,
    query_preview: Optional[str] = None,
    threat_type: Optional[str] = None,
    blocked: bool = False,
) -> None:
    """
    Write an audit event.

    Audit failures intentionally never crash the primary application action.
    """
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (
                    username,
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    action,
                    details,
                    ip_address,
                    max(0, int(tokens_used)),
                    max(0.0, float(cost_usd)),
                    model_used,
                    query_preview,
                    threat_type,
                    1 if blocked else 0,
                ),
            )

            conn.commit()

    except Exception:
        logger.exception("Audit logging failed")


def log_query_event(
    username: str,
    query: str,
    answer: str,
    chunks_used: int,
    latency_ms: int,
    tokens_used: int,
    cost_usd: float,
    model_used: str,
    *,
    cache_hit: bool = False,
    grounded: Optional[bool] = None,
    citations_count: int = 0,
) -> None:
    """Record a complete RAG query telemetry event."""

    query_preview = (
        query[:200] + "..."
        if len(query) > 200
        else query
    )

    answer_preview = (
        answer[:300] + "..."
        if len(answer) > 300
        else answer
    )

    details = (
        f"answer={answer_preview} | "
        f"chunks={chunks_used} | "
        f"latency_ms={latency_ms} | "
        f"cache_hit={cache_hit} | "
        f"grounded={grounded} | "
        f"citations={citations_count}"
    )

    _log_audit(
        username=username,
        action="QUERY_EXECUTED",
        details=details,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        model_used=model_used,
        query_preview=query_preview,
    )


# =============================================================================
# USER REGISTRATION
# =============================================================================

def register_user(
    username: str,
    password: str,
    role: str = "viewer",
) -> Tuple[bool, str]:
    """
    Register a new user.

    SECURITY:
        Public registration can only create Viewer accounts.
        Privileged roles must be granted by an administrator.
    """
    username = username.strip()

    username_error = _validate_username(username)

    if username_error:
        return False, username_error

    password_error = _validate_password(password)

    if password_error:
        return False, password_error

    # Never allow public registration to create privileged accounts.
    if role != "viewer":
        logger.warning(
            "Attempted privileged self-registration: username=%s role=%s",
            username,
            role,
        )
        return False, "New accounts can only be registered as Viewer"

    if not _validate_role(role):
        return False, "Invalid user role"

    password_hash = hash_password(password)

    if not password_hash:
        return False, "Unable to securely hash password"

    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    username,
                    password_hash,
                    role
                )
                VALUES (?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    role,
                ),
            )

            conn.commit()

        _log_audit(
            username=username,
            action="USER_REGISTERED",
            details="New Viewer account registered",
        )

        return True, "Registration successful. Please sign in."

    except sqlite3.IntegrityError:
        return False, "Username already exists"

    except Exception:
        logger.exception("User registration failed")
        return False, "Registration failed due to an internal error"


# =============================================================================
# LOGIN
# =============================================================================

def login_user(
    username: str,
    password: str,
) -> Tuple[bool, str, Optional[str]]:
    """
    Authenticate a user.

    Returns:
        (success, message, role)
    """
    username = username.strip()

    if not username or not password:
        return False, "Username and password are required", None

    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    id,
                    username,
                    password_hash,
                    role,
                    is_active,
                    failed_login_attempts,
                    last_failed_login
                FROM users
                WHERE username = ?
                """,
                (username,),
            )

            user = cursor.fetchone()

            if not user:
                _log_audit(
                    username=username,
                    action="LOGIN_FAILED",
                    details="User not found",
                    blocked=True,
                )

                return False, "Invalid username or password", None

            (
                user_id,
                db_username,
                stored_hash,
                role,
                is_active,
                failed_attempts,
                last_failed_login,
            ) = user

            if not is_active:
                _log_audit(
                    username=username,
                    action="LOGIN_FAILED",
                    details="Account inactive",
                    blocked=True,
                )

                return False, "Account is inactive", None

            # -------------------------------------------------------------
            # Brute-force lockout
            # -------------------------------------------------------------

            if failed_attempts >= MAX_FAILED_LOGIN_ATTEMPTS:
                lockout_end: Optional[datetime] = None

                if isinstance(last_failed_login, datetime):
                    lockout_end = (
                        last_failed_login
                        + timedelta(minutes=LOCKOUT_MINUTES)
                    )

                elif last_failed_login:
                    try:
                        parsed = datetime.fromisoformat(
                            str(last_failed_login)
                        )

                        lockout_end = (
                            parsed
                            + timedelta(minutes=LOCKOUT_MINUTES)
                        )

                    except ValueError:
                        logger.warning(
                            "Invalid last_failed_login for user %s",
                            username,
                        )

                if lockout_end and datetime.now() < lockout_end:
                    _log_audit(
                        username=username,
                        action="LOGIN_BLOCKED",
                        details="Account temporarily locked",
                        blocked=True,
                    )

                    return (
                        False,
                        "Account temporarily locked. Try again later.",
                        None,
                    )

                # Lockout expired.
                conn.execute(
                    """
                    UPDATE users
                    SET failed_login_attempts = 0,
                        last_failed_login = NULL
                    WHERE id = ?
                    """,
                    (user_id,),
                )

                conn.commit()

            # -------------------------------------------------------------
            # Password verification
            # -------------------------------------------------------------

            if not verify_password(stored_hash, password):
                conn.execute(
                    """
                    UPDATE users
                    SET failed_login_attempts =
                            failed_login_attempts + 1,
                        last_failed_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (user_id,),
                )

                conn.commit()

                _log_audit(
                    username=username,
                    action="LOGIN_FAILED",
                    details="Invalid password",
                    blocked=True,
                )

                return False, "Invalid username or password", None

            # -------------------------------------------------------------
            # Successful login
            # -------------------------------------------------------------

            conn.execute(
                """
                UPDATE users
                SET failed_login_attempts = 0,
                    last_failed_login = NULL,
                    last_login = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (user_id,),
            )

            conn.commit()

        _log_audit(
            username=db_username,
            action="LOGIN_SUCCESS",
            details=f"Role: {role}",
        )

        return (
            True,
            f"Welcome back, {db_username}!",
            role,
        )

    except Exception:
        logger.exception("Login failed")
        return (
            False,
            "Authentication service temporarily unavailable",
            None,
        )


# =============================================================================
# USER QUERIES
# =============================================================================

def get_user_by_username(
    username: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve a user by username without exposing password hash."""
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    id,
                    username,
                    role,
                    created_at,
                    last_login,
                    is_active
                FROM users
                WHERE username = ?
                """,
                (username,),
            ).fetchone()

            return dict(row) if row else None

    except Exception:
        logger.exception("Could not retrieve user")
        return None


def get_user_by_id(
    user_id: int,
) -> Optional[Dict[str, Any]]:
    """Retrieve a user by numeric ID."""
    if user_id <= 0:
        return None

    try:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    id,
                    username,
                    role,
                    created_at,
                    last_login,
                    is_active
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

            return dict(row) if row else None

    except Exception:
        logger.exception("Could not retrieve user by ID")
        return None


def get_all_users() -> List[Dict[str, Any]]:
    """Return all users without password hashes."""
    try:
        with get_db_connection() as conn:
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
                ORDER BY created_at DESC
                """
            ).fetchall()

            return [dict(row) for row in rows]

    except Exception:
        logger.exception("Could not retrieve users")
        return []


# =============================================================================
# USER ADMINISTRATION
# =============================================================================

def update_user_role(
    user_id: int,
    new_role: str,
    updated_by: str,
) -> bool:
    """Change a user's role."""
    if user_id <= 0 or not _validate_role(new_role):
        return False

    try:
        with get_db_connection() as conn:
            target = conn.execute(
                "SELECT username, role FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()

            if not target:
                return False

            target_username = target["username"]
            old_role = target["role"]

            if old_role == new_role:
                return True

            # Prevent the final active admin from being demoted.
            if old_role == "admin" and new_role != "admin":
                admin_count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM users
                    WHERE role = 'admin'
                      AND is_active = 1
                    """
                ).fetchone()[0]

                if admin_count <= 1:
                    logger.warning(
                        "Attempt to demote final active admin: %s",
                        target_username,
                    )
                    return False

            conn.execute(
                """
                UPDATE users
                SET role = ?
                WHERE id = ?
                """,
                (new_role, user_id),
            )

            conn.commit()

        _log_audit(
            username=updated_by,
            action="ROLE_CHANGED",
            details=(
                f"User '{target_username}' "
                f"role changed {old_role} -> {new_role}"
            ),
        )

        return True

    except Exception:
        logger.exception("User role update failed")
        return False


def update_user_active_status(
    user_id: int,
    is_active: bool,
    updated_by: str,
) -> bool:
    """Activate or deactivate a user account."""
    if user_id <= 0:
        return False

    try:
        with get_db_connection() as conn:
            target = conn.execute(
                "SELECT username, role, is_active FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()

            if not target:
                return False

            username = target["username"]

            # Don't allow the last active admin to deactivate itself.
            if (
                not is_active
                and target["role"] == "admin"
                and target["is_active"]
            ):
                admin_count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM users
                    WHERE role = 'admin'
                      AND is_active = 1
                    """
                ).fetchone()[0]

                if admin_count <= 1:
                    return False

            conn.execute(
                """
                UPDATE users
                SET is_active = ?
                WHERE id = ?
                """,
                (1 if is_active else 0, user_id),
            )

            conn.commit()

        _log_audit(
            username=updated_by,
            action=(
                "USER_ACTIVATED"
                if is_active
                else "USER_DEACTIVATED"
            ),
            details=f"User '{username}' account status changed",
        )

        return True

    except Exception:
        logger.exception("Account status update failed")
        return False


def delete_user(
    user_id: int,
    deleted_by: str,
) -> bool:
    """
    Permanently delete a user.

    NOTE:
        For enterprise compliance, deactivation is preferred.
        This function remains for administrative cleanup/testing.
    """
    if user_id <= 0:
        return False

    try:
        with get_db_connection() as conn:
            target = conn.execute(
                "SELECT username, role FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()

            if not target:
                return False

            username = target["username"]
            role = target["role"]

            # Never allow deleting the last active admin.
            if role == "admin":
                admin_count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM users
                    WHERE role = 'admin'
                      AND is_active = 1
                    """
                ).fetchone()[0]

                if admin_count <= 1:
                    return False

            conn.execute(
                "DELETE FROM users WHERE id = ?",
                (user_id,),
            )

            conn.commit()

        _log_audit(
            username=deleted_by,
            action="USER_DELETED",
            details=f"User '{username}' permanently deleted",
        )

        return True

    except Exception:
        logger.exception("User deletion failed")
        return False


# =============================================================================
# ACCESS REQUEST WORKFLOW
# =============================================================================

def create_access_request(
    user_id: int,
    from_role: str,
    to_role: str,
    reason: str,
) -> Tuple[bool, str]:
    """Create a role-escalation request."""
    reason = reason.strip()

    if user_id <= 0:
        return False, "Invalid user ID"

    if not _validate_role(from_role) or not _validate_role(to_role):
        return False, "Invalid role"

    if not reason or len(reason) < 10:
        return False, "Please provide a reason of at least 10 characters"

    if not settings.role_at_least(to_role, from_role):
        return False, "Requested role cannot be lower than current role"

    if from_role == to_role:
        return False, "You already have this role"

    try:
        with get_db_connection() as conn:
            user = conn.execute(
                """
                SELECT username, role, is_active
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

            if not user:
                return False, "User not found"

            if not user["is_active"]:
                return False, "Inactive users cannot request access"

            # Trust the database's current role rather than a role supplied
            # by the client.
            actual_role = user["role"]

            if actual_role != from_role:
                return False, "Current role does not match account"

            # Avoid duplicate pending requests.
            existing = conn.execute(
                """
                SELECT id
                FROM access_requests
                WHERE user_id = ?
                  AND status = 'pending'
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()

            if existing:
                return False, "You already have a pending access request"

            conn.execute(
                """
                INSERT INTO access_requests (
                    user_id,
                    from_role,
                    to_role,
                    reason,
                    status
                )
                VALUES (?, ?, ?, ?, 'pending')
                """,
                (
                    user_id,
                    from_role,
                    to_role,
                    reason,
                ),
            )

            conn.commit()

            username = user["username"]

        _log_audit(
            username=username,
            action="ACCESS_REQUEST_CREATED",
            details=(
                f"Requested role {from_role} -> {to_role}; "
                f"reason={reason[:200]}"
            ),
        )

        return True, "Access request submitted for admin approval"

    except Exception:
        logger.exception("Access request creation failed")
        return False, "Unable to create access request"


def get_pending_access_requests() -> List[Dict[str, Any]]:
    """Return pending access requests with user information."""
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    ar.id,
                    ar.user_id,
                    ar.from_role,
                    ar.to_role,
                    ar.reason,
                    ar.status,
                    ar.created_at,
                    ar.approved_by,
                    ar.approved_at,
                    u.username
                FROM access_requests ar
                JOIN users u
                    ON u.id = ar.user_id
                WHERE ar.status = 'pending'
                ORDER BY ar.created_at ASC
                """
            ).fetchall()

            return [dict(row) for row in rows]

    except Exception:
        logger.exception("Could not retrieve access requests")
        return []


def approve_access_request(
    request_id: int,
    admin_username: str,
    approved: bool,
) -> Tuple[bool, str]:
    """
    Approve or reject an access request atomically.
    """
    if request_id <= 0 or not admin_username:
        return False, "Invalid request"

    try:
        with get_db_connection() as conn:
            request = conn.execute(
                """
                SELECT
                    ar.id,
                    ar.user_id,
                    ar.from_role,
                    ar.to_role,
                    ar.status,
                    u.username,
                    u.role AS current_role
                FROM access_requests ar
                JOIN users u
                    ON u.id = ar.user_id
                WHERE ar.id = ?
                  AND ar.status = 'pending'
                """,
                (request_id,),
            ).fetchone()

            if not request:
                return False, "Request not found or already processed"

            target_username = request["username"]
            old_role = request["current_role"]
            requested_role = request["to_role"]

            # Current database state wins over stale request state.
            if old_role != request["from_role"]:
                conn.execute(
                    """
                    UPDATE access_requests
                    SET status = 'rejected',
                        approved_by = ?,
                        approved_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (admin_username, request_id),
                )

                conn.commit()

                return False, "Request is stale because the user's role changed"

            status = "approved" if approved else "rejected"

            if approved:
                conn.execute(
                    """
                    UPDATE users
                    SET role = ?
                    WHERE id = ?
                    """,
                    (requested_role, request["user_id"]),
                )

            conn.execute(
                """
                UPDATE access_requests
                SET status = ?,
                    approved_by = ?,
                    approved_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    status,
                    admin_username,
                    request_id,
                ),
            )

            conn.commit()

        _log_audit(
            username=admin_username,
            action=(
                "ACCESS_REQUEST_APPROVED"
                if approved
                else "ACCESS_REQUEST_REJECTED"
            ),
            details=(
                f"Request #{request_id}: "
                f"{target_username} "
                f"{old_role} -> {requested_role}"
            ),
        )

        return True, f"Access request {status}"

    except Exception:
        logger.exception("Access request decision failed")
        return False, "Unable to process access request"


# =============================================================================
# AUDIT QUERYING
# =============================================================================

def get_audit_log(
    username: Optional[str] = None,
    limit: int = 100,
    action_filter: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Retrieve audit events using safe parameterized filtering."""
    limit = max(1, min(limit, 5000))

    try:
        with get_db_connection() as conn:
            query = """
                SELECT *
                FROM audit_log
                WHERE 1 = 1
            """

            params: List[Any] = []

            if username:
                query += " AND username = ?"
                params.append(username)

            if action_filter:
                query += " AND action LIKE ?"
                params.append(f"%{action_filter}%")

            if start_date:
                query += " AND timestamp >= ?"
                params.append(
                    start_date.isoformat(sep=" ")
                    if isinstance(start_date, datetime)
                    else str(start_date)
                )

            if end_date:
                query += " AND timestamp <= ?"
                params.append(
                    end_date.isoformat(sep=" ")
                    if isinstance(end_date, datetime)
                    else str(end_date)
                )

            query += """
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            """

            params.append(limit)

            rows = conn.execute(
                query,
                tuple(params),
            ).fetchall()

            return [dict(row) for row in rows]

    except Exception:
        logger.exception("Audit log retrieval failed")
        return []


# Backward-compatible alias.
def _get_audit_logs(
    limit: int = 100,
    username: Optional[str] = None,
    action_filter: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Backward-compatible audit-log helper."""
    return get_audit_log(
        username=username,
        limit=limit,
        action_filter=action_filter,
        start_date=start_date,
        end_date=end_date,
    )


# =============================================================================
# RBAC
# =============================================================================

def check_permission(
    user_role: str,
    required_permission: str,
) -> bool:
    """Return True if a role has a permission."""
    return settings.has_permission(
        user_role,
        required_permission,
    )


def require_permission(
    user_role: str,
    permission: str,
) -> Tuple[bool, str]:
    """Return permission result with a useful message."""
    if check_permission(user_role, permission):
        return True, "Permission granted"

    role_config = settings.ROLES.get(user_role, {})
    permissions = role_config.get("permissions", [])

    return (
        False,
        (
            f"Access denied for permission '{permission}'. "
            f"Current role: {user_role}. "
            f"Available permissions: "
            f"{', '.join(permissions) if permissions else 'none'}"
        ),
    )


# =============================================================================
# DATABASE STATISTICS
# =============================================================================

def get_database_stats() -> Dict[str, Any]:
    """Return operational database statistics."""
    try:
        with get_db_connection() as conn:
            users_by_role = conn.execute(
                """
                SELECT
                    role,
                    COUNT(*) AS count,
                    SUM(
                        CASE
                            WHEN is_active = 1 THEN 1
                            ELSE 0
                        END
                    ) AS active
                FROM users
                GROUP BY role
                ORDER BY role
                """
            ).fetchall()

            top_actions = conn.execute(
                """
                SELECT
                    action,
                    COUNT(*) AS count
                FROM audit_log
                GROUP BY action
                ORDER BY count DESC
                LIMIT 10
                """
            ).fetchall()

            pending_requests = conn.execute(
                """
                SELECT COUNT(*)
                FROM access_requests
                WHERE status = 'pending'
                """
            ).fetchone()[0]

            total_documents = conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]

            active_documents = conn.execute(
                """
                SELECT COUNT(*)
                FROM documents
                WHERE status = 'active'
                """
            ).fetchone()[0]

            total_queries = conn.execute(
                """
                SELECT COUNT(*)
                FROM audit_log
                WHERE action = 'QUERY_EXECUTED'
                """
            ).fetchone()[0]

            blocked_queries = conn.execute(
                """
                SELECT COUNT(*)
                FROM audit_log
                WHERE blocked = 1
                """
            ).fetchone()[0]

            return {
                "users_by_role": [
                    dict(row)
                    for row in users_by_role
                ],
                "top_actions": [
                    dict(row)
                    for row in top_actions
                ],
                "pending_requests": pending_requests,
                "total_documents": total_documents,
                "active_documents": active_documents,
                "total_queries": total_queries,
                "blocked_events": blocked_queries,
                "db_size_mb": round(
                    DB_FILE.stat().st_size / (1024 * 1024),
                    2,
                )
                if DB_FILE.exists()
                else 0.0,
                "database_path": str(DB_FILE),
            }

    except Exception as exc:
        logger.exception("Could not calculate database statistics")
        return {"error": str(exc)}


# =============================================================================
# DATABASE BACKUP
# =============================================================================

def backup_database(
    backup_path: Optional[Path] = None,
) -> bool:
    """Create a consistent SQLite backup using SQLite's backup API."""
    try:
        if not DB_FILE.exists():
            logger.warning("Cannot backup database; file does not exist")
            return False

        if backup_path is None:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            backup_path = (
                DB_FILE.parent
                / f"{DB_FILE.stem}_backup_{timestamp}.db"
            )

        backup_path = Path(backup_path).resolve()
        backup_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with sqlite3.connect(str(DB_FILE)) as source:
            with sqlite3.connect(str(backup_path)) as destination:
                source.backup(destination)

        logger.info(
            "Database backup created: %s",
            backup_path,
        )

        return True

    except Exception:
        logger.exception("Database backup failed")
        return False


# =============================================================================
# INITIALIZATION
# =============================================================================

_DB_READY = init_database()


def database_ready() -> bool:
    """Return current database readiness state."""
    return _DB_READY and DB_FILE.exists()


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# Existing app.py used private function names. Keep aliases temporarily so
# the application can be migrated incrementally.

_hash_password = hash_password
_verify_password = verify_password
_register_user = register_user
_login_user = login_user
_get_all_users = get_all_users
_update_user_role = update_user_role
_delete_user = delete_user


# =============================================================================
# DEVELOPMENT DIAGNOSTICS
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("PolicyGuard AI - Authentication Database Diagnostics")
    print("=" * 72)

    print(f"Database: {DB_FILE}")
    print(f"Ready   : {database_ready()}")

    if not database_ready():
        raise SystemExit("Database initialization failed")

    stats = get_database_stats()

    print()
    print("Database Statistics")
    print("-" * 72)

    if "error" in stats:
        print(f"ERROR: {stats['error']}")
    else:
        print(f"Database size     : {stats.get('db_size_mb', 0)} MB")
        print(f"Documents         : {stats.get('total_documents', 0)}")
        print(f"Active documents  : {stats.get('active_documents', 0)}")
        print(f"Queries           : {stats.get('total_queries', 0)}")
        print(f"Blocked events    : {stats.get('blocked_events', 0)}")
        print(
            f"Pending requests  : "
            f"{stats.get('pending_requests', 0)}"
        )

        print()
        print("Users by role:")
        for item in stats.get("users_by_role", []):
            print(
                f"  {item['role']:8} "
                f"total={item['count']} "
                f"active={item['active']}"
            )

    print()
    print("RBAC")
    print("-" * 72)

    for role in settings.ROLES:
        print(f"{role.upper()}:")
        permissions = settings.ROLES[role].get(
            "permissions",
            [],
        )

        for permission in permissions:
            print(f"  - {permission}")

    print()
    print("=" * 72)
    print("Database diagnostics complete.")
    print("=" * 72)



# =============================================================================
# ENTERPRISE HARDENING LAYER
# =============================================================================
#
# The original service API is intentionally preserved above.  The definitions
# below harden the service in-place so existing app.py imports remain
# compatible while privileged mutations become server-side authorized.
#
# Key guarantees:
# - tenant/organization identity is stored with security-sensitive records
# - privileged mutations require an active administrator in the same tenant
# - client-supplied roles are never trusted for authorization
# - legacy databases are upgraded without deleting data
# - init_db() exists as a stable bootstrap compatibility API
# - password hashes are never returned by user-query APIs
# - audit records are tenant scoped
# =============================================================================

ENTERPRISE_SCHEMA_VERSION = 2

# A process-local dummy bcrypt hash is used only to reduce username-enumeration
# timing differences on failed logins.  It contains no usable application
# credential and is regenerated on each process start.
_DUMMY_BCRYPT_HASH = bcrypt.hashpw(
    os.urandom(32),
    bcrypt.gensalt(rounds=12),
)


def _enterprise_add_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    """Add a migration column only when it is absent."""
    columns = _get_table_columns(conn, table)
    if column not in columns:
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
        )


def _enterprise_schema_upgrade() -> bool:
    """
    Upgrade legacy installations in-place.

    This function is deliberately non-destructive: it never drops or recreates
    existing authentication data.
    """
    try:
        with get_db_connection() as conn:
            conn.executescript(_SCHEMA_SQL)

            _enterprise_add_column(
                conn, "users", "organization_id",
                "TEXT NOT NULL DEFAULT 'default'",
            )
            _enterprise_add_column(
                conn, "audit_log", "organization_id",
                "TEXT NOT NULL DEFAULT 'default'",
            )
            _enterprise_add_column(
                conn, "documents", "organization_id",
                "TEXT NOT NULL DEFAULT 'default'",
            )
            _enterprise_add_column(
                conn, "access_requests", "organization_id",
                "TEXT NOT NULL DEFAULT 'default'",
            )
            _enterprise_add_column(
                conn, "query_cache", "organization_id",
                "TEXT NOT NULL DEFAULT 'default'",
            )

            for table in (
                "users",
                "audit_log",
                "documents",
                "access_requests",
                "query_cache",
            ):
                conn.execute(
                    f"""
                    UPDATE {table}
                    SET organization_id = ?
                    WHERE organization_id IS NULL
                       OR TRIM(organization_id) = ''
                    """,
                    (DEFAULT_ORGANIZATION_ID,),
                )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_users_org_role
                ON users(organization_id, role)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_org_timestamp
                ON audit_log(organization_id, timestamp)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_org_status
                ON documents(organization_id, status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_requests_org_status
                ON access_requests(organization_id, status)
                """
            )

            conn.execute(
                f"PRAGMA user_version = {ENTERPRISE_SCHEMA_VERSION}"
            )
            conn.commit()

        logger.info(
            "Enterprise auth schema ready: version=%s db=%s",
            ENTERPRISE_SCHEMA_VERSION,
            DB_FILE,
        )
        return True

    except Exception:
        logger.exception("Enterprise auth schema upgrade failed")
        return False


def init_db() -> bool:
    """
    Stable database bootstrap API.

    Kept intentionally simple so maintenance scripts can safely run:
        from src.auth.database import init_db
        init_db()
    """
    global _ENTERPRISE_DB_READY
    _ENTERPRISE_DB_READY = _enterprise_schema_upgrade()
    return _ENTERPRISE_DB_READY


# Run the non-destructive upgrade once on import.  Do not delete or recreate
# the existing database if migration fails.
_ENTERPRISE_DB_READY = _enterprise_schema_upgrade()


def database_ready() -> bool:
    """Return whether the authentication database is initialized and usable."""
    return bool(
        DB_FILE.exists()
        and _ENTERPRISE_DB_READY
    )


def _normalize_org(organization_id: Optional[str]) -> str:
    """Normalize and validate a tenant identifier."""
    value = (organization_id or DEFAULT_ORGANIZATION_ID).strip()
    if not value:
        value = DEFAULT_ORGANIZATION_ID

    if len(value) > ORGANIZATION_MAX_LENGTH:
        raise ValueError("Organization ID is too long")

    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise ValueError("Organization ID contains unsupported characters")

    return value


def get_user_role(username: str) -> Optional[str]:
    """Return the active user's authoritative database role."""
    if not username:
        return None

    try:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT role
                FROM users
                WHERE username = ?
                  AND is_active = 1
                """,
                (username.strip(),),
            ).fetchone()

        return row["role"] if row else None

    except Exception:
        logger.exception("Could not retrieve user role")
        return None


def get_user_organization(username: str) -> Optional[str]:
    """Return the active user's tenant."""
    if not username:
        return None

    try:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT organization_id
                FROM users
                WHERE username = ?
                  AND is_active = 1
                """,
                (username.strip(),),
            ).fetchone()

        return row["organization_id"] if row else None

    except Exception:
        logger.exception("Could not retrieve user organization")
        return None


def is_admin(username: str) -> bool:
    """Authoritative server-side administrator check."""
    return get_user_role(username) == "admin"


def _require_admin_actor(username: str) -> Optional[str]:
    """
    Return the admin's tenant when the actor is an active administrator.

    Returning the tenant makes it harder for callers to accidentally perform
    a privileged action across organization boundaries.
    """
    if not username:
        return None

    try:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT role, organization_id
                FROM users
                WHERE username = ?
                  AND is_active = 1
                """,
                (username.strip(),),
            ).fetchone()

        if not row or row["role"] != "admin":
            return None

        return row["organization_id"]

    except Exception:
        logger.exception("Administrator authorization check failed")
        return None


def _audit_v2(
    username: str,
    action: str,
    details: str = "",
    *,
    organization_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
    model_used: Optional[str] = None,
    query_preview: Optional[str] = None,
    threat_type: Optional[str] = None,
    blocked: bool = False,
) -> None:
    """Tenant-aware audit writer that never breaks the primary operation."""
    try:
        org = _normalize_org(organization_id)
        details = str(details or "")[:AUDIT_PREVIEW_MAX_LENGTH]
        query_preview = (
            str(query_preview)[:AUDIT_PREVIEW_MAX_LENGTH]
            if query_preview is not None
            else None
        )

        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (
                    username,
                    action,
                    organization_id,
                    timestamp,
                    details,
                    ip_address,
                    tokens_used,
                    cost_usd,
                    model_used,
                    query_preview,
                    threat_type,
                    blocked
                )
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    action,
                    org,
                    details,
                    ip_address,
                    max(0, int(tokens_used)),
                    max(0.0, float(cost_usd)),
                    model_used,
                    query_preview,
                    threat_type,
                    1 if blocked else 0,
                ),
            )
            conn.commit()

    except Exception:
        logger.exception("Tenant-aware audit logging failed")


def register_user(
    username: str,
    password: str,
    role: str = "viewer",
    organization_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Register a new account.

    Public registration is deliberately restricted to Viewer accounts.
    Editors/Admins must be provisioned or promoted by an administrator.
    """
    username = (username or "").strip()

    try:
        org = _normalize_org(organization_id)
    except ValueError as exc:
        return False, str(exc)

    username_error = _validate_username(username)
    if username_error:
        return False, username_error

    password_error = _validate_password(password)
    if password_error:
        return False, password_error

    if role != "viewer":
        return False, "New accounts can only be registered as Viewer"

    password_hash = hash_password(password)
    if not password_hash:
        return False, "Unable to securely hash password"

    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    username,
                    password_hash,
                    role,
                    organization_id
                )
                VALUES (?, ?, ?, ?)
                """,
                (username, password_hash, role, org),
            )
            conn.commit()

        _audit_v2(
            username,
            "USER_REGISTERED",
            "New Viewer account registered",
            organization_id=org,
        )
        return True, "Registration successful. Please sign in."

    except sqlite3.IntegrityError:
        return False, "Username already exists"

    except Exception:
        logger.exception("User registration failed")
        return False, "Registration failed due to an internal error"


def login_user(
    username: str,
    password: str,
) -> Tuple[bool, str, Optional[str]]:
    """
    Authenticate against the authoritative database role.

    Failed authentication never reveals whether a username exists.
    """
    username = (username or "").strip()

    if not username or not password:
        return False, "Username and password are required", None

    try:
        with get_db_connection() as conn:
            user = conn.execute(
                """
                SELECT
                    id,
                    username,
                    password_hash,
                    role,
                    organization_id,
                    is_active,
                    failed_login_attempts,
                    last_failed_login
                FROM users
                WHERE username = ?
                """,
                (username,),
            ).fetchone()

            if not user:
                # Keep the failure path cryptographically similar to an
                # existing-user password verification path without embedding
                # a reusable password/hash pair in source code.
                try:
                    bcrypt.checkpw(
                        password.encode("utf-8"),
                        _DUMMY_BCRYPT_HASH,
                    )
                except Exception:
                    pass

                _audit_v2(
                    username,
                    "LOGIN_FAILED",
                    "Invalid credentials",
                    blocked=True,
                    organization_id=DEFAULT_ORGANIZATION_ID,
                )
                return False, "Invalid username or password", None

            org = user["organization_id"]

            if not user["is_active"]:
                _audit_v2(
                    username,
                    "LOGIN_FAILED",
                    "Account inactive",
                    blocked=True,
                    organization_id=org,
                )
                return False, "Account is inactive", None

            failed_attempts = int(user["failed_login_attempts"] or 0)
            last_failed = user["last_failed_login"]

            if failed_attempts >= MAX_FAILED_LOGIN_ATTEMPTS:
                lockout_end = None

                if isinstance(last_failed, datetime):
                    lockout_end = last_failed + timedelta(
                        minutes=LOCKOUT_MINUTES
                    )
                elif last_failed:
                    try:
                        lockout_end = (
                            datetime.fromisoformat(str(last_failed))
                            + timedelta(minutes=LOCKOUT_MINUTES)
                        )
                    except ValueError:
                        logger.warning(
                            "Could not parse last_failed_login for %s",
                            username,
                        )

                if lockout_end and datetime.now() < lockout_end:
                    _audit_v2(
                        username,
                        "LOGIN_BLOCKED",
                        "Temporary brute-force lockout",
                        blocked=True,
                        organization_id=org,
                    )
                    return (
                        False,
                        "Account temporarily locked. Try again later.",
                        None,
                    )

                conn.execute(
                    """
                    UPDATE users
                    SET failed_login_attempts = 0,
                        last_failed_login = NULL
                    WHERE id = ?
                    """,
                    (user["id"],),
                )

            valid = verify_password(user["password_hash"], password)

            if not valid:
                conn.execute(
                    """
                    UPDATE users
                    SET failed_login_attempts =
                            failed_login_attempts + 1,
                        last_failed_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (user["id"],),
                )
                conn.commit()

                _audit_v2(
                    username,
                    "LOGIN_FAILED",
                    "Invalid credentials",
                    blocked=True,
                    organization_id=org,
                )
                return False, "Invalid username or password", None

            conn.execute(
                """
                UPDATE users
                SET failed_login_attempts = 0,
                    last_failed_login = NULL,
                    last_login = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (user["id"],),
            )
            conn.commit()

        _audit_v2(
            username,
            "LOGIN_SUCCESS",
            f"Role: {user['role']}",
            organization_id=org,
        )
        return True, f"Welcome back, {username}!", user["role"]

    except Exception:
        logger.exception("Login failed")
        return False, "Authentication service temporarily unavailable", None


def get_user_by_username(
    username: str,
    organization_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a user without exposing the password hash."""
    try:
        with get_db_connection() as conn:
            if organization_id:
                org = _normalize_org(organization_id)
                row = conn.execute(
                    """
                    SELECT
                        id,
                        username,
                        role,
                        organization_id,
                        created_at,
                        last_login,
                        is_active
                    FROM users
                    WHERE username = ?
                      AND organization_id = ?
                    """,
                    (username.strip(), org),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT
                        id,
                        username,
                        role,
                        organization_id,
                        created_at,
                        last_login,
                        is_active
                    FROM users
                    WHERE username = ?
                    """,
                    (username.strip(),),
                ).fetchone()
        return dict(row) if row else None
    except Exception:
        logger.exception("Could not retrieve user")
        return None


def get_user_by_id(
    user_id: int,
    organization_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a user by ID without exposing the password hash."""
    if user_id <= 0:
        return None

    try:
        with get_db_connection() as conn:
            if organization_id:
                org = _normalize_org(organization_id)
                row = conn.execute(
                    """
                    SELECT
                        id,
                        username,
                        role,
                        organization_id,
                        created_at,
                        last_login,
                        is_active
                    FROM users
                    WHERE id = ?
                      AND organization_id = ?
                    """,
                    (user_id, org),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT
                        id,
                        username,
                        role,
                        organization_id,
                        created_at,
                        last_login,
                        is_active
                    FROM users
                    WHERE id = ?
                    """,
                    (user_id,),
                ).fetchone()
        return dict(row) if row else None
    except Exception:
        logger.exception("Could not retrieve user by ID")
        return None


def get_all_users(
    organization_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return users, optionally constrained to one tenant."""
    try:
        with get_db_connection() as conn:
            if organization_id:
                org = _normalize_org(organization_id)
                rows = conn.execute(
                    """
                    SELECT
                        id,
                        username,
                        role,
                        organization_id,
                        created_at,
                        last_login,
                        is_active
                    FROM users
                    WHERE organization_id = ?
                    ORDER BY created_at DESC
                    """,
                    (org,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        id,
                        username,
                        role,
                        organization_id,
                        created_at,
                        last_login,
                        is_active
                    FROM users
                    ORDER BY created_at DESC
                    """
                ).fetchall()

        return [dict(row) for row in rows]
    except Exception:
        logger.exception("Could not retrieve users")
        return []


def update_user_role(
    user_id: int,
    new_role: str,
    updated_by: str,
) -> bool:
    """Change a role only when an active same-tenant admin requests it."""
    if user_id <= 0 or not _validate_role(new_role):
        return False

    actor_org = _require_admin_actor(updated_by)
    if not actor_org:
        logger.warning("Unauthorized role-change attempt by %s", updated_by)
        return False

    try:
        with get_db_connection() as conn:
            target = conn.execute(
                """
                SELECT username, role, organization_id
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

            if not target or target["organization_id"] != actor_org:
                return False

            old_role = target["role"]
            target_username = target["username"]

            if old_role == new_role:
                return True

            if old_role == "admin" and new_role != "admin":
                admin_count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM users
                    WHERE organization_id = ?
                      AND role = 'admin'
                      AND is_active = 1
                    """,
                    (actor_org,),
                ).fetchone()[0]

                if admin_count <= 1:
                    logger.warning(
                        "Attempt to demote final active admin: %s",
                        target_username,
                    )
                    return False

            conn.execute(
                "UPDATE users SET role = ? WHERE id = ?",
                (new_role, user_id),
            )
            conn.commit()

        _audit_v2(
            updated_by,
            "ROLE_CHANGED",
            f"User '{target_username}' role changed "
            f"{old_role} -> {new_role}",
            organization_id=actor_org,
        )
        return True

    except Exception:
        logger.exception("User role update failed")
        return False


def update_user_active_status(
    user_id: int,
    is_active: bool,
    updated_by: str,
) -> bool:
    """Activate/deactivate a same-tenant user as an administrator."""
    if user_id <= 0:
        return False

    actor_org = _require_admin_actor(updated_by)
    if not actor_org:
        return False

    try:
        with get_db_connection() as conn:
            target = conn.execute(
                """
                SELECT username, role, is_active, organization_id
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

            if not target or target["organization_id"] != actor_org:
                return False

            if (
                not is_active
                and target["role"] == "admin"
                and target["is_active"]
            ):
                admin_count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM users
                    WHERE organization_id = ?
                      AND role = 'admin'
                      AND is_active = 1
                    """,
                    (actor_org,),
                ).fetchone()[0]

                if admin_count <= 1:
                    return False

            conn.execute(
                "UPDATE users SET is_active = ? WHERE id = ?",
                (1 if is_active else 0, user_id),
            )
            conn.commit()

        _audit_v2(
            updated_by,
            "USER_ACTIVATED" if is_active else "USER_DEACTIVATED",
            f"User '{target['username']}' account status changed",
            organization_id=actor_org,
        )
        return True

    except Exception:
        logger.exception("Account status update failed")
        return False


def delete_user(
    user_id: int,
    deleted_by: str,
) -> bool:
    """
    Permanently delete a user as a same-tenant admin.

    Prefer deactivation for normal enterprise lifecycle management.
    """
    if user_id <= 0:
        return False

    actor_org = _require_admin_actor(deleted_by)
    if not actor_org:
        return False

    try:
        with get_db_connection() as conn:
            target = conn.execute(
                """
                SELECT username, role, organization_id
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

            if not target or target["organization_id"] != actor_org:
                return False

            if target["role"] == "admin":
                admin_count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM users
                    WHERE organization_id = ?
                      AND role = 'admin'
                      AND is_active = 1
                    """,
                    (actor_org,),
                ).fetchone()[0]

                if admin_count <= 1:
                    return False

            conn.execute(
                "DELETE FROM users WHERE id = ?",
                (user_id,),
            )
            conn.commit()

        _audit_v2(
            deleted_by,
            "USER_DELETED",
            f"User '{target['username']}' permanently deleted",
            organization_id=actor_org,
        )
        return True

    except Exception:
        logger.exception("User deletion failed")
        return False


def create_access_request(
    user_id: int,
    from_role: str,
    to_role: str,
    reason: str,
) -> Tuple[bool, str]:
    """Create a role request using the user's actual current role."""
    reason = (reason or "").strip()

    if user_id <= 0:
        return False, "Invalid user ID"
    if not _validate_role(from_role) or not _validate_role(to_role):
        return False, "Invalid role"
    if len(reason) < 10:
        return False, "Please provide a reason of at least 10 characters"
    if len(reason) > ACCESS_REASON_MAX_LENGTH:
        return False, "Reason is too long"
    if not settings.role_at_least(to_role, from_role):
        return False, "Requested role cannot be lower than current role"
    if from_role == to_role:
        return False, "You already have this role"

    try:
        with get_db_connection() as conn:
            user = conn.execute(
                """
                SELECT username, role, is_active, organization_id
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

            if not user:
                return False, "User not found"
            if not user["is_active"]:
                return False, "Inactive users cannot request access"
            if user["role"] != from_role:
                return False, "Current role does not match account"

            existing = conn.execute(
                """
                SELECT id
                FROM access_requests
                WHERE user_id = ?
                  AND status = 'pending'
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()

            if existing:
                return False, "You already have a pending access request"

            conn.execute(
                """
                INSERT INTO access_requests (
                    user_id,
                    organization_id,
                    from_role,
                    to_role,
                    reason,
                    status
                )
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (
                    user_id,
                    user["organization_id"],
                    from_role,
                    to_role,
                    reason,
                ),
            )
            conn.commit()

        _audit_v2(
            user["username"],
            "ACCESS_REQUEST_CREATED",
            f"Requested role {from_role} -> {to_role}; "
            f"reason={reason[:200]}",
            organization_id=user["organization_id"],
        )
        return True, "Access request submitted for admin approval"

    except Exception:
        logger.exception("Access request creation failed")
        return False, "Unable to create access request"


def get_pending_access_requests(
    organization_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return pending requests, optionally restricted to one tenant."""
    try:
        with get_db_connection() as conn:
            if organization_id:
                org = _normalize_org(organization_id)
                rows = conn.execute(
                    """
                    SELECT
                        ar.id,
                        ar.user_id,
                        ar.organization_id,
                        ar.from_role,
                        ar.to_role,
                        ar.reason,
                        ar.status,
                        ar.created_at,
                        ar.approved_by,
                        ar.approved_at,
                        u.username
                    FROM access_requests ar
                    JOIN users u ON u.id = ar.user_id
                    WHERE ar.status = 'pending'
                      AND ar.organization_id = ?
                    ORDER BY ar.created_at ASC
                    """,
                    (org,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        ar.id,
                        ar.user_id,
                        ar.organization_id,
                        ar.from_role,
                        ar.to_role,
                        ar.reason,
                        ar.status,
                        ar.created_at,
                        ar.approved_by,
                        ar.approved_at,
                        u.username
                    FROM access_requests ar
                    JOIN users u ON u.id = ar.user_id
                    WHERE ar.status = 'pending'
                    ORDER BY ar.created_at ASC
                    """
                ).fetchall()

        return [dict(row) for row in rows]

    except Exception:
        logger.exception("Could not retrieve access requests")
        return []


def approve_access_request(
    request_id: int,
    admin_username: str,
    approved: bool,
) -> Tuple[bool, str]:
    """Approve/reject a request atomically as a same-tenant admin."""
    if request_id <= 0:
        return False, "Invalid request"

    actor_org = _require_admin_actor(admin_username)
    if not actor_org:
        return False, "Administrator approval is required"

    try:
        with get_db_connection() as conn:
            request = conn.execute(
                """
                SELECT
                    ar.id,
                    ar.user_id,
                    ar.organization_id,
                    ar.from_role,
                    ar.to_role,
                    ar.status,
                    u.username,
                    u.role AS current_role,
                    u.is_active
                FROM access_requests ar
                JOIN users u ON u.id = ar.user_id
                WHERE ar.id = ?
                  AND ar.status = 'pending'
                """,
                (request_id,),
            ).fetchone()

            if not request:
                return False, "Request not found or already processed"

            if request["organization_id"] != actor_org:
                return False, "Request belongs to another organization"

            if not request["is_active"]:
                approved = False

            if request["current_role"] != request["from_role"]:
                conn.execute(
                    """
                    UPDATE access_requests
                    SET status = 'rejected',
                        approved_by = ?,
                        approved_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                      AND status = 'pending'
                    """,
                    (admin_username, request_id),
                )
                conn.commit()
                return False, "Request is stale because the user's role changed"

            status = "approved" if approved else "rejected"

            if approved:
                conn.execute(
                    """
                    UPDATE users
                    SET role = ?
                    WHERE id = ?
                      AND organization_id = ?
                    """,
                    (
                        request["to_role"],
                        request["user_id"],
                        actor_org,
                    ),
                )

            conn.execute(
                """
                UPDATE access_requests
                SET status = ?,
                    approved_by = ?,
                    approved_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND organization_id = ?
                  AND status = 'pending'
                """,
                (status, admin_username, request_id, actor_org),
            )
            conn.commit()

        _audit_v2(
            admin_username,
            "ACCESS_REQUEST_APPROVED" if approved else "ACCESS_REQUEST_REJECTED",
            f"Request #{request_id}: {request['username']} "
            f"{request['from_role']} -> {request['to_role']}",
            organization_id=actor_org,
        )
        return True, f"Access request {status}"

    except Exception:
        logger.exception("Access request decision failed")
        return False, "Unable to process access request"


def get_audit_log(
    username: Optional[str] = None,
    limit: int = 100,
    action_filter: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    organization_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query audit records with optional tenant scoping."""
    limit = max(1, min(int(limit), 5000))

    try:
        with get_db_connection() as conn:
            query = "SELECT * FROM audit_log WHERE 1 = 1"
            params: List[Any] = []

            if organization_id:
                query += " AND organization_id = ?"
                params.append(_normalize_org(organization_id))

            if username:
                query += " AND username = ?"
                params.append(username)

            if action_filter:
                query += " AND action LIKE ?"
                params.append(f"%{action_filter}%")

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    except Exception:
        logger.exception("Audit log retrieval failed")
        return []


def _get_audit_logs(
    limit: int = 100,
    username: Optional[str] = None,
    action_filter: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    organization_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Backward-compatible audit-log helper."""
    return get_audit_log(
        username=username,
        limit=limit,
        action_filter=action_filter,
        start_date=start_date,
        end_date=end_date,
        organization_id=organization_id,
    )


# Preserve legacy private names used by older app.py code.
_hash_password = hash_password
_verify_password = verify_password
_register_user = register_user
_login_user = login_user
_get_all_users = get_all_users
_update_user_role = update_user_role
_delete_user = delete_user
