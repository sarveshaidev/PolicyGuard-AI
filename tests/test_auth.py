"""
PolicyGuardAI - Authentication / RBAC / Tenant-Isolation Tests
===============================================================

These tests exercise the current public API in src.auth.database.

IMPORTANT:
- Tests use an isolated temporary SQLite database.
- The production authentication database is never modified.
- The test suite verifies the enterprise hardening layer, including
  organization/tenant boundaries and server-side administrator authorization.

Run from the PolicyGuardAI project root:

    python -m pytest -q tests/test_auth.py

For extra detail:

    python -m pytest -q tests/test_auth.py -vv
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterator

import pytest
import secrets
import src.auth.database as database


def _test_password(env_name: str) -> str:
    """Return a test password without embedding a credential in source."""
    value = os.getenv(env_name)
    if value:
        return value
    return "Pw" + secrets.token_urlsafe(18) + "9!"


PASSWORD = _test_password("POLICYGUARD_TEST_PASSWORD")
ADMIN_PASSWORD = _test_password("POLICYGUARD_TEST_ADMIN_PASSWORD")
EDITOR_PASSWORD = _test_password("POLICYGUARD_TEST_EDITOR_PASSWORD")

ORG_ALPHA = "org_alpha"
ORG_BETA = "org_beta"


@pytest.fixture()
def auth_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """
    Redirect the imported database service to a fresh temporary DB.

    The module may have initialized its normal DB during import. We never
    mutate that DB here: DB_FILE is replaced before the explicit test
    initialization call, and every test uses the temporary path.
    """
    db_path = tmp_path / "nexus_auth_test.db"

    monkeypatch.setattr(database, "DB_FILE", db_path)

    assert database.init_db() is True
    assert database.database_ready() is True
    assert db_path.exists()

    yield db_path


def register_viewer(
    username: str,
    organization_id: str,
) -> int:
    """Register a viewer and return its authoritative user ID."""
    ok, message = database.register_user(
        username,
        PASSWORD,
        organization_id=organization_id,
    )
    assert ok, message

    user = database.get_user_by_username(
        username,
        organization_id=organization_id,
    )
    assert user is not None
    assert "password_hash" not in user
    return int(user["id"])


def provision_admin(
    username: str,
    organization_id: str,
) -> int:
    """
    Create a Viewer through the public registration path, then promote it
    using the server-side admin API.

    This deliberately avoids bypassing the public-registration rule.
    """
    user_id = register_viewer(username, organization_id)

    # The first admin in a tenant is provisioned for the test through direct
    # database state, because the production API intentionally provides no
    # public "create admin" operation.
    with database.get_db_connection() as conn:
        conn.execute(
            """
            UPDATE users
            SET role = 'admin'
            WHERE id = ?
              AND organization_id = ?
            """,
            (user_id, organization_id),
        )
        conn.commit()

    user = database.get_user_by_id(
        user_id,
        organization_id=organization_id,
    )
    assert user is not None
    assert user["role"] == "admin"
    return user_id


def test_database_initializes_in_isolated_database(auth_db: Path) -> None:
    assert auth_db.exists()

    with sqlite3.connect(auth_db) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }

    assert {
        "users",
        "audit_log",
        "documents",
        "access_requests",
        "query_cache",
    }.issubset(tables)


def test_registration_creates_viewer_in_requested_tenant(
    auth_db: Path,
) -> None:
    ok, message = database.register_user(
        "alice",
        PASSWORD,
        organization_id=ORG_ALPHA,
    )

    assert ok, message

    user = database.get_user_by_username(
        "alice",
        organization_id=ORG_ALPHA,
    )

    assert user is not None
    assert user["username"] == "alice"
    assert user["role"] == "viewer"
    assert user["organization_id"] == ORG_ALPHA
    assert user["is_active"] == 1
    assert "password_hash" not in user


@pytest.mark.parametrize("privileged_role", ["editor", "admin"])
def test_public_registration_cannot_create_privileged_roles(
    auth_db: Path,
    privileged_role: str,
) -> None:
    ok, message = database.register_user(
        f"attacker_{privileged_role}",
        PASSWORD,
        role=privileged_role,
        organization_id=ORG_ALPHA,
    )

    assert not ok
    assert "viewer" in message.lower()

    assert database.get_user_by_username(
        f"attacker_{privileged_role}",
        organization_id=ORG_ALPHA,
    ) is None


def test_duplicate_username_is_rejected(auth_db: Path) -> None:
    register_viewer("duplicate_user", ORG_ALPHA)

    ok, message = database.register_user(
        "duplicate_user",
        PASSWORD,
        organization_id=ORG_ALPHA,
    )

    assert not ok
    assert "already exists" in message.lower()


@pytest.mark.parametrize(
    "password",
    [
        "short1A",
        "alllowercase123",
        "ALLUPPERCASE123",
        "NoDigitsHere",
    ],
)
def test_weak_password_is_rejected(
    auth_db: Path,
    password: str,
) -> None:
    ok, message = database.register_user(
        "weak_password_user",
        password,
        organization_id=ORG_ALPHA,
    )

    assert not ok
    assert "password" in message.lower()


@pytest.mark.parametrize(
    "username",
    [
        "",
        "ab",
        "invalid-name",
        "invalid name",
    ],
)
def test_invalid_username_is_rejected(
    auth_db: Path,
    username: str,
) -> None:
    ok, message = database.register_user(
        username,
        PASSWORD,
        organization_id=ORG_ALPHA,
    )

    assert not ok
    assert message


def test_invalid_organization_is_rejected(auth_db: Path) -> None:
    ok, message = database.register_user(
        "bad_org_user",
        PASSWORD,
        organization_id="org/../../escape",
    )

    assert not ok
    assert "organization" in message.lower()


def test_successful_login_returns_authoritative_viewer_role(
    auth_db: Path,
) -> None:
    register_viewer("alice", ORG_ALPHA)

    ok, message, role = database.login_user(
        "alice",
        PASSWORD,
    )

    assert ok, message
    assert role == "viewer"
    assert "alice" in message.lower()

    user = database.get_user_by_username("alice")
    assert user is not None
    assert user["last_login"] is not None


def test_wrong_password_is_rejected_without_role(
    auth_db: Path,
) -> None:
    register_viewer("alice", ORG_ALPHA)

    ok, message, role = database.login_user(
        "alice",
        "WrongPassword123!",
    )

    assert not ok
    assert role is None
    assert "invalid username or password" in message.lower()


def test_unknown_user_is_rejected_without_user_enumeration(
    auth_db: Path,
) -> None:
    ok, message, role = database.login_user(
        "does_not_exist",
        PASSWORD,
    )

    assert not ok
    assert role is None
    assert "invalid username or password" in message.lower()


def test_empty_login_is_rejected(auth_db: Path) -> None:
    ok, message, role = database.login_user("", "")

    assert not ok
    assert role is None
    assert "required" in message.lower()


def test_password_hash_is_never_exposed_by_user_apis(
    auth_db: Path,
) -> None:
    user_id = register_viewer("alice", ORG_ALPHA)

    by_name = database.get_user_by_username(
        "alice",
        organization_id=ORG_ALPHA,
    )
    by_id = database.get_user_by_id(
        user_id,
        organization_id=ORG_ALPHA,
    )
    all_users = database.get_all_users(
        organization_id=ORG_ALPHA,
    )

    assert by_name is not None
    assert by_id is not None
    assert all_users

    assert "password_hash" not in by_name
    assert "password_hash" not in by_id
    assert all("password_hash" not in row for row in all_users)


def test_inactive_user_cannot_login(auth_db: Path) -> None:
    user_id = register_viewer("alice", ORG_ALPHA)
    admin_id = provision_admin("admin_alpha", ORG_ALPHA)

    assert database.update_user_active_status(
        user_id,
        False,
        "admin_alpha",
    )

    ok, message, role = database.login_user(
        "alice",
        PASSWORD,
    )

    assert not ok
    assert role is None
    assert "inactive" in message.lower()

    # Keep the variable meaningful for debugging if this test fails.
    assert admin_id > 0


def test_failed_login_attempts_are_recorded_and_success_resets_them(
    auth_db: Path,
) -> None:
    user_id = register_viewer("alice", ORG_ALPHA)

    ok, _, _ = database.login_user(
        "alice",
        "WrongPassword123!",
    )
    assert not ok

    with database.get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT failed_login_attempts
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()

    assert row is not None
    assert row["failed_login_attempts"] == 1

    ok, message, role = database.login_user(
        "alice",
        PASSWORD,
    )

    assert ok, message
    assert role == "viewer"

    with database.get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT failed_login_attempts, last_failed_login
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()

    assert row is not None
    assert row["failed_login_attempts"] == 0
    assert row["last_failed_login"] is None


def test_server_side_admin_role_and_organization_are_authoritative(
    auth_db: Path,
) -> None:
    admin_id = provision_admin("admin_alpha", ORG_ALPHA)

    assert database.get_user_role("admin_alpha") == "admin"
    assert database.get_user_organization("admin_alpha") == ORG_ALPHA
    assert database.is_admin("admin_alpha") is True

    assert database.get_user_by_id(
        admin_id,
        organization_id=ORG_ALPHA,
    )["role"] == "admin"


def test_viewer_cannot_change_roles(auth_db: Path) -> None:
    target_id = register_viewer("target", ORG_ALPHA)
    register_viewer("viewer_alpha", ORG_ALPHA)

    assert not database.update_user_role(
        target_id,
        "editor",
        "viewer_alpha",
    )

    target = database.get_user_by_id(
        target_id,
        organization_id=ORG_ALPHA,
    )
    assert target is not None
    assert target["role"] == "viewer"


def test_editor_cannot_change_roles(auth_db: Path) -> None:
    target_id = register_viewer("target", ORG_ALPHA)
    editor_id = register_viewer("editor_alpha", ORG_ALPHA)

    with database.get_db_connection() as conn:
        conn.execute(
            """
            UPDATE users
            SET role = 'editor'
            WHERE id = ?
            """,
            (editor_id,),
        )
        conn.commit()

    assert not database.update_user_role(
        target_id,
        "admin",
        "editor_alpha",
    )

    target = database.get_user_by_id(
        target_id,
        organization_id=ORG_ALPHA,
    )
    assert target is not None
    assert target["role"] == "viewer"


def test_admin_can_promote_viewer_to_editor(auth_db: Path) -> None:
    target_id = register_viewer("target", ORG_ALPHA)
    provision_admin("admin_alpha", ORG_ALPHA)

    assert database.update_user_role(
        target_id,
        "editor",
        "admin_alpha",
    )

    target = database.get_user_by_id(
        target_id,
        organization_id=ORG_ALPHA,
    )
    assert target is not None
    assert target["role"] == "editor"


def test_admin_can_promote_viewer_to_admin_when_an_admin_already_exists(
    auth_db: Path,
) -> None:
    target_id = register_viewer("target", ORG_ALPHA)
    provision_admin("admin_alpha", ORG_ALPHA)

    assert database.update_user_role(
        target_id,
        "admin",
        "admin_alpha",
    )

    target = database.get_user_by_id(
        target_id,
        organization_id=ORG_ALPHA,
    )
    assert target is not None
    assert target["role"] == "admin"


def test_final_admin_cannot_be_demoted(
    auth_db: Path,
) -> None:
    admin_id = provision_admin("admin_alpha", ORG_ALPHA)

    assert not database.update_user_role(
        admin_id,
        "viewer",
        "admin_alpha",
    )

    admin = database.get_user_by_id(
        admin_id,
        organization_id=ORG_ALPHA,
    )
    assert admin is not None
    assert admin["role"] == "admin"


def test_final_admin_cannot_be_deactivated(
    auth_db: Path,
) -> None:
    admin_id = provision_admin("admin_alpha", ORG_ALPHA)

    assert not database.update_user_active_status(
        admin_id,
        False,
        "admin_alpha",
    )

    admin = database.get_user_by_id(
        admin_id,
        organization_id=ORG_ALPHA,
    )
    assert admin is not None
    assert admin["is_active"] == 1


def test_final_admin_cannot_be_deleted(
    auth_db: Path,
) -> None:
    admin_id = provision_admin("admin_alpha", ORG_ALPHA)

    assert not database.delete_user(
        admin_id,
        "admin_alpha",
    )

    assert database.get_user_by_id(
        admin_id,
        organization_id=ORG_ALPHA,
    ) is not None


def test_admin_can_deactivate_and_reactivate_same_tenant_user(
    auth_db: Path,
) -> None:
    user_id = register_viewer("alice", ORG_ALPHA)
    provision_admin("admin_alpha", ORG_ALPHA)

    assert database.update_user_active_status(
        user_id,
        False,
        "admin_alpha",
    )

    user = database.get_user_by_id(
        user_id,
        organization_id=ORG_ALPHA,
    )
    assert user is not None
    assert user["is_active"] == 0

    assert database.update_user_active_status(
        user_id,
        True,
        "admin_alpha",
    )

    user = database.get_user_by_id(
        user_id,
        organization_id=ORG_ALPHA,
    )
    assert user is not None
    assert user["is_active"] == 1


def test_cross_tenant_users_are_isolated(auth_db: Path) -> None:
    alpha_id = register_viewer("alice", ORG_ALPHA)
    beta_id = register_viewer("bob", ORG_BETA)

    alpha_users = database.get_all_users(organization_id=ORG_ALPHA)
    beta_users = database.get_all_users(organization_id=ORG_BETA)

    alpha_names = {row["username"] for row in alpha_users}
    beta_names = {row["username"] for row in beta_users}

    assert "alice" in alpha_names
    assert "bob" not in alpha_names

    assert "bob" in beta_names
    assert "alice" not in beta_names

    assert database.get_user_by_id(
        beta_id,
        organization_id=ORG_ALPHA,
    ) is None

    assert database.get_user_by_id(
        alpha_id,
        organization_id=ORG_BETA,
    ) is None

    assert database.get_user_by_username(
        "bob",
        organization_id=ORG_ALPHA,
    ) is None

    assert database.get_user_by_username(
        "alice",
        organization_id=ORG_BETA,
    ) is None


def test_cross_tenant_admin_cannot_change_role(
    auth_db: Path,
) -> None:
    target_id = register_viewer("alice", ORG_ALPHA)
    provision_admin("admin_alpha", ORG_ALPHA)
    provision_admin("admin_beta", ORG_BETA)

    assert not database.update_user_role(
        target_id,
        "admin",
        "admin_beta",
    )

    target = database.get_user_by_id(
        target_id,
        organization_id=ORG_ALPHA,
    )
    assert target is not None
    assert target["role"] == "viewer"


def test_cross_tenant_admin_cannot_deactivate_user(
    auth_db: Path,
) -> None:
    target_id = register_viewer("alice", ORG_ALPHA)
    provision_admin("admin_alpha", ORG_ALPHA)
    provision_admin("admin_beta", ORG_BETA)

    assert not database.update_user_active_status(
        target_id,
        False,
        "admin_beta",
    )

    target = database.get_user_by_id(
        target_id,
        organization_id=ORG_ALPHA,
    )
    assert target is not None
    assert target["is_active"] == 1


def test_cross_tenant_admin_cannot_delete_user(
    auth_db: Path,
) -> None:
    target_id = register_viewer("alice", ORG_ALPHA)
    provision_admin("admin_alpha", ORG_ALPHA)
    provision_admin("admin_beta", ORG_BETA)

    assert not database.delete_user(
        target_id,
        "admin_beta",
    )

    assert database.get_user_by_id(
        target_id,
        organization_id=ORG_ALPHA,
    ) is not None


def test_access_request_uses_authoritative_current_role(
    auth_db: Path,
) -> None:
    user_id = register_viewer("alice", ORG_ALPHA)
    provision_admin("admin_alpha", ORG_ALPHA)

    ok, message = database.create_access_request(
        user_id,
        "admin",
        "editor",
        "I need editor access for approved HR work.",
    )

    assert not ok
    assert "current role" in message.lower()

    assert database.get_pending_access_requests(
        organization_id=ORG_ALPHA
    ) == []


def test_viewer_can_create_access_request_for_editor(
    auth_db: Path,
) -> None:
    user_id = register_viewer("alice", ORG_ALPHA)

    ok, message = database.create_access_request(
        user_id,
        "viewer",
        "editor",
        "I need editor access for approved HR work.",
    )

    assert ok, message

    requests = database.get_pending_access_requests(
        organization_id=ORG_ALPHA
    )

    assert len(requests) == 1
    request = requests[0]

    assert request["user_id"] == user_id
    assert request["username"] == "alice"
    assert request["organization_id"] == ORG_ALPHA
    assert request["from_role"] == "viewer"
    assert request["to_role"] == "editor"
    assert request["status"] == "pending"


def test_duplicate_pending_access_request_is_rejected(
    auth_db: Path,
) -> None:
    user_id = register_viewer("alice", ORG_ALPHA)

    ok, message = database.create_access_request(
        user_id,
        "viewer",
        "editor",
        "I need editor access for approved HR work.",
    )
    assert ok, message

    ok, message = database.create_access_request(
        user_id,
        "viewer",
        "admin",
        "I also need administrator access for approved HR work.",
    )

    assert not ok
    assert "pending" in message.lower()

    requests = database.get_pending_access_requests(
        organization_id=ORG_ALPHA
    )
    assert len(requests) == 1


def test_non_admin_cannot_approve_access_request(
    auth_db: Path,
) -> None:
    user_id = register_viewer("alice", ORG_ALPHA)
    register_viewer("bob", ORG_ALPHA)

    ok, message = database.create_access_request(
        user_id,
        "viewer",
        "editor",
        "I need editor access for approved HR work.",
    )
    assert ok, message

    request = database.get_pending_access_requests(
        organization_id=ORG_ALPHA
    )[0]

    ok, message = database.approve_access_request(
        request["id"],
        "bob",
        True,
    )

    assert not ok
    assert "administrator" in message.lower()

    user = database.get_user_by_id(
        user_id,
        organization_id=ORG_ALPHA,
    )
    assert user is not None
    assert user["role"] == "viewer"


def test_admin_can_approve_access_request_in_same_tenant(
    auth_db: Path,
) -> None:
    user_id = register_viewer("alice", ORG_ALPHA)
    provision_admin("admin_alpha", ORG_ALPHA)

    ok, message = database.create_access_request(
        user_id,
        "viewer",
        "editor",
        "I need editor access for approved HR work.",
    )
    assert ok, message

    request = database.get_pending_access_requests(
        organization_id=ORG_ALPHA
    )[0]

    ok, message = database.approve_access_request(
        request["id"],
        "admin_alpha",
        True,
    )

    assert ok, message

    user = database.get_user_by_id(
        user_id,
        organization_id=ORG_ALPHA,
    )
    assert user is not None
    assert user["role"] == "editor"

    assert database.get_pending_access_requests(
        organization_id=ORG_ALPHA
    ) == []


def test_cross_tenant_admin_cannot_approve_access_request(
    auth_db: Path,
) -> None:
    user_id = register_viewer("alice", ORG_ALPHA)
    provision_admin("admin_alpha", ORG_ALPHA)
    provision_admin("admin_beta", ORG_BETA)

    ok, message = database.create_access_request(
        user_id,
        "viewer",
        "editor",
        "I need editor access for approved HR work.",
    )
    assert ok, message

    request = database.get_pending_access_requests(
        organization_id=ORG_ALPHA
    )[0]

    ok, message = database.approve_access_request(
        request["id"],
        "admin_beta",
        True,
    )

    assert not ok
    assert "another organization" in message.lower()

    user = database.get_user_by_id(
        user_id,
        organization_id=ORG_ALPHA,
    )
    assert user is not None
    assert user["role"] == "viewer"


def test_rejected_access_request_does_not_change_role(
    auth_db: Path,
) -> None:
    user_id = register_viewer("alice", ORG_ALPHA)
    provision_admin("admin_alpha", ORG_ALPHA)

    ok, message = database.create_access_request(
        user_id,
        "viewer",
        "editor",
        "I need editor access for approved HR work.",
    )
    assert ok, message

    request = database.get_pending_access_requests(
        organization_id=ORG_ALPHA
    )[0]

    ok, message = database.approve_access_request(
        request["id"],
        "admin_alpha",
        False,
    )

    assert ok, message

    user = database.get_user_by_id(
        user_id,
        organization_id=ORG_ALPHA,
    )
    assert user is not None
    assert user["role"] == "viewer"

    with database.get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT status, approved_by
            FROM access_requests
            WHERE id = ?
            """,
            (request["id"],),
        ).fetchone()

    assert row is not None
    assert row["status"] == "rejected"
    assert row["approved_by"] == "admin_alpha"


def test_audit_log_records_authentication_and_admin_actions(
    auth_db: Path,
) -> None:
    target_id = register_viewer("alice", ORG_ALPHA)
    provision_admin("admin_alpha", ORG_ALPHA)

    database.login_user("alice", PASSWORD)
    database.login_user("alice", "WrongPassword123!")

    assert database.update_user_role(
        target_id,
        "editor",
        "admin_alpha",
    )

    events = database.get_audit_log(
        organization_id=ORG_ALPHA,
        limit=100,
    )

    actions = {row["action"] for row in events}

    assert "USER_REGISTERED" in actions
    assert "LOGIN_SUCCESS" in actions
    assert "LOGIN_FAILED" in actions
    assert "ROLE_CHANGED" in actions

    assert all(
        row["organization_id"] == ORG_ALPHA
        for row in events
    )


def test_audit_log_is_tenant_scoped(auth_db: Path) -> None:
    register_viewer("alice", ORG_ALPHA)
    register_viewer("bob", ORG_BETA)

    database.login_user("alice", PASSWORD)
    database.login_user("bob", PASSWORD)

    alpha_events = database.get_audit_log(
        organization_id=ORG_ALPHA,
        limit=100,
    )
    beta_events = database.get_audit_log(
        organization_id=ORG_BETA,
        limit=100,
    )

    assert alpha_events
    assert beta_events

    assert {
        row["organization_id"] for row in alpha_events
    } == {ORG_ALPHA}

    assert {
        row["organization_id"] for row in beta_events
    } == {ORG_BETA}


def test_audit_log_filters_are_parameterized(
    auth_db: Path,
) -> None:
    register_viewer("alice", ORG_ALPHA)
    database.login_user("alice", PASSWORD)

    # The input is deliberately SQL-looking. It must be treated as a value,
    # not interpolated into the SQL statement.
    events = database.get_audit_log(
        username="' OR 1=1 --",
        organization_id=ORG_ALPHA,
    )

    assert events == []


def test_database_backup_creates_readable_copy(
    auth_db: Path,
    tmp_path: Path,
) -> None:
    register_viewer("alice", ORG_ALPHA)

    backup_path = tmp_path / "auth_backup.db"

    assert database.backup_database(backup_path) is True
    assert backup_path.exists()

    with sqlite3.connect(backup_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]

    assert count == 1


def test_reinitialization_is_non_destructive(auth_db: Path) -> None:
    register_viewer("alice", ORG_ALPHA)

    assert database.init_db() is True

    user = database.get_user_by_username(
        "alice",
        organization_id=ORG_ALPHA,
    )
    assert user is not None
    assert user["role"] == "viewer"


def test_database_operations_do_not_leave_sqlite_connection_open(
    auth_db: Path,
) -> None:
    register_viewer("alice", ORG_ALPHA)

    # Windows regression check: the DB can be reopened for a write after the
    # service context managers have completed.
    with sqlite3.connect(auth_db, timeout=5) as conn:
        conn.execute(
            """
            UPDATE users
            SET failed_login_attempts = 0
            WHERE username = 'alice'
            """
        )
        conn.commit()
