
#!/usr/bin/env python3
"""
PolicyGuard AI - Access Request Management Module
==================================================

Handles role escalation requests with approval workflow,
validation, audit logging, and RBAC enforcement.

Security principles:
- Never trust the role supplied by the client.
- Never allow a user to approve their own request.
- Only authorized administrators may approve/reject requests.
- Only valid upward role transitions are permitted.
- Keep request state consistent with the database.
- Avoid returning fabricated request objects.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import re

from pydantic import BaseModel, Field, field_validator, model_validator

from config.settings import settings
from src.auth.database import (
    create_access_request as db_create_request,
    get_pending_access_requests as db_get_pending,
    approve_access_request as db_approve_request,
    get_user_by_id,
    get_user_by_username,
    _log_audit,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


class RequestStatus(str, Enum):
    """Access request lifecycle states."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class AccessRequest(BaseModel):
    """
    Represents a role escalation request.

    Attributes:
        id: Database-assigned request ID.
        user_id: ID of user requesting role change.
        username: Cached username for display.
        from_role: Current role of user.
        to_role: Requested role.
        reason: Justification for role change.
        status: Current request state.
        created_at: Request submission timestamp.
        approved_by: Admin who processed the request.
        approved_at: Decision timestamp.
    """

    id: Optional[int] = Field(
        default=None,
        description="Database-assigned request ID",
    )
    user_id: int = Field(
        ...,
        description="ID of user making request",
        ge=1,
    )
    username: Optional[str] = Field(
        default=None,
        description="Cached username for display",
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Tenant organization identifier",
        min_length=1,
        max_length=128,
    )
    from_role: str = Field(
        ...,
        description="Current user role",
    )
    to_role: str = Field(
        ...,
        description="Requested role",
    )
    reason: str = Field(
        ...,
        description="Justification for role change",
        min_length=10,
        max_length=500,
    )
    status: RequestStatus = Field(
        default=RequestStatus.PENDING,
        description="Request lifecycle state",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Request submission time",
    )
    approved_by: Optional[str] = Field(
        default=None,
        description="Admin who processed request",
    )
    approved_at: Optional[datetime] = Field(
        default=None,
        description="Decision timestamp",
    )

    @field_validator("from_role", "to_role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        """Ensure the role exists in application configuration."""
        value = value.strip().lower()
        valid_roles = set(settings.ROLES.keys())

        if value not in valid_roles:
            raise ValueError(
                f"Role must be one of {sorted(valid_roles)}"
            )

        return value

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        """
        Validate request reason.

        SQL injection protection is handled by parameterized database
        queries, so destructive-looking SQL fragments are not stripped
        here. Removing characters from user input can corrupt legitimate
        audit/request text.
        """
        sanitized = value.strip()

        if len(sanitized) < 10:
            raise ValueError("Reason must be at least 10 characters")

        return sanitized

    @model_validator(mode="after")
    def validate_role_escalation(self) -> "AccessRequest":
        """Ensure the requested role is strictly higher than current role."""
        allowed, reason = AccessRequestService.can_request_escalation(
            self.from_role,
            self.to_role,
        )

        if not allowed:
            raise ValueError(reason)

        return self

    @property
    def is_pending(self) -> bool:
        """Return True when the request is awaiting approval."""
        return self.status == RequestStatus.PENDING

    @property
    def is_resolved(self) -> bool:
        """Return True when the request has reached a final state."""
        return self.status in {
            RequestStatus.APPROVED,
            RequestStatus.REJECTED,
            RequestStatus.EXPIRED,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the request into an API-safe dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.username,
            "organization_id": self.organization_id,
            "from_role": self.from_role,
            "to_role": self.to_role,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
            "approved_by": self.approved_by,
            "approved_at": (
                self.approved_at.isoformat()
                if self.approved_at
                else None
            ),
        }


# =============================================================================
# ROLE HELPERS
# =============================================================================


def _role_levels() -> Dict[str, int]:
    """
    Return an explicit role hierarchy.

    Configuration ordering should not silently determine authorization
    semantics. This function uses the application's standard hierarchy
    while remaining compatible with custom ROLES configuration.
    """
    configured_roles = set(settings.ROLES.keys())

    preferred_order = ["viewer", "editor", "admin"]

    ordered = [
        role for role in preferred_order
        if role in configured_roles
    ]

    # Preserve any custom roles after the standard roles so they remain
    # supported without changing the standard application's hierarchy.
    ordered.extend(
        role for role in settings.ROLES.keys()
        if role not in ordered
    )

    return {role: level for level, role in enumerate(ordered)}


def _is_admin(user: Optional[Dict[str, Any]]) -> bool:
    """Return True when a user record represents an active administrator."""
    if not user:
        return False

    role = str(user.get("role", "")).strip().lower()
    is_active = user.get("is_active", True)

    return role == "admin" and bool(is_active)


def _normalize_org(value: Any) -> str:
    """Normalize a tenant id and fail closed on malformed values."""
    org = "default" if value is None else str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:@-]{1,128}", org):
        raise ValueError("Invalid organization_id")
    return org


def _same_org(user_a: Optional[Dict[str, Any]], user_b: Optional[Dict[str, Any]]) -> bool:
    """Require both users to belong to the same tenant."""
    if not user_a or not user_b:
        return False
    return _normalize_org(user_a.get("organization_id")) == _normalize_org(user_b.get("organization_id"))


# =============================================================================
# ACCESS REQUEST SERVICE
# =============================================================================


class AccessRequestService:
    """
    Service layer for access request management.

    Responsibilities:
    - Request validation.
    - Current-role verification.
    - Duplicate pending-request prevention.
    - Approval/rejection authorization.
    - Audit logging.
    """

    @staticmethod
    def create_request(
        user_id: int,
        from_role: str,
        to_role: str,
        reason: str,
        username: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[AccessRequest]]:
        """
        Create a new access request.

        The supplied from_role is checked against the user's actual
        database role so callers cannot request an escalation by
        pretending to have a different current role.
        """
        try:
            if not isinstance(user_id, int) or user_id < 1:
                return False, "Invalid user ID", None

            user = get_user_by_id(user_id)

            if not user:
                return False, "User not found", None

            if not bool(user.get("is_active", True)):
                return False, "User account is inactive", None

            organization_id = _normalize_org(user.get("organization_id"))
            actual_role = str(user.get("role", "")).strip().lower()
            requested_from_role = str(from_role).strip().lower()
            requested_to_role = str(to_role).strip().lower()

            if actual_role != requested_from_role:
                logger.warning(
                    "Access request role mismatch for user_id=%s: "
                    "supplied=%s actual=%s",
                    user_id,
                    requested_from_role,
                    actual_role,
                )
                return False, "Current user role does not match request", None

            allowed, policy_message = (
                AccessRequestService.can_request_escalation(
                    actual_role,
                    requested_to_role,
                )
            )

            if not allowed:
                return False, policy_message, None

            # Use the database username as the authoritative value.
            authoritative_username = (
                username
                or user.get("username")
                or f"user_{user_id}"
            )

            # Check for an existing pending request.
            pending = AccessRequestService.get_user_pending_requests(user_id)

            if pending:
                return (
                    False,
                    "You already have a pending access request",
                    None,
                )

            request = AccessRequest(
                user_id=user_id,
                username=authoritative_username,
                from_role=actual_role,
                to_role=requested_to_role,
                reason=reason,
                organization_id=organization_id,
            )

            success, message = db_create_request(
                user_id=user_id,
                from_role=actual_role,
                to_role=requested_to_role,
                reason=request.reason,
            )

            if not success:
                return False, message, None

            # Retrieve the newly created request so the returned object
            # contains the database-assigned ID.
            created = AccessRequestService._find_latest_user_request(
                user_id=user_id,
                to_role=requested_to_role,
                reason=request.reason,
            )

            returned_request = created or request

            _log_audit(
                username=authoritative_username,
                action="ACCESS_REQUEST_CREATED",
                details=(
                    f"Requested {requested_to_role} access: "
                    f"{request.reason[:100]}"
                ),
            )

            return True, message, returned_request

        except ValueError as exc:
            logger.warning("Access request validation error: %s", exc)
            return False, str(exc), None

        except Exception as exc:
            logger.exception(
                "Unexpected error creating access request: %s",
                exc,
            )
            return False, "Failed to create access request", None

    @staticmethod
    def get_user_pending_requests(
        user_id: int,
    ) -> List[AccessRequest]:
        """Return all pending requests belonging to a user."""
        try:
            if not isinstance(user_id, int) or user_id < 1:
                return []

            user = get_user_by_id(user_id)
            if not user:
                return []
            organization_id = _normalize_org(user.get("organization_id"))
            pending = db_get_pending()

            requests: List[AccessRequest] = []

            for item in pending:
                if item.get("user_id") != user_id:
                    continue
                item_org = _normalize_org(item.get("organization_id"))
                if item_org != organization_id:
                    continue

                status = str(item.get("status", "")).lower()

                if status != RequestStatus.PENDING.value:
                    continue

                try:
                    requests.append(AccessRequest.model_validate(item))
                except ValueError as exc:
                    logger.warning(
                        "Skipping invalid access request %s: %s",
                        item.get("id"),
                        exc,
                    )

            return requests

        except Exception as exc:
            logger.exception(
                "Error retrieving pending requests for user %s: %s",
                user_id,
                exc,
            )
            return []

    @staticmethod
    def get_all_pending_requests(
        admin_username: str,
    ) -> List[AccessRequest]:
        """
        Return pending requests for an authorized administrator.
        """
        try:
            admin = get_user_by_username(admin_username)

            if not _is_admin(admin):
                logger.warning(
                    "Unauthorized pending-request access attempt by %s",
                    admin_username,
                )
                return []

            admin_org = _normalize_org(admin.get("organization_id"))
            pending = db_get_pending()
            requests: List[AccessRequest] = []

            for item in pending:
                try:
                    item_org = _normalize_org(item.get("organization_id"))
                    if item_org != admin_org:
                        continue
                    requests.append(AccessRequest.model_validate(item))
                except ValueError as exc:
                    logger.warning(
                        "Skipping invalid pending request %s: %s",
                        item.get("id"),
                        exc,
                    )

            _log_audit(
                username=admin_username,
                action="ACCESS_REQUESTS_VIEWED",
                details="Admin viewed pending access requests",
            )

            return requests

        except Exception as exc:
            logger.exception(
                "Error retrieving pending access requests: %s",
                exc,
            )
            return []

    @staticmethod
    def process_request(
        request_id: int,
        admin_username: str,
        approve: bool,
        admin_note: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[AccessRequest]]:
        """
        Approve or reject an access request.

        The request is loaded before modification so the returned object
        represents the real database request rather than a fabricated
        placeholder.
        """
        try:
            if not isinstance(request_id, int) or request_id < 1:
                return False, "Invalid request ID", None

            admin = get_user_by_username(admin_username)

            if not _is_admin(admin):
                _log_audit(
                    username=admin_username,
                    action="ACCESS_REQUEST_UNAUTHORIZED",
                    details=(
                        f"Unauthorized attempt to process "
                        f"request {request_id}"
                    ),
                )
                return False, "Admin permission required", None

            request = get_request_by_id(request_id)

            if request is None:
                return False, "Access request not found", None

            request_user = get_user_by_id(request.user_id)
            if not _same_org(admin, request_user):
                _log_audit(
                    username=admin_username,
                    action="ACCESS_REQUEST_CROSS_TENANT_BLOCKED",
                    details=f"Attempted to process request {request_id}",
                )
                return False, "Access request is outside administrator tenant", None

            if not request.is_pending:
                return (
                    False,
                    f"Request is already {request.status.value}",
                    request,
                )

            # Prevent an administrator from approving their own request.
            if request.username and request.username == admin_username:
                _log_audit(
                    username=admin_username,
                    action="ACCESS_REQUEST_SELF_APPROVAL_BLOCKED",
                    details=f"Attempted to process request {request_id}",
                )
                return (
                    False,
                    "Administrators cannot approve their own access request",
                    None,
                )

            note = (admin_note or "").strip()

            if len(note) > 500:
                return False, "Admin note must be 500 characters or fewer", None

            success, message = db_approve_request(
                request_id=request_id,
                admin_username=admin_username,
                approved=bool(approve),
            )

            if not success:
                return False, message, None

            status = (
                RequestStatus.APPROVED
                if approve
                else RequestStatus.REJECTED
            )

            updated = request.model_copy(
                update={
                    "status": status,
                    "approved_by": admin_username,
                    "approved_at": datetime.now(),
                }
            )

            action = (
                "ACCESS_REQUEST_APPROVED"
                if approve
                else "ACCESS_REQUEST_REJECTED"
            )

            details = f"Request {request_id} {action.lower()}"

            if note:
                details += f" | Note: {note[:500]}"

            _log_audit(
                username=admin_username,
                action=action,
                details=details,
            )

            return True, message, updated

        except Exception as exc:
            logger.exception(
                "Error processing access request %s: %s",
                request_id,
                exc,
            )
            return False, "Error processing access request", None

    @staticmethod
    def can_request_escalation(
        user_role: str,
        target_role: str,
    ) -> Tuple[bool, str]:
        """
        Determine whether a role escalation is permitted by policy.
        """
        user_role = str(user_role).strip().lower()
        target_role = str(target_role).strip().lower()

        levels = _role_levels()

        if user_role not in levels:
            return False, f"Invalid current role: {user_role}"

        if target_role not in levels:
            return False, f"Invalid target role: {target_role}"

        if levels[target_role] <= levels[user_role]:
            return False, "Can only request higher-level roles"

        # PolicyGuard's standard policy: only editors can request admin.
        if target_role == "admin" and user_role != "editor":
            return False, "Only editors can request admin access"

        return True, "Escalation allowed"

    @staticmethod
    def _find_latest_user_request(
        user_id: int,
        to_role: str,
        reason: str,
    ) -> Optional[AccessRequest]:
        """
        Locate the newly-created request from the pending request list.

        This is intentionally best-effort because the database API currently
        returns only (success, message) from create_access_request().
        """
        try:
            pending = db_get_pending()

            matches = [
                AccessRequest.model_validate(item)
                for item in pending
                if item.get("user_id") == user_id
                and str(item.get("to_role", "")).lower() == to_role
                and str(item.get("reason", "")).strip() == reason.strip()
                and str(item.get("status", "")).lower()
                == RequestStatus.PENDING.value
            ]

            if not matches:
                return None

            return max(
                matches,
                key=lambda request: request.id or 0,
            )

        except Exception as exc:
            logger.warning(
                "Unable to retrieve newly-created access request: %s",
                exc,
            )
            return None


# =============================================================================
# REQUEST LOOKUP / EXPIRATION
# =============================================================================


def get_request_by_id(request_id: int) -> Optional[AccessRequest]:
    """
    Fetch a specific pending access request.

    The current database interface exposes pending requests, so this
    function searches that authoritative collection instead of returning
    a fabricated object.
    """
    if not isinstance(request_id, int) or request_id < 1:
        return None

    try:
        pending = db_get_pending()

        for item in pending:
            if item.get("id") != request_id:
                continue

            return AccessRequest.model_validate(item)

        return None

    except Exception as exc:
        logger.exception(
            "Error retrieving access request %s: %s",
            request_id,
            exc,
        )
        return None


def expire_old_requests(days: int = 30) -> int:
    """
    Count pending requests that have exceeded the expiration period.

    IMPORTANT:
    The current database API does not expose an atomic
    expire_access_request(s) operation. Therefore this function does not
    pretend to update the database.

    Returns:
        Number of requests that are eligible for expiration.

    A dedicated database operation should be added before this function
    is used as an actual cleanup job.
    """
    if not isinstance(days, int) or days < 1:
        raise ValueError("days must be a positive integer")

    cutoff = datetime.now() - timedelta(days=days)
    eligible = 0

    try:
        for request in db_get_pending():
            created_at = request.get("created_at")

            if not created_at:
                continue

            try:
                if isinstance(created_at, datetime):
                    created = created_at
                else:
                    created = datetime.fromisoformat(
                        str(created_at)
                    )

                if created < cutoff:
                    eligible += 1

            except (TypeError, ValueError):
                logger.warning(
                    "Invalid created_at value for request %s: %r",
                    request.get("id"),
                    created_at,
                )

        logger.info(
            "%s pending access request(s) are eligible for expiration "
            "(older than %s days)",
            eligible,
            days,
        )

        return eligible

    except Exception as exc:
        logger.exception(
            "Error checking expired access requests: %s",
            exc,
        )
        return 0


# =============================================================================
# NOTIFICATION HOOKS
# =============================================================================


def notify_admins_of_request(request: AccessRequest) -> bool:
    """
    Placeholder notification hook for administrators.

    Production integrations can connect this function to email, Slack,
    Teams, or an internal notification system.
    """
    try:
        logger.info(
            "Access request notification: user=%s, %s -> %s",
            request.username or request.user_id,
            request.from_role,
            request.to_role,
        )

        logger.info(
            "Access request reason: %s",
            request.reason[:100],
        )

        # TODO: Integrate an actual notification provider.
        return True

    except Exception as exc:
        logger.exception(
            "Admin notification error: %s",
            exc,
        )
        return False


def notify_user_of_decision(
    request: AccessRequest,
    approved: bool,
) -> bool:
    """Placeholder notification hook for request decisions."""
    try:
        status = "approved" if approved else "rejected"

        logger.info(
            "Access request %s: request_id=%s user_id=%s",
            status,
            request.id,
            request.user_id,
        )

        # TODO: Integrate email/in-app notification.
        return True

    except Exception as exc:
        logger.exception(
            "User notification error: %s",
            exc,
        )
        return False


# =============================================================================
# TEST / DEMO
# =============================================================================


def demo_access_request_flow() -> None:
    """Demonstrate the access request workflow."""
    print("\nAccess Request Demo Flow\n")
    print("=" * 60)

    print("1. Editor requests admin access:")

    success, message, request = AccessRequestService.create_request(
        user_id=2,
        from_role="editor",
        to_role="admin",
        reason=(
            "Need admin access to manage team permissions "
            "and audit logs for Q4 review"
        ),
        username="jane_editor",
    )

    print(f"   Result: {message}")

    if request:
        print(f"   Request ID: {request.id}")
        print(f"   Status: {request.status.value}")

    print("\n2. Admin views pending requests:")

    pending = AccessRequestService.get_all_pending_requests(
        admin_username="admin"
    )

    print(f"   Pending requests: {len(pending)}")

    for req in pending:
        print(
            f"   • {req.username or req.user_id}: "
            f"{req.from_role} -> {req.to_role}"
        )
        print(f"     Reason: {req.reason[:80]}...")

    if pending and pending[0].id:
        print("\n3. Admin approves request:")

        req = pending[0]

        success, message, updated = (
            AccessRequestService.process_request(
                request_id=req.id,
                admin_username="admin",
                approve=True,
                admin_note="Approved for Q4 audit responsibilities",
            )
        )

        print(f"   Result: {message}")

        if updated:
            print(f"   New status: {updated.status.value}")
            print(f"   Approved by: {updated.approved_by}")

    print("\n4. Check escalation policy:")

    test_cases = [
        ("viewer", "editor"),
        ("viewer", "admin"),
        ("editor", "admin"),
        ("admin", "editor"),
    ]

    for from_role, to_role in test_cases:
        allowed, reason = (
            AccessRequestService.can_request_escalation(
                from_role,
                to_role,
            )
        )

        status = "ALLOWED" if allowed else "DENIED"

        print(
            f"   [{status}] {from_role} -> {to_role}: {reason}"
        )

    print("\n" + "=" * 60)
    print("Access request demo complete.\n")


if __name__ == "__main__":
    demo_access_request_flow()

