
#!/usr/bin/env python3
"""
PolicyGuard AI - Model Context Protocol (MCP) Server
=====================================================

Production-ready MCP-style server for HR agent tool calling.

Features:
- Pydantic-based input/output validation
- HR-specific tools
- Sync and async handler support
- Role-based authorization
- Execution timeouts
- Thread-safe tool registry
- Structured logging
- Execution metrics
- Mock implementations with production integration hooks

Important:
The HR handlers in this file are MOCK implementations.
They must be replaced with real email/calendar/ATS/HRIS integrations
before production use.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, ValidationError, model_validator

# =============================================================================
# PROJECT PATH SETUP
# =============================================================================

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.exceptions import RAGException, ValidationError as AppValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

ROLE_LEVELS = {
    "viewer": 1,
    "editor": 2,
    "admin": 3,
}


def _utc_now_iso() -> str:
    """Return current UTC time as ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def _normalize_role(role: Any) -> Optional[str]:
    """Normalize a user role safely."""
    if role is None:
        return None

    role_value = str(role).strip().lower()

    if role_value in ROLE_LEVELS:
        return role_value

    return None


def _normalize_organization_id(value: Any) -> Optional[str]:
    """Validate the tenant identifier supplied by the authenticated context."""
    if value is None:
        return None
    value = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:@-]{1,128}", value):
        return None
    return value


def _safe_error_message(error: Exception, limit: int = 200) -> str:
    """Return a bounded error message safe for API responses."""
    message = str(error).strip()

    if not message:
        message = error.__class__.__name__

    return message[:limit]


def _generate_id(prefix: str) -> str:
    """Generate a collision-resistant identifier."""
    return f"{prefix}_{uuid.uuid4().hex}"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ToolInput(BaseModel):
    """Base class for all tool inputs."""

    @model_validator(mode="after")
    def validate_non_empty(self) -> "ToolInput":
        """Reject empty strings in supplied string fields."""
        for field_name, value in self.model_dump().items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(
                    f"{field_name} cannot be empty"
                )

        return self


class ToolOutput(BaseModel):
    """Base standardized tool response."""

    success: bool = Field(
        ...,
        description="Whether the tool execution succeeded",
    )

    message: str = Field(
        ...,
        description="Human-readable status message",
    )

    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured response data",
    )

    error_code: Optional[str] = Field(
        default=None,
        description="Error code if success=False",
    )

    timestamp: str = Field(
        default_factory=_utc_now_iso,
        description="Response timestamp",
    )

    def model_dump_for_api(self) -> Dict[str, Any]:
        """
        Export the complete public output.

        Unlike the original implementation, subclass-specific fields are
        retained (for example email_id, meeting_id, jobs, etc.).
        """
        payload = self.model_dump()

        if self.success:
            payload["error_code"] = None

        return payload


# =============================================================================
# TOOL INPUT / OUTPUT MODELS
# =============================================================================

class SendEmailInput(ToolInput):
    """Input schema for sending emails."""

    to: str = Field(
        ...,
        description="Primary recipient email address",
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    )

    subject: str = Field(
        ...,
        description="Email subject line",
        min_length=1,
        max_length=200,
    )

    body: str = Field(
        ...,
        description="Email body content",
        min_length=10,
    )

    cc: Optional[str] = Field(
        default=None,
        description="CC recipients (comma-separated emails)",
    )

    bcc: Optional[str] = Field(
        default=None,
        description="BCC recipients (comma-separated emails)",
    )

    priority: str = Field(
        default="normal",
        description="Email priority",
        pattern=r"^(low|normal|high)$",
    )


class SendEmailOutput(ToolOutput):
    """Output schema for email sending."""

    email_id: Optional[str] = Field(
        default=None,
        description="Unique email identifier",
    )

    sent_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp when email was sent",
    )

    recipients_count: Optional[int] = Field(
        default=None,
        description="Total number of recipients",
    )


class ScheduleInterviewInput(ToolInput):
    """Input schema for scheduling interviews."""

    candidate_name: str = Field(
        ...,
        description="Candidate full name",
        min_length=2,
        max_length=200,
    )

    candidate_email: str = Field(
        ...,
        description="Candidate email address",
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    )

    interview_date: str = Field(
        ...,
        description="Interview date YYYY-MM-DD",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )

    interview_time: str = Field(
        ...,
        description="Interview time HH:MM",
        pattern=r"^\d{2}:\d{2}$",
    )

    interviewers: List[str] = Field(
        ...,
        description="Interviewer email addresses",
        min_length=1,
    )

    position: str = Field(
        ...,
        description="Job position title",
        min_length=2,
        max_length=200,
    )

    duration_minutes: int = Field(
        default=60,
        description="Interview duration",
        ge=15,
        le=240,
    )

    meeting_type: str = Field(
        default="video",
        description="Meeting type",
        pattern=r"^(video|phone|in-person)$",
    )

    @model_validator(mode="after")
    def validate_datetime(self) -> "ScheduleInterviewInput":
        """Validate actual calendar date/time values."""
        try:
            datetime.strptime(
                self.interview_date,
                "%Y-%m-%d",
            )

            datetime.strptime(
                self.interview_time,
                "%H:%M",
            )

        except ValueError as error:
            raise ValueError(
                "interview_date/interview_time must contain valid "
                "calendar date and time values"
            ) from error

        for email in self.interviewers:
            if not isinstance(email, str):
                raise ValueError(
                    "interviewers must contain email strings"
                )

            if not email.strip():
                raise ValueError(
                    "interviewer email cannot be empty"
                )

        return self


class ScheduleInterviewOutput(ToolOutput):
    """Output schema for interview scheduling."""

    meeting_id: Optional[str] = None
    calendar_link: Optional[str] = None
    ics_attachment: Optional[str] = None


class GetCandidateInput(ToolInput):
    """Input schema for candidate lookup."""

    candidate_id: str = Field(
        ...,
        description="Candidate ID or email address",
        min_length=1,
        max_length=200,
    )

    include_resume: bool = False

    include_assessments: bool = False


class GetCandidateOutput(ToolOutput):
    """Output schema for candidate data."""

    candidate_data: Optional[Dict[str, Any]] = None


class SearchJobsInput(ToolInput):
    """Input schema for job search."""

    keywords: Optional[str] = Field(
        default=None,
        max_length=200,
    )

    department: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    location: Optional[str] = Field(
        default=None,
        max_length=150,
    )

    job_type: Optional[str] = Field(
        default=None,
        pattern=r"^(full-time|part-time|contract|internship)$",
    )

    min_experience: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
    )

    remote_ok: Optional[bool] = None

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
    )


class SearchJobsOutput(ToolOutput):
    """Output schema for job search results."""

    jobs: Optional[List[Dict[str, Any]]] = None
    total_count: Optional[int] = None


class LeaveRequestInput(ToolInput):
    """Input schema for leave requests."""

    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    leave_type: str = Field(
        ...,
        pattern=r"^(annual|sick|maternity|paternity|unpaid|bereavement)$",
    )

    start_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )

    end_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    emergency_contact: Optional[str] = Field(
        default=None,
        max_length=300,
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "LeaveRequestInput":
        """Validate actual dates and ordering."""
        try:
            start = datetime.strptime(
                self.start_date,
                "%Y-%m-%d",
            )

            end = datetime.strptime(
                self.end_date,
                "%Y-%m-%d",
            )

        except ValueError as error:
            raise ValueError(
                "start_date and end_date must be valid dates"
            ) from error

        if end < start:
            raise ValueError(
                "End date must be on or after start date"
            )

        return self


class LeaveRequestOutput(ToolOutput):
    """Output schema for leave requests."""

    request_id: Optional[str] = None
    approval_status: Optional[str] = None
    days_requested: Optional[int] = None
    balance_after: Optional[float] = None


# =============================================================================
# MCP TOOL
# =============================================================================

class MCPTool:
    """Represents one registered MCP-style tool."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Type[ToolInput],
        output_schema: Type[ToolOutput],
        handler: Callable,
        timeout_seconds: int = 30,
        requires_auth: Union[bool, str] = False,
    ):
        if not name or not name.strip():
            raise ValueError(
                "Tool name cannot be empty"
            )

        if not callable(handler):
            raise TypeError(
                "handler must be callable"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero"
            )

        if isinstance(requires_auth, str):
            normalized_role = _normalize_role(
                requires_auth
            )

            if normalized_role is None:
                raise ValueError(
                    "requires_auth must be False, True, "
                    "or one of viewer/editor/admin"
                )

            self.requires_auth: Union[bool, str] = normalized_role

        elif isinstance(requires_auth, bool):
            self.requires_auth = requires_auth

        else:
            raise ValueError(
                "requires_auth must be False, True, "
                "or a role string"
            )

        self.name = name.strip()
        self.description = description.strip()
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.handler = handler
        self.timeout_seconds = int(timeout_seconds)

        logger.debug(
            "Tool registered: %s timeout=%ss auth=%s",
            self.name,
            self.timeout_seconds,
            self.requires_auth,
        )

    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------

    def validate_input(
        self,
        input_data: Dict[str, Any],
    ) -> ToolInput:
        """Validate tool input with Pydantic."""
        if not isinstance(input_data, dict):
            raise AppValidationError(
                f"Invalid input for tool '{self.name}': "
                "input_data must be an object"
            )

        try:
            return self.input_schema(
                **input_data
            )

        except ValidationError as error:
            logger.warning(
                "Input validation failed for %s: %s",
                self.name,
                error,
            )

            raise AppValidationError(
                f"Invalid input for tool '{self.name}': "
                f"{error}"
            ) from error

    # -------------------------------------------------------------------------
    # Authorization
    # -------------------------------------------------------------------------

    def check_authorization(
        self,
        user_context: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Check authorization.

        IMPORTANT:
        A protected tool is denied when no user context is supplied.
        The previous implementation accidentally allowed this case.
        """
        if not self.requires_auth:
            return True

        if not user_context or not isinstance(
            user_context,
            dict,
        ):
            return False

        user_role = _normalize_role(
            user_context.get("role")
        )
        organization_id = _normalize_organization_id(
            user_context.get("organization_id")
        )

        if user_role is None or organization_id is None:
            return False

        if self.requires_auth is True:
            # True means any authenticated/recognized role.
            return True

        required_role = _normalize_role(
            self.requires_auth
        )

        if required_role is None:
            return False

        return (
            ROLE_LEVELS[user_role]
            >= ROLE_LEVELS[required_role]
        )

    # Backward-compatible private name.
    def _check_authorization(
        self,
        user_role: Optional[str],
    ) -> bool:
        """Backward-compatible authorization helper."""
        normalized = _normalize_role(user_role)

        if not self.requires_auth:
            return True

        if normalized is None:
            return False

        if self.requires_auth is True:
            return True

        required = _normalize_role(
            self.requires_auth
        )

        if required is None:
            return False

        return (
            ROLE_LEVELS[normalized]
            >= ROLE_LEVELS[required]
        )

    # -------------------------------------------------------------------------
    # Handler execution
    # -------------------------------------------------------------------------

    async def _run_async_handler(
        self,
        validated_input: ToolInput,
        user_context: Optional[Dict[str, Any]],
    ) -> Any:
        """Run an async handler with a timeout."""
        return await asyncio.wait_for(
            self.handler(
                validated_input,
                user_context=user_context,
            ),
            timeout=self.timeout_seconds,
        )

    def _run_async_from_sync(
        self,
        validated_input: ToolInput,
        user_context: Optional[Dict[str, Any]],
    ) -> Any:
        """
        Execute an async handler from synchronous code.

        asyncio.run() cannot be called while another event loop is already
        running. This helper uses a dedicated thread in that situation.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._run_async_handler(
                    validated_input,
                    user_context,
                )
            )

        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, BaseException] = {}

        def runner() -> None:
            try:
                result_holder["result"] = asyncio.run(
                    self._run_async_handler(
                        validated_input,
                        user_context,
                    )
                )
            except BaseException as error:
                error_holder["error"] = error

        thread = threading.Thread(
            target=runner,
            name=f"mcp-async-{self.name}",
            daemon=True,
        )

        thread.start()
        thread.join(
            timeout=self.timeout_seconds + 1
        )

        if thread.is_alive():
            raise TimeoutError(
                f"Async tool exceeded timeout: "
                f"{self.timeout_seconds}s"
            )

        if "error" in error_holder:
            raise error_holder["error"]

        return result_holder.get("result")

    def _run_sync_handler(
        self,
        validated_input: ToolInput,
        user_context: Optional[Dict[str, Any]],
    ) -> Any:
        """
        Execute sync handler in a worker thread.

        The worker is not forcibly killed on timeout because Python cannot
        safely terminate an arbitrary running thread. The future is cancelled
        where possible and the executor is shut down without waiting.
        """
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"mcp-{self.name}",
        )

        future = executor.submit(
            self.handler,
            validated_input,
            user_context=user_context,
        )

        try:
            return future.result(
                timeout=self.timeout_seconds
            )

        finally:
            future.cancel()

            # Do not wait for an already-running handler after timeout.
            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

    def execute(
        self,
        input_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None,
    ) -> ToolOutput:
        """Validate, authorize, execute, and validate tool output."""

        validated_input = self.validate_input(
            input_data
        )

        # Authorization is always checked for protected tools.
        if self.requires_auth and not self.check_authorization(
            user_context
        ):
            return self.output_schema(
                success=False,
                message=(
                    f"Access denied: tool '{self.name}' "
                    "requires appropriate authorization"
                ),
                error_code="AUTH_REQUIRED",
            )

        try:
            if asyncio.iscoroutinefunction(
                self.handler
            ):
                result = self._run_async_from_sync(
                    validated_input,
                    user_context,
                )
            else:
                result = self._run_sync_handler(
                    validated_input,
                    user_context,
                )

            if isinstance(
                result,
                self.output_schema,
            ):
                return result

            if isinstance(result, dict):
                return self.output_schema(
                    **result
                )

            logger.error(
                "Handler returned unexpected type for %s: %s",
                self.name,
                type(result).__name__,
            )

            return self.output_schema(
                success=False,
                message=(
                    "Internal error: invalid handler response"
                ),
                error_code="HANDLER_ERROR",
            )

        except (
            asyncio.TimeoutError,
            TimeoutError,
        ):
            logger.error(
                "Tool timeout: %s exceeded %ss",
                self.name,
                self.timeout_seconds,
            )

            return self.output_schema(
                success=False,
                message=(
                    f"Tool execution timed out after "
                    f"{self.timeout_seconds} seconds"
                ),
                error_code="TIMEOUT",
            )

        except Exception as error:
            logger.error(
                "Tool execution error: %s - %s",
                self.name,
                error,
                exc_info=True,
            )

            return self.output_schema(
                success=False,
                message=(
                    "Tool execution failed: "
                    f"{_safe_error_message(error)}"
                ),
                error_code="EXECUTION_ERROR",
            )

    def get_schema_for_discovery(
        self,
    ) -> Dict[str, Any]:
        """Return tool discovery metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": (
                self.input_schema.model_json_schema()
            ),
            "output_schema": (
                self.output_schema.model_json_schema()
            ),
            "requires_auth": self.requires_auth,
            "timeout_seconds": self.timeout_seconds,
            "mock_only": True,
        }


# =============================================================================
# MCP SERVER
# =============================================================================

class MCPServer:
    """Thread-safe MCP-style HR tool server."""

    def __init__(
        self,
        enable_metrics: bool = True,
    ):
        self.tools: Dict[str, MCPTool] = {}
        self.enable_metrics = bool(
            enable_metrics
        )
        self.is_mock_server = True

        self._metrics: Dict[
            str,
            List[float],
        ] = {}

        self._lock = threading.RLock()

        self._register_default_tools()

        logger.info(
            "MCPServer initialized: %s tools, metrics=%s",
            len(self.tools),
            self.enable_metrics,
        )

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def _register_default_tools(self) -> None:
        """Register built-in HR tools."""

        self.register_tool(
            name="send_email",
            description=(
                "Send an email to a candidate, employee, "
                "or external party via configured email service"
            ),
            input_schema=SendEmailInput,
            output_schema=SendEmailOutput,
            handler=self._handle_send_email,
            timeout_seconds=15,
            requires_auth="editor",
        )

        self.register_tool(
            name="schedule_interview",
            description=(
                "Schedule an interview with a candidate "
                "and interviewers via calendar integration"
            ),
            input_schema=ScheduleInterviewInput,
            output_schema=ScheduleInterviewOutput,
            handler=self._handle_schedule_interview,
            timeout_seconds=30,
            requires_auth="editor",
        )

        self.register_tool(
            name="get_candidate",
            description=(
                "Retrieve candidate information from ATS "
                "by ID or email"
            ),
            input_schema=GetCandidateInput,
            output_schema=GetCandidateOutput,
            handler=self._handle_get_candidate,
            timeout_seconds=10,
            requires_auth="viewer",
        )

        self.register_tool(
            name="search_jobs",
            description=(
                "Search for open job positions with filters"
            ),
            input_schema=SearchJobsInput,
            output_schema=SearchJobsOutput,
            handler=self._handle_search_jobs,
            timeout_seconds=10,
            requires_auth="viewer",
        )

        self.register_tool(
            name="request_leave",
            description=(
                "Submit a leave request for an employee "
                "via HRIS integration"
            ),
            input_schema=LeaveRequestInput,
            output_schema=LeaveRequestOutput,
            handler=self._handle_request_leave,
            timeout_seconds=20,
            requires_auth="editor",
        )

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Type[ToolInput],
        output_schema: Type[ToolOutput],
        handler: Callable,
        timeout_seconds: int = 30,
        requires_auth: Union[bool, str] = False,
    ) -> None:
        """Register or replace a tool."""
        if not isinstance(
            name,
            str,
        ):
            raise TypeError(
                "Tool name must be a string"
            )

        if not issubclass(
            input_schema,
            ToolInput,
        ):
            raise TypeError(
                "input_schema must inherit from ToolInput"
            )

        if not issubclass(
            output_schema,
            ToolOutput,
        ):
            raise TypeError(
                "output_schema must inherit from ToolOutput"
            )

        with self._lock:
            if name in self.tools:
                logger.warning(
                    "Overwriting existing tool: %s",
                    name,
                )

            self.tools[name] = MCPTool(
                name=name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                handler=handler,
                timeout_seconds=timeout_seconds,
                requires_auth=requires_auth,
            )

            if self.enable_metrics:
                self._metrics.setdefault(
                    name,
                    [],
                )

    def get_tool(
        self,
        name: str,
    ) -> Optional[MCPTool]:
        """Get a registered tool."""
        with self._lock:
            return self.tools.get(name)

    def list_tools(
        self,
        include_schemas: bool = True,
    ) -> List[Dict[str, Any]]:
        """List registered tools."""
        with self._lock:
            tools = list(
                self.tools.values()
            )

        if include_schemas:
            return [
                tool.get_schema_for_discovery()
                for tool in tools
            ]

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "requires_auth": tool.requires_auth,
            }
            for tool in tools
        ]

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def execute_tool(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        user_context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """Execute a named tool and return an API-ready dictionary."""

        start = time.perf_counter()

        tool = self.get_tool(
            tool_name
        )

        if tool is None:
            with self._lock:
                available = list(
                    self.tools.keys()
                )

            logger.warning(
                "Tool not found: %s",
                tool_name,
            )

            return ToolOutput(
                success=False,
                message=(
                    f"Tool '{tool_name}' not found. "
                    f"Available: {available}"
                ),
                error_code="TOOL_NOT_FOUND",
            ).model_dump_for_api()

        try:
            if tool.requires_auth and (
                not isinstance(user_context, dict)
                or _normalize_organization_id(user_context.get("organization_id")) is None
            ):
                output = tool.output_schema(
                    success=False,
                    message="Authenticated tenant context is required",
                    error_code="TENANT_CONTEXT_REQUIRED",
                )
            else:
                output = tool.execute(
                    input_data,
                    user_context,
                )

        except AppValidationError as error:
            # Validation should be a client/input error, not an internal
            # server error.
            logger.warning(
                "Tool input rejected: %s - %s",
                tool_name,
                error,
            )

            output = tool.output_schema(
                success=False,
                message=_safe_error_message(
                    error
                ),
                error_code="INVALID_INPUT",
            )

        except Exception as error:
            logger.error(
                "Unhandled tool execution error: %s - %s",
                tool_name,
                error,
                exc_info=True,
            )

            output = tool.output_schema(
                success=False,
                message=(
                    f"Internal error executing "
                    f"'{tool_name}'"
                ),
                error_code="INTERNAL_ERROR",
            )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        if self.enable_metrics:
            with self._lock:
                values = self._metrics.setdefault(
                    tool_name,
                    [],
                )

                values.append(elapsed_ms)

                if len(values) > 100:
                    del values[:-100]

        logger.info(
            "Tool execution complete: %s success=%s %.2fms",
            tool_name,
            output.success,
            elapsed_ms,
        )

        return output.model_dump_for_api()

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def get_metrics(
        self,
        tool_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return execution latency metrics."""
        if not self.enable_metrics:
            return {
                "metrics_enabled": False
            }

        with self._lock:
            if tool_name is not None:
                values = list(
                    self._metrics.get(
                        tool_name,
                        [],
                    )
                )

                if not values:
                    return {
                        "tool": tool_name,
                        "count": 0,
                    }

                return self._calculate_metrics(
                    tool_name,
                    values,
                )

            snapshot = {
                name: list(values)
                for name, values
                in self._metrics.items()
            }

        return {
            "metrics_enabled": True,
            "tools": {
                name: self._calculate_metrics(
                    name,
                    values,
                )
                for name, values
                in snapshot.items()
            },
        }

    @staticmethod
    def _calculate_metrics(
        tool_name: str,
        values: List[float],
    ) -> Dict[str, Any]:
        """Calculate latency statistics without requiring NumPy."""
        if not values:
            return {
                "tool": tool_name,
                "count": 0,
            }

        ordered = sorted(values)
        count = len(ordered)

        def percentile(
            percentile_value: float,
        ) -> float:
            if count == 1:
                return ordered[0]

            position = (
                percentile_value
                / 100
            ) * (count - 1)

            lower = int(position)
            upper = min(
                lower + 1,
                count - 1,
            )

            fraction = position - lower

            return (
                ordered[lower]
                + (
                    ordered[upper]
                    - ordered[lower]
                )
                * fraction
            )

        return {
            "tool": tool_name,
            "count": count,
            "mean_ms": sum(ordered) / count,
            "p50_ms": percentile(50),
            "p95_ms": percentile(95),
            "p99_ms": percentile(99),
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
        }

    # =============================================================================
    # MOCK TOOL HANDLERS
    # =============================================================================

    def _handle_send_email(
        self,
        input: SendEmailInput,
        user_context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> SendEmailOutput:
        """
        Mock email handler.

        Replace with a real provider such as SMTP, SES, or SendGrid before
        production deployment.
        """
        try:
            recipients = [
                input.to.strip()
            ]

            if input.cc:
                recipients.extend(
                    item.strip()
                    for item in input.cc.split(",")
                    if item.strip()
                )

            if input.bcc:
                recipients.extend(
                    item.strip()
                    for item in input.bcc.split(",")
                    if item.strip()
                )

            email_id = _generate_id(
                "email"
            )

            time.sleep(0.1)

            sent_at = _utc_now_iso()

            logger.info(
                "Mock email sent: to=%s subject=%s recipients=%s",
                input.to,
                input.subject[:50],
                len(recipients),
            )

            return SendEmailOutput(
                success=True,
                message=(
                    f"Email sent successfully to "
                    f"{input.to}"
                ),
                data={
                    "email_id": email_id,
                    "to": input.to,
                    "subject": input.subject,
                    "priority": input.priority,
                    "recipients_count": len(
                        recipients
                    ),
                },
                email_id=email_id,
                sent_at=sent_at,
                recipients_count=len(
                    recipients
                ),
            )

        except Exception as error:
            logger.error(
                "Email handler error: %s",
                error,
                exc_info=True,
            )

            return SendEmailOutput(
                success=False,
                message=(
                    f"Failed to send email: "
                    f"{_safe_error_message(error)}"
                ),
                error_code="EMAIL_SEND_FAILED",
            )

    def _handle_schedule_interview(
        self,
        input: ScheduleInterviewInput,
        user_context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ScheduleInterviewOutput:
        """Mock interview scheduling handler."""
        try:
            meeting_id = _generate_id(
                "int"
            )

            start_datetime = datetime.strptime(
                f"{input.interview_date} "
                f"{input.interview_time}",
                "%Y-%m-%d %H:%M",
            )

            end_datetime = (
                start_datetime
                + timedelta(
                    minutes=input.duration_minutes
                )
            )

            event_details = {
                "summary": (
                    f"Interview: "
                    f"{input.position} - "
                    f"{input.candidate_name}"
                ),
                "start": start_datetime.isoformat(),
                "end": end_datetime.isoformat(),
                "attendees": [
                    input.candidate_email,
                    *input.interviewers,
                ],
                "location": (
                    "Video Conference"
                    if input.meeting_type == "video"
                    else None
                ),
                "description": (
                    f"Position: {input.position}\n"
                    f"Candidate: {input.candidate_name}"
                ),
            }

            calendar_link = (
                "https://calendar.policyguard.ai/meet/"
                f"{meeting_id}"
            )

            ics_content = self._generate_ics_event(
                event_details
            )

            return ScheduleInterviewOutput(
                success=True,
                message=(
                    f"Interview scheduled for "
                    f"{input.interview_date} "
                    f"at {input.interview_time}"
                ),
                data={
                    "meeting_id": meeting_id,
                    "candidate": input.candidate_name,
                    "position": input.position,
                    "datetime": event_details["start"],
                    "end_datetime": event_details["end"],
                    "duration_minutes": input.duration_minutes,
                    "meeting_type": input.meeting_type,
                    "attendees_count": len(
                        event_details["attendees"]
                    ),
                },
                meeting_id=meeting_id,
                calendar_link=calendar_link,
                ics_attachment=ics_content,
            )

        except Exception as error:
            logger.error(
                "Interview scheduling error: %s",
                error,
                exc_info=True,
            )

            return ScheduleInterviewOutput(
                success=False,
                message=(
                    f"Failed to schedule interview: "
                    f"{_safe_error_message(error)}"
                ),
                error_code="SCHEDULING_FAILED",
            )

    def _handle_get_candidate(
        self,
        input: GetCandidateInput,
        user_context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> GetCandidateOutput:
        """Mock candidate lookup handler."""
        try:
            candidate_id = (
                input.candidate_id.strip().lower()
            )

            mock_candidates = {
                "john.doe@example.com": {
                    "candidate_id": "CAND-2026-001",
                    "name": "John Doe",
                    "email": "john.doe@example.com",
                    "phone": "+1-555-0123",
                    "position_applied": "Senior Software Engineer",
                    "status": "Interview Scheduled",
                    "stage": "Technical Interview",
                    "applied_date": "2026-08-15",
                    "last_updated": "2026-09-10",
                    "skills": [
                        "Python",
                        "JavaScript",
                        "SQL",
                        "AWS",
                    ],
                    "experience_years": 5,
                    "education": "BS Computer Science",
                    "location": "San Francisco, CA",
                    "resume_url": (
                        "https://storage.policyguard.ai/"
                        "resumes/john_doe.pdf"
                    ),
                    "cover_letter_url": (
                        "https://storage.policyguard.ai/"
                        "letters/john_doe.pdf"
                    ),
                    "interviews": [
                        {
                            "date": "2026-09-05",
                            "type": "Phone Screen",
                            "result": "Passed",
                        },
                        {
                            "date": "2026-09-20",
                            "type": "Technical",
                            "result": "Scheduled",
                        },
                    ],
                },
                "jane.smith@example.com": {
                    "candidate_id": "CAND-2026-002",
                    "name": "Jane Smith",
                    "email": "jane.smith@example.com",
                    "phone": "+1-555-0456",
                    "position_applied": "HR Manager",
                    "status": "Offer Extended",
                    "stage": "Final Review",
                    "applied_date": "2026-07-20",
                    "last_updated": "2026-09-12",
                    "skills": [
                        "HR Management",
                        "Recruiting",
                        "Employee Relations",
                    ],
                    "experience_years": 8,
                    "education": "MBA Human Resources",
                    "location": "New York, NY",
                },
            }

            candidate = mock_candidates.get(
                candidate_id
            )

            if candidate is None:
                return GetCandidateOutput(
                    success=True,
                    message=(
                        "No candidate found for identifier: "
                        f"{input.candidate_id}"
                    ),
                    data={
                        "not_found": True,
                        "searched_id": input.candidate_id,
                    },
                    candidate_data=None,
                )

            candidate_data = dict(
                candidate
            )

            if not input.include_resume:
                candidate_data.pop(
                    "resume_url",
                    None,
                )

            if not input.include_assessments:
                candidate_data.pop(
                    "assessments",
                    None,
                )

            return GetCandidateOutput(
                success=True,
                message=(
                    f"Candidate found: "
                    f"{candidate_data['name']}"
                ),
                data=candidate_data,
                candidate_data=candidate_data,
            )

        except Exception as error:
            logger.error(
                "Candidate lookup error: %s",
                error,
                exc_info=True,
            )

            return GetCandidateOutput(
                success=False,
                message=(
                    f"Failed to retrieve candidate: "
                    f"{_safe_error_message(error)}"
                ),
                error_code="CANDIDATE_LOOKUP_FAILED",
            )

    def _handle_search_jobs(
        self,
        input: SearchJobsInput,
        user_context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> SearchJobsOutput:
        """Mock job search handler."""
        try:
            all_jobs = [
                {
                    "job_id": "JOB-2026-001",
                    "title": "Senior Software Engineer",
                    "department": "Engineering",
                    "location": "Remote",
                    "job_type": "full-time",
                    "experience_required": 5,
                    "posted_date": "2026-09-01",
                    "status": "Open",
                    "openings": 3,
                    "description": (
                        "Build scalable backend systems..."
                    ),
                    "skills_required": [
                        "Python",
                        "AWS",
                        "Microservices",
                    ],
                    "salary_range": "$140k-$180k",
                },
                {
                    "job_id": "JOB-2026-002",
                    "title": "HR Manager",
                    "department": "Human Resources",
                    "location": "New York, NY",
                    "job_type": "full-time",
                    "experience_required": 7,
                    "posted_date": "2026-08-25",
                    "status": "Open",
                    "openings": 1,
                    "description": (
                        "Lead HR initiatives and recruiting..."
                    ),
                    "skills_required": [
                        "HR Management",
                        "Recruiting",
                        "Compliance",
                    ],
                    "salary_range": "$90k-$120k",
                },
                {
                    "job_id": "JOB-2026-003",
                    "title": "Data Analyst",
                    "department": "Analytics",
                    "location": "San Francisco, CA",
                    "job_type": "full-time",
                    "experience_required": 3,
                    "posted_date": "2026-09-05",
                    "status": "Open",
                    "openings": 2,
                    "description": (
                        "Analyze business metrics and create reports..."
                    ),
                    "skills_required": [
                        "SQL",
                        "Python",
                        "Tableau",
                    ],
                    "salary_range": "$80k-$110k",
                },
                {
                    "job_id": "JOB-2026-004",
                    "title": "Frontend Developer",
                    "department": "Engineering",
                    "location": "Remote",
                    "job_type": "contract",
                    "experience_required": 4,
                    "posted_date": "2026-09-08",
                    "status": "Open",
                    "openings": 1,
                    "description": (
                        "Build responsive user interfaces..."
                    ),
                    "skills_required": [
                        "React",
                        "TypeScript",
                        "CSS",
                    ],
                    "salary_range": "$70-$90/hour",
                },
            ]

            jobs = list(all_jobs)

            if input.department:
                department = (
                    input.department.strip().lower()
                )

                jobs = [
                    job
                    for job in jobs
                    if job["department"].lower()
                    == department
                ]

            if input.location:
                location = (
                    input.location.strip().lower()
                )

                jobs = [
                    job
                    for job in jobs
                    if location
                    in job["location"].lower()
                ]

            if input.job_type:
                jobs = [
                    job
                    for job in jobs
                    if job["job_type"]
                    == input.job_type
                ]

            if input.min_experience is not None:
                jobs = [
                    job
                    for job in jobs
                    if job["experience_required"]
                    >= input.min_experience
                ]

            if input.remote_ok is True:
                jobs = [
                    job
                    for job in jobs
                    if job["location"].lower()
                    == "remote"
                ]
            elif input.remote_ok is False:
                jobs = [
                    job
                    for job in jobs
                    if job["location"].lower()
                    != "remote"
                ]

            if input.keywords:
                keywords = [
                    keyword.lower()
                    for keyword
                    in input.keywords.split()
                    if keyword.strip()
                ]

                def matches_keywords(
                    job: Dict[str, Any],
                ) -> bool:
                    searchable = " ".join(
                        [
                            str(job.get("title", "")),
                            str(job.get("description", "")),
                            " ".join(
                                map(
                                    str,
                                    job.get(
                                        "skills_required",
                                        [],
                                    ),
                                )
                            ),
                        ]
                    ).lower()

                    return any(
                        keyword in searchable
                        for keyword in keywords
                    )

                jobs = [
                    job
                    for job in jobs
                    if matches_keywords(job)
                ]

            total_count = len(jobs)
            limited_jobs = jobs[: input.limit]

            return SearchJobsOutput(
                success=True,
                message=(
                    f"Found {len(limited_jobs)} open positions"
                    + (
                        f" (of {total_count} total)"
                        if total_count > len(limited_jobs)
                        else ""
                    )
                ),
                data={
                    "count": len(limited_jobs),
                    "total_available": total_count,
                    "filters_applied": {
                        key: value
                        for key, value
                        in input.model_dump().items()
                        if value is not None
                    },
                },
                jobs=limited_jobs,
                total_count=total_count,
            )

        except Exception as error:
            logger.error(
                "Job search error: %s",
                error,
                exc_info=True,
            )

            return SearchJobsOutput(
                success=False,
                message=(
                    f"Failed to search jobs: "
                    f"{_safe_error_message(error)}"
                ),
                error_code="JOB_SEARCH_FAILED",
            )

    def _handle_request_leave(
        self,
        input: LeaveRequestInput,
        user_context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> LeaveRequestOutput:
        """Mock leave request handler."""
        try:
            start = datetime.strptime(
                input.start_date,
                "%Y-%m-%d",
            )

            end = datetime.strptime(
                input.end_date,
                "%Y-%m-%d",
            )

            days_requested = (
                end - start
            ).days + 1

            request_id = _generate_id(
                "leave"
            )

            mock_balances = {
                "annual": 15.0,
                "sick": 10.0,
                "maternity": 12.0,
                "paternity": 2.0,
                "bereavement": 5.0,
                "unpaid": 999.0,
            }

            current_balance = mock_balances.get(
                input.leave_type,
                0.0,
            )

            if input.leave_type == "unpaid":
                balance_after = current_balance
            else:
                balance_after = max(
                    0.0,
                    current_balance
                    - days_requested,
                )

            approval_status = (
                "auto_approved"
                if (
                    days_requested <= 3
                    and input.leave_type
                    in {
                        "sick",
                        "bereavement",
                    }
                )
                else "pending"
            )

            return LeaveRequestOutput(
                success=True,
                message=(
                    f"Leave request submitted for "
                    f"{input.leave_type} from "
                    f"{input.start_date} to "
                    f"{input.end_date}"
                ),
                data={
                    "request_id": request_id,
                    "employee_id": input.employee_id,
                    "leave_type": input.leave_type,
                    "start_date": input.start_date,
                    "end_date": input.end_date,
                    "days_requested": days_requested,
                    "reason": input.reason,
                    "current_balance": current_balance,
                    "balance_after_approval": balance_after,
                    "approval_workflow": approval_status,
                },
                request_id=request_id,
                approval_status=approval_status,
                days_requested=days_requested,
                balance_after=balance_after,
            )

        except Exception as error:
            logger.error(
                "Leave request error: %s",
                error,
                exc_info=True,
            )

            return LeaveRequestOutput(
                success=False,
                message=(
                    f"Failed to submit leave request: "
                    f"{_safe_error_message(error)}"
                ),
                error_code="LEAVE_REQUEST_FAILED",
            )

    # =============================================================================
    # HELPERS
    # =============================================================================

    def _add_minutes_to_time(
        self,
        iso_datetime: str,
        minutes: int,
    ) -> str:
        """Add minutes to an ISO datetime."""
        dt = datetime.fromisoformat(
            iso_datetime
        )

        return (
            dt + timedelta(minutes=minutes)
        ).isoformat()

    def _generate_ics_event(
        self,
        event: Dict[str, Any],
    ) -> str:
        """Generate a basic RFC 5545-compatible ICS event."""
        def escape_ics(value: str) -> str:
            return (
                str(value)
                .replace("\\", "\\\\")
                .replace(";", "\\;")
                .replace(",", "\\,")
                .replace("\r\n", "\\n")
                .replace("\n", "\\n")
            )

        def format_ics_datetime(
            value: str,
        ) -> str:
            dt = datetime.fromisoformat(
                value
            )

            # Treat mock scheduling values as UTC for the generated
            # demonstration event.
            if dt.tzinfo is None:
                dt = dt.replace(
                    tzinfo=timezone.utc
                )

            dt = dt.astimezone(
                timezone.utc
            )

            return dt.strftime(
                "%Y%m%dT%H%M%SZ"
            )

        uid = (
            f"{uuid.uuid4().hex}"
            "@policyguard.ai"
        )

        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//PolicyGuard AI//Interview Scheduler//EN",
            "CALSCALE:GREGORIAN",
            "METHOD:REQUEST",
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            f"DTSTART:{format_ics_datetime(event['start'])}",
            f"DTEND:{format_ics_datetime(event['end'])}",
            f"SUMMARY:{escape_ics(event.get('summary', 'Interview'))}",
        ]

        if event.get("location"):
            lines.append(
                f"LOCATION:{escape_ics(event['location'])}"
            )

        if event.get("description"):
            lines.append(
                f"DESCRIPTION:{escape_ics(event['description'])}"
            )

        for attendee in event.get(
            "attendees",
            [],
        ):
            lines.append(
                f"ATTENDEE;RSVP=TRUE:MAILTO:{attendee}"
            )

        lines.extend(
            [
                "END:VEVENT",
                "END:VCALENDAR",
            ]
        )

        return "\r\n".join(lines) + "\r\n"

    def shutdown(self) -> None:
        """Shutdown server resources."""
        logger.info(
            "Shutting down MCPServer"
        )
        logger.info(
            "MCPServer shutdown complete"
        )


# =============================================================================
# GLOBAL INSTANCE MANAGEMENT
# =============================================================================

_mcp_server: Optional[MCPServer] = None
_server_lock = threading.RLock()


def get_mcp_server(
    enable_metrics: bool = True,
) -> MCPServer:
    """Get or create the global MCP server singleton."""
    global _mcp_server

    with _server_lock:
        if _mcp_server is None:
            _mcp_server = MCPServer(
                enable_metrics=enable_metrics
            )

        return _mcp_server


def reset_mcp_server() -> None:
    """Reset the global MCP server singleton."""
    global _mcp_server

    with _server_lock:
        server = _mcp_server
        _mcp_server = None

        if server is not None:
            try:
                server.shutdown()
            except Exception as error:
                logger.warning(
                    "MCP server shutdown failed: %s",
                    error,
                )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def execute_mcp_tool(
    tool_name: str,
    input_data: Dict[str, Any],
    user_context: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """Execute a tool using the global MCP server."""
    return get_mcp_server().execute_tool(
        tool_name,
        input_data,
        user_context,
    )


def list_available_tools(
    include_schemas: bool = True,
) -> List[Dict[str, Any]]:
    """List available tools."""
    return get_mcp_server().list_tools(
        include_schemas
    )


def register_custom_tool(
    name: str,
    description: str,
    input_schema: Type[ToolInput],
    output_schema: Type[ToolOutput],
    handler: Callable,
    **kwargs: Any,
) -> None:
    """Register a custom tool."""
    get_mcp_server().register_tool(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        handler=handler,
        **kwargs,
    )


def get_server_metrics(
    tool_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Get MCP server metrics."""
    return get_mcp_server().get_metrics(
        tool_name
    )


# =============================================================================
# TEST / DEMO
# =============================================================================

def test_mcp_server() -> None:
    """Run a basic MCP server smoke test."""
    print("\nTesting MCP Server")
    print("=" * 70)

    server = get_mcp_server()

    print(
        f"Tools registered: {len(server.tools)}"
    )

    print(
        f"Metrics enabled: "
        f"{server.enable_metrics}"
    )

    print("=" * 70)

    # -------------------------------------------------------------------------
    # Tool discovery
    # -------------------------------------------------------------------------

    print("\nAvailable tools")
    print("-" * 70)

    for tool in server.list_tools(
        include_schemas=False
    ):
        print(
            f"- {tool['name']} "
            f"(auth={tool['requires_auth']})"
        )

    # -------------------------------------------------------------------------
    # Authorization test
    # -------------------------------------------------------------------------

    print("\nAuthorization test")
    print("-" * 70)

    denied = execute_mcp_tool(
        "get_candidate",
        {
            "candidate_id": (
                "john.doe@example.com"
            )
        },
    )

    print(
        f"No context -> "
        f"success={denied['success']} "
        f"error={denied.get('error_code')}"
    )

    allowed = execute_mcp_tool(
        "get_candidate",
        {
            "candidate_id": (
                "john.doe@example.com"
            )
        },
        user_context={
            "role": "viewer"
        },
    )

    print(
        f"Viewer -> "
        f"success={allowed['success']}"
    )

    # -------------------------------------------------------------------------
    # Email
    # -------------------------------------------------------------------------

    print("\nEmail test")
    print("-" * 70)

    result = execute_mcp_tool(
        "send_email",
        {
            "to": "candidate@example.com",
            "subject": (
                "Interview Invitation"
            ),
            "body": (
                "Dear Candidate,\n\n"
                "You are invited for an interview."
            ),
            "cc": "hr@company.com",
            "priority": "high",
        },
        user_context={
            "role": "editor"
        },
    )

    print(
        f"Success: {result['success']}"
    )
    print(
        f"Email ID: "
        f"{result.get('email_id')}"
    )

    # -------------------------------------------------------------------------
    # Job search
    # -------------------------------------------------------------------------

    print("\nJob search test")
    print("-" * 70)

    result = execute_mcp_tool(
        "search_jobs",
        {
            "department": "Engineering",
            "job_type": "full-time",
            "limit": 5,
        },
        user_context={
            "role": "viewer"
        },
    )

    print(
        f"Success: {result['success']}"
    )
    print(
        f"Jobs: "
        f"{len(result.get('jobs') or [])}"
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    print("\nValidation test")
    print("-" * 70)

    result = execute_mcp_tool(
        "send_email",
        {
            "to": "invalid-email",
            "subject": "Test",
            "body": "Test body",
        },
    )

    print(
        f"Success: {result['success']}"
    )
    print(
        f"Error: "
        f"{result.get('error_code')}"
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    print("\nMetrics")
    print("-" * 70)

    metrics = get_server_metrics()

    print(
        f"Metrics enabled: "
        f"{metrics.get('metrics_enabled')}"
    )

    for name, values in metrics.get(
        "tools",
        {},
    ).items():
        if values.get("count", 0):
            print(
                f"- {name}: "
                f"{values['count']} calls, "
                f"mean={values['mean_ms']:.1f}ms, "
                f"p95={values['p95_ms']:.1f}ms"
            )

    print("\n" + "=" * 70)
    print("MCP server smoke test complete")


if __name__ == "__main__":
    test_mcp_server()

