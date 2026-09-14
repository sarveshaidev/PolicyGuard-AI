
#!/usr/bin/env python3
"""
PolicyGuard AI - Production Observability Module
=================================================

Provides:
- Optional LangSmith tracing
- Thread-safe local metrics
- Prometheus-compatible metric export
- Structured audit logging
- Safe synchronous/asynchronous trace delivery
- Dashboard/health statistics
- JSON and Prometheus exports

Important:
- LangSmith is optional. The application must continue working if it is
  unavailable or misconfigured.
- User query/answer content is truncated before external tracing.
- Local metrics are kept in bounded ring buffers.
- This module does not own application business logic.

Author: PolicyGuard AI Team
Version: 2.0.0
Last Updated: 2026-09-13
"""

import asyncio
import inspect
import json
import logging
import hashlib
import secrets
import math
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
# CONSTANTS
# =============================================================================

DEFAULT_METRIC_BUFFER_SIZE = 1000

# Prevent very large prompts/responses from being sent to an external
# observability provider.
MAX_TRACE_QUERY_CHARS = 2_000
MAX_TRACE_ANSWER_CHARS = 2_000

SUPPORTED_EXPORT_FORMATS = {"json", "prometheus"}


# =============================================================================
# HELPERS
# =============================================================================

def _utc_now() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float, otherwise return None."""
    try:
        result = float(value)

        if not math.isfinite(result):
            return None

        return result
    except (TypeError, ValueError):
        return None


def _truncate_text(value: Any, max_chars: int) -> str:
    """Convert arbitrary input to bounded text."""
    if value is None:
        return ""

    text = str(value)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "…"


def _sanitize_metric_name(metric_name: str) -> str:
    """
    Convert a metric name to a Prometheus-compatible-ish name.

    This does not attempt to implement every Prometheus naming rule, but
    prevents spaces and common punctuation from producing invalid output.
    """
    if not isinstance(metric_name, str):
        raise ValueError("metric_name must be a string")

    metric_name = metric_name.strip()

    if not metric_name:
        raise ValueError("metric_name cannot be empty")

    result = []

    for char in metric_name:
        if char.isalnum() or char == "_":
            result.append(char)
        else:
            result.append("_")

    return "".join(result).strip("_") or "metric"


# =============================================================================
# PRIVACY HELPERS
# =============================================================================

def _pseudonymous_id(value: Any) -> str:
    """Return a stable non-reversible identifier for external observability."""
    text = str(value or "").strip()
    if not text:
        return "anonymous"
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]


def _safe_external_metadata(value: Any, max_items: int = 30) -> Any:
    """Recursively bound metadata and remove obvious secret-bearing fields."""
    secret_keys = {"api_key", "token", "authorization", "password", "secret", "access_token", "refresh_token", "cookie"}
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                break
            key_text = str(key)
            if key_text.lower() in secret_keys or any(term in key_text.lower() for term in ("api_key", "password", "authorization", "access_token", "refresh_token")):
                continue
            result[key_text[:100]] = _safe_external_metadata(item, max_items)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_external_metadata(item, max_items) for item in list(value)[:max_items]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _truncate_text(value, 500) if isinstance(value, str) else value
    return _truncate_text(value, 500)

# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """
    Thread-safe bounded metrics collector.

    Values are stored in ring buffers so memory usage remains bounded.
    """

    def __init__(self, buffer_size: int = DEFAULT_METRIC_BUFFER_SIZE):
        if isinstance(buffer_size, bool) or not isinstance(buffer_size, int):
            raise ValueError("buffer_size must be an integer")

        if buffer_size <= 0:
            raise ValueError("buffer_size must be greater than zero")

        self.buffer_size = buffer_size

        self._metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.buffer_size)
        )

        self._lock = threading.RLock()

    def record(
        self,
        metric_name: str,
        value: Union[int, float],
    ) -> bool:
        """Record one numeric metric value."""
        try:
            name = _sanitize_metric_name(metric_name)
            numeric_value = _safe_float(value)

            if numeric_value is None:
                logger.warning(
                    "Rejected invalid metric value: %r=%r",
                    metric_name,
                    value,
                )
                return False

            with self._lock:
                self._metrics[name].append(numeric_value)

            return True

        except ValueError as exc:
            logger.warning("Metric record rejected: %s", exc)
            return False

    def record_batch(
        self,
        metric_name: str,
        values: List[Union[int, float]],
    ) -> int:
        """Record multiple metric values and return the number accepted."""
        if not isinstance(values, (list, tuple)):
            raise ValueError("values must be a list or tuple")

        accepted = 0

        for value in values:
            if self.record(metric_name, value):
                accepted += 1

        return accepted

    def get(self, metric_name: str) -> List[Union[int, float]]:
        """Return a snapshot of metric values."""
        try:
            name = _sanitize_metric_name(metric_name)
        except ValueError:
            return []

        with self._lock:
            return list(self._metrics.get(name, []))

    def names(self) -> List[str]:
        """Return all known metric names."""
        with self._lock:
            return list(self._metrics.keys())

    def stats(self, metric_name: str) -> Dict[str, float]:
        """Calculate descriptive statistics for a metric."""
        try:
            name = _sanitize_metric_name(metric_name)
        except ValueError:
            return {
                "count": 0,
                "error": "Invalid metric name",
            }

        with self._lock:
            values = list(self._metrics.get(name, []))

        if not values:
            return {
                "count": 0,
                "error": "No data",
            }

        values = [float(value) for value in values]
        values.sort()

        count = len(values)
        mean = sum(values) / count

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / count

        def percentile(percent: float) -> float:
            if count == 1:
                return values[0]

            position = (count - 1) * (percent / 100.0)
            lower = int(position)
            upper = min(lower + 1, count - 1)

            if lower == upper:
                return values[lower]

            fraction = position - lower

            return (
                values[lower]
                + (values[upper] - values[lower]) * fraction
            )

        return {
            "count": count,
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
            "min": float(values[0]),
            "max": float(values[-1]),
            "p50": float(percentile(50)),
            "p90": float(percentile(90)),
            "p95": float(percentile(95)),
            "p99": float(percentile(99)),
        }

    def to_prometheus_format(self) -> str:
        """
        Export current metric summaries in Prometheus text format.

        These are local summary gauges, not native Prometheus counters.
        """
        lines: List[str] = []

        with self._lock:
            metric_names = list(self._metrics.keys())

        for name in metric_names:
            stats = self.stats(name)

            if "error" in stats:
                continue

            prom_name = f"policyguard_{_sanitize_metric_name(name)}"

            lines.append(
                f"# HELP {prom_name} "
                f"Statistics for {name.replace('_', ' ')}"
            )
            lines.append(
                f"# TYPE {prom_name} gauge"
            )

            lines.append(
                f'{prom_name}{{stat="mean"}} '
                f'{stats["mean"]:.6f}'
            )
            lines.append(
                f'{prom_name}{{stat="p50"}} '
                f'{stats["p50"]:.6f}'
            )
            lines.append(
                f'{prom_name}{{stat="p95"}} '
                f'{stats["p95"]:.6f}'
            )
            lines.append(
                f'{prom_name}{{stat="p99"}} '
                f'{stats["p99"]:.6f}'
            )
            lines.append(
                f'{prom_name}{{stat="count"}} '
                f'{int(stats["count"])}'
            )
            lines.append("")

        return "\n".join(lines)

    def clear(self, metric_name: Optional[str] = None) -> None:
        """Clear one metric or all metrics."""
        with self._lock:
            if metric_name is not None:
                try:
                    name = _sanitize_metric_name(metric_name)
                except ValueError:
                    return

                self._metrics.pop(name, None)
            else:
                self._metrics.clear()


# =============================================================================
# OBSERVABILITY MANAGER
# =============================================================================

class ObservabilityManager:
    """
    Unified observability manager.

    LangSmith remains optional. Local metrics and audit logging continue to
    function even when LangSmith is unavailable.
    """

    def __init__(self):
        """Initialize observability."""
        self.metrics_enabled = bool(
            getattr(settings, "ENABLE_METRICS", True)
        )

        self.langsmith_enabled = bool(
            getattr(settings, "ENABLE_TRACING", False)
            and getattr(settings, "LANGSMITH_API_KEY", None)
            and getattr(settings, "LANGSMITH_PROJECT", None)
        )

        self.metrics = (
            MetricsCollector(
                buffer_size=DEFAULT_METRIC_BUFFER_SIZE
            )
            if self.metrics_enabled
            else None
        )

        self.langsmith_client = None

        self._trace_lock = threading.RLock()

        # Background trace futures/tasks are tracked so shutdown can clean
        # them up safely.
        self._trace_futures = set()
        self._trace_futures_lock = threading.RLock()

        if self.langsmith_enabled:
            self._init_langsmith()

        logger.info(
            "ObservabilityManager initialized: "
            "LangSmith=%s, Metrics=%s",
            self.langsmith_enabled,
            self.metrics_enabled,
        )

    # -------------------------------------------------------------------------
    # LANGSMITH
    # -------------------------------------------------------------------------

    def _init_langsmith(self) -> None:
        """Initialize the optional LangSmith client."""
        try:
            from langsmith import Client

            endpoint = getattr(
                settings,
                "LANGSMITH_ENDPOINT",
                None,
            )

            api_key = getattr(
                settings,
                "LANGSMITH_API_KEY",
                None,
            )

            # Do not assume a particular Client constructor signature.
            # LangSmith versions differ.
            client_kwargs: Dict[str, Any] = {}

            if api_key:
                client_kwargs["api_key"] = api_key

            if endpoint:
                client_kwargs["api_url"] = endpoint

            try:
                self.langsmith_client = Client(**client_kwargs)
            except TypeError:
                # Older/newer versions may not accept api_url.
                client_kwargs.pop("api_url", None)
                self.langsmith_client = Client(**client_kwargs)

            logger.info(
                "LangSmith client initialized: project=%s endpoint=%s",
                getattr(settings, "LANGSMITH_PROJECT", None),
                endpoint or "default",
            )

        except ImportError:
            logger.warning(
                "langsmith package is not installed. "
                "LangSmith tracing has been disabled."
            )
            self.langsmith_enabled = False

        except Exception as exc:
            logger.warning(
                "LangSmith initialization failed; tracing disabled: %s",
                exc,
            )
            self.langsmith_client = None
            self.langsmith_enabled = False

    def start_trace(
        self,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Start a lightweight trace context.

        The actual LangSmith network call occurs at end_trace(), so starting
        a trace does not introduce network latency.
        """
        if not self.langsmith_enabled or self.langsmith_client is None:
            return None

        if not isinstance(operation, str) or not operation.strip():
            logger.warning("Cannot start trace with empty operation")
            return None

        trace_context = {
            "operation": _sanitize_metric_name(operation)[:100],
            "project": getattr(
                settings,
                "LANGSMITH_PROJECT",
                None,
            ),
            "metadata": _safe_external_metadata(metadata or {}),
            "parent_run_id": parent_run_id,
            "start_time": _utc_now(),
            "inputs": {},
            "outputs": {},
            "error": None,
        }

        logger.debug(
            "Trace started: %s",
            trace_context["operation"],
        )

        return trace_context

    def end_trace(
        self,
        trace_context: Optional[Dict[str, Any]],
        output: Any = None,
        error: Optional[Exception] = None,
        send_async: bool = True,
    ) -> None:
        """
        End a trace.

        If an asyncio event loop is running, asynchronous delivery uses it.
        Otherwise the trace is sent synchronously. This avoids the original
        RuntimeError caused by asyncio.create_task() without a running loop.
        """
        if (
            not self.langsmith_enabled
            or self.langsmith_client is None
            or not trace_context
        ):
            return

        try:
            start_time = trace_context.get("start_time")

            if not isinstance(start_time, datetime):
                start_time = _utc_now()

            end_time = _utc_now()

            duration_ms = (
                end_time - start_time
            ).total_seconds() * 1000

            metadata = dict(
                trace_context.get("metadata") or {}
            )

            metadata.update(
                {
                    "duration_ms": round(duration_ms, 3),
                    "success": error is None,
                }
            )

            if error is not None:
                metadata["error_type"] = type(error).__name__

            inputs = dict(
                trace_context.get("inputs") or {}
            )

            outputs = dict(
                trace_context.get("outputs") or {}
            )

            # Never send raw HR policy, resume, employee, or user content to
            # external observability by default. Keep only lengths and hashes.
            if "query" in inputs:
                query_text = str(inputs.get("query") or "")
                inputs = {
                    "query_length": len(query_text),
                    "query_fingerprint": _pseudonymous_id(query_text),
                }

            if "answer" in outputs:
                answer_text = str(outputs.get("answer") or "")
                outputs = {
                    "answer_length": len(answer_text),
                    "answer_fingerprint": _pseudonymous_id(answer_text),
                }

            if not outputs and output is not None:
                output_text = str(output)
                outputs = {
                    "result_length": len(output_text),
                    "result_fingerprint": _pseudonymous_id(output_text),
                }

            inputs = _safe_external_metadata(inputs)
            outputs = _safe_external_metadata(outputs)

            trace_data = {
                "name": trace_context["operation"],
                "run_type": "chain",
                "inputs": inputs,
                "outputs": outputs,
                "start_time": start_time,
                "end_time": end_time,
                "extra": {
                    "metadata": metadata,
                },
            }

            if error is not None:
                trace_data["error"] = _truncate_text(
                    str(error),
                    2_000,
                )

            parent_run_id = trace_context.get("parent_run_id")

            if parent_run_id:
                trace_data["parent_run_id"] = parent_run_id

            project_name = trace_context.get("project")

            if project_name:
                trace_data["project_name"] = project_name

            if send_async:
                self._dispatch_trace_async_or_background(
                    trace_data
                )
            else:
                self._send_trace_sync(trace_data)

            logger.debug(
                "Trace ended: %s (%.1fms, success=%s)",
                trace_context["operation"],
                duration_ms,
                error is None,
            )

        except Exception as exc:
            # Observability must never break the actual RAG request.
            logger.warning(
                "Trace finalization failed: %s",
                exc,
            )

    def _dispatch_trace_async_or_background(
        self,
        trace_data: Dict[str, Any],
    ) -> None:
        """
        Dispatch a trace without assuming an asyncio event loop exists.
        """
        try:
            loop = asyncio.get_running_loop()

        except RuntimeError:
            # No running loop. Send synchronously rather than creating an
            # orphan coroutine/task.
            self._send_trace_sync(trace_data)
            return

        task = loop.create_task(
            self._send_trace_async(trace_data)
        )

        with self._trace_futures_lock:
            self._trace_futures.add(task)

        task.add_done_callback(
            self._trace_task_done
        )

    def _trace_task_done(self, task: asyncio.Task) -> None:
        """Remove completed async trace task from tracking."""
        with self._trace_futures_lock:
            self._trace_futures.discard(task)

        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug(
                "Background trace task failed: %s",
                exc,
            )

    async def _send_trace_async(
        self,
        trace_data: Dict[str, Any],
    ) -> None:
        """Send a trace without blocking the asyncio event loop."""
        try:
            await asyncio.to_thread(
                self._send_trace_sync,
                trace_data,
            )
        except Exception as exc:
            logger.warning(
                "Async trace send failed: %s",
                exc,
            )

    def _send_trace_sync(
        self,
        trace_data: Dict[str, Any],
    ) -> bool:
        """
        Send trace to LangSmith.

        Uses runtime signature detection where possible because LangSmith
        client APIs can vary between installed versions.
        """
        client = self.langsmith_client

        if client is None:
            return False

        try:
            create_run = getattr(
                client,
                "create_run",
                None,
            )

            if create_run is None:
                logger.warning(
                    "Installed LangSmith client does not expose create_run"
                )
                return False

            kwargs = dict(trace_data)

            # Some LangSmith versions don't accept every optional parameter.
            try:
                create_run(**kwargs)
            except TypeError as exc:
                # Retry with the most portable core fields.
                logger.debug(
                    "Retrying LangSmith create_run with compatibility "
                    "parameters after TypeError: %s",
                    exc,
                )

                portable_kwargs = {
                    "name": kwargs.get("name"),
                    "run_type": kwargs.get(
                        "run_type",
                        "chain",
                    ),
                    "inputs": kwargs.get(
                        "inputs",
                        {},
                    ),
                    "outputs": kwargs.get(
                        "outputs",
                        {},
                    ),
                    "start_time": kwargs.get(
                        "start_time"
                    ),
                    "end_time": kwargs.get(
                        "end_time"
                    ),
                    "extra": kwargs.get(
                        "extra",
                        {},
                    ),
                }

                if kwargs.get("error"):
                    portable_kwargs["error"] = kwargs["error"]

                if kwargs.get("project_name"):
                    portable_kwargs["project_name"] = (
                        kwargs["project_name"]
                    )

                if kwargs.get("parent_run_id"):
                    portable_kwargs["parent_run_id"] = (
                        kwargs["parent_run_id"]
                    )

                create_run(**portable_kwargs)

            return True

        except Exception as exc:
            logger.warning(
                "LangSmith trace send failed: %s",
                exc,
            )
            return False

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------

    def record_metric(
        self,
        metric_name: str,
        value: Union[int, float],
    ) -> bool:
        """Record one local metric."""
        if not self.metrics_enabled or self.metrics is None:
            return False

        return self.metrics.record(
            metric_name,
            value,
        )

    # -------------------------------------------------------------------------
    # QUERY OBSERVABILITY
    # -------------------------------------------------------------------------

    def record_query_event(
        self,
        username: str,
        query: str,
        answer: str,
        chunks_used: int,
        latency_ms: float,
        tokens_used: int,
        cost_usd: float,
        model_used: str,
        cache_hit: bool,
        error: Optional[Exception] = None,
        organization_id: Optional[str] = None,
        namespace: str = "policy",
    ) -> None:
        """
        Record a complete RAG query event.

        This method is intentionally best-effort. Observability failures
        should never cause the user's RAG request to fail.
        """
        # ---------------------------------------------------------------------
        # Tenant scope is metadata only here; retrieval authorization remains
        # enforced by the RAG/security layers. Never put raw tenant content in
        # external traces.
        safe_org = _pseudonymous_id(organization_id)
        safe_namespace = _sanitize_metric_name(str(namespace or "policy"))[:50]

        # ---------------------------------------------------------------------
        # Validate/normalize numeric values
        # ---------------------------------------------------------------------

        safe_latency = _safe_float(latency_ms) or 0.0
        safe_cost = _safe_float(cost_usd) or 0.0

        try:
            safe_chunks = max(0, int(chunks_used))
        except (TypeError, ValueError):
            safe_chunks = 0

        try:
            safe_tokens = max(0, int(tokens_used))
        except (TypeError, ValueError):
            safe_tokens = 0

        # ---------------------------------------------------------------------
        # Local metrics
        # ---------------------------------------------------------------------

        if self.metrics_enabled and self.metrics:
            self.metrics.record(
                "query_latency_ms",
                safe_latency,
            )

            self.metrics.record(
                "tokens_per_query",
                safe_tokens,
            )

            self.metrics.record(
                "cost_per_query",
                safe_cost,
            )

            self.metrics.record(
                "cache_hit_indicator",
                1.0 if cache_hit else 0.0,
            )

            self.metrics.record(
                "error_indicator",
                1.0 if error is not None else 0.0,
            )

            self.metrics.record(
                "chunks_per_query",
                safe_chunks,
            )

        # ---------------------------------------------------------------------
        # Database audit log
        # ---------------------------------------------------------------------

        try:
            from src.auth.database import log_query_event

            log_query_event(
                username=str(username),
                query=str(query),
                answer=str(answer),
                chunks_used=safe_chunks,
                latency_ms=int(safe_latency),
                tokens_used=safe_tokens,
                cost_usd=safe_cost,
                model_used=str(model_used),
            )

        except ImportError:
            logger.debug(
                "Database audit logging is unavailable"
            )

        except Exception as exc:
            # Database logging must not break the RAG response.
            logger.warning(
                "Database audit logging failed: %s",
                exc,
            )

        # ---------------------------------------------------------------------
        # Structured application log
        # ---------------------------------------------------------------------

        log_level = (
            logging.ERROR
            if error is not None
            else logging.INFO
        )

        log_msg = (
            "Query event | "
            "user=%s | "
            "latency_ms=%.0f | "
            "tokens=%s | "
            "cost_usd=%.6f | "
            "cache_hit=%s | "
            "chunks=%s | "
            "model=%s"
        )

        if error is not None:
            log_msg += " | error=%s"

            logger.log(
                log_level,
                log_msg,
                username,
                safe_latency,
                safe_tokens,
                safe_cost,
                bool(cache_hit),
                safe_chunks,
                model_used,
                type(error).__name__,
            )
        else:
            logger.log(
                log_level,
                log_msg,
                username,
                safe_latency,
                safe_tokens,
                safe_cost,
                bool(cache_hit),
                safe_chunks,
                model_used,
            )

        # ---------------------------------------------------------------------
        # LangSmith trace
        # ---------------------------------------------------------------------

        if self.langsmith_enabled:
            trace_ctx = self.start_trace(
                operation="rag_query",
                metadata={
                    "user_fingerprint": _pseudonymous_id(username),
                    "organization_fingerprint": safe_org,
                    "namespace": safe_namespace,
                    "query_length": len(str(query)),
                    "answer_length": len(str(answer)),
                    "cache_hit": bool(cache_hit),
                    "model": _truncate_text(
                        model_used,
                        200,
                    ),
                    "chunks_used": safe_chunks,
                    "tokens_used": safe_tokens,
                    "latency_ms": round(
                        safe_latency,
                        3,
                    ),
                    "cost_usd": safe_cost,
                },
            )

            if trace_ctx:
                trace_ctx["inputs"] = {
                    "query": _truncate_text(
                        query,
                        MAX_TRACE_QUERY_CHARS,
                    )
                }

                trace_ctx["outputs"] = {
                    "answer": _truncate_text(
                        answer,
                        MAX_TRACE_ANSWER_CHARS,
                    )
                }

                self.end_trace(
                    trace_ctx,
                    output=answer,
                    error=error,
                    send_async=True,
                )

    # -------------------------------------------------------------------------
    # DASHBOARD
    # -------------------------------------------------------------------------

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Return observability dashboard information.
        """
        dashboard: Dict[str, Any] = {
            "system": {
                "langsmith_enabled": self.langsmith_enabled,
                "metrics_enabled": self.metrics_enabled,
                "timestamp": _utc_now().isoformat(),
                "version": "2.0.0",
            },
            "health": {},
            "metrics": {},
        }

        if not self.metrics_enabled or self.metrics is None:
            return dashboard

        metric_names = [
            "query_latency_ms",
            "tokens_per_query",
            "cost_per_query",
            "cache_hit_indicator",
            "error_indicator",
            "chunks_per_query",
        ]

        stats = {
            name: self.metrics.stats(name)
            for name in metric_names
        }

        dashboard["metrics"] = stats

        latency_stats = stats["query_latency_ms"]
        cost_stats = stats["cost_per_query"]
        cache_stats = stats["cache_hit_indicator"]
        error_stats = stats["error_indicator"]

        dashboard["health"] = {
            "avg_latency_ms": latency_stats.get(
                "mean",
                0.0,
            ),
            "p95_latency_ms": latency_stats.get(
                "p95",
                0.0,
            ),
            "avg_cost_per_query": cost_stats.get(
                "mean",
                0.0,
            ),
            "cache_hit_rate": (
                cache_stats.get("mean", 0.0) * 100
            ),
            "error_rate": (
                error_stats.get("mean", 0.0) * 100
            ),
            "total_queries": latency_stats.get(
                "count",
                0,
            ),
        }

        return dashboard

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------

    def export_metrics(
        self,
        filepath: Optional[Union[str, Path]] = None,
        format: str = "json",
    ) -> Optional[Path]:
        """
        Export local metrics.

        Supported formats:
            json
            prometheus
        """
        if not self.metrics_enabled or self.metrics is None:
            logger.warning(
                "Metrics are disabled; nothing to export"
            )
            return None

        export_format = str(format).strip().lower()

        if export_format not in SUPPORTED_EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported export format: {format}. "
                f"Use one of: {sorted(SUPPORTED_EXPORT_FORMATS)}"
            )

        if filepath is None:
            export_dir = project_root / "data" / "traces"
            export_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            extension = (
                "prom"
                if export_format == "prometheus"
                else "json"
            )

            filepath = (
                export_dir
                / f"metrics_{timestamp}.{extension}"
            )

        else:
            filepath = Path(filepath).expanduser()
            filepath.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        try:
            if export_format == "prometheus":
                content = self.metrics.to_prometheus_format()

                filepath.write_text(
                    content,
                    encoding="utf-8",
                )

            else:
                metric_names = self.metrics.names()

                export_data = {
                    "schema_version": 2,
                    "exported_at": _utc_now().isoformat(),
                    "settings": {
                        "langsmith_enabled": (
                            self.langsmith_enabled
                        ),
                        "metrics_enabled": (
                            self.metrics_enabled
                        ),
                        "buffer_size": (
                            self.metrics.buffer_size
                        ),
                    },
                    "stats": {
                        name: self.metrics.stats(name)
                        for name in metric_names
                    },
                }

                with filepath.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        export_data,
                        file,
                        indent=2,
                        ensure_ascii=False,
                    )

            logger.info(
                "Metrics exported to %s (%s)",
                filepath,
                export_format,
            )

            return filepath

        except (OSError, TypeError, ValueError) as exc:
            logger.error(
                "Metrics export failed: %s",
                exc,
            )
            return None

    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Gracefully release observability resources.

        Pending async trace tasks are cancelled when possible. Metrics are
        cleared because they are in-memory operational data.
        """
        logger.info(
            "Shutting down ObservabilityManager..."
        )

        # Cancel outstanding asyncio tasks associated with this manager.
        with self._trace_futures_lock:
            pending_tasks = list(
                self._trace_futures
            )

            self._trace_futures.clear()

        for task in pending_tasks:
            try:
                if not task.done():
                    task.cancel()
            except Exception:
                pass

        # Disable new trace dispatch.
        self.langsmith_enabled = False
        self.langsmith_client = None

        # Clear local metrics.
        if self.metrics is not None:
            self.metrics.clear()

        logger.info(
            "ObservabilityManager shutdown complete"
        )

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        self.shutdown()
        return False


# =============================================================================
# GLOBAL INSTANCE MANAGEMENT
# =============================================================================

_observability: Optional[ObservabilityManager] = None
_obs_lock = threading.RLock()


def get_observability() -> ObservabilityManager:
    """Get or create the global observability singleton."""
    global _observability

    with _obs_lock:
        if _observability is None:
            _observability = ObservabilityManager()

        return _observability


def reset_observability() -> None:
    """Reset the global observability singleton, primarily for tests."""
    global _observability

    with _obs_lock:
        manager = _observability

        if manager is not None:
            manager.shutdown()

        _observability = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def start_trace(
    operation: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Start a trace using the global observability manager."""
    return get_observability().start_trace(
        operation=operation,
        metadata=metadata,
    )


def end_trace(
    trace_context: Optional[Dict[str, Any]],
    output: Any = None,
    error: Optional[Exception] = None,
    send_async: bool = True,
) -> None:
    """End a trace using the global observability manager."""
    get_observability().end_trace(
        trace_context=trace_context,
        output=output,
        error=error,
        send_async=send_async,
    )


def record_query_event(**kwargs) -> None:
    """Record a query event using the global manager."""
    get_observability().record_query_event(**kwargs)


def record_metric(
    metric_name: str,
    value: Union[int, float],
) -> bool:
    """Record a metric using the global manager."""
    return get_observability().record_metric(
        metric_name,
        value,
    )


def get_dashboard_data() -> Dict[str, Any]:
    """Get dashboard data from the global manager."""
    return get_observability().get_dashboard_data()


def export_metrics(
    filepath: Optional[Union[str, Path]] = None,
    format: str = "json",
) -> Optional[Path]:
    """Export metrics using the global manager."""
    return get_observability().export_metrics(
        filepath=filepath,
        format=format,
    )


# =============================================================================
# TEST / DEMO
# =============================================================================

def test_observability() -> None:
    """Run a basic observability self-test."""
    print("\n📊 Testing Observability Manager")
    print("=" * 70)

    # Use a local manager for a deterministic test.
    obs = ObservabilityManager()

    print(
        f"LangSmith enabled: {obs.langsmith_enabled}"
    )

    print(
        f"Metrics enabled: {obs.metrics_enabled}"
    )

    print(
        f"Project: "
        f"{getattr(settings, 'LANGSMITH_PROJECT', 'N/A')}"
    )

    print("=" * 70)

    # -------------------------------------------------------------------------
    # Test 1: Metrics
    # -------------------------------------------------------------------------

    print("\n1. Metrics recording")
    print("-" * 70)

    test_data = [
        (
            "query_latency_ms",
            [234, 189, 456, 123, 567, 201],
        ),
        (
            "tokens_per_query",
            [150, 200, 175, 300, 125, 220],
        ),
        (
            "cost_per_query",
            [
                0.000015,
                0.000020,
                0.000017,
                0.000030,
                0.000012,
                0.000022,
            ],
        ),
        (
            "cache_hit_indicator",
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        ),
        (
            "error_indicator",
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ),
    ]

    for metric_name, values in test_data:
        accepted = (
            obs.metrics.record_batch(
                metric_name,
                values,
            )
            if obs.metrics
            else 0
        )

        stats = (
            obs.metrics.stats(metric_name)
            if obs.metrics
            else {}
        )

        print(
            f"   {metric_name}: "
            f"accepted={accepted}, "
            f"count={stats.get('count', 0)}, "
            f"mean={stats.get('mean', 0):.6f}, "
            f"p95={stats.get('p95', 0):.6f}"
        )

    # -------------------------------------------------------------------------
    # Test 2: Query event
    # -------------------------------------------------------------------------

    print("\n2. Query event recording")
    print("-" * 70)

    obs.record_query_event(
        username="test_user",
        query="What is the company leave policy?",
        answer=(
            "Please consult the approved company leave policy."
        ),
        chunks_used=3,
        latency_ms=234.5,
        tokens_used=150,
        cost_usd=0.000015,
        model_used=(
            "meta-llama/llama-3.1-8b-instruct"
        ),
        cache_hit=True,
    )

    print(
        "   ✅ Query event recorded"
    )

    # -------------------------------------------------------------------------
    # Test 3: Dashboard
    # -------------------------------------------------------------------------

    print("\n3. Dashboard data")
    print("-" * 70)

    dashboard = obs.get_dashboard_data()

    print(
        "   System:",
        dashboard["system"],
    )

    if dashboard.get("health"):
        health = dashboard["health"]

        print(
            f"   Avg latency: "
            f"{health['avg_latency_ms']:.1f} ms"
        )

        print(
            f"   P95 latency: "
            f"{health['p95_latency_ms']:.1f} ms"
        )

        print(
            f"   Avg cost: "
            f"${health['avg_cost_per_query']:.6f}"
        )

        print(
            f"   Cache hit rate: "
            f"{health['cache_hit_rate']:.1f}%"
        )

        print(
            f"   Error rate: "
            f"{health['error_rate']:.1f}%"
        )

        print(
            f"   Total queries: "
            f"{health['total_queries']}"
        )

    # -------------------------------------------------------------------------
    # Test 4: Export
    # -------------------------------------------------------------------------

    print("\n4. Metrics export")
    print("-" * 70)

    json_path = obs.export_metrics(
        format="json"
    )

    if json_path:
        print(
            f"   ✅ JSON: {json_path}"
        )

    prom_path = obs.export_metrics(
        format="prometheus"
    )

    if prom_path:
        print(
            f"   ✅ Prometheus: {prom_path}"
        )

    # -------------------------------------------------------------------------
    # Test 5: Trace context
    # -------------------------------------------------------------------------

    print("\n5. Trace context")

    trace_ctx = obs.start_trace(
        "test_operation",
        metadata={"test": True},
    )

    if trace_ctx:
        trace_ctx["inputs"] = {
            "query": "Test query"
        }

        trace_ctx["outputs"] = {
            "answer": "Test result"
        }

        time.sleep(0.01)

        obs.end_trace(
            trace_ctx,
            output="Test result",
            send_async=False,
        )

        print(
            "   ✅ Trace lifecycle completed"
        )

    else:
        print(
            "   ℹ️ LangSmith unavailable; "
            "trace lifecycle skipped"
        )

    obs.shutdown()

    print("\n" + "=" * 70)
    print("✅ Observability test complete")


if __name__ == "__main__":
    test_observability()

