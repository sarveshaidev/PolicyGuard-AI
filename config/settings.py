#!/usr/bin/env python3
"""
PolicyGuard AI - Enterprise Production Configuration
======================================================

Centralized, type-safe application configuration for PolicyGuard AI.

Design goals:
- Pydantic v2 / pydantic-settings
- Environment-variable driven configuration
- Windows/Linux/macOS compatible paths
- No hard-coded production secrets
- OpenRouter-compatible LLM configuration
- Centralized RAG/retrieval configuration
- Security/rate-limit configuration
- Recruitment configuration
- Evaluation/RAGAS configuration
- Feature flags
- Safe configuration validation
- Backward-compatible helper methods

IMPORTANT:
    Never commit the real .env file to source control.

Primary LLM:
    meta-llama/llama-3.1-8b-instruct

Embedding model:
    BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Central application configuration.

    Configuration precedence:

        1. Explicit environment variables
        2. .env file
        3. Safe defaults

    Secrets are never intentionally exposed through __str__ or repr().
    """

    # =========================================================================
    # PYDANTIC SETTINGS CONFIGURATION
    # =========================================================================

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_nested_delimiter="__",
        validate_default=True,
    )

    # =========================================================================
    # APPLICATION
    # =========================================================================

    APP_NAME: str = Field(
        default="PolicyGuard AI",
        min_length=1,
        max_length=100,
    )

    APP_VERSION: str = Field(
        default="1.0.0",
        min_length=1,
        max_length=30,
    )

    DEBUG: bool = Field(
        default=False,
        description="Enable development/debug behavior.",
    )

    LOG_LEVEL: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = Field(default="INFO")

    # =========================================================================
    # API / SECRETS
    # =========================================================================

    OPENROUTER_API_KEY: Optional[str] = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
    )

    # Keep this explicit because the supplied .env uses OPENROUTER_MODEL.
    # All three routing settings intentionally default to the same model.
    OPENROUTER_MODEL: str = Field(
        default="meta-llama/llama-3.1-8b-instruct",
    )

    HF_TOKEN: Optional[str] = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    LANGSMITH_API_KEY: Optional[str] = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    LANGSMITH_ENDPOINT: str = Field(
        default="https://api.smith.langchain.com",
    )

    LANGSMITH_PROJECT: str = Field(
        default="PolicyGuard_AI",
    )

    LANGSMITH_TRACING: bool = Field(
        default=False,
    )

    # Must be changed in .env for production.
    SECRET_KEY: str = Field(
        default="",
        repr=False,
        exclude=True,
    )

    ALGORITHM: str = Field(
        default="HS256",
    )

    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=1,
        le=1440,
    )

    # =========================================================================
    # DATABASE
    # =========================================================================

    DATABASE_URL: str = Field(
        default="sqlite:///./nexus_auth.db",
    )

    USE_PGVECTOR: bool = Field(
        default=False,
    )

    POSTGRES_URL: Optional[str] = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    REDIS_URL: Optional[str] = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    # =========================================================================
    # NEO4J
    # =========================================================================

    NEO4J_URI: Optional[str] = Field(
        default=None,
    )

    NEO4J_USER: Optional[str] = Field(
        default=None,
    )

    NEO4J_PASSWORD: Optional[str] = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    # =========================================================================
    # LLM
    # =========================================================================

    # All query-routing models intentionally point to your selected
    # OpenRouter model unless explicitly overridden.
    CHAT_MODEL_SIMPLE: str = Field(
        default="meta-llama/llama-3.1-8b-instruct",
    )

    CHAT_MODEL_REASONING: str = Field(
        default="meta-llama/llama-3.1-8b-instruct",
    )

    CHAT_MODEL_BALANCED: str = Field(
        default="meta-llama/llama-3.1-8b-instruct",
    )

    VISION_MODEL: str = Field(
        default="llava-hf/llava-1.5-7b-hf",
    )

    MAX_TOKENS: int = Field(
        default=1000,
        ge=100,
        le=4000,
    )

    TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
    )

    TOP_P: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
    )

    MAX_RETRIES: int = Field(
        default=2,
        ge=0,
        le=5,
    )

    # =========================================================================
    # EMBEDDINGS
    # =========================================================================

    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5",
    )

    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        ge=1,
        le=512,
    )

    EMBEDDING_MAX_WORKERS: int = Field(
        default=4,
        ge=1,
        le=32,
    )

    # =========================================================================
    # VECTOR / RAG CONFIGURATION
    # =========================================================================

    VECTOR_DB_PATH_ENV: Optional[str] = Field(
        default=None,
        validation_alias="VECTOR_DB_PATH",
    )

    TOP_K: int = Field(
        default=5,
        ge=1,
        le=100,
    )

    RERANK_TOP_K: int = Field(
        default=3,
        ge=1,
        le=50,
    )

    HYBRID_ALPHA: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
    )

    CHUNK_SIZE: int = Field(
        default=300,
        ge=100,
        le=5000,
    )

    CHUNK_OVERLAP: int = Field(
        default=30,
        ge=0,
        le=1000,
    )

    CACHE_SIMILARITY_THRESHOLD: float = Field(
        default=0.90,
        ge=0.50,
        le=1.0,
    )

    CACHE_TTL_SECONDS: int = Field(
        default=1800,
        ge=60,
        le=604800,
    )

    # =========================================================================
    # SECURITY
    # =========================================================================

    MAX_QUERY_LENGTH: int = Field(
        default=1500,
        ge=10,
        le=10000,
    )

    MIN_QUERY_LENGTH: int = Field(
        default=5,
        ge=1,
        le=1000,
    )

    MAX_UPLOAD_SIZE_MB: int = Field(
        default=25,
        ge=1,
        le=500,
    )

    RATE_LIMIT_PER_MINUTE: int = Field(
        default=30,
        ge=1,
        le=1000,
    )

    SECURITY_MAX_SPECIAL_CHAR_RATIO: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
    )

    SECURITY_MAX_REPEATED_CHARS: int = Field(
        default=5,
        ge=2,
        le=100,
    )

    SECURITY_PII_REDACTION_ENABLED: bool = Field(
        default=True,
    )

    # =========================================================================
    # COST CONTROL
    # =========================================================================

    ENABLE_COST_TRACKING: bool = Field(
        default=True,
    )

    MAX_COST_PER_QUERY: float = Field(
        default=0.005,
        ge=0.0,
        le=100.0,
    )

    DAILY_BUDGET_LIMIT: float = Field(
        default=2.0,
        ge=0.0,
        le=100000.0,
    )

    # =========================================================================
    # OCR
    # =========================================================================

    OCR_MAX_WORKERS: int = Field(
        default=2,
        ge=1,
        le=32,
    )

    OCR_LANGUAGES: str = Field(
        default="en",
    )

    OCR_USE_GPU: bool = Field(
        default=False,
    )

    OCR_ENABLE_PREPROCESSING: bool = Field(
        default=True,
    )

    OCR_CONFIDENCE_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    OCR_PDF_DPI: int = Field(
        default=300,
        ge=72,
        le=600,
    )

    # =========================================================================
    # CROSS ENCODER
    # =========================================================================

    CROSS_ENCODER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    CROSS_ENCODER_BATCH_SIZE: int = Field(
        default=32,
        ge=1,
        le=512,
    )

    # =========================================================================
    # MCP
    # =========================================================================

    MCP_MAX_WORKERS: int = Field(
        default=4,
        ge=1,
        le=32,
    )

    # =========================================================================
    # EVALUATION / OBSERVABILITY
    # =========================================================================

    ENABLE_METRICS: bool = Field(
        default=True,
    )

    EVAL_DATASET_PATH_ENV: Optional[str] = Field(
        default=None,
        validation_alias="EVAL_DATASET_PATH",
    )

    EVAL_METRICS: List[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ],
    )

    USE_HEURISTIC_EVALUATION: bool = Field(
        default=False,
    )

    # =========================================================================
    # PATH ENVIRONMENT OVERRIDES
    #
    # We intentionally don't call these DATA_DIR etc. because those names are
    # already used by our computed project-root properties below.
    # =========================================================================

    DATA_DIR_ENV: Optional[str] = Field(
        default=None,
        validation_alias="DATA_DIR",
    )

    TRACE_DIR_ENV: Optional[str] = Field(
        default=None,
        validation_alias="TRACE_DIR",
    )

    UPLOAD_DIR_ENV: Optional[str] = Field(
        default=None,
        validation_alias="UPLOAD_DIR",
    )

    LOG_DIR_ENV: Optional[str] = Field(
        default=None,
        validation_alias="LOG_DIR",
    )

    # =========================================================================
    # STREAMLIT
    # =========================================================================

    STREAMLIT_SERVER_PORT: int = Field(
        default=8501,
        ge=1,
        le=65535,
    )

    STREAMLIT_SERVER_HEADLESS: bool = Field(
        default=True,
    )

    STREAMLIT_SERVER_ADDRESS: str = Field(
        default="0.0.0.0",
    )

    # =========================================================================
    # FEATURE FLAGS
    # =========================================================================

    ENABLE_HYBRID_SEARCH: bool = Field(
        default=True,
    )

    ENABLE_CROSS_ENCODER_RERANKING: bool = Field(
        default=True,
    )

    ENABLE_SEMANTIC_CACHE: bool = Field(
        default=True,
    )

    ENABLE_MCP_SERVER: bool = Field(
        default=True,
    )

    ENABLE_ADVANCED_OCR: bool = Field(
        default=True,
    )

    # =========================================================================
    # RBAC
    # =========================================================================

    ROLES: Dict[str, Dict[str, List[str]]] = Field(
        default_factory=lambda: {
            "viewer": {
                "permissions": [
                    "chat",
                    "view_policies",
                    "view_own_history",
                ]
            },
            "editor": {
                "permissions": [
                    "chat",
                    "view_policies",
                    "view_own_history",
                    "upload_docs",
                    "manage_documents",
                    "jd_matching",
                    "screen_candidates",
                    "view_recruitment",
                ]
            },
            "admin": {
                "permissions": ["*"]
            },
        }
    )

    # =========================================================================
    # PROJECT PATHS
    # =========================================================================

    @property
    def BASE_DIR(self) -> Path:
        """Absolute project root directory."""
        return Path(__file__).resolve().parent.parent

    @staticmethod
    def _resolve_project_path(
        base_dir: Path,
        configured_path: Optional[str],
        default_relative: str,
    ) -> Path:
        """
        Resolve an environment-configured path.

        Relative paths are resolved against the project root.
        Absolute paths are preserved.
        """
        raw = configured_path or default_relative
        path = Path(raw).expanduser()

        if not path.is_absolute():
            path = base_dir / path

        return path.resolve()

    @property
    def DATA_DIR(self) -> Path:
        """Canonical document storage directory."""
        return self._resolve_project_path(
            self.BASE_DIR,
            self.DATA_DIR_ENV,
            "data/documents",
        )

    @property
    def TRACE_DIR(self) -> Path:
        """Tracing/observability directory."""
        return self._resolve_project_path(
            self.BASE_DIR,
            self.TRACE_DIR_ENV,
            "data/traces",
        )

    @property
    def VECTOR_DB_PATH(self) -> Path:
        """FAISS/BM25 vector database directory."""
        return self._resolve_project_path(
            self.BASE_DIR,
            self.VECTOR_DB_PATH_ENV,
            "data/vector_db",
        )

    @property
    def UPLOAD_DIR(self) -> Path:
        """Uploaded-file working directory."""
        return self._resolve_project_path(
            self.BASE_DIR,
            self.UPLOAD_DIR_ENV,
            "data/uploads",
        )

    @property
    def LOG_DIR(self) -> Path:
        """Application log directory."""
        return self._resolve_project_path(
            self.BASE_DIR,
            self.LOG_DIR_ENV,
            "logs",
        )

    @property
    def EVAL_DATASET_PATH(self) -> Path:
        """Ground-truth evaluation dataset."""
        configured = self.EVAL_DATASET_PATH_ENV

        return self._resolve_project_path(
            self.BASE_DIR,
            configured,
            "data/ground_truth.json",
        )

    # =========================================================================
    # DERIVED CONFIGURATION
    # =========================================================================

    @property
    def VECTOR_DIMENSION(self) -> Optional[int]:
        """
        Expected embedding dimension.

        BGE-small-en-v1.5 produces 384-dimensional embeddings.

        This is kept as a derived configuration value rather than forcing
        vector_store.py to hard-code the dimension.
        """
        model = self.EMBEDDING_MODEL.lower()

        known_dimensions = {
            "baai/bge-small-en-v1.5": 384,
        }

        return known_dimensions.get(model)

    @property
    def ROLE_HIERARCHY(self) -> Dict[str, int]:
        """
        Role hierarchy used by application-level RBAC.

        Higher number = greater privilege.
        """
        return {
            "viewer": 1,
            "editor": 2,
            "admin": 3,
        }

    @property
    def OPENROUTER_CHAT_MODEL(self) -> str:
        """
        Canonical LLM model used by the application.

        This keeps compatibility with code that may use either
        OPENROUTER_MODEL or CHAT_MODEL_*.
        """
        return self.OPENROUTER_MODEL

    @property
    def ENABLE_TRACING(self) -> bool:
        """
        Backward-compatible alias for LANGSMITH_TRACING.

        Some older orchestration code refers to ENABLE_TRACING while the
        canonical configuration name is LANGSMITH_TRACING.
        """
        return self.LANGSMITH_TRACING

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def normalize_log_level(cls, value: Any) -> str:
        """Normalize logging level to uppercase."""
        if value is None:
            return "INFO"

        value = str(value).strip().upper()

        allowed = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if value not in allowed:
            raise ValueError(
                f"LOG_LEVEL must be one of: {sorted(allowed)}"
            )

        return value

    @field_validator("OPENROUTER_BASE_URL", mode="before")
    @classmethod
    def normalize_openrouter_url(cls, value: Any) -> str:
        """Normalize OpenRouter base URL."""
        value = str(value or "").strip().rstrip("/")

        if not value:
            raise ValueError("OPENROUTER_BASE_URL cannot be empty")

        if not value.startswith(("http://", "https://")):
            raise ValueError(
                "OPENROUTER_BASE_URL must start with http:// or https://"
            )

        return value

    @field_validator("EVAL_METRICS", mode="before")
    @classmethod
    def parse_eval_metrics(cls, value: Any) -> List[str]:
        """
        Parse evaluation metrics from:

            ["faithfulness", "answer_relevancy"]

        or:

            ["faithfulness","answer_relevancy"]

        or:

            faithfulness,answer_relevancy
        """
        if value is None:
            return [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]

        if isinstance(value, list):
            metrics = value

        elif isinstance(value, str):
            raw = value.strip()

            if not raw:
                return []

            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)

                    if not isinstance(parsed, list):
                        raise ValueError(
                            "EVAL_METRICS JSON value must be a list"
                        )

                    metrics = parsed

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "EVAL_METRICS must be valid JSON list or "
                        "comma-separated values"
                    ) from exc
            else:
                metrics = raw.split(",")

        else:
            raise ValueError(
                "EVAL_METRICS must be a list or string"
            )

        cleaned = []

        for metric in metrics:
            metric_name = str(metric).strip()

            if metric_name and metric_name not in cleaned:
                cleaned.append(metric_name)

        return cleaned

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info) -> int:
        """Prevent overlap from being greater than chunk size."""
        chunk_size = info.data.get("CHUNK_SIZE", 300)

        if value >= chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
            )

        return value

    @field_validator("MIN_QUERY_LENGTH")
    @classmethod
    def validate_min_query_length(cls, value: int, info) -> int:
        """Ensure minimum query length doesn't exceed maximum."""
        max_length = info.data.get("MAX_QUERY_LENGTH", 1500)

        if value >= max_length:
            raise ValueError(
                "MIN_QUERY_LENGTH must be smaller than MAX_QUERY_LENGTH"
            )

        return value

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        """
        Cross-field production validation.

        We deliberately do NOT require an OpenRouter key during object
        construction so that:
        - tests can import settings
        - local development can start
        - UI can display configuration state

        Runtime code should call validate_runtime_requirements()
        before making an LLM request.
        """

        if self.USE_PGVECTOR and not self.POSTGRES_URL:
            raise ValueError(
                "USE_PGVECTOR=true requires POSTGRES_URL"
            )

        if self.ENABLE_SEMANTIC_CACHE and self.CACHE_TTL_SECONDS <= 0:
            raise ValueError(
                "CACHE_TTL_SECONDS must be greater than zero"
            )

        # Production traffic must never send credentials over plain HTTP.
        if self.is_production() and self.OPENROUTER_BASE_URL.startswith("http://"):
            raise ValueError(
                "OPENROUTER_BASE_URL must use HTTPS when DEBUG=false."
            )

        for model_name, model_value in (
            ("OPENROUTER_MODEL", self.OPENROUTER_MODEL),
            ("CHAT_MODEL_SIMPLE", self.CHAT_MODEL_SIMPLE),
            ("CHAT_MODEL_REASONING", self.CHAT_MODEL_REASONING),
            ("CHAT_MODEL_BALANCED", self.CHAT_MODEL_BALANCED),
            ("EMBEDDING_MODEL", self.EMBEDDING_MODEL),
        ):
            if not str(model_value).strip():
                raise ValueError(f"{model_name} cannot be empty")

        if self.LANGSMITH_TRACING and not self.LANGSMITH_API_KEY:
            logger.warning(
                "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not configured. "
                "Tracing will not be operational."
            )

        if self.MAX_COST_PER_QUERY > self.DAILY_BUDGET_LIMIT:
            logger.warning(
                "MAX_COST_PER_QUERY exceeds DAILY_BUDGET_LIMIT. "
                "A single query could theoretically exceed the daily budget."
            )

        # Ensure directories exist.
        #
        # This is intentionally limited to application-owned directories.
        for directory in (
            self.DATA_DIR,
            self.TRACE_DIR,
            self.VECTOR_DB_PATH,
            self.UPLOAD_DIR,
            self.LOG_DIR,
        ):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning(
                    "Could not create directory %s: %s",
                    directory,
                    exc,
                )

        return self

    # =========================================================================
    # RUNTIME VALIDATION
    # =========================================================================

    def validate_runtime_requirements(self) -> List[str]:
        """
        Return runtime configuration warnings/errors.

        This method does not raise, making it suitable for health checks,
        admin dashboards, and startup diagnostics.
        """
        issues: List[str] = []

        if not self.OPENROUTER_API_KEY:
            issues.append(
                "OPENROUTER_API_KEY is not configured. "
                "LLM requests cannot be executed."
            )

        if self.is_production():
            if not self.SECRET_KEY:
                issues.append(
                    "SECRET_KEY is not configured in production."
                )
            elif len(self.SECRET_KEY) < 32:
                issues.append(
                    "SECRET_KEY must contain at least 32 characters."
                )

            if self.OPENROUTER_BASE_URL.startswith("http://"):
                issues.append(
                    "OPENROUTER_BASE_URL must use HTTPS in production."
                )

        elif self.SECRET_KEY and len(self.SECRET_KEY) < 32:
            issues.append(
                "SECRET_KEY should contain at least 32 characters."
            )

        if self.LANGSMITH_TRACING and not self.LANGSMITH_API_KEY:
            issues.append(
                "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is missing."
            )

        if self.USE_PGVECTOR and not self.POSTGRES_URL:
            issues.append(
                "USE_PGVECTOR is enabled but POSTGRES_URL is missing."
            )

        if not self.OPENROUTER_MODEL.strip():
            issues.append("OPENROUTER_MODEL is empty.")

        if self.VECTOR_DIMENSION is None:
            issues.append(
                f"Unknown embedding dimension for model '{self.EMBEDDING_MODEL}'. "
                "Verify vector-store compatibility before indexing."
            )

        return issues

    def is_ready_for_llm(self) -> bool:
        """Return whether the application can make an OpenRouter request."""
        return bool(
            self.OPENROUTER_API_KEY
            and self.OPENROUTER_BASE_URL
            and self.OPENROUTER_MODEL
        )

    def require_runtime_ready(self) -> None:
        """
        Fail closed when production startup/runtime requirements are missing.

        Importing Settings remains safe for tests and diagnostics; deployment
        code can explicitly call this gate before serving user traffic.
        """
        issues = self.validate_runtime_requirements()

        if issues:
            raise RuntimeError(
                "PolicyGuard AI runtime configuration is not ready:\n- "
                + "\n- ".join(issues)
            )

    def is_production(self) -> bool:
        """Return whether production behavior is enabled."""
        return not self.DEBUG

    # =========================================================================
    # MODEL ROUTING
    # =========================================================================

    def get_model_for_query(
        self,
        query: str,
        complexity: Optional[float] = None,
    ) -> str:
        """
        Select an LLM model based on query complexity.

        Your current project intentionally uses the same OpenRouter model
        for all routing levels:

            meta-llama/llama-3.1-8b-instruct

        Keeping the routing abstraction allows the architecture to evolve
        later without rewriting the orchestration layer.
        """
        if not query:
            return self.OPENROUTER_MODEL

        if complexity is None:
            word_count = len(query.split())

            reasoning_keywords = (
                "why",
                "how",
                "analyze",
                "analyse",
                "compare",
                "evaluate",
                "explain",
                "difference",
                "recommend",
                "summarize",
                "assess",
            )

            has_reasoning = any(
                keyword in query.lower()
                for keyword in reasoning_keywords
            )

            if word_count < 15 and not has_reasoning:
                return self.CHAT_MODEL_SIMPLE

            return self.CHAT_MODEL_REASONING

        complexity = max(0.0, min(1.0, complexity))

        if complexity > 0.7:
            return self.CHAT_MODEL_REASONING

        if complexity > 0.4:
            return self.CHAT_MODEL_BALANCED

        return self.CHAT_MODEL_SIMPLE

    # =========================================================================
    # OPENROUTER HELPERS
    # =========================================================================

    def get_openrouter_headers(self) -> Dict[str, str]:
        """
        Return headers required for OpenRouter.

        IMPORTANT:
            HF_TOKEN must NEVER overwrite the OpenRouter Authorization
            header. They are credentials for different services.
        """
        if not self.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured."
            )

        return {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://policyguard-ai.local",
            "X-Title": self.APP_NAME,
        }

    def get_api_headers(self) -> Dict[str, str]:
        """
        Backward-compatible alias for existing application code.

        Returns OpenRouter headers only.
        """
        return self.get_openrouter_headers()

    def get_huggingface_token(self) -> Optional[str]:
        """Return the optional Hugging Face token."""
        return self.HF_TOKEN

    # =========================================================================
    # RBAC HELPERS
    # =========================================================================

    def has_permission(
        self,
        role: str,
        permission: str,
    ) -> bool:
        """Check whether a role has a specific permission."""
        role_config = self.ROLES.get(role)

        if not role_config:
            return False

        permissions = role_config.get("permissions", [])

        return "*" in permissions or permission in permissions

    def role_at_least(
        self,
        current_role: str,
        required_role: str,
    ) -> bool:
        """Check role hierarchy."""
        current_level = self.ROLE_HIERARCHY.get(current_role, 0)
        required_level = self.ROLE_HIERARCHY.get(required_role, 999)

        return current_level >= required_level

    # =========================================================================
    # SERIALIZATION / DISPLAY
    # =========================================================================

    def safe_dict(self) -> Dict[str, Any]:
        """
        Return a safe configuration snapshot.

        Secrets are intentionally excluded.
        """
        return {
            "app_name": self.APP_NAME,
            "app_version": self.APP_VERSION,
            "debug": self.DEBUG,
            "log_level": self.LOG_LEVEL,
            "openrouter_base_url": self.OPENROUTER_BASE_URL,
            "openrouter_model": self.OPENROUTER_MODEL,
            "embedding_model": self.EMBEDDING_MODEL,
            "vector_dimension": self.VECTOR_DIMENSION,
            "database_url": self._redact_database_url(
                self.DATABASE_URL
            ),
            "use_pgvector": self.USE_PGVECTOR,
            "top_k": self.TOP_K,
            "rerank_top_k": self.RERANK_TOP_K,
            "hybrid_alpha": self.HYBRID_ALPHA,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "semantic_cache_enabled": self.ENABLE_SEMANTIC_CACHE,
            "cross_encoder_enabled": self.ENABLE_CROSS_ENCODER_RERANKING,
            "hybrid_search_enabled": self.ENABLE_HYBRID_SEARCH,
            "advanced_ocr_enabled": self.ENABLE_ADVANCED_OCR,
            "mcp_enabled": self.ENABLE_MCP_SERVER,
            "metrics_enabled": self.ENABLE_METRICS,
            "langsmith_tracing": self.LANGSMITH_TRACING,
            "cost_tracking": self.ENABLE_COST_TRACKING,
            "data_dir": str(self.DATA_DIR),
            "vector_db_path": str(self.VECTOR_DB_PATH),
            "eval_dataset_path": str(self.EVAL_DATASET_PATH),
        }

    @staticmethod
    def _redact_database_url(url: str) -> str:
        """
        Hide database credentials in diagnostics.

        Example:
            postgresql://user:password@host/db

        becomes:

            postgresql://user:***@host/db
        """
        if "://" not in url:
            return url

        try:
            scheme, remainder = url.split("://", 1)

            if "@" not in remainder:
                return url

            credentials, host = remainder.split("@", 1)

            if ":" in credentials:
                username, _password = credentials.split(":", 1)
                credentials = f"{username}:***"

            return f"{scheme}://{credentials}@{host}"

        except Exception:
            return "***REDACTED***"

    def __str__(self) -> str:
        return (
            f"Settings("
            f"APP_NAME={self.APP_NAME!r}, "
            f"APP_VERSION={self.APP_VERSION!r}, "
            f"DEBUG={self.DEBUG}, "
            f"MODEL={self.OPENROUTER_MODEL!r}"
            f")"
        )

    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# GLOBAL SETTINGS SINGLETON
# =============================================================================

settings = Settings()


def get_settings() -> Settings:
    """Return the application settings singleton."""
    return settings


def reload_settings() -> Settings:
    """
    Reload configuration.

    Primarily useful for tests and controlled development workflows.
    """
    global settings

    settings = Settings()

    return settings


def validate_settings() -> List[str]:
    """
    Return configuration warnings/errors.

    Suitable for startup diagnostics.
    """
    return settings.validate_runtime_requirements()


# =============================================================================
# CLI DIAGNOSTICS
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("PolicyGuard AI - Configuration Diagnostics")
    print("=" * 72)

    print(f"Application : {settings.APP_NAME}")
    print(f"Version     : {settings.APP_VERSION}")
    print(f"Debug       : {settings.DEBUG}")
    print(f"Environment : {'PRODUCTION' if settings.is_production() else 'DEVELOPMENT'}")

    print()
    print("LLM")
    print(f"  Provider  : OpenRouter")
    print(f"  Model     : {settings.OPENROUTER_MODEL}")
    print(f"  Ready     : {settings.is_ready_for_llm()}")

    print()
    print("Embeddings")
    print(f"  Model     : {settings.EMBEDDING_MODEL}")
    print(f"  Dimension : {settings.VECTOR_DIMENSION}")

    print()
    print("RAG")
    print(f"  TOP_K     : {settings.TOP_K}")
    print(f"  RERANK_K  : {settings.RERANK_TOP_K}")
    print(f"  Alpha     : {settings.HYBRID_ALPHA}")
    print(f"  Chunk     : {settings.CHUNK_SIZE}")
    print(f"  Overlap   : {settings.CHUNK_OVERLAP}")

    print()
    print("Paths")
    print(f"  Base      : {settings.BASE_DIR}")
    print(f"  Data      : {settings.DATA_DIR}")
    print(f"  Vector DB : {settings.VECTOR_DB_PATH}")
    print(f"  Uploads   : {settings.UPLOAD_DIR}")
    print(f"  Logs      : {settings.LOG_DIR}")
    print(f"  Eval      : {settings.EVAL_DATASET_PATH}")

    print()
    print("Features")
    print(f"  Hybrid Search       : {settings.ENABLE_HYBRID_SEARCH}")
    print(f"  Cross Encoder       : {settings.ENABLE_CROSS_ENCODER_RERANKING}")
    print(f"  Semantic Cache      : {settings.ENABLE_SEMANTIC_CACHE}")
    print(f"  Advanced OCR        : {settings.ENABLE_ADVANCED_OCR}")
    print(f"  MCP                 : {settings.ENABLE_MCP_SERVER}")
    print(f"  Metrics             : {settings.ENABLE_METRICS}")
    print(f"  LangSmith Tracing   : {settings.LANGSMITH_TRACING}")

    print()
    print("Configuration Issues")

    issues = validate_settings()

    if issues:
        for issue in issues:
            print(f"  [!] {issue}")
    else:
        print("  [OK] No configuration issues detected.")

    print("=" * 72)