
#!/usr/bin/env python3
"""
PolicyGuard AI - LLM-Based Answer Judge Module
===============================================

Production evaluator using an OpenRouter-compatible LLM for:

- Faithfulness scoring
- Relevance scoring
- Robust JSON response parsing
- Retry and exponential backoff
- Thread-safe rate limiting
- Synchronous and asynchronous evaluation
- Batch evaluation
- Deterministic heuristic fallback
- Graceful shutdown

The judge is intentionally best-effort:
if the external LLM is unavailable, evaluation falls back to local heuristics
instead of breaking the main application.

Author: PolicyGuard AI Team
Version: 2.0.0
Last Updated: 2026-09-13
"""

import asyncio
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT_DELAY = 0.5
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_TIMEOUT_SECONDS = 30

MIN_SCORE = 1.0
MAX_SCORE = 5.0
DEFAULT_SCORE = 3.0

MAX_QUESTION_CHARS = 8_000
MAX_CONTEXT_CHARS = 20_000
MAX_ANSWER_CHARS = 8_000

SUPPORTED_SCORE_KEYS = (
    "faithfulness",
    "relevance",
)


# =============================================================================
# HELPERS
# =============================================================================

def _clamp_score(value: Any) -> float:
    """
    Convert a model-produced score into the supported 1-5 range.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return DEFAULT_SCORE

    if score != score:  # NaN
        return DEFAULT_SCORE

    if score == float("inf") or score == float("-inf"):
        return DEFAULT_SCORE

    return max(
        MIN_SCORE,
        min(MAX_SCORE, score),
    )


def _truncate_text(value: Any, max_chars: int) -> str:
    """Convert arbitrary input to bounded text."""
    if value is None:
        return ""

    text = str(value)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "…"


def _normalize_context(
    context: Union[str, List[str]],
) -> str:
    """Normalize context strings/lists into one bounded string."""
    if isinstance(context, list):
        parts = [
            str(item)
            for item in context
            if item is not None
        ]
        context_text = "\n\n".join(parts)
    elif context is None:
        context_text = ""
    else:
        context_text = str(context)

    return _truncate_text(
        context_text,
        MAX_CONTEXT_CHARS,
    )


def _validate_evaluation_input(
    question: str,
    context: Union[str, List[str]],
    answer: str,
) -> bool:
    """Validate basic evaluator inputs."""
    if not isinstance(question, str):
        return False

    if not isinstance(answer, str):
        return False

    if not isinstance(context, (str, list)):
        return False

    if not question.strip():
        return False

    if not answer.strip():
        return False

    if len(question) > MAX_QUESTION_CHARS:
        return False
    if len(answer) > MAX_ANSWER_CHARS:
        return False
    if isinstance(context, list):
        if sum(len(str(item)) for item in context if item is not None) > MAX_CONTEXT_CHARS:
            return False
    elif len(context) > MAX_CONTEXT_CHARS:
        return False

    return True


# =============================================================================
# OPENROUTER CLIENT
# =============================================================================

class OpenRouterClient:
    """
    Thread-safe OpenRouter-compatible client.

    The client uses a small thread pool for async calls because the requests
    library is synchronous.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_workers: int = 4,
    ):
        """
        Initialize the OpenRouter client.
        """
        self.api_key = (
            api_key
            if api_key is not None
            else getattr(
                settings,
                "OPENROUTER_API_KEY",
                None,
            )
        )

        self.base_url = (
            base_url
            if base_url is not None
            else getattr(
                settings,
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            )
        )

        # Prefer the dedicated simple/chat model if available.
        self.model = (
            model
            if model is not None
            else getattr(
                settings,
                "CHAT_MODEL_SIMPLE",
                getattr(
                    settings,
                    "OPENROUTER_MODEL",
                    "meta-llama/llama-3.1-8b-instruct",
                ),
            )
        )

        if isinstance(max_retries, bool) or not isinstance(
            max_retries,
            int,
        ):
            raise ValueError(
                "max_retries must be an integer"
            )

        if max_retries < 1:
            raise ValueError(
                "max_retries must be at least 1"
            )

        if rate_limit_delay < 0:
            raise ValueError(
                "rate_limit_delay cannot be negative"
            )

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero"
            )

        if max_workers < 1:
            raise ValueError(
                "max_workers must be at least 1"
            )

        self.max_retries = max_retries
        self.rate_limit_delay = float(
            rate_limit_delay
        )
        self.timeout = float(timeout)

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="policyguard-llm-judge",
        )

        # Protects request timing so concurrent requests don't bypass the
        # configured rate limit.
        self._rate_lock = threading.Lock()
        self._last_request_time = 0.0

        self._shutdown = False
        self._shutdown_lock = threading.RLock()

        if str(self.base_url).lower().startswith("http://"):
            raise ValueError("OpenRouter base_url must use HTTPS")

        if not self.model or not str(self.model).strip():
            raise ValueError("LLM judge model cannot be empty")

        if not self.api_key:
            logger.warning(
                "OPENROUTER_API_KEY is not configured; "
                "LLM judge will use heuristic fallback"
            )

        logger.info(
            "OpenRouterClient initialized: model=%s retries=%s "
            "rate_limit=%.2fs timeout=%.1fs",
            self.model,
            self.max_retries,
            self.rate_limit_delay,
            self.timeout,
        )

    # -------------------------------------------------------------------------
    # RATE LIMITING
    # -------------------------------------------------------------------------

    def _enforce_rate_limit(self) -> None:
        """
        Enforce a process-local minimum delay between requests.

        The lock is held while calculating/updating the timestamp so concurrent
        workers cannot all observe the same old timestamp and fire together.
        """
        with self._rate_lock:
            now = time.monotonic()

            elapsed = now - self._last_request_time

            if elapsed < self.rate_limit_delay:
                time.sleep(
                    self.rate_limit_delay - elapsed
                )

            self._last_request_time = time.monotonic()

    # -------------------------------------------------------------------------
    # JSON PARSING
    # -------------------------------------------------------------------------

    def _parse_json_response(
        self,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse evaluator JSON robustly.

        Supports:
        - Plain JSON
        - Markdown JSON code blocks
        - Embedded JSON objects
        - Simple score extraction as a last resort
        """
        if not content or not isinstance(content, str):
            return None

        text = content.strip()

        # 1. Direct JSON.
        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # 2. Markdown code block.
        code_block_matches = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        for candidate in code_block_matches:
            try:
                parsed = json.loads(candidate)

                if isinstance(parsed, dict):
                    return parsed

            except json.JSONDecodeError:
                continue

        # 3. Locate a JSON object containing one of our expected keys.
        decoder = json.JSONDecoder()

        for match in re.finditer(
            r"\{",
            text,
        ):
            start = match.start()

            try:
                parsed, _ = decoder.raw_decode(
                    text[start:]
                )

                if (
                    isinstance(parsed, dict)
                    and any(
                        key in parsed
                        for key in SUPPORTED_SCORE_KEYS
                    )
                ):
                    return parsed

            except json.JSONDecodeError:
                continue

        # 4. Last-resort score extraction.
        result: Dict[str, float] = {}

        for key in SUPPORTED_SCORE_KEYS:
            pattern = (
                rf'["\']?{re.escape(key)}["\']?'
                rf'\s*[:=]\s*'
                rf'([1-5](?:\.\d+)?)'
            )

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                try:
                    result[key] = float(
                        match.group(1)
                    )
                except ValueError:
                    pass

        if result:
            return result

        logger.warning(
            "Failed to parse LLM judge response: %s",
            text[:300],
        )

        return None

    # -------------------------------------------------------------------------
    # REQUEST
    # -------------------------------------------------------------------------

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 300,
    ) -> Optional[str]:
        """
        Make a synchronous OpenRouter request with retry handling.
        """
        if not self.api_key:
            logger.debug(
                "OpenRouter API key unavailable"
            )
            return None

        with self._shutdown_lock:
            if self._shutdown:
                logger.warning(
                    "OpenRouterClient is already shut down"
                )
                return None

        try:
            import requests
        except ImportError:
            logger.error(
                "requests package is not installed"
            )
            return None

        base_url = str(
            self.base_url
        ).rstrip("/")

        url = (
            f"{base_url}/chat/completions"
        )

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
            "HTTP-Referer": (
                "https://policyguard-ai.local"
            ),
            "X-Title": (
                "PolicyGuard AI - LLM Judge"
            ),
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "response_format": {
                "type": "json_object"
            },
        }

        for attempt in range(
            self.max_retries
        ):
            try:
                self._enforce_rate_limit()

                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                # -------------------------------------------------------------
                # Success
                # -------------------------------------------------------------

                if response.status_code == 200:
                    try:
                        result = response.json()
                    except ValueError:
                        logger.warning(
                            "OpenRouter returned invalid JSON"
                        )
                        return None

                    choices = result.get(
                        "choices",
                        [],
                    )

                    if not isinstance(
                        choices,
                        list
                    ) or not choices:
                        logger.warning(
                            "OpenRouter response contained no choices"
                        )
                        return None

                    message = choices[0].get(
                        "message",
                        {},
                    )

                    content = message.get(
                        "content",
                        "",
                    )

                    if not isinstance(
                        content,
                        str
                    ):
                        return None

                    return content

                # -------------------------------------------------------------
                # Rate limited
                # -------------------------------------------------------------

                if response.status_code == 429:
                    retry_after = (
                        response.headers.get(
                            "Retry-After"
                        )
                    )

                    try:
                        wait_time = float(
                            retry_after
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        wait_time = (
                            self.rate_limit_delay
                            * (2 ** attempt)
                        )

                    wait_time = min(
                        max(wait_time, 0.5),
                        30.0,
                    )

                    if attempt < self.max_retries - 1:
                        logger.warning(
                            "OpenRouter rate limited "
                            "(429); retrying in %.1fs "
                            "(attempt %s/%s)",
                            wait_time,
                            attempt + 1,
                            self.max_retries,
                        )

                        time.sleep(
                            wait_time
                        )
                        continue

                    logger.warning(
                        "OpenRouter rate limit persisted "
                        "after %s attempts",
                        self.max_retries,
                    )
                    return None

                # -------------------------------------------------------------
                # Server errors
                # -------------------------------------------------------------

                if response.status_code >= 500:
                    wait_time = min(
                        0.5 * (2 ** attempt),
                        10.0,
                    )

                    if attempt < self.max_retries - 1:
                        logger.warning(
                            "OpenRouter server error %s; "
                            "retrying in %.1fs",
                            response.status_code,
                            wait_time,
                        )

                        time.sleep(
                            wait_time
                        )
                        continue

                    logger.error(
                        "OpenRouter server error %s "
                        "after %s attempts",
                        response.status_code,
                        self.max_retries,
                    )
                    return None

                # -------------------------------------------------------------
                # Client errors
                # -------------------------------------------------------------

                response_preview = (
                    response.text[:500]
                    if response.text
                    else ""
                )

                logger.error(
                    "OpenRouter API error %s: %s",
                    response.status_code,
                    response_preview,
                )

                return None

            except requests.Timeout as exc:
                if attempt < self.max_retries - 1:
                    wait_time = min(
                        0.5 * (2 ** attempt),
                        10.0,
                    )

                    logger.warning(
                        "OpenRouter timeout: %s; "
                        "retrying in %.1fs",
                        exc,
                        wait_time,
                    )

                    time.sleep(
                        wait_time
                    )
                    continue

                logger.error(
                    "OpenRouter timed out after %s attempts",
                    self.max_retries,
                )
                return None

            except requests.RequestException as exc:
                if attempt < self.max_retries - 1:
                    wait_time = min(
                        0.5 * (2 ** attempt),
                        10.0,
                    )

                    logger.warning(
                        "OpenRouter request error: %s; "
                        "retrying in %.1fs",
                        exc,
                        wait_time,
                    )

                    time.sleep(
                        wait_time
                    )
                    continue

                logger.error(
                    "OpenRouter request failed after "
                    "%s attempts: %s",
                    self.max_retries,
                    exc,
                )
                return None

            except Exception as exc:
                logger.exception(
                    "Unexpected OpenRouter request failure: %s",
                    exc,
                )
                return None

        return None

    async def _make_request_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 300,
    ) -> Optional[str]:
        """
        Async request wrapper.

        The requests library is synchronous, so it runs in the client's
        thread pool.
        """
        with self._shutdown_lock:
            if self._shutdown:
                return None

        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            self._executor,
            self._make_request,
            messages,
            temperature,
            max_tokens,
        )

    # -------------------------------------------------------------------------
    # EVALUATION PROMPT
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_evaluation_prompt(
        question: str,
        context: str,
        answer: str,
    ) -> str:
        """
        Build a strict evaluation prompt.

        The evaluator is instructed to judge only the supplied context rather
        than using outside knowledge.
        """
        return f"""
You are a strict evaluator for a policy question-answering system.

Evaluate the ANSWER using ONLY the supplied CONTEXT.

QUESTION:
{_truncate_text(question, MAX_QUESTION_CHARS)}

CONTEXT:
{_truncate_text(context, MAX_CONTEXT_CHARS)}

ANSWER:
{_truncate_text(answer, MAX_ANSWER_CHARS)}

Score both dimensions from 1 to 5:

- faithfulness:
  5 = every important factual claim is clearly supported by the context
  4 = mostly supported, with only minor omissions or imprecision
  3 = partially supported or contains some unsupported detail
  2 = substantial unsupported content
  1 = largely ungrounded or contradicts the context

- relevance:
  5 = directly and completely answers the question
  4 = directly answers it with minor omissions
  3 = partially answers it
  2 = mostly misses the question
  1 = irrelevant

Important:
- Do not reward information merely because it is generally true.
- Do not use outside knowledge.
- Penalize unsupported factual claims.
- If the context does not contain enough information to answer the question,
  do not assume missing facts.
- Return ONLY valid JSON.

Required JSON:
{{
  "faithfulness": 1,
  "relevance": 1
}}
""".strip()

    # -------------------------------------------------------------------------
    # SINGLE EVALUATION
    # -------------------------------------------------------------------------

    def evaluate_single(
        self,
        question: str,
        context: str,
        answer: str,
        temperature: float = 0.1,
    ) -> Optional[Dict[str, float]]:
        """
        Evaluate one question/context/answer pair.

        Returns:
            Dict containing faithfulness and relevance, or None if the
            external evaluation fails.
        """
        if not _validate_evaluation_input(
            question,
            context,
            answer,
        ):
            logger.warning(
                "Invalid input supplied to evaluate_single"
            )
            return None

        prompt = self._build_evaluation_prompt(
            question,
            context,
            answer,
        )

        response_content = self._make_request(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_tokens=300,
        )

        if not response_content:
            return None

        parsed = self._parse_json_response(
            response_content
        )

        if not parsed:
            return None

        result: Dict[str, float] = {}

        for key in SUPPORTED_SCORE_KEYS:
            if key in parsed:
                result[key] = _clamp_score(
                    parsed[key]
                )
            else:
                logger.warning(
                    "LLM judge omitted score: %s",
                    key,
                )
                result[key] = DEFAULT_SCORE

        return result

    async def evaluate_single_async(
        self,
        question: str,
        context: str,
        answer: str,
        temperature: float = 0.1,
    ) -> Optional[Dict[str, float]]:
        """
        Async evaluation of one item.

        Fixed from the original implementation: Python coroutines do not
        support JavaScript-style `.then()`.
        """
        if not _validate_evaluation_input(
            question,
            context,
            answer,
        ):
            logger.warning(
                "Invalid input supplied to evaluate_single_async"
            )
            return None

        prompt = self._build_evaluation_prompt(
            question,
            context,
            answer,
        )

        response_content = (
            await self._make_request_async(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                max_tokens=300,
            )
        )

        if not response_content:
            return None

        parsed = self._parse_json_response(
            response_content
        )

        if not parsed:
            return None

        result: Dict[str, float] = {}

        for key in SUPPORTED_SCORE_KEYS:
            result[key] = _clamp_score(
                parsed.get(
                    key,
                    DEFAULT_SCORE,
                )
            )

        return result

    # -------------------------------------------------------------------------
    # BATCH
    # -------------------------------------------------------------------------

    def evaluate_batch(
        self,
        items: List[Dict[str, str]],
        temperature: float = 0.1,
    ) -> List[Optional[Dict[str, float]]]:
        """
        Evaluate multiple items sequentially.

        Sequential mode intentionally respects the client-side rate limit.
        """
        if not isinstance(items, list):
            raise ValueError(
                "items must be a list"
            )

        results: List[
            Optional[Dict[str, float]]
        ] = []

        for index, item in enumerate(
            items,
            start=1,
        ):
            if not isinstance(item, dict):
                results.append(None)
                continue

            logger.debug(
                "Evaluating item %s/%s",
                index,
                len(items),
            )

            result = self.evaluate_single(
                question=item.get(
                    "question",
                    "",
                ),
                context=item.get(
                    "context",
                    "",
                ),
                answer=item.get(
                    "answer",
                    "",
                ),
                temperature=temperature,
            )

            results.append(result)

        return results

    async def evaluate_batch_async(
        self,
        items: List[Dict[str, str]],
        temperature: float = 0.1,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> List[
        Optional[Dict[str, float]]
    ]:
        """
        Evaluate multiple items concurrently with bounded concurrency.
        """
        if not isinstance(items, list):
            raise ValueError(
                "items must be a list"
            )

        if (
            isinstance(max_concurrent, bool)
            or not isinstance(
                max_concurrent,
                int,
            )
            or max_concurrent < 1
        ):
            raise ValueError(
                "max_concurrent must be a positive integer"
            )

        semaphore = asyncio.Semaphore(
            max_concurrent
        )

        async def evaluate_one(
            item: Dict[str, str],
        ) -> Optional[Dict[str, float]]:
            if not isinstance(item, dict):
                return None

            async with semaphore:
                return await self.evaluate_single_async(
                    question=item.get(
                        "question",
                        "",
                    ),
                    context=item.get(
                        "context",
                        "",
                    ),
                    answer=item.get(
                        "answer",
                        "",
                    ),
                    temperature=temperature,
                )

        tasks = [
            evaluate_one(item)
            for item in items
        ]

        if not tasks:
            return []

        return list(
            await asyncio.gather(
                *tasks,
                return_exceptions=False,
            )
        )

    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------

    def shutdown(self) -> None:
        """Gracefully shut down the client's thread pool."""
        with self._shutdown_lock:
            if self._shutdown:
                return

            self._shutdown = True

        try:
            self._executor.shutdown(
                wait=True
            )
        except Exception as exc:
            logger.warning(
                "OpenRouterClient executor shutdown failed: %s",
                exc,
            )

        logger.info(
            "OpenRouterClient shutdown complete"
        )


# =============================================================================
# LLM JUDGE
# =============================================================================

class LLMJudge:
    """
    High-level LLM answer evaluator.

    If the external judge cannot be used, local deterministic heuristics are
    returned instead.
    """

    def __init__(
        self,
        client: Optional[OpenRouterClient] = None,
        use_async: bool = False,
    ):
        self.client = (
            client
            if client is not None
            else OpenRouterClient()
        )

        self.use_async = bool(
            use_async
        )

        self._available = bool(
            self.client.api_key
        )

        if not self._available:
            logger.warning(
                "LLMJudge is running in heuristic-fallback mode"
            )

    def is_available(self) -> bool:
        """Return whether an API key is configured."""
        return self._available

    def evaluate(
        self,
        question: str,
        context: Union[str, List[str]],
        answer: str,
        temperature: float = 0.1,
    ) -> Dict[str, float]:
        """
        Evaluate an answer.

        External LLM evaluation is preferred; local heuristic evaluation is
        used when the external judge is unavailable or fails.
        """
        context_text = _normalize_context(
            context
        )

        if not _validate_evaluation_input(
            question,
            context_text,
            answer,
        ):
            return {
                "faithfulness": MIN_SCORE,
                "relevance": MIN_SCORE,
            }

        if self._available:
            try:
                result = self.client.evaluate_single(
                    question=question,
                    context=context_text,
                    answer=answer,
                    temperature=temperature,
                )

                if result:
                    logger.info(
                        "LLM Judge: faithfulness=%.1f "
                        "relevance=%.1f",
                        result["faithfulness"],
                        result["relevance"],
                    )

                    return result

            except Exception as exc:
                logger.warning(
                    "LLM evaluation failed; "
                    "using heuristic fallback: %s",
                    exc,
                )

        return self._heuristic_evaluate(
            question,
            context_text,
            answer,
        )

    # -------------------------------------------------------------------------
    # HEURISTIC EVALUATION
    # -------------------------------------------------------------------------

    def _heuristic_evaluate(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> Dict[str, float]:
        """
        Deterministic local fallback.

        This is NOT equivalent to LLM judging. It is only intended to keep
        evaluation operational when the API is unavailable.

        Faithfulness is based on content-word overlap between answer/context.

        Relevance combines question keyword coverage and answer/context
        relationship to reduce obvious false positives.
        """
        def tokenize(text: str) -> List[str]:
            return re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower(),
            )

        stop_words = {
            "the",
            "and",
            "for",
            "that",
            "this",
            "with",
            "from",
            "what",
            "when",
            "where",
            "which",
            "who",
            "how",
            "does",
            "can",
            "could",
            "would",
            "should",
            "about",
            "have",
            "has",
            "are",
            "was",
            "were",
            "your",
            "you",
            "our",
            "their",
            "they",
            "them",
            "into",
            "than",
            "then",
            "there",
            "here",
            "also",
        }

        question_words = {
            word
            for word in tokenize(question)
            if len(word) > 2
            and word not in stop_words
        }

        context_words = {
            word
            for word in tokenize(context)
            if len(word) > 2
            and word not in stop_words
        }

        answer_words = {
            word
            for word in tokenize(answer)
            if len(word) > 2
            and word not in stop_words
        }

        # ---------------------------------------------------------------------
        # Faithfulness
        # ---------------------------------------------------------------------

        if not answer_words:
            faithfulness_ratio = 0.0

        elif not context_words:
            faithfulness_ratio = 0.0

        else:
            faithfulness_ratio = (
                len(
                    answer_words
                    & context_words
                )
                / len(answer_words)
            )

        # ---------------------------------------------------------------------
        # Relevance
        # ---------------------------------------------------------------------

        if not question_words:
            relevance_ratio = 1.0

        else:
            question_coverage = (
                len(
                    question_words
                    & answer_words
                )
                / len(question_words)
            )

            # If the answer is clearly related to context, give a modest
            # secondary signal. This prevents generic overlap from dominating.
            context_overlap = (
                len(
                    answer_words
                    & context_words
                )
                / max(
                    len(answer_words),
                    1,
                )
            )

            relevance_ratio = (
                0.75 * question_coverage
                + 0.25 * context_overlap
            )

        def scale_to_1_5(
            score: float,
        ) -> float:
            if score >= 0.80:
                return 5.0

            if score >= 0.60:
                return 4.0

            if score >= 0.40:
                return 3.0

            if score >= 0.20:
                return 2.0

            return 1.0

        return {
            "faithfulness": scale_to_1_5(
                faithfulness_ratio
            ),
            "relevance": scale_to_1_5(
                relevance_ratio
            ),
        }

    # -------------------------------------------------------------------------
    # BATCH
    # -------------------------------------------------------------------------

    def evaluate_batch(
        self,
        items: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> List[Dict[str, float]]:
        """Evaluate multiple items with automatic fallback."""
        if not isinstance(items, list):
            raise ValueError(
                "items must be a list"
            )

        results = []

        for item in items:
            if not isinstance(item, dict):
                results.append(
                    {
                        "faithfulness": MIN_SCORE,
                        "relevance": MIN_SCORE,
                    }
                )
                continue

            results.append(
                self.evaluate(
                    question=item.get(
                        "question",
                        "",
                    ),
                    context=item.get(
                        "context",
                        [],
                    ),
                    answer=item.get(
                        "answer",
                        "",
                    ),
                    temperature=temperature,
                )
            )

        return results

    async def evaluate_async(
        self,
        question: str,
        context: Union[str, List[str]],
        answer: str,
        temperature: float = 0.1,
    ) -> Dict[str, float]:
        """Async answer evaluation with heuristic fallback."""
        context_text = _normalize_context(
            context
        )

        if not _validate_evaluation_input(
            question,
            context_text,
            answer,
        ):
            return {
                "faithfulness": MIN_SCORE,
                "relevance": MIN_SCORE,
            }

        if self._available:
            try:
                result = (
                    await self.client.evaluate_single_async(
                        question=question,
                        context=context_text,
                        answer=answer,
                        temperature=temperature,
                    )
                )

                if result:
                    logger.info(
                        "Async LLM Judge: "
                        "faithfulness=%.1f relevance=%.1f",
                        result["faithfulness"],
                        result["relevance"],
                    )

                    return result

            except Exception as exc:
                logger.warning(
                    "Async LLM evaluation failed; "
                    "using heuristic fallback: %s",
                    exc,
                )

        return self._heuristic_evaluate(
            question,
            context_text,
            answer,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return evaluator configuration and availability."""
        return {
            "available": self._available,
            "model": (
                self.client.model
                if self.client
                else None
            ),
            "base_url": (
                self.client.base_url
                if self.client
                else None
            ),
            "use_async": self.use_async,
            "max_retries": (
                self.client.max_retries
                if self.client
                else 0
            ),
            "rate_limit_delay": (
                self.client.rate_limit_delay
                if self.client
                else 0
            ),
        }

    def shutdown(self) -> None:
        """Shut down the underlying client."""
        if self.client:
            self.client.shutdown()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_judge: Optional[LLMJudge] = None
_judge_lock = threading.RLock()


def get_llm_judge(
    use_async: bool = False,
) -> LLMJudge:
    """Get or create the global LLM judge singleton."""
    global _judge

    with _judge_lock:
        if _judge is None:
            _judge = LLMJudge(
                use_async=use_async
            )

        return _judge


def reset_llm_judge() -> None:
    """Reset the global judge instance."""
    global _judge

    with _judge_lock:
        judge = _judge

        if judge is not None:
            judge.shutdown()

        _judge = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def evaluate_answer(
    question: str,
    context: Union[str, List[str]],
    answer: str,
    temperature: float = 0.1,
) -> Dict[str, float]:
    """Evaluate one answer using the global judge."""
    return get_llm_judge().evaluate(
        question,
        context,
        answer,
        temperature,
    )


def evaluate_answers_batch(
    items: List[Dict[str, Any]],
    temperature: float = 0.1,
) -> List[Dict[str, float]]:
    """Evaluate multiple answers using the global judge."""
    return get_llm_judge().evaluate_batch(
        items,
        temperature,
    )


async def evaluate_answer_async(
    question: str,
    context: Union[str, List[str]],
    answer: str,
    temperature: float = 0.1,
) -> Dict[str, float]:
    """Evaluate one answer asynchronously."""
    return await get_llm_judge(
        use_async=True
    ).evaluate_async(
        question,
        context,
        answer,
        temperature,
    )


# =============================================================================
# TEST / DEMO
# =============================================================================

def test_llm_judge() -> None:
    """Run basic judge tests."""
    print("\n🎯 Testing LLM Judge")
    print("=" * 70)

    judge = LLMJudge()

    print(
        f"Judge available: "
        f"{judge.is_available()}"
    )

    print(
        f"Config: "
        f"{judge.get_stats()}"
    )

    print("=" * 70)

    test_cases = [
        {
            "question": "What is the leave policy?",
            "context": (
                "Employees are entitled to 20 days of paid "
                "leave per year. Leave accrues monthly. "
                "Requests must be submitted 2 weeks in advance "
                "through the HR portal."
            ),
            "answer": (
                "Employees get 20 days of paid leave annually, "
                "accruing monthly, with 2 weeks advance notice "
                "required via the HR portal."
            ),
        },
        {
            "question": "What is the leave policy?",
            "context": (
                "Employees are entitled to 20 days of paid "
                "leave per year."
            ),
            "answer": (
                "The office is open Monday through Friday, "
                "9am to 5pm."
            ),
        },
        {
            "question": "How do I request time off?",
            "context": (
                "All leave requests must be submitted at "
                "least 2 weeks in advance through the HR portal."
            ),
            "answer": (
                "Submit your request through the HR portal "
                "at least 2 weeks before your desired dates."
            ),
        },
    ]

    print("\n📝 Test Evaluations")
    print("-" * 70)

    for index, test in enumerate(
        test_cases,
        start=1,
    ):
        result = judge.evaluate(
            question=test["question"],
            context=test["context"],
            answer=test["answer"],
        )

        print(
            f"{index}. "
            f"faithfulness="
            f"{result['faithfulness']:.1f}, "
            f"relevance="
            f"{result['relevance']:.1f}"
        )

    print("\n📝 Testing Async Evaluation")
    print("-" * 70)

    async def async_test():
        return await judge.evaluate_async(
            question=(
                "How do I request time off?"
            ),
            context=(
                "Leave requests must be submitted "
                "through the HR portal."
            ),
            answer=(
                "Submit your leave request through "
                "the HR portal."
            ),
        )

    try:
        async_result = asyncio.run(
            async_test()
        )

        print(
            f"   Async result: "
            f"faithfulness="
            f"{async_result['faithfulness']:.1f}, "
            f"relevance="
            f"{async_result['relevance']:.1f}"
        )

    except Exception as exc:
        print(
            f"   ⚠️ Async test failed: {exc}"
        )

    print("\n📝 Testing Batch Evaluation")
    print("-" * 70)

    batch_items = [
        {
            "question": item["question"],
            "context": item["context"],
            "answer": item["answer"],
        }
        for item in test_cases
    ]

    batch_results = judge.evaluate_batch(
        batch_items
    )

    for index, result in enumerate(
        batch_results,
        start=1,
    ):
        print(
            f"   {index}. "
            f"faithfulness="
            f"{result['faithfulness']:.1f}, "
            f"relevance="
            f"{result['relevance']:.1f}"
        )

    judge.shutdown()

    print("\n" + "=" * 70)
    print("✅ LLM judge test complete")


if __name__ == "__main__":
    test_llm_judge()

