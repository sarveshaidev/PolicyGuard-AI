
#!/usr/bin/env python3
"""
PolicyGuard AI - RAGAS Evaluation Module
=========================================

Production-ready RAG quality evaluator with:

- RAGAS metrics when RAGAS is installed/configured
- Heuristic fallback when RAGAS is unavailable
- Faithfulness
- Answer relevancy
- Context precision
- Context recall when ground truth is available
- Batch evaluation
- Ground-truth dataset loading
- JSON result export
- Thread-safe global evaluator
- Graceful degradation when optional dependencies fail

Important:
RAGAS is an evaluation dependency, not a runtime dependency for the main
RAG pipeline. Evaluation failures must never break normal application use.

Author: PolicyGuard AI Team
Version: 2.0.0
Last Updated: 2026-09-13
"""

import json
import logging
import math
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

MIN_SCORE = 0.0
MAX_SCORE = 1.0

MAX_QUESTION_CHARS = 8_000
MAX_ANSWER_CHARS = 8_000
MAX_CONTEXT_CHARS = 30_000

DEFAULT_POLICYGUARD_VERSION = "2.0.0"


# =============================================================================
# HELPERS
# =============================================================================

def _utc_now() -> datetime:
    """Return timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def _clamp_score(value: Any) -> Optional[float]:
    """
    Convert a metric value into a finite 0-1 float.

    Returns None for invalid values.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(score):
        return None

    return max(
        MIN_SCORE,
        min(MAX_SCORE, score),
    )


def _truncate_text(
    value: Any,
    max_chars: int,
) -> str:
    """Convert a value to bounded text."""
    if value is None:
        return ""

    text = str(value)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "…"


def _normalize_contexts(
    contexts: Any,
) -> List[str]:
    """
    Normalize retrieved contexts.

    RAGAS expects contexts as a list of strings.
    """
    if contexts is None:
        return []

    if isinstance(contexts, str):
        return (
            [_truncate_text(contexts, MAX_CONTEXT_CHARS)]
            if contexts.strip()
            else []
        )

    if not isinstance(contexts, (list, tuple)):
        return []

    normalized: List[str] = []

    for context in contexts:
        if context is None:
            continue

        text = str(context).strip()

        if not text:
            continue

        normalized.append(
            _truncate_text(
                text,
                MAX_CONTEXT_CHARS,
            )
        )

    return normalized


def _validate_query(
    question: Any,
    answer: Any,
    contexts: Any,
) -> bool:
    """Validate basic evaluation input."""
    if not isinstance(question, str):
        return False

    if not isinstance(answer, str):
        return False

    if not isinstance(contexts, (str, list, tuple)):
        return False

    if not question.strip():
        return False

    if not answer.strip():
        return False

    if len(question) > MAX_QUESTION_CHARS:
        return False
    if len(answer) > MAX_ANSWER_CHARS:
        return False
    if isinstance(contexts, (list, tuple)):
        if sum(len(str(item)) for item in contexts if item is not None) > MAX_CONTEXT_CHARS * 10:
            return False
    elif isinstance(contexts, str) and len(contexts) > MAX_CONTEXT_CHARS:
        return False

    return True


# =============================================================================
# HEURISTIC EVALUATOR
# =============================================================================

class HeuristicEvaluator:
    """
    Lightweight local fallback evaluator.

    This is intentionally not presented as equivalent to RAGAS. It provides
    useful approximate metrics when RAGAS or its LLM dependencies are absent.
    """

    STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "with",
        "from",
        "by",
        "as",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "into",
        "than",
        "then",
        "there",
        "here",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "have",
        "has",
        "had",
        "your",
        "you",
        "our",
        "we",
        "they",
        "their",
        "them",
    }

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Tokenize text into meaningful lowercase words."""
        import re

        if not text:
            return []

        tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            str(text).lower(),
        )

        return [
            token
            for token in tokens
            if len(token) > 2
            and token not in cls.STOPWORDS
        ]

    @classmethod
    def _token_set(cls, text: str) -> set:
        """Return normalized unique tokens."""
        return set(
            cls._tokenize(text)
        )

    @classmethod
    def faithfulness_score(
        cls,
        answer: str,
        contexts: List[str],
    ) -> float:
        """
        Approximate faithfulness using answer/context token overlap.
        """
        if not answer.strip():
            return 0.0

        if not contexts:
            return 0.0

        answer_tokens = cls._token_set(
            answer
        )

        if not answer_tokens:
            return 1.0

        context_tokens = set()

        for context in contexts:
            context_tokens.update(
                cls._tokenize(context)
            )

        if not context_tokens:
            return 0.0

        overlap = len(
            answer_tokens
            & context_tokens
        )

        return _clamp_score(
            overlap / len(answer_tokens)
        ) or 0.0

    @classmethod
    def relevancy_score(
        cls,
        question: str,
        answer: str,
    ) -> float:
        """
        Approximate answer relevancy using question-keyword coverage.
        """
        if not question.strip() or not answer.strip():
            return 0.0

        question_tokens = cls._token_set(
            question
        )

        answer_tokens = cls._token_set(
            answer
        )

        if not question_tokens:
            return 1.0

        if not answer_tokens:
            return 0.0

        overlap = len(
            question_tokens
            & answer_tokens
        )

        return _clamp_score(
            overlap / len(question_tokens)
        ) or 0.0

    @classmethod
    def context_precision_score(
        cls,
        question: str,
        contexts: List[str],
        relevant_indices: Optional[List[int]] = None,
    ) -> float:
        """
        Approximate context precision.

        If relevant_indices are supplied, earlier relevant chunks receive
        higher reciprocal-rank weight.

        Without ground-truth indices, this uses query/context similarity
        rather than blindly assuming the first three chunks are relevant.
        """
        if not question.strip() or not contexts:
            return 0.0

        question_tokens = cls._token_set(
            question
        )

        if not question_tokens:
            return 1.0

        # ---------------------------------------------------------------------
        # Ground-truth ranking information available
        # ---------------------------------------------------------------------

        if relevant_indices is not None:
            valid_indices = sorted(
                {
                    index
                    for index in relevant_indices
                    if isinstance(index, int)
                    and 0 <= index < len(contexts)
                }
            )

            if not valid_indices:
                return 0.0

            reciprocal_ranks = [
                1.0 / (index + 1)
                for index in valid_indices
            ]

            return _clamp_score(
                sum(reciprocal_ranks)
                / len(reciprocal_ranks)
            ) or 0.0

        # ---------------------------------------------------------------------
        # No ground-truth indices: estimate query/context relevance
        # ---------------------------------------------------------------------

        similarities: List[float] = []

        for context in contexts:
            context_tokens = cls._token_set(
                context
            )

            if not context_tokens:
                similarities.append(0.0)
                continue

            overlap = len(
                question_tokens
                & context_tokens
            )

            similarities.append(
                overlap / len(question_tokens)
            )

        if not similarities:
            return 0.0

        # Weighted average favors higher-ranked retrieved chunks.
        weights = [
            1.0 / (index + 1)
            for index in range(
                len(similarities)
            )
        ]

        weighted_score = (
            sum(
                score * weight
                for score, weight in zip(
                    similarities,
                    weights,
                )
            )
            / sum(weights)
        )

        return _clamp_score(
            weighted_score
        ) or 0.0

    @classmethod
    def context_recall_score(
        cls,
        ground_truth: str,
        contexts: List[str],
    ) -> float:
        """
        Approximate context recall using ground-truth token coverage.
        """
        if not ground_truth.strip():
            return 0.0

        if not contexts:
            return 0.0

        ground_truth_tokens = cls._token_set(
            ground_truth
        )

        if not ground_truth_tokens:
            return 1.0

        context_tokens = set()

        for context in contexts:
            context_tokens.update(
                cls._tokenize(context)
            )

        if not context_tokens:
            return 0.0

        covered = len(
            ground_truth_tokens
            & context_tokens
        )

        return _clamp_score(
            covered / len(ground_truth_tokens)
        ) or 0.0

    @classmethod
    def evaluate(
        cls,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run all available heuristic evaluations.

        Context recall is returned only when ground truth is actually
        supplied. We deliberately do not invent a default 0.7 score.
        """
        scores = {
            "faithfulness": cls.faithfulness_score(
                answer,
                contexts,
            ),
            "answer_relevancy": cls.relevancy_score(
                question,
                answer,
            ),
            "context_precision": cls.context_precision_score(
                question,
                contexts,
            ),
        }

        if ground_truth:
            scores["context_recall"] = (
                cls.context_recall_score(
                    ground_truth,
                    contexts,
                )
            )

        return scores


# =============================================================================
# RAGAS EVALUATOR
# =============================================================================

class RAGASEvaluator:
    """
    Production RAG quality evaluator.

    RAGAS is optional. If it cannot be imported, configured, or executed,
    the evaluator automatically falls back to local heuristic metrics.
    """

    def __init__(
        self,
        use_ragas: Optional[bool] = None,
        llm_model: Optional[str] = None,
    ):
        self.llm_model = (
            llm_model
            or getattr(
                settings,
                "CHAT_MODEL_REASONING",
                getattr(
                    settings,
                    "OPENROUTER_MODEL",
                    "meta-llama/llama-3.1-8b-instruct",
                ),
            )
        )

        self.use_ragas = (
            bool(use_ragas)
            if use_ragas is not None
            else bool(
                getattr(
                    settings,
                    "ENABLE_EVALUATION",
                    True,
                )
            )
        )

        self.ragas_available = False
        self.llm_available = False

        self.heuristic = HeuristicEvaluator()

        self._ragas_metrics: Dict[str, Any] = {}
        self._ragas_evaluate_fn = None
        self._llm_client = None

        self._shutdown = False
        self._shutdown_lock = threading.RLock()

        if self.use_ragas:
            self._init_ragas()

            if self.ragas_available:
                self._init_llm_for_ragas()

        logger.info(
            "RAGASEvaluator initialized: "
            "ragas=%s llm=%s model=%s use_ragas=%s",
            self.ragas_available,
            self.llm_available,
            self.llm_model,
            self.use_ragas,
        )

    # -------------------------------------------------------------------------
    # RAGAS INITIALIZATION
    # -------------------------------------------------------------------------

    def _init_ragas(self) -> None:
        """
        Detect RAGAS and load supported metrics.

        RAGAS has changed its public API across versions, so imports are
        intentionally isolated here.
        """
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )

            self._ragas_evaluate_fn = (
                ragas_evaluate
            )

            self._ragas_metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }

            self.ragas_available = True

            logger.info(
                "RAGAS library detected and metrics loaded"
            )

        except ImportError as exc:
            self.ragas_available = False

            logger.warning(
                "RAGAS unavailable; using heuristic fallback: %s",
                exc,
            )

        except Exception as exc:
            self.ragas_available = False

            logger.warning(
                "RAGAS initialization failed; "
                "using heuristic fallback: %s",
                exc,
            )

    def _init_llm_for_ragas(self) -> None:
        """
        Configure LangChain's OpenAI-compatible client for OpenRouter.

        If this optional integration is unavailable, RAGAS may still work
        depending on the installed RAGAS version/configuration.
        """
        if not self.ragas_available:
            return

        api_key = getattr(
            settings,
            "OPENROUTER_API_KEY",
            None,
        )

        base_url = getattr(
            settings,
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        )

        if str(base_url).lower().startswith("http://"):
            logger.warning("RAGAS LLM base URL must use HTTPS; external evaluation disabled")
            return

        if not api_key:
            logger.warning(
                "OPENROUTER_API_KEY not configured; "
                "RAGAS LLM client unavailable"
            )
            return

        try:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": self.llm_model,
                "temperature": 0.1,
                "api_key": api_key,
                "base_url": str(
                    base_url
                ).rstrip("/"),
                "timeout": 30,
            }

            try:
                self._llm_client = ChatOpenAI(
                    **kwargs
                )
            except TypeError:
                # Compatibility with older langchain-openai versions.
                kwargs.pop("api_key", None)
                kwargs.pop("base_url", None)

                kwargs["openai_api_key"] = api_key
                kwargs["openai_api_base"] = (
                    str(base_url).rstrip("/")
                )

                self._llm_client = ChatOpenAI(
                    **kwargs
                )

            self.llm_available = True

            logger.info(
                "RAGAS LLM configured: model=%s",
                self.llm_model,
            )

        except ImportError as exc:
            self.llm_available = False

            logger.warning(
                "langchain_openai is not installed; "
                "RAGAS LLM integration unavailable: %s",
                exc,
            )

        except Exception as exc:
            self.llm_available = False

            logger.warning(
                "RAGAS LLM initialization failed: %s",
                exc,
            )

    # -------------------------------------------------------------------------
    # SINGLE QUERY
    # -------------------------------------------------------------------------

    def evaluate_query(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate one RAG query.

        RAGAS is attempted first. Any RAGAS failure falls back to heuristics.
        """
        normalized_contexts = _normalize_contexts(
            contexts
        )

        normalized_ground_truth = (
            str(ground_truth).strip()
            if ground_truth is not None
            else None
        )

        if not _validate_query(
            question,
            answer,
            normalized_contexts,
        ):
            logger.warning(
                "Invalid RAG evaluation input"
            )

            return self.heuristic.evaluate(
                question=str(question or ""),
                answer=str(answer or ""),
                contexts=normalized_contexts,
                ground_truth=normalized_ground_truth,
            )

        if self.ragas_available:
            try:
                scores = self._evaluate_with_ragas(
                    question=question,
                    answer=answer,
                    contexts=normalized_contexts,
                    ground_truth=normalized_ground_truth,
                )

                if scores:
                    return scores

            except Exception as exc:
                logger.warning(
                    "RAGAS evaluation failed; "
                    "falling back to heuristics: %s",
                    exc,
                )

        return self.heuristic.evaluate(
            question=question,
            answer=answer,
            contexts=normalized_contexts,
            ground_truth=normalized_ground_truth,
        )

    # -------------------------------------------------------------------------
    # RAGAS EXECUTION
    # -------------------------------------------------------------------------

    def _evaluate_with_ragas(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Execute RAGAS evaluation.

        Only context recall is included when ground truth is available.
        """
        if self._ragas_evaluate_fn is None:
            raise RuntimeError(
                "RAGAS evaluate function is unavailable"
            )

        try:
            from datasets import Dataset
        except ImportError as exc:
            raise RuntimeError(
                "datasets package is required for RAGAS evaluation"
            ) from exc

        eval_data: Dict[str, List[Any]] = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        if ground_truth:
            eval_data["ground_truth"] = [
                ground_truth
            ]

        dataset = Dataset.from_dict(
            eval_data
        )

        metrics_to_use = []

        for metric_name in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
        ):
            metric = self._ragas_metrics.get(
                metric_name
            )

            if metric is not None:
                metrics_to_use.append(
                    metric
                )

        if ground_truth:
            metric = self._ragas_metrics.get(
                "context_recall"
            )

            if metric is not None:
                metrics_to_use.append(
                    metric
                )

        if not metrics_to_use:
            raise RuntimeError(
                "No RAGAS metrics are available"
            )

        # ---------------------------------------------------------------------
        # Call RAGAS.
        #
        # Some RAGAS versions accept llm=None while others behave differently.
        # Try the configured LLM first when available, then retry without the
        # explicit llm argument if the installed version rejects it.
        # ---------------------------------------------------------------------

        if self.llm_available and self._llm_client is not None:
            try:
                result = self._ragas_evaluate_fn(
                    dataset=dataset,
                    metrics=metrics_to_use,
                    llm=self._llm_client,
                )
            except TypeError:
                result = self._ragas_evaluate_fn(
                    dataset=dataset,
                    metrics=metrics_to_use,
                )
        else:
            result = self._ragas_evaluate_fn(
                dataset=dataset,
                metrics=metrics_to_use,
            )

        return self._extract_ragas_scores(
            result=result,
            expected_metrics=[
                metric_name
                for metric_name in (
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                )
                if (
                    metric_name != "context_recall"
                    or ground_truth
                )
            ],
        )

    # -------------------------------------------------------------------------
    # SCORE EXTRACTION
    # -------------------------------------------------------------------------

    def _extract_ragas_scores(
        self,
        result: Any,
        expected_metrics: List[str],
    ) -> Dict[str, float]:
        """
        Extract scores from different RAGAS EvaluationResult versions.

        RAGAS has returned dict-like objects, dataframes, and other wrappers
        across releases, so extraction is deliberately defensive.
        """
        scores: Dict[str, float] = {}

        # ---------------------------------------------------------------------
        # Convert result to a mapping-like object where possible.
        # ---------------------------------------------------------------------

        for metric_name in expected_metrics:
            raw_value = None
            found = False

            # 1. Standard mapping access.
            if isinstance(result, dict):
                if metric_name in result:
                    raw_value = result[
                        metric_name
                    ]
                    found = True

            # 2. RAGAS EvaluationResult often supports .get().
            if not found and hasattr(
                result,
                "get",
            ):
                try:
                    raw_value = result.get(
                        metric_name
                    )
                    found = raw_value is not None
                except Exception:
                    pass

            # 3. DataFrame-like result.
            if not found and hasattr(
                result,
                "to_pandas",
            ):
                try:
                    dataframe = result.to_pandas()

                    if metric_name in dataframe.columns:
                        column = dataframe[
                            metric_name
                        ]

                        if len(column) > 0:
                            raw_value = column.iloc[0]
                            found = True

                except Exception as exc:
                    logger.debug(
                        "Could not extract %s from "
                        "RAGAS dataframe: %s",
                        metric_name,
                        exc,
                    )

            # 4. Attribute access.
            if not found and hasattr(
                result,
                metric_name,
            ):
                try:
                    raw_value = getattr(
                        result,
                        metric_name,
                    )
                    found = True
                except Exception:
                    pass

            if not found:
                logger.warning(
                    "RAGAS result did not contain metric: %s",
                    metric_name,
                )
                continue

            # -----------------------------------------------------------------
            # Handle arrays/series.
            # -----------------------------------------------------------------

            if isinstance(
                raw_value,
                (list, tuple),
            ):
                if not raw_value:
                    continue

                raw_value = raw_value[0]

            else:
                # numpy arrays / pandas series
                try:
                    if hasattr(
                        raw_value,
                        "iloc",
                    ):
                        raw_value = raw_value.iloc[0]
                    elif hasattr(
                        raw_value,
                        "__len__",
                    ) and not isinstance(
                        raw_value,
                        (str, bytes, dict),
                    ):
                        length = len(raw_value)

                        if length > 0:
                            raw_value = raw_value[0]

                except (
                    TypeError,
                    IndexError,
                    KeyError,
                ):
                    pass

            score = _clamp_score(
                raw_value
            )

            if score is not None:
                scores[metric_name] = score

        if not scores:
            raise RuntimeError(
                "RAGAS returned no usable metric scores"
            )

        logger.debug(
            "RAGAS evaluation scores: %s",
            scores,
        )

        return scores

    # -------------------------------------------------------------------------
    # BATCH
    # -------------------------------------------------------------------------

    def evaluate_batch(
        self,
        queries: List[Dict[str, Any]],
        progress_callback: Optional[
            Callable[[int, int], None]
        ] = None,
    ) -> Dict[str, float]:
        """
        Evaluate multiple queries and return aggregate metrics.

        `evaluated_queries` counts successfully evaluated query records,
        rather than counting metric values and dividing by the number of
        metrics.
        """
        if not isinstance(
            queries,
            list,
        ):
            raise ValueError(
                "queries must be a list"
            )

        if not queries:
            return {
                "error": "No queries to evaluate",
                "num_queries": 0,
                "evaluated_queries": 0,
            }

        all_scores: Dict[
            str,
            List[float],
        ] = defaultdict(list)

        evaluated_queries = 0

        total_queries = len(
            queries
        )

        for index, query in enumerate(
            queries,
            start=1,
        ):
            if progress_callback:
                try:
                    progress_callback(
                        index,
                        total_queries,
                    )
                except Exception as exc:
                    logger.warning(
                        "Progress callback failed: %s",
                        exc,
                    )

            if not isinstance(
                query,
                dict,
            ):
                logger.warning(
                    "Skipping invalid query %s/%s",
                    index,
                    total_queries,
                )
                continue

            try:
                scores = self.evaluate_query(
                    question=query.get(
                        "question",
                        "",
                    ),
                    answer=query.get(
                        "answer",
                        "",
                    ),
                    contexts=query.get(
                        "contexts",
                        [],
                    ),
                    ground_truth=query.get(
                        "ground_truth"
                    ),
                )

                valid_metric_count = 0

                for metric, score in scores.items():
                    normalized_score = _clamp_score(
                        score
                    )

                    if normalized_score is None:
                        continue

                    all_scores[
                        metric
                    ].append(
                        normalized_score
                    )

                    valid_metric_count += 1

                if valid_metric_count > 0:
                    evaluated_queries += 1

            except Exception as exc:
                logger.warning(
                    "Failed to evaluate query %s/%s: %s",
                    index,
                    total_queries,
                    exc,
                )

        # ---------------------------------------------------------------------
        # Aggregate
        # ---------------------------------------------------------------------

        result: Dict[str, float] = {}

        for metric in METRIC_NAMES:
            values = all_scores.get(
                metric,
                [],
            )

            if values:
                result[metric] = (
                    sum(values)
                    / len(values)
                )

        result["num_queries"] = total_queries
        result["evaluated_queries"] = (
            evaluated_queries
        )

        # Number of observations per metric is useful when some queries don't
        # have ground truth and therefore don't produce context_recall.
        result["metric_counts"] = {
            metric: len(
                all_scores.get(
                    metric,
                    [],
                )
            )
            for metric in METRIC_NAMES
        }

        logger.info(
            "Batch evaluation complete: "
            "queries=%s evaluated=%s "
            "faithfulness=%.3f relevancy=%.3f",
            total_queries,
            evaluated_queries,
            result.get(
                "faithfulness",
                0.0,
            ),
            result.get(
                "answer_relevancy",
                0.0,
            ),
        )

        return result

    # -------------------------------------------------------------------------
    # GROUND TRUTH
    # -------------------------------------------------------------------------

    def load_ground_truth(
        self,
        filepath: Optional[
            Union[str, Path]
        ] = None,
    ) -> List[Dict[str, str]]:
        """
        Load a ground-truth JSON dataset.

        Expected format:

        [
          {
            "question": "What is X?",
            "ground_truth": "X is..."
          }
        ]
        """
        if filepath is None:
            filepath = getattr(
                settings,
                "EVAL_DATASET_PATH",
                project_root
                / "data"
                / "evaluation_dataset.json",
            )

        path = Path(
            filepath
        ).expanduser()

        try:
            if not path.exists():
                logger.warning(
                    "Ground truth file not found: %s",
                    path,
                )
                return []

            if not path.is_file():
                logger.warning(
                    "Ground truth path is not a file: %s",
                    path,
                )
                return []

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(
                    file
                )

            if not isinstance(
                data,
                list,
            ):
                logger.error(
                    "Ground truth file must contain a JSON array"
                )
                return []

            valid_entries: List[
                Dict[str, str]
            ] = []

            for entry in data:
                if not isinstance(
                    entry,
                    dict,
                ):
                    continue

                question = entry.get(
                    "question"
                )

                ground_truth = entry.get(
                    "ground_truth"
                )

                if (
                    not isinstance(
                        question,
                        str,
                    )
                    or not question.strip()
                ):
                    continue

                if (
                    not isinstance(
                        ground_truth,
                        str,
                    )
                    or not ground_truth.strip()
                ):
                    continue

                if len(question) > MAX_QUESTION_CHARS or len(ground_truth) > MAX_ANSWER_CHARS:
                    continue

                valid_entries.append(
                    {
                        "question": question.strip(),
                        "ground_truth": ground_truth.strip(),
                    }
                )

            invalid_count = (
                len(data)
                - len(valid_entries)
            )

            if invalid_count:
                logger.warning(
                    "Filtered %s invalid ground-truth entries",
                    invalid_count,
                )

            logger.info(
                "Loaded %s ground-truth pairs from %s",
                len(valid_entries),
                path,
            )

            return valid_entries

        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid JSON in ground-truth file %s: %s",
                path,
                exc,
            )
            return []

        except OSError as exc:
            logger.error(
                "Could not read ground-truth file %s: %s",
                path,
                exc,
            )
            return []

        except Exception as exc:
            logger.exception(
                "Unexpected ground-truth loading error: %s",
                exc,
            )
            return []

    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------

    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        filepath: Optional[
            Union[str, Path]
        ] = None,
        include_metadata: bool = True,
    ) -> Optional[Path]:
        """
        Save evaluation results as JSON.
        """
        try:
            if filepath is None:
                export_dir = (
                    project_root
                    / "data"
                    / "traces"
                )

                export_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                filepath = (
                    export_dir
                    / f"evaluation_{timestamp}.json"
                )

            else:
                filepath = (
                    Path(
                        filepath
                    )
                    .expanduser()
                )

                filepath.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            export_data: Dict[str, Any] = {
                "schema_version": 2,
                "evaluated_at": (
                    _utc_now().isoformat()
                ),
                "evaluator_config": {
                    "ragas_available": (
                        self.ragas_available
                    ),
                    "llm_available": (
                        self.llm_available
                    ),
                    "llm_model": self.llm_model,
                    "use_ragas": self.use_ragas,
                    "heuristic_fallback": True,
                },
                "results": results,
            }

            if include_metadata:
                export_data[
                    "metadata"
                ] = {
                    "policyguard_version": (
                        DEFAULT_POLICYGUARD_VERSION
                    ),
                    "embedding_model": getattr(
                        settings,
                        "EMBEDDING_MODEL",
                        None,
                    ),
                    "chat_model": getattr(
                        settings,
                        "CHAT_MODEL_SIMPLE",
                        None,
                    ),
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
                    default=str,
                )

            logger.info(
                "Evaluation results saved to %s",
                filepath,
            )

            return filepath

        except OSError as exc:
            logger.error(
                "Could not save evaluation results: %s",
                exc,
            )
            return None

        except Exception as exc:
            logger.exception(
                "Unexpected evaluation result save error: %s",
                exc,
            )
            return None

    # -------------------------------------------------------------------------
    # STATS
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return evaluator configuration and capabilities."""
        if self.ragas_available:
            supported_metrics = list(
                self._ragas_metrics.keys()
            )
        else:
            supported_metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]

        return {
            "ragas_available": (
                self.ragas_available
            ),
            "llm_available": (
                self.llm_available
            ),
            "llm_model": self.llm_model,
            "use_ragas": self.use_ragas,
            "heuristic_fallback": True,
            "supported_metrics": supported_metrics,
            "shutdown": self._shutdown,
        }

    # -------------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Release optional LLM resources.
        """
        with self._shutdown_lock:
            if self._shutdown:
                return

            self._shutdown = True

        client = self._llm_client
        self._llm_client = None
        self.llm_available = False

        if client is not None:
            try:
                close_method = getattr(
                    client,
                    "close",
                    None,
                )

                if callable(close_method):
                    close_method()

                else:
                    # Some LangChain clients expose async cleanup instead.
                    close_async = getattr(
                        client,
                        "aclose",
                        None,
                    )

                    if callable(close_async):
                        logger.debug(
                            "RAGAS LLM client exposes async close; "
                            "cleanup is managed by its runtime"
                        )

            except Exception as exc:
                logger.warning(
                    "RAGAS LLM client cleanup failed: %s",
                    exc,
                )

        logger.info(
            "RAGASEvaluator shutdown complete"
        )


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_evaluator: Optional[
    RAGASEvaluator
] = None

_eval_lock = threading.RLock()


def get_ragas_evaluator(
    use_ragas: Optional[bool] = None,
    llm_model: Optional[str] = None,
) -> RAGASEvaluator:
    """
    Get/create the global RAGAS evaluator.

    The first call determines the singleton configuration.
    """
    global _evaluator

    with _eval_lock:
        if (
            _evaluator is None
            or _evaluator._shutdown
        ):
            _evaluator = RAGASEvaluator(
                use_ragas=use_ragas,
                llm_model=llm_model,
            )

        return _evaluator


def reset_ragas_evaluator() -> None:
    """Reset the global evaluator, primarily for tests."""
    global _evaluator

    with _eval_lock:
        evaluator = _evaluator

        if evaluator is not None:
            evaluator.shutdown()

        _evaluator = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def evaluate_rag_query(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate one RAG query."""
    return get_ragas_evaluator().evaluate_query(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )


def evaluate_rag_batch(
    queries: List[Dict[str, Any]],
    progress_callback: Optional[
        Callable[[int, int], None]
    ] = None,
) -> Dict[str, float]:
    """Evaluate a batch of RAG queries."""
    return get_ragas_evaluator().evaluate_batch(
        queries=queries,
        progress_callback=progress_callback,
    )


def load_ground_truth_dataset(
    filepath: Optional[
        Union[str, Path]
    ] = None,
) -> List[Dict[str, str]]:
    """Load a ground-truth dataset."""
    return get_ragas_evaluator().load_ground_truth(
        filepath
    )


def save_evaluation_results(
    results: Dict[str, Any],
    filepath: Optional[
        Union[str, Path]
    ] = None,
) -> Optional[Path]:
    """Save evaluation results."""
    return get_ragas_evaluator().save_evaluation_results(
        results,
        filepath,
    )


def get_evaluator_stats() -> Dict[str, Any]:
    """Return evaluator status and capability information."""
    return get_ragas_evaluator().get_stats()


# =============================================================================
# TEST / DEMO
# =============================================================================

def test_ragas_evaluator() -> None:
    """Run a comprehensive local evaluator test."""
    print("\n📊 Testing RAGAS Evaluator")
    print("=" * 70)

    evaluator = RAGASEvaluator()

    print(
        f"RAGAS available: "
        f"{evaluator.ragas_available}"
    )

    print(
        f"LLM available: "
        f"{evaluator.llm_available}"
    )

    print(
        f"Model: "
        f"{evaluator.llm_model}"
    )

    print(
        f"Stats: "
        f"{evaluator.get_stats()}"
    )

    print("=" * 70)

    # -------------------------------------------------------------------------
    # Test 1: Single evaluation
    # -------------------------------------------------------------------------

    print("\n📝 Test 1: Single Query Evaluation")
    print("-" * 70)

    test_query = {
        "question": (
            "What is the company leave policy?"
        ),
        "answer": (
            "Employees are entitled to 20 days "
            "of paid leave per year. Leave accrues "
            "monthly and must be requested at least "
            "2 weeks in advance through the HR portal."
        ),
        "contexts": [
            (
                "Employees are entitled to 20 days "
                "of paid leave per year."
            ),
            (
                "Leave accrues on a monthly basis "
                "throughout the employment period."
            ),
            (
                "All leave requests must be submitted "
                "at least 2 weeks in advance through "
                "the HR portal."
            ),
        ],
        "ground_truth": (
            "Employees get 20 days of paid leave "
            "annually, accruing monthly, with 2 weeks "
            "advance notice required via the HR portal."
        ),
    }

    scores = evaluator.evaluate_query(
        question=test_query["question"],
        answer=test_query["answer"],
        contexts=test_query["contexts"],
        ground_truth=test_query["ground_truth"],
    )

    print(
        f"Question: "
        f"{test_query['question']}"
    )

    print(
        "Scores (0.0-1.0, higher is better):"
    )

    for metric in METRIC_NAMES:
        if metric not in scores:
            continue

        score = scores[metric]

        bar_len = int(
            max(
                0,
                min(
                    10,
                    round(score * 10),
                ),
            )
        )

        bar = (
            "█" * bar_len
            + "░" * (10 - bar_len)
        )

        print(
            f"  {metric:20s}: "
            f"{score:.2f} [{bar}]"
        )

    # -------------------------------------------------------------------------
    # Test 2: Batch
    # -------------------------------------------------------------------------

    print("\n📝 Test 2: Batch Evaluation")
    print("-" * 70)

    test_queries = [
        {
            "question": "What is the leave policy?",
            "answer": (
                "Employees get 20 days paid leave per year."
            ),
            "contexts": [
                (
                    "Employees are entitled to 20 days "
                    "of paid leave per year."
                )
            ],
            "ground_truth": (
                "20 days paid leave annually"
            ),
        },
        {
            "question": "How do I request time off?",
            "answer": (
                "Submit the request through the HR portal "
                "2 weeks in advance."
            ),
            "contexts": [
                (
                    "All leave requests must be submitted "
                    "at least 2 weeks in advance through "
                    "the HR portal."
                )
            ],
            "ground_truth": (
                "Use the HR portal with 2 weeks notice"
            ),
        },
        {
            "question": "What are the office hours?",
            "answer": (
                "Office hours are 9am to 5pm, "
                "Monday through Friday."
            ),
            "contexts": [
                (
                    "Standard office hours are 9:00 AM "
                    "to 5:00 PM, Monday through Friday."
                )
            ],
            "ground_truth": (
                "9am-5pm, Monday-Friday"
            ),
        },
    ]

    def progress_callback(
        current: int,
        total: int,
    ) -> None:
        print(
            f"  Progress: {current}/{total}",
            end="\r",
        )

    batch_scores = evaluator.evaluate_batch(
        test_queries,
        progress_callback=progress_callback,
    )

    print()

    print(
        f"Evaluated queries: "
        f"{batch_scores.get('evaluated_queries', 0)}"
    )

    for metric in METRIC_NAMES:
        if metric not in batch_scores:
            continue

        score = batch_scores[metric]

        print(
            f"  {metric:20s}: "
            f"{score:.3f}"
        )

    # -------------------------------------------------------------------------
    # Test 3: Ground truth
    # -------------------------------------------------------------------------

    print("\n📝 Test 3: Ground Truth Loading")
    print("-" * 70)

    ground_truth = evaluator.load_ground_truth()

    if ground_truth:
        print(
            f"Loaded {len(ground_truth)} ground-truth pairs"
        )

        sample = ground_truth[0]

        print(
            f"Sample question: "
            f"{sample['question'][:60]}..."
        )

        print(
            f"Sample answer: "
            f"{sample['ground_truth'][:60]}..."
        )

    else:
        print(
            "No ground-truth dataset found."
        )

        print(
            f"Expected path: "
            f"{getattr(settings, 'EVAL_DATASET_PATH', 'not configured')}"
        )

        print(
            'Expected format: '
            '[{"question": "What is X?", '
            '"ground_truth": "X is..."}]'
        )

    # -------------------------------------------------------------------------
    # Test 4: Save results
    # -------------------------------------------------------------------------

    print("\n📝 Test 4: Save Evaluation Results")
    print("-" * 70)

    export_path = evaluator.save_evaluation_results(
        batch_scores
    )

    if export_path:
        print(
            f"Saved to: {export_path}"
        )

        try:
            print(
                f"File size: "
                f"{export_path.stat().st_size} bytes"
            )
        except OSError:
            pass

    else:
        print(
            "Failed to save results"
        )

    # -------------------------------------------------------------------------
    # Test 5: Forced heuristic mode
    # -------------------------------------------------------------------------

    print("\n📝 Test 5: Heuristic Fallback")
    print("-" * 70)

    heuristic_evaluator = RAGASEvaluator(
        use_ragas=False
    )

    heuristic_scores = (
        heuristic_evaluator.evaluate_query(
            question=test_query["question"],
            answer=test_query["answer"],
            contexts=test_query["contexts"],
            ground_truth=test_query["ground_truth"],
        )
    )

    for metric, score in heuristic_scores.items():
        print(
            f"  {metric:20s}: "
            f"{score:.2f}"
        )

    heuristic_evaluator.shutdown()

    evaluator.shutdown()

    print("\n" + "=" * 70)
    print("✅ RAGAS evaluator test complete")


if __name__ == "__main__":
    test_ragas_evaluator()

