"""Shared, content-safe resilience policy for long real-model evaluations.

The policy is intentionally small and deterministic.  Provider SDK retries
remain responsible for one logical call; this module governs the slower
campaign-level retry window that survives process restarts through atomic
checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from rag_kb.domain import ErrorCode
from tools.evaluation_campaign_state import digest


POLICY_SCHEMA = "large_evaluation_resilience_policy_v1"
HEARTBEAT_INTERVAL_SECONDS = 30.0
LONG_STAGE_STALL_TIMEOUT_SECONDS = 7_200.0
SHORT_STAGE_STALL_TIMEOUT_SECONDS = 900.0
MAX_STALL_RESTARTS = 3


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    backoff_seconds: tuple[float, ...]
    maximum_backoff_seconds: float

    def __post_init__(self) -> None:
        if (
            self.max_attempts < 1
            or not self.backoff_seconds
            or any(delay < 0 for delay in self.backoff_seconds)
            or self.maximum_backoff_seconds < 0
        ):
            raise ValueError("evaluation retry policy is invalid")

    def delay_after(self, attempt_count: int) -> float:
        """Return the delay after a failed one-based attempt."""

        if attempt_count < 1:
            raise ValueError("evaluation retry attempt must be positive")
        index = min(attempt_count - 1, len(self.backoff_seconds) - 1)
        return min(self.backoff_seconds[index], self.maximum_backoff_seconds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "backoff_seconds": list(self.backoff_seconds),
            "maximum_backoff_seconds": self.maximum_backoff_seconds,
        }


# A case may contain several internally retried model rounds.  Twelve
# campaign attempts give a several-hour unattended outage window while still
# bounding duplicate cost and eventually failing closed.
PROVIDER_CASE_RETRY_POLICY = RetryPolicy(
    max_attempts=12,
    backoff_seconds=(30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0),
    maximum_backoff_seconds=1800.0,
)

# Graph builds checkpoint every committed chunk.  Use a larger total resume
# budget because a 1,441-chunk build can encounter several independent
# transport interruptions without losing completed episodes.
GRAPH_BUILD_MAX_RESUMES = 64
GRAPH_BUILD_RETRY_POLICY = RetryPolicy(
    max_attempts=GRAPH_BUILD_MAX_RESUMES,
    backoff_seconds=(30.0, 60.0, 120.0, 300.0, 600.0, 900.0),
    maximum_backoff_seconds=900.0,
)

RESILIENCE_POLICY = {
    "schema_version": POLICY_SCHEMA,
    "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
    "stall_recovery": {
        "long_stage_timeout_seconds": LONG_STAGE_STALL_TIMEOUT_SECONDS,
        "short_stage_timeout_seconds": SHORT_STAGE_STALL_TIMEOUT_SECONDS,
        "max_restarts": MAX_STALL_RESTARTS,
    },
    "provider_case": PROVIDER_CASE_RETRY_POLICY.as_dict(),
    "graph_build": {
        "max_resumes": GRAPH_BUILD_MAX_RESUMES,
        **GRAPH_BUILD_RETRY_POLICY.as_dict(),
    },
}
RESILIENCE_POLICY_SHA256 = digest(RESILIENCE_POLICY)


_RETRYABLE_CODES = frozenset(
    {
        ErrorCode.CHAT_PROVIDER_UNAVAILABLE.value,
        ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE.value,
        ErrorCode.GRAPH_PROVIDER_UNAVAILABLE.value,
        ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED.value,
        ErrorCode.RETRIEVAL_DEADLINE_EXCEEDED.value,
    }
)
_SAFE_DIAGNOSTIC_KEYS = frozenset({"check", "http_status", "retryable"})


def stable_error_code(error: BaseException) -> str:
    """Return a content-safe stable code without persisting exception text."""

    value = getattr(error, "code", None)
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str) and value:
        return value
    value = getattr(error, "error_code", None)
    if isinstance(value, str) and value:
        return value
    return type(error).__name__


def safe_failure_summary(error: BaseException) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": type(error).__name__,
        "code": stable_error_code(error),
    }
    diagnostic = getattr(error, "diagnostic", None)
    if isinstance(diagnostic, Mapping):
        safe = {
            key: diagnostic[key]
            for key in _SAFE_DIAGNOSTIC_KEYS
            if isinstance(diagnostic.get(key), (str, int, bool))
        }
        if safe:
            result["diagnostic"] = safe
    return result


def is_retryable_provider_failure(error: BaseException) -> bool:
    code = stable_error_code(error)
    if code in _RETRYABLE_CODES:
        return True
    diagnostic = getattr(error, "diagnostic", None)
    if isinstance(diagnostic, Mapping) and diagnostic.get("retryable") is True:
        return True
    # Embedding profile validation deliberately collapses transport details
    # to this stable code at the API boundary.
    return code == "provider_validation_failed"


def timestamp_after(delay_seconds: float) -> str:
    if delay_seconds < 0:
        raise ValueError("evaluation retry delay must be non-negative")
    return (datetime.now(UTC) + timedelta(seconds=delay_seconds)).isoformat()


def seconds_until(timestamp: object) -> float:
    if not isinstance(timestamp, str) or not timestamp:
        return 0.0
    try:
        target = datetime.fromisoformat(timestamp)
    except ValueError:
        return 0.0
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    return max(0.0, (target - datetime.now(UTC)).total_seconds())
