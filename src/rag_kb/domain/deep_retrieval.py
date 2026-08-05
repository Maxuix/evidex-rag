"""Versioned, framework-independent contracts for bounded deep retrieval.

This module deliberately contains no Pydantic (or model/provider) dependency.  It
defines the immutable facts that a future planner and coverage assessor may pass
between application layers.  Execution policy remains server-owned: a goal can
carry text to retrieve, but it cannot carry a strategy, scope, filter, provider,
or budget override.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, is_dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping, Sequence
from uuid import UUID

from rag_kb.domain.errors import ErrorCode


PLAN_WIRE_VERSION = "multi_query_plan_v1"
GOAL_WIRE_VERSION = "query_goal_v1"
COVERAGE_WIRE_VERSION = "coverage_report_v1"
BUDGET_WIRE_VERSION = "deep_retrieval_budget_v1"
EVIDENCE_KEY_VERSION = "evidence_key_v1"
SINGLE_GOAL_FALLBACK_VERSION = "single_goal_fallback_v1"
ASSESSMENT_FAILURE_VERSION = "coverage_assessment_failure_v1"
CAPABILITY_WIRE_VERSION = "deep_retrieval_capability_v1"
FAILURE_WIRE_VERSION = "deep_retrieval_failure_v1"

SINGLE_GOAL_FALLBACK_REASONS = frozenset(
    {"plan_invalid", "plan_empty", "plan_budget_exceeded"}
)
ASSESSMENT_FAILURE_REASONS = frozenset(
    {"coverage_invalid", "coverage_budget_exceeded", "coverage_unavailable"}
)

MAX_GOALS = 4
MAX_QUERY_VARIANTS_PER_GOAL = 1
MAX_ADAPTIVE_WAVES = 0
MAX_REPAIRS = 1
MAX_OUTPUT_TOKENS = 2048
MAX_DEADLINE_SECONDS = 120.0
MAX_PARALLELISM = 1
MAX_RETRIEVAL_CALLS = 1
MAX_EVIDENCE_KEYS_PER_GOAL = 10
MAX_EVIDENCE_KEYS_PER_REPORT = MAX_GOALS * MAX_EVIDENCE_KEYS_PER_GOAL
MAX_USAGE_TOKENS = 1_000_000

_IDENTITY_RE = re.compile(r"^[^\s]{1,256}$")
_GOAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class WorkflowDepth(StrEnum):
    """The workflow depth dimension, independent from retrieval strategy."""

    STANDARD = "standard"
    DEEP = "deep"


class RetrievalMode(StrEnum):
    """The retrieval lane dimension, independent from workflow depth."""

    VECTOR = "vector"
    HYBRID = "hybrid"


class GoalStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    COMPLETED = "completed"


class CoverageStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    MISSING = "missing"
    CONFLICT = "conflict"


class StructuredOutcome(StrEnum):
    ACCEPTED = "accepted"
    SINGLE_GOAL_FALLBACK = "single_goal_fallback"
    ASSESSMENT_FAILURE = "assessment_failure"


class DeepRetrievalErrorCode(StrEnum):
    """Future adapter error vocabulary, isolated from the public API enum.

    Phase 1 has no production route, so publishing these through the global
    ``ErrorCode`` would alter OpenAPI prematurely.  Later phases must map a
    selected code explicitly when they add a guarded endpoint.
    """

    CAPABILITY_NOT_ENABLED = "CAPABILITY_NOT_ENABLED"
    STRUCTURED_WIRE_INVALID = "STRUCTURED_WIRE_INVALID"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    INVALID_DEPENDENCY = "INVALID_DEPENDENCY"
    PLAN_HASH_CONFLICT = "PLAN_HASH_CONFLICT"
    REVISION_MISMATCH = "REVISION_MISMATCH"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    RETRIEVAL_DEADLINE_EXCEEDED = "RETRIEVAL_DEADLINE_EXCEEDED"
    ASSESSMENT_STRUCTURE_FAILED = "ASSESSMENT_STRUCTURE_FAILED"


class DeepRetrievalControlReason(StrEnum):
    """Successful terminal control outcomes are not transport failures."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def _text(value: str, *, name: str, maximum: int = 256) -> str:
    if isinstance(value, UUID):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds the {maximum}-character limit")
    return normalized


def _identity(value: str, *, name: str) -> str:
    normalized = _text(value, name=name)
    if not _IDENTITY_RE.fullmatch(normalized):
        raise ValueError(f"{name} contains invalid whitespace")
    return normalized


def _query_text(value: str, *, name: str = "query") -> str:
    normalized = unicodedata.normalize("NFC", _text(value, name=name, maximum=2048))
    if any(unicodedata.category(character) in {"Cc", "Cf"} for character in normalized):
        raise ValueError(f"{name} contains a control character")
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _query_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _unique(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _jsonable(value: Any) -> Any:
    """Convert immutable contract facts into deterministic JSON-compatible data."""

    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {
            field_name: _jsonable(getattr(value, field_name))
            for field_name in value.__dataclass_fields__  # type: ignore[attr-defined]
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def canonical_json(value: Any) -> str:
    """Return the stable UTF-8 JSON representation used by contract hashes."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(value: Any) -> str:
    """Return a namespaced SHA-256 hash for a canonical contract value."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DeepRetrievalCapabilitySnapshot:
    """Content-safe, process-level declaration for future guarded routes."""

    configuration_fingerprint: str
    deep_workflow_enabled: bool = False
    static_multi_query_v1_enabled: bool = False
    wire_version: str = CAPABILITY_WIRE_VERSION
    plan_wire_version: str = PLAN_WIRE_VERSION
    budget_wire_version: str = BUDGET_WIRE_VERSION

    def __post_init__(self) -> None:
        if self.wire_version != CAPABILITY_WIRE_VERSION:
            raise ValueError("unsupported capability snapshot version")
        if self.plan_wire_version != PLAN_WIRE_VERSION:
            raise ValueError("unsupported plan capability version")
        if self.budget_wire_version != BUDGET_WIRE_VERSION:
            raise ValueError("unsupported budget capability version")
        if not isinstance(self.configuration_fingerprint, str) or not _HASH_RE.fullmatch(
            self.configuration_fingerprint
        ):
            raise ValueError("configuration_fingerprint must be a SHA-256 identity")
        if not isinstance(self.deep_workflow_enabled, bool):
            raise ValueError("deep_workflow_enabled must be a boolean")
        if not isinstance(self.static_multi_query_v1_enabled, bool):
            raise ValueError("static_multi_query_v1_enabled must be a boolean")
        if self.static_multi_query_v1_enabled and not self.deep_workflow_enabled:
            raise ValueError("static multi-query requires the deep workflow capability")


_FAILURE_POLICIES: dict[DeepRetrievalErrorCode, tuple[int, bool, str]] = {
    DeepRetrievalErrorCode.CAPABILITY_NOT_ENABLED: (409, False, "capability"),
    DeepRetrievalErrorCode.STRUCTURED_WIRE_INVALID: (422, False, "structured_wire"),
    DeepRetrievalErrorCode.BUDGET_EXCEEDED: (409, False, "budget"),
    DeepRetrievalErrorCode.INVALID_DEPENDENCY: (422, False, "dependency"),
    DeepRetrievalErrorCode.PLAN_HASH_CONFLICT: (409, False, "plan_hash"),
    DeepRetrievalErrorCode.REVISION_MISMATCH: (409, False, "revision"),
    DeepRetrievalErrorCode.PROVIDER_UNAVAILABLE: (503, True, "provider"),
    DeepRetrievalErrorCode.RETRIEVAL_DEADLINE_EXCEEDED: (503, True, "deadline"),
    DeepRetrievalErrorCode.ASSESSMENT_STRUCTURE_FAILED: (502, False, "assessment"),
}

_FAILURE_FIELDS = frozenset(
    {
        "workflow_depth",
        "wire_version",
        "goal_count",
        "query_count",
        "retrieval_call_count",
        "repair_count",
        "deadline_seconds",
        "plan_hash",
        "index_revision_id",
    }
)


@dataclass(frozen=True, slots=True)
class DeepRetrievalFailureFact:
    """Bounded future Problem Details input with no free-form model content."""

    code: DeepRetrievalErrorCode
    check: str
    status: int
    retryable: bool
    field: str | None = None
    count: int | None = None
    limit: int | None = None
    wire_version: str = FAILURE_WIRE_VERSION

    def __post_init__(self) -> None:
        if self.wire_version != FAILURE_WIRE_VERSION:
            raise ValueError("unsupported deep-retrieval failure version")
        try:
            code = DeepRetrievalErrorCode(self.code)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported deep-retrieval error code") from exc
        object.__setattr__(self, "code", code)
        expected_status, expected_retryable, expected_check = _FAILURE_POLICIES[code]
        if (
            isinstance(self.status, bool)
            or not isinstance(self.status, int)
            or not isinstance(self.retryable, bool)
            or not isinstance(self.check, str)
        ):
            raise ValueError("failure policy fields use strict scalar types")
        if (self.status, self.retryable, self.check) != (
            expected_status,
            expected_retryable,
            expected_check,
        ):
            raise ValueError("failure status, retryability, and check must match the code")
        if self.field is not None and self.field not in _FAILURE_FIELDS:
            raise ValueError("failure field is not allowlisted")
        for name, value in (("count", self.count), ("limit", self.limit)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"failure {name} must be a non-negative integer")


def deep_retrieval_failure(
    code: DeepRetrievalErrorCode,
    *,
    field: str | None = None,
    count: int | None = None,
    limit: int | None = None,
) -> DeepRetrievalFailureFact:
    status, retryable, check = _FAILURE_POLICIES[DeepRetrievalErrorCode(code)]
    return DeepRetrievalFailureFact(
        code=code,
        check=check,
        status=status,
        retryable=retryable,
        field=field,
        count=count,
        limit=limit,
    )


@dataclass(frozen=True, slots=True)
class ServerBudgetSnapshot:
    """The server-resolved resource budget frozen with one plan.

    The hard bounds here are intentionally stricter than a future settings
    object.  A caller cannot use a wire payload to increase any of them.
    """

    wire_version: str = BUDGET_WIRE_VERSION
    max_goals: int = MAX_GOALS
    max_query_variants_per_goal: int = MAX_QUERY_VARIANTS_PER_GOAL
    max_adaptive_waves: int = MAX_ADAPTIVE_WAVES
    max_repairs: int = MAX_REPAIRS
    deadline_seconds: float = 30.0
    max_parallelism: int = 1
    max_retrieval_calls: int = 1
    max_output_tokens: int = MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        if self.wire_version != BUDGET_WIRE_VERSION:
            raise ValueError("unsupported deep-retrieval budget version")
        if isinstance(self.max_goals, bool) or not 1 <= self.max_goals <= MAX_GOALS:
            raise ValueError("max_goals must be between 1 and 4")
        if (
            isinstance(self.max_query_variants_per_goal, bool)
            or not isinstance(self.max_query_variants_per_goal, int)
            or self.max_query_variants_per_goal != MAX_QUERY_VARIANTS_PER_GOAL
        ):
            raise ValueError("Phase 1 allows exactly one query variant per goal")
        if (
            isinstance(self.max_adaptive_waves, bool)
            or not isinstance(self.max_adaptive_waves, int)
            or self.max_adaptive_waves != MAX_ADAPTIVE_WAVES
        ):
            raise ValueError("adaptive waves are disabled in Phase 1")
        if isinstance(self.max_repairs, bool) or not 0 <= self.max_repairs <= MAX_REPAIRS:
            raise ValueError("max_repairs must be between 0 and 1")
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(float(self.deadline_seconds))
            or not 0 < self.deadline_seconds <= MAX_DEADLINE_SECONDS
        ):
            raise ValueError("deadline_seconds must be finite and between 0 and 120")
        if (
            isinstance(self.max_parallelism, bool)
            or not isinstance(self.max_parallelism, int)
            or self.max_parallelism != MAX_PARALLELISM
        ):
            raise ValueError("Phase 1 allows exactly one concurrent retrieval")
        if (
            isinstance(self.max_retrieval_calls, bool)
            or not isinstance(self.max_retrieval_calls, int)
            or self.max_retrieval_calls != MAX_RETRIEVAL_CALLS
        ):
            raise ValueError("Phase 1 allows exactly one retrieval call")
        if (
            isinstance(self.max_output_tokens, bool)
            or not 1 <= self.max_output_tokens <= MAX_OUTPUT_TOKENS
        ):
            raise ValueError("max_output_tokens must be between 1 and 2048")
        object.__setattr__(self, "deadline_seconds", float(self.deadline_seconds))

    # Vocabulary used by some future callers; keeping aliases avoids introducing
    # a second, subtly different budget schema.
    @property
    def goals_per_plan(self) -> int:
        return self.max_goals

    @property
    def query_variants_per_goal(self) -> int:
        return self.max_query_variants_per_goal

    @property
    def adaptive_waves(self) -> int:
        return self.max_adaptive_waves

    @property
    def repair_count(self) -> int:
        return self.max_repairs

    @property
    def parallelism(self) -> int:
        return self.max_parallelism

    @property
    def retrieval_calls(self) -> int:
        return self.max_retrieval_calls


# Explicit aliases make the server-owned nature clear while preserving a small
# public vocabulary for later phases.
DeepRetrievalBudgetSnapshot = ServerBudgetSnapshot
BudgetSnapshot = ServerBudgetSnapshot


@dataclass(frozen=True, slots=True)
class CanonicalEvidenceKey:
    """Stable cross-query identity for one admitted representation.

    ``chunk_or_group_id`` is deliberately one field: a key can identify either
    a chunk representation or an Evidence Group, but never an unscoped asset.
    """

    revision_id: str
    target_id: str
    chunk_or_group_id: str
    representation_id: str
    representation_kind: str = "chunk"
    plan_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("revision_id", self.revision_id),
            ("target_id", self.target_id),
            ("chunk_or_group_id", self.chunk_or_group_id),
            ("representation_id", self.representation_id),
        ):
            object.__setattr__(self, name, _identity(value, name=name))
        if self.representation_kind not in {"chunk", "group"}:
            raise ValueError("representation_kind must be chunk or group")
        if self.plan_id is not None:
            object.__setattr__(self, "plan_id", _identity(self.plan_id, name="plan_id"))

    @property
    def index_revision_id(self) -> str:
        return self.revision_id

    @property
    def chunk_or_group(self) -> str:
        return self.chunk_or_group_id

    @property
    def representation(self) -> str:
        return self.representation_id

    @property
    def canonical(self) -> str:
        # Canonical JSON length-escapes arbitrary identity text through JSON's
        # grammar.  Delimiter concatenation would allow IDs containing ``|``
        # or ``=`` to collide while still satisfying the identity contract.
        return canonical_json(
            {
                "wire_version": EVIDENCE_KEY_VERSION,
                "revision_id": self.revision_id,
                "target_id": self.target_id,
                "chunk_or_group_id": self.chunk_or_group_id,
                "representation_id": self.representation_id,
                "representation_kind": self.representation_kind,
                "plan_id": self.plan_id,
            }
        )

    @property
    def key(self) -> str:
        return self.canonical

    def as_dict(self) -> dict[str, str]:
        value = {
            "wire_version": EVIDENCE_KEY_VERSION,
            "revision_id": self.revision_id,
            "target_id": self.target_id,
            "chunk_or_group_id": self.chunk_or_group_id,
            "representation_id": self.representation_id,
            "representation_kind": self.representation_kind,
        }
        if self.plan_id is not None:
            value["plan_id"] = self.plan_id
        return value


@dataclass(frozen=True, slots=True)
class Goal:
    """One server-authorized retrieval objective in a plan."""

    goal_id: str
    question: str
    query: str | None = None
    query_variants: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    intent: str | None = None
    status: GoalStatus = GoalStatus.PENDING
    wire_version: str = GOAL_WIRE_VERSION

    def __post_init__(self) -> None:
        if self.wire_version != GOAL_WIRE_VERSION:
            raise ValueError("unsupported query-goal version")
        goal_id = _text(self.goal_id, name="goal_id", maximum=64)
        if not _GOAL_ID_RE.fullmatch(goal_id):
            raise ValueError("goal_id contains unsupported characters")
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "question", _text(self.question, name="question", maximum=2048))
        if self.intent is not None:
            object.__setattr__(self, "intent", _text(self.intent, name="intent", maximum=256))
        variants = tuple(_query_text(item, name="query_variant") for item in self.query_variants)
        if len(variants) > MAX_QUERY_VARIANTS_PER_GOAL:
            raise ValueError("query variants exceed the Phase 1 budget")
        if len(variants) != len(set(variants)):
            raise ValueError("query variants must be unique")
        query = self.query
        if query is not None:
            query = _query_text(query)
        if query is None and variants:
            query = variants[0]
        elif query is not None and not variants:
            variants = (query,)
        elif query is not None and variants and variants[0] != query:
            raise ValueError("query must equal the sole query variant")
        if query is None:
            raise ValueError("goal requires a non-empty query")
        if len(variants) != MAX_QUERY_VARIANTS_PER_GOAL:
            raise ValueError("goal must contain exactly one query variant")
        dependencies = tuple(_text(item, name="depends_on goal id", maximum=64) for item in self.depends_on)
        _unique(dependencies, name="depends_on")
        if self.goal_id in dependencies:
            raise ValueError("goal cannot depend on itself")
        if not isinstance(self.status, GoalStatus):
            try:
                object.__setattr__(self, "status", GoalStatus(self.status))
            except (TypeError, ValueError) as exc:
                raise ValueError("unsupported goal status") from exc
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "query_variants", variants)
        object.__setattr__(self, "depends_on", dependencies)


def _check_dag(goals: Sequence[Goal]) -> None:
    by_id = {goal.goal_id: goal for goal in goals}
    if len(by_id) != len(goals):
        raise ValueError("goal ids must be unique")
    for goal in goals:
        unknown = [dependency for dependency in goal.depends_on if dependency not in by_id]
        if unknown:
            raise ValueError(f"unknown goal dependency: {unknown[0]}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(goal_id: str) -> None:
        if goal_id in visiting:
            raise ValueError("goal dependencies must form a DAG")
        if goal_id in visited:
            return
        visiting.add(goal_id)
        for dependency in by_id[goal_id].depends_on:
            visit(dependency)
        visiting.remove(goal_id)
        visited.add(goal_id)

    for goal in goals:
        visit(goal.goal_id)


@dataclass(frozen=True, slots=True)
class PlanEnvelope:
    """Strict, replayable planner output plus server-owned execution snapshot."""

    plan_id: str
    question_ref: str
    workflow_depth: WorkflowDepth
    retrieval_mode: RetrievalMode
    goals: tuple[Goal, ...]
    budget: ServerBudgetSnapshot
    capability_fingerprint: str | None = None
    wire_version: str = PLAN_WIRE_VERSION
    created_at: datetime | None = None
    model_capability_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.wire_version != PLAN_WIRE_VERSION:
            raise ValueError("unsupported multi-query plan version")
        object.__setattr__(self, "plan_id", _identity(self.plan_id, name="plan_id"))
        object.__setattr__(self, "question_ref", _identity(self.question_ref, name="question_ref"))
        capability = self.capability_fingerprint
        model_capability = self.model_capability_fingerprint
        if capability is None:
            capability = model_capability
        elif model_capability is not None and capability != model_capability:
            raise ValueError("capability fingerprints must agree")
        if capability is None:
            raise ValueError("plan requires a capability fingerprint")
        capability = _identity(capability, name="capability_fingerprint")
        if not _HASH_RE.fullmatch(capability):
            raise ValueError("capability_fingerprint must be a SHA-256 identity")
        object.__setattr__(self, "capability_fingerprint", capability)
        object.__setattr__(self, "model_capability_fingerprint", capability)
        try:
            object.__setattr__(self, "workflow_depth", WorkflowDepth(self.workflow_depth))
            object.__setattr__(self, "retrieval_mode", RetrievalMode(self.retrieval_mode))
        except (TypeError, ValueError) as exc:
            raise ValueError("workflow_depth and retrieval_mode are unsupported") from exc
        if not isinstance(self.budget, ServerBudgetSnapshot):
            raise ValueError("plan budget must be a server-owned snapshot")
        goals = tuple(self.goals)
        if not 1 <= len(goals) <= min(MAX_GOALS, self.budget.max_goals):
            raise ValueError("plan goal count exceeds the server budget")
        _check_dag(goals)
        if len(goals) > 1 and len({_query_key(goal.query) for goal in goals}) < 2:
            raise ValueError("multi-goal plans require at least two distinct queries")
        object.__setattr__(self, "goals", goals)
        if self.created_at is not None and self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

    @property
    def canonical_payload(self) -> dict[str, Any]:
        """Hash input, excluding trace timestamps and any provider response facts."""

        ordered_goals = sorted(self.goals, key=lambda goal: goal.goal_id)
        return {
            "wire_version": self.wire_version,
            "plan_id": self.plan_id,
            "question_ref": self.question_ref,
            "workflow_depth": self.workflow_depth.value,
            "retrieval_mode": self.retrieval_mode.value,
            "capability_fingerprint": self.capability_fingerprint,
            "budget": _jsonable(self.budget),
            "goals": [
                {
                    "wire_version": goal.wire_version,
                    "goal_id": goal.goal_id,
                    "question": goal.question,
                    "query": goal.query,
                    "query_variants": sorted(goal.query_variants),
                    "depends_on": sorted(goal.depends_on),
                    "intent": goal.intent,
                    "status": goal.status.value,
                }
                for goal in ordered_goals
            ],
        }

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.canonical_payload)

    @property
    def canonical_hash(self) -> str:
        return canonical_hash(self.canonical_payload)

    @property
    def hash(self) -> str:
        return self.canonical_hash

    @property
    def plan_hash(self) -> str:
        return self.canonical_hash


@dataclass(frozen=True, slots=True)
class GoalCoverage:
    goal_id: str
    status: CoverageStatus
    evidence_keys: tuple[CanonicalEvidenceKey, ...] = ()
    missing_aspects: tuple[str, ...] = ()
    conflict_keys: tuple[CanonicalEvidenceKey, ...] = ()
    suggested_query: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _text(self.goal_id, name="goal_id", maximum=64))
        try:
            object.__setattr__(self, "status", CoverageStatus(self.status))
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported coverage status") from exc
        evidence = tuple(self.evidence_keys)
        conflicts = tuple(self.conflict_keys)
        if len(evidence) > MAX_EVIDENCE_KEYS_PER_GOAL:
            raise ValueError("goal evidence exceeds the server budget")
        if len(conflicts) > MAX_EVIDENCE_KEYS_PER_GOAL:
            raise ValueError("goal conflict evidence exceeds the server budget")
        if len({item.canonical for item in evidence}) != len(evidence):
            raise ValueError("coverage evidence keys must be unique")
        if len({item.canonical for item in conflicts}) != len(conflicts):
            raise ValueError("coverage conflict keys must be unique")
        if any(not isinstance(item, CanonicalEvidenceKey) for item in (*evidence, *conflicts)):
            raise ValueError("coverage evidence must use canonical evidence keys")
        missing = tuple(_text(item, name="missing_aspect", maximum=256) for item in self.missing_aspects)
        _unique(missing, name="missing_aspects")
        suggested = self.suggested_query
        if suggested is not None:
            suggested = _text(suggested, name="suggested_query", maximum=2048)
        # These combinations are deliberately fail-closed.  A supported goal
        # has no unresolved aspect; a missing goal has no admitted evidence;
        # conflict keeps at least two representations so callers cannot silently
        # choose one source.
        if self.status is CoverageStatus.SUPPORTED and (not evidence or missing or conflicts):
            raise ValueError("supported coverage requires evidence and no missing/conflict facts")
        if self.status is CoverageStatus.PARTIAL and (not evidence or not missing or conflicts):
            raise ValueError("partial coverage requires evidence and missing aspects")
        if self.status is CoverageStatus.MISSING and (evidence or not missing or conflicts):
            raise ValueError("missing coverage requires missing aspects and no evidence")
        if self.status is CoverageStatus.CONFLICT and (
            len(evidence) < 2
            or len(conflicts) < 2
            or missing
            or not {item.canonical for item in conflicts}.issubset(
                {item.canonical for item in evidence}
            )
        ):
            raise ValueError("conflict coverage requires two evidence keys and conflict facts")
        object.__setattr__(self, "evidence_keys", evidence)
        object.__setattr__(self, "conflict_keys", conflicts)
        object.__setattr__(self, "missing_aspects", missing)
        object.__setattr__(self, "suggested_query", suggested)

    @property
    def admitted_evidence_keys(self) -> tuple[CanonicalEvidenceKey, ...]:
        return self.evidence_keys

    @property
    def coverage(self) -> CoverageStatus:
        return self.status


CoverageItem = GoalCoverage


@dataclass(frozen=True, slots=True)
class ModelUsageFact:
    """Content-safe model usage facts; provider request/response bodies are absent."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in (("input_tokens", self.input_tokens), ("output_tokens", self.output_tokens)):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= MAX_USAGE_TOKENS
            ):
                raise ValueError(f"{name} must be between 0 and {MAX_USAGE_TOKENS}")


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Coverage assessment for every goal in one plan."""

    plan_id: str
    goals: tuple[GoalCoverage, ...]
    plan_goal_ids: tuple[str, ...] = ()
    allowed_evidence_keys: tuple[CanonicalEvidenceKey, ...] = ()
    admitted_evidence_keys: tuple[CanonicalEvidenceKey, ...] = ()
    repair_count: int = 0
    usage: ModelUsageFact = ModelUsageFact()
    wire_version: str = COVERAGE_WIRE_VERSION

    def __post_init__(self) -> None:
        if self.wire_version != COVERAGE_WIRE_VERSION:
            raise ValueError("unsupported coverage report version")
        object.__setattr__(self, "plan_id", _identity(self.plan_id, name="plan_id"))
        goals = tuple(self.goals)
        goal_ids = [goal.goal_id for goal in goals]
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("coverage goals must be unique")
        if any(not isinstance(goal, GoalCoverage) for goal in goals):
            raise ValueError("coverage goals must use GoalCoverage")
        plan_goal_ids = tuple(
            _text(item, name="plan_goal_id", maximum=64)
            for item in self.plan_goal_ids
        )
        if (
            not 1 <= len(plan_goal_ids) <= MAX_GOALS
            or len(plan_goal_ids) != len(set(plan_goal_ids))
            or set(plan_goal_ids) != set(goal_ids)
        ):
            raise ValueError("coverage goals must exactly match the frozen plan")
        allowed = tuple(self.allowed_evidence_keys)
        admitted = tuple(self.admitted_evidence_keys)
        if len(allowed) > MAX_EVIDENCE_KEYS_PER_REPORT:
            raise ValueError("coverage allowlist exceeds the server budget")
        if len(admitted) > MAX_EVIDENCE_KEYS_PER_REPORT:
            raise ValueError("coverage admitted evidence exceeds the server budget")
        for name, values in (("allowed_evidence_keys", allowed), ("admitted_evidence_keys", admitted)):
            if any(not isinstance(item, CanonicalEvidenceKey) for item in values):
                raise ValueError(f"{name} must contain canonical evidence keys")
            if len({item.canonical for item in values}) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        allowed_set = {item.canonical for item in allowed}
        all_evidence = {item.canonical: item for goal in goals for item in goal.evidence_keys}
        if not set(all_evidence).issubset(allowed_set):
            raise ValueError("coverage contains evidence outside the allowlist")
        if admitted:
            if {item.canonical for item in admitted} != set(all_evidence):
                raise ValueError("admitted evidence must equal per-goal evidence")
        else:
            admitted = tuple(all_evidence[key] for key in sorted(all_evidence))
        if any(
            item.plan_id != self.plan_id
            for item in (*allowed, *admitted, *all_evidence.values())
        ):
            raise ValueError("coverage evidence must be bound to this plan")
        if isinstance(self.repair_count, bool) or not 0 <= self.repair_count <= MAX_REPAIRS:
            raise ValueError("coverage repair_count must be between 0 and 1")
        if not isinstance(self.usage, ModelUsageFact):
            raise ValueError("coverage usage must be a ModelUsageFact")
        object.__setattr__(self, "goals", goals)
        object.__setattr__(self, "plan_goal_ids", plan_goal_ids)
        object.__setattr__(self, "allowed_evidence_keys", allowed)
        object.__setattr__(self, "admitted_evidence_keys", admitted)

    @property
    def items(self) -> tuple[GoalCoverage, ...]:
        return self.goals

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "wire_version": self.wire_version,
            "plan_id": self.plan_id,
            "plan_goal_ids": sorted(self.plan_goal_ids),
            "goals": [
                {
                    "goal_id": item.goal_id,
                    "status": item.status.value,
                    "evidence_keys": sorted(
                        (key.as_dict() for key in item.evidence_keys),
                        key=lambda value: canonical_json(value),
                    ),
                    "missing_aspects": sorted(item.missing_aspects),
                    "conflict_keys": sorted(
                        (key.as_dict() for key in item.conflict_keys),
                        key=lambda value: canonical_json(value),
                    ),
                    "suggested_query": item.suggested_query,
                }
                for item in sorted(self.goals, key=lambda value: value.goal_id)
            ],
            "allowed_evidence_keys": sorted((key.as_dict() for key in self.allowed_evidence_keys), key=lambda value: canonical_json(value)),
            "admitted_evidence_keys": sorted((key.as_dict() for key in self.admitted_evidence_keys), key=lambda value: canonical_json(value)),
            "repair_count": self.repair_count,
            "usage": _jsonable(self.usage),
        }

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.canonical_payload)

    @property
    def canonical_hash(self) -> str:
        return canonical_hash(self.canonical_payload)

    @property
    def report_hash(self) -> str:
        return self.canonical_hash


@dataclass(frozen=True, slots=True)
class SingleGoalFallback:
    """Versioned fact emitted when a plan wire cannot be repaired safely."""

    question_ref: str
    retrieval_mode: RetrievalMode = RetrievalMode.VECTOR
    reason: str = "plan_invalid"
    wire_version: str = SINGLE_GOAL_FALLBACK_VERSION
    outcome: StructuredOutcome = StructuredOutcome.SINGLE_GOAL_FALLBACK
    goal_count: int = 1
    query_variant_count: int = 1

    def __post_init__(self) -> None:
        if self.wire_version != SINGLE_GOAL_FALLBACK_VERSION:
            raise ValueError("unsupported single-goal fallback version")
        object.__setattr__(self, "question_ref", _identity(self.question_ref, name="question_ref"))
        if self.reason not in SINGLE_GOAL_FALLBACK_REASONS:
            raise ValueError("fallback reason is not allowlisted")
        object.__setattr__(self, "retrieval_mode", RetrievalMode(self.retrieval_mode))
        if self.outcome is not StructuredOutcome.SINGLE_GOAL_FALLBACK:
            raise ValueError("fallback outcome must be single_goal_fallback")
        if self.goal_count != 1 or self.query_variant_count != 1:
            raise ValueError("single-goal fallback facts must report one goal and query")


PlanFallback = SingleGoalFallback
SingleGoalFallbackFact = SingleGoalFallback


@dataclass(frozen=True, slots=True)
class CoverageAssessmentFailure:
    """Versioned fact emitted when coverage parse/repair remains invalid."""

    reason: str = "coverage_invalid"
    repair_count: int = MAX_REPAIRS
    wire_version: str = ASSESSMENT_FAILURE_VERSION
    outcome: StructuredOutcome = StructuredOutcome.ASSESSMENT_FAILURE
    error_code: ErrorCode = ErrorCode.CHAT_ASSESSMENT_INVALID

    def __post_init__(self) -> None:
        if self.wire_version != ASSESSMENT_FAILURE_VERSION:
            raise ValueError("unsupported assessment failure version")
        if self.reason not in ASSESSMENT_FAILURE_REASONS:
            raise ValueError("assessment failure reason is not allowlisted")
        if isinstance(self.repair_count, bool) or not 0 <= self.repair_count <= MAX_REPAIRS:
            raise ValueError("assessment repair_count must be between 0 and 1")
        if self.outcome is not StructuredOutcome.ASSESSMENT_FAILURE:
            raise ValueError("assessment failure outcome is fixed")
        if self.error_code is not ErrorCode.CHAT_ASSESSMENT_INVALID:
            raise ValueError("assessment failure uses CHAT_ASSESSMENT_INVALID")


AssessmentFailure = CoverageAssessmentFailure
CoverageAssessmentFailureFact = CoverageAssessmentFailure


def single_goal_fallback(question_ref: str, *, retrieval_mode: RetrievalMode = RetrievalMode.VECTOR, reason: str = "plan_invalid") -> SingleGoalFallback:
    """Build the only permitted fallback for an invalid plan wire."""

    return SingleGoalFallback(question_ref=question_ref, retrieval_mode=retrieval_mode, reason=reason)


def coverage_assessment_failure(*, reason: str = "coverage_invalid", repair_count: int = MAX_REPAIRS) -> CoverageAssessmentFailure:
    """Build a coverage assessment failure without exposing model output."""

    return CoverageAssessmentFailure(reason=reason, repair_count=repair_count)


def plan_canonical_hash(plan: PlanEnvelope) -> str:
    return plan.canonical_hash


def coverage_canonical_hash(report: CoverageReport) -> str:
    return report.canonical_hash


__all__ = [
    "ASSESSMENT_FAILURE_VERSION",
    "ASSESSMENT_FAILURE_REASONS",
    "BUDGET_WIRE_VERSION",
    "CAPABILITY_WIRE_VERSION",
    "COVERAGE_WIRE_VERSION",
    "EVIDENCE_KEY_VERSION",
    "FAILURE_WIRE_VERSION",
    "GOAL_WIRE_VERSION",
    "MAX_ADAPTIVE_WAVES",
    "MAX_DEADLINE_SECONDS",
    "MAX_EVIDENCE_KEYS_PER_GOAL",
    "MAX_EVIDENCE_KEYS_PER_REPORT",
    "MAX_GOALS",
    "MAX_OUTPUT_TOKENS",
    "MAX_PARALLELISM",
    "MAX_QUERY_VARIANTS_PER_GOAL",
    "MAX_REPAIRS",
    "MAX_RETRIEVAL_CALLS",
    "MAX_USAGE_TOKENS",
    "PLAN_WIRE_VERSION",
    "SINGLE_GOAL_FALLBACK_VERSION",
    "SINGLE_GOAL_FALLBACK_REASONS",
    "AssessmentFailure",
    "BudgetSnapshot",
    "CanonicalEvidenceKey",
    "CoverageAssessmentFailure",
    "CoverageAssessmentFailureFact",
    "CoverageItem",
    "CoverageReport",
    "CoverageStatus",
    "DeepRetrievalCapabilitySnapshot",
    "DeepRetrievalControlReason",
    "DeepRetrievalErrorCode",
    "DeepRetrievalFailureFact",
    "DeepRetrievalBudgetSnapshot",
    "Goal",
    "GoalCoverage",
    "GoalStatus",
    "ModelUsageFact",
    "PlanEnvelope",
    "PlanFallback",
    "RetrievalMode",
    "ServerBudgetSnapshot",
    "SingleGoalFallback",
    "SingleGoalFallbackFact",
    "StructuredOutcome",
    "WorkflowDepth",
    "canonical_hash",
    "canonical_json",
    "coverage_assessment_failure",
    "coverage_canonical_hash",
    "deep_retrieval_failure",
    "plan_canonical_hash",
    "single_goal_fallback",
]
