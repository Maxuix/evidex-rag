"""Strict Pydantic wire models for the bounded deep-retrieval contracts."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from rag_kb.domain.deep_retrieval import (
    ASSESSMENT_FAILURE_VERSION,
    BUDGET_WIRE_VERSION,
    CAPABILITY_WIRE_VERSION,
    COVERAGE_WIRE_VERSION,
    EVIDENCE_KEY_VERSION,
    FAILURE_WIRE_VERSION,
    GOAL_WIRE_VERSION,
    MAX_ADAPTIVE_WAVES,
    MAX_DEADLINE_SECONDS,
    MAX_EVIDENCE_KEYS_PER_GOAL,
    MAX_EVIDENCE_KEYS_PER_REPORT,
    MAX_GOALS,
    MAX_OUTPUT_TOKENS,
    MAX_PARALLELISM,
    MAX_QUERY_VARIANTS_PER_GOAL,
    MAX_REPAIRS,
    MAX_RETRIEVAL_CALLS,
    MAX_USAGE_TOKENS,
    PLAN_WIRE_VERSION,
    SINGLE_GOAL_FALLBACK_VERSION,
    CanonicalEvidenceKey,
    CoverageAssessmentFailure,
    CoverageReport,
    CoverageStatus,
    DeepRetrievalCapabilitySnapshot,
    DeepRetrievalErrorCode,
    DeepRetrievalFailureFact,
    Goal,
    GoalCoverage,
    GoalStatus,
    ModelUsageFact,
    PlanEnvelope,
    RetrievalMode,
    ServerBudgetSnapshot,
    SingleGoalFallback,
    StructuredOutcome,
    WorkflowDepth,
)
from rag_kb.schemas.common import PublicSchema


class DeepRetrievalSchema(PublicSchema):
    """Strict/frozen base shared by every future deep-retrieval wire."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        use_enum_values=False,
    )

Identity = Annotated[
    str, Field(min_length=1, max_length=256, pattern=r"^[^\s]{1,256}$")
]
GoalId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")]
Question = Annotated[str, Field(min_length=1, max_length=2048)]
Fingerprint = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class DeepRetrievalCapabilitySnapshotWire(DeepRetrievalSchema):
    wire_version: Literal[CAPABILITY_WIRE_VERSION] = CAPABILITY_WIRE_VERSION
    deep_workflow_enabled: bool = False
    static_multi_query_v1_enabled: bool = False
    plan_wire_version: Literal[PLAN_WIRE_VERSION] = PLAN_WIRE_VERSION
    budget_wire_version: Literal[BUDGET_WIRE_VERSION] = BUDGET_WIRE_VERSION
    configuration_fingerprint: Annotated[
        str, Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ]

    @model_validator(mode="after")
    def validate_capability_dependencies(self) -> "DeepRetrievalCapabilitySnapshotWire":
        self.to_domain()
        return self

    def to_domain(self) -> DeepRetrievalCapabilitySnapshot:
        return DeepRetrievalCapabilitySnapshot(
            configuration_fingerprint=self.configuration_fingerprint,
            deep_workflow_enabled=self.deep_workflow_enabled,
            static_multi_query_v1_enabled=self.static_multi_query_v1_enabled,
            wire_version=self.wire_version,
            plan_wire_version=self.plan_wire_version,
            budget_wire_version=self.budget_wire_version,
        )

    @classmethod
    def from_domain(
        cls, value: DeepRetrievalCapabilitySnapshot
    ) -> "DeepRetrievalCapabilitySnapshotWire":
        return cls.model_validate(
            {
                "wire_version": value.wire_version,
                "deep_workflow_enabled": value.deep_workflow_enabled,
                "static_multi_query_v1_enabled": value.static_multi_query_v1_enabled,
                "plan_wire_version": value.plan_wire_version,
                "budget_wire_version": value.budget_wire_version,
                "configuration_fingerprint": value.configuration_fingerprint,
            }
        )


FailureField = Literal[
    "workflow_depth",
    "wire_version",
    "goal_count",
    "query_count",
    "retrieval_call_count",
    "repair_count",
    "deadline_seconds",
    "plan_hash",
    "index_revision_id",
]


class DeepRetrievalFailureDetailWire(DeepRetrievalSchema):
    check: Literal[
        "capability",
        "structured_wire",
        "budget",
        "dependency",
        "plan_hash",
        "revision",
        "provider",
        "deadline",
        "assessment",
    ]
    field: FailureField | None = None
    count: Annotated[int, Field(ge=0)] | None = None
    limit: Annotated[int, Field(ge=0)] | None = None


class DeepRetrievalFailureWire(DeepRetrievalSchema):
    wire_version: Literal[FAILURE_WIRE_VERSION] = FAILURE_WIRE_VERSION
    code: DeepRetrievalErrorCode
    status: Annotated[int, Field(ge=400, le=599)]
    retryable: bool
    detail: DeepRetrievalFailureDetailWire

    @field_validator("code", mode="before")
    @classmethod
    def parse_error_code(cls, value: object) -> DeepRetrievalErrorCode:
        return value if isinstance(value, DeepRetrievalErrorCode) else DeepRetrievalErrorCode(value)

    @model_validator(mode="after")
    def validate_error_policy(self) -> "DeepRetrievalFailureWire":
        self.to_domain()
        return self

    def to_domain(self) -> DeepRetrievalFailureFact:
        return DeepRetrievalFailureFact(
            code=self.code,
            check=self.detail.check,
            status=self.status,
            retryable=self.retryable,
            field=self.detail.field,
            count=self.detail.count,
            limit=self.detail.limit,
            wire_version=self.wire_version,
        )

    @classmethod
    def from_domain(cls, value: DeepRetrievalFailureFact) -> "DeepRetrievalFailureWire":
        return cls.model_validate(
            {
                "wire_version": value.wire_version,
                "code": value.code,
                "status": value.status,
                "retryable": value.retryable,
                "detail": {
                    "check": value.check,
                    "field": value.field,
                    "count": value.count,
                    "limit": value.limit,
                },
            }
        )


class ServerBudgetSnapshotWire(DeepRetrievalSchema):
    wire_version: Literal[BUDGET_WIRE_VERSION] = BUDGET_WIRE_VERSION
    max_goals: Annotated[int, Field(ge=1, le=MAX_GOALS)] = MAX_GOALS
    max_query_variants_per_goal: Literal[MAX_QUERY_VARIANTS_PER_GOAL] = MAX_QUERY_VARIANTS_PER_GOAL
    max_adaptive_waves: Literal[MAX_ADAPTIVE_WAVES] = MAX_ADAPTIVE_WAVES
    max_repairs: Annotated[int, Field(ge=0, le=MAX_REPAIRS)] = MAX_REPAIRS
    deadline_seconds: Annotated[
        float, Field(gt=0, le=MAX_DEADLINE_SECONDS, allow_inf_nan=False)
    ] = 30.0
    max_parallelism: Literal[MAX_PARALLELISM] = MAX_PARALLELISM
    max_retrieval_calls: Literal[MAX_RETRIEVAL_CALLS] = MAX_RETRIEVAL_CALLS
    max_output_tokens: Annotated[int, Field(ge=1, le=MAX_OUTPUT_TOKENS)] = MAX_OUTPUT_TOKENS

    @model_validator(mode="after")
    def validate_finite_deadline(self) -> "ServerBudgetSnapshotWire":
        if not math.isfinite(self.deadline_seconds):
            raise ValueError("deadline_seconds must be finite")
        return self

    def to_domain(self) -> ServerBudgetSnapshot:
        return ServerBudgetSnapshot(**self.model_dump())

    @classmethod
    def from_domain(cls, value: ServerBudgetSnapshot) -> "ServerBudgetSnapshotWire":
        return cls.model_validate(value.__dict__ if hasattr(value, "__dict__") else {
            "wire_version": value.wire_version,
            "max_goals": value.max_goals,
            "max_query_variants_per_goal": value.max_query_variants_per_goal,
            "max_adaptive_waves": value.max_adaptive_waves,
            "max_repairs": value.max_repairs,
            "deadline_seconds": value.deadline_seconds,
            "max_parallelism": value.max_parallelism,
            "max_retrieval_calls": value.max_retrieval_calls,
            "max_output_tokens": value.max_output_tokens,
        })


BudgetSnapshotWire = ServerBudgetSnapshotWire
DeepRetrievalBudgetSnapshotWire = ServerBudgetSnapshotWire


class CanonicalEvidenceKeyWire(DeepRetrievalSchema):
    wire_version: Literal[EVIDENCE_KEY_VERSION] = EVIDENCE_KEY_VERSION
    revision_id: Identity
    target_id: Identity
    chunk_or_group_id: Identity
    representation_id: Identity
    representation_kind: Literal["chunk", "group"] = "chunk"
    plan_id: Identity | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> "CanonicalEvidenceKeyWire":
        CanonicalEvidenceKey(
            revision_id=self.revision_id,
            target_id=self.target_id,
            chunk_or_group_id=self.chunk_or_group_id,
            representation_id=self.representation_id,
            representation_kind=self.representation_kind,
            plan_id=self.plan_id,
        )
        return self

    def to_domain(self) -> CanonicalEvidenceKey:
        return CanonicalEvidenceKey(
            revision_id=self.revision_id,
            target_id=self.target_id,
            chunk_or_group_id=self.chunk_or_group_id,
            representation_id=self.representation_id,
            representation_kind=self.representation_kind,
            plan_id=self.plan_id,
        )

    @classmethod
    def from_domain(cls, value: CanonicalEvidenceKey) -> "CanonicalEvidenceKeyWire":
        return cls.model_validate(value.as_dict())


EvidenceKeyWire = CanonicalEvidenceKeyWire


class GoalWire(DeepRetrievalSchema):
    wire_version: Literal[GOAL_WIRE_VERSION] = GOAL_WIRE_VERSION
    goal_id: GoalId
    question: Question
    query: Question | None = None
    query_variants: Annotated[tuple[Question, ...], Field(max_length=MAX_QUERY_VARIANTS_PER_GOAL)] = ()
    depends_on: Annotated[tuple[GoalId, ...], Field(max_length=MAX_GOALS - 1)] = ()
    intent: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    status: GoalStatus = GoalStatus.PENDING

    @field_validator("status", mode="before")
    @classmethod
    def parse_goal_status(cls, value: object) -> GoalStatus:
        return value if isinstance(value, GoalStatus) else GoalStatus(value)

    @model_validator(mode="after")
    def normalize_one_query(self) -> "GoalWire":
        variants = tuple(self.query_variants)
        if len(variants) != len(set(variants)):
            raise ValueError("query_variants must be unique")
        query = self.query
        if query is None and variants:
            query = variants[0]
        elif query is not None and not variants:
            variants = (query,)
        elif query is not None and variants and variants[0] != query:
            raise ValueError("query must equal the sole query variant")
        if query is None or len(variants) != MAX_QUERY_VARIANTS_PER_GOAL:
            raise ValueError("Goal requires exactly one non-empty query variant")
        if self.goal_id in self.depends_on:
            raise ValueError("goal cannot depend on itself")
        normalized = Goal(
            goal_id=self.goal_id,
            question=self.question,
            query=query,
            query_variants=variants,
            depends_on=tuple(self.depends_on),
            intent=self.intent,
            status=self.status,
            wire_version=self.wire_version,
        )
        return self.model_copy(
            update={
                "goal_id": normalized.goal_id,
                "question": normalized.question,
                "query": normalized.query,
                "query_variants": normalized.query_variants,
                "depends_on": normalized.depends_on,
                "intent": normalized.intent,
                "status": normalized.status,
            }
        )

    def to_domain(self) -> Goal:
        return Goal(
            goal_id=self.goal_id,
            question=self.question,
            query=self.query,
            query_variants=tuple(self.query_variants),
            depends_on=tuple(self.depends_on),
            intent=self.intent,
            status=self.status,
            wire_version=self.wire_version,
        )

    @classmethod
    def from_domain(cls, value: Goal) -> "GoalWire":
        return cls.model_validate(
            {
                "wire_version": value.wire_version,
                "goal_id": value.goal_id,
                "question": value.question,
                "query": value.query,
                "query_variants": value.query_variants,
                "depends_on": value.depends_on,
                "intent": value.intent,
                "status": value.status,
            }
        )


QueryGoalWire = GoalWire


class PlanEnvelopeWire(DeepRetrievalSchema):
    wire_version: Literal[PLAN_WIRE_VERSION] = PLAN_WIRE_VERSION
    plan_id: Identity
    question_ref: Identity
    workflow_depth: WorkflowDepth
    retrieval_mode: RetrievalMode
    goals: Annotated[tuple[GoalWire, ...], Field(min_length=1, max_length=MAX_GOALS)]
    budget: ServerBudgetSnapshotWire
    capability_fingerprint: Fingerprint | None = None
    created_at: datetime | None = None
    model_capability_fingerprint: Fingerprint | None = None

    @field_validator("workflow_depth", mode="before")
    @classmethod
    def parse_workflow_depth(cls, value: object) -> WorkflowDepth:
        return value if isinstance(value, WorkflowDepth) else WorkflowDepth(value)

    @field_validator("retrieval_mode", mode="before")
    @classmethod
    def parse_retrieval_mode(cls, value: object) -> RetrievalMode:
        return value if isinstance(value, RetrievalMode) else RetrievalMode(value)

    @model_validator(mode="after")
    def normalize_capability_fingerprint(self) -> "PlanEnvelopeWire":
        capability = self.capability_fingerprint
        model_capability = self.model_capability_fingerprint
        if capability is None:
            capability = model_capability
        elif model_capability is not None and capability != model_capability:
            raise ValueError("capability fingerprints must agree")
        if capability is None:
            raise ValueError("plan requires a capability fingerprint")
        return self.model_copy(
            update={
                "capability_fingerprint": capability,
                "model_capability_fingerprint": capability,
            }
        )

    @model_validator(mode="after")
    def validate_dependencies(self) -> "PlanEnvelopeWire":
        goals = tuple(goal.to_domain() for goal in self.goals)
        # Domain construction performs duplicate/unknown/cycle validation and
        # also protects against a model constructed directly from a dict.
        PlanEnvelope(
            plan_id=self.plan_id,
            question_ref=self.question_ref,
            workflow_depth=self.workflow_depth,
            retrieval_mode=self.retrieval_mode,
            goals=goals,
            budget=self.budget.to_domain(),
            capability_fingerprint=self.capability_fingerprint,
            wire_version=self.wire_version,
            created_at=self.created_at,
            model_capability_fingerprint=self.model_capability_fingerprint,
        )
        return self

    def to_domain(self) -> PlanEnvelope:
        return PlanEnvelope(
            plan_id=self.plan_id,
            question_ref=self.question_ref,
            workflow_depth=self.workflow_depth,
            retrieval_mode=self.retrieval_mode,
            goals=tuple(goal.to_domain() for goal in self.goals),
            budget=self.budget.to_domain(),
            capability_fingerprint=self.capability_fingerprint,
            wire_version=self.wire_version,
            created_at=self.created_at,
        )

    @classmethod
    def from_domain(cls, value: PlanEnvelope) -> "PlanEnvelopeWire":
        return cls.model_validate(
            {
                "wire_version": value.wire_version,
                "plan_id": value.plan_id,
                "question_ref": value.question_ref,
                "workflow_depth": value.workflow_depth,
                "retrieval_mode": value.retrieval_mode,
                "goals": tuple(GoalWire.from_domain(goal) for goal in value.goals),
                "budget": ServerBudgetSnapshotWire.from_domain(value.budget),
                "capability_fingerprint": value.capability_fingerprint,
                "created_at": value.created_at,
                "model_capability_fingerprint": value.model_capability_fingerprint,
            }
        )

    @property
    def canonical_json(self) -> str:
        return self.to_domain().canonical_json

    @property
    def canonical_hash(self) -> str:
        return self.to_domain().canonical_hash

    @property
    def plan_hash(self) -> str:
        return self.canonical_hash


MultiQueryPlanWire = PlanEnvelopeWire


class ModelUsageFactWire(DeepRetrievalSchema):
    input_tokens: Annotated[int, Field(ge=0, le=MAX_USAGE_TOKENS)] = 0
    output_tokens: Annotated[int, Field(ge=0, le=MAX_USAGE_TOKENS)] = 0

    def to_domain(self) -> ModelUsageFact:
        return ModelUsageFact(**self.model_dump())


class GoalCoverageWire(DeepRetrievalSchema):
    goal_id: GoalId
    status: CoverageStatus
    evidence_keys: Annotated[
        tuple[CanonicalEvidenceKeyWire, ...],
        Field(max_length=MAX_EVIDENCE_KEYS_PER_GOAL),
    ] = ()
    missing_aspects: Annotated[tuple[str, ...], Field(max_length=32)] = ()
    conflict_keys: Annotated[
        tuple[CanonicalEvidenceKeyWire, ...],
        Field(max_length=MAX_EVIDENCE_KEYS_PER_GOAL),
    ] = ()
    suggested_query: Question | None = None

    @field_validator("status", mode="before")
    @classmethod
    def parse_coverage_status(cls, value: object) -> CoverageStatus:
        return value if isinstance(value, CoverageStatus) else CoverageStatus(value)

    @model_validator(mode="after")
    def validate_combinations(self) -> "GoalCoverageWire":
        evidence = tuple(key.to_domain() for key in self.evidence_keys)
        conflicts = tuple(key.to_domain() for key in self.conflict_keys)
        GoalCoverage(
            goal_id=self.goal_id,
            status=self.status,
            evidence_keys=evidence,
            missing_aspects=tuple(self.missing_aspects),
            conflict_keys=conflicts,
            suggested_query=self.suggested_query,
        )
        return self

    def to_domain(self) -> GoalCoverage:
        return GoalCoverage(
            goal_id=self.goal_id,
            status=self.status,
            evidence_keys=tuple(key.to_domain() for key in self.evidence_keys),
            missing_aspects=tuple(self.missing_aspects),
            conflict_keys=tuple(key.to_domain() for key in self.conflict_keys),
            suggested_query=self.suggested_query,
        )

    @classmethod
    def from_domain(cls, value: GoalCoverage) -> "GoalCoverageWire":
        return cls.model_validate(
            {
                "goal_id": value.goal_id,
                "status": value.status,
                "evidence_keys": tuple(CanonicalEvidenceKeyWire.from_domain(key) for key in value.evidence_keys),
                "missing_aspects": value.missing_aspects,
                "conflict_keys": tuple(CanonicalEvidenceKeyWire.from_domain(key) for key in value.conflict_keys),
                "suggested_query": value.suggested_query,
            }
        )


CoverageItemWire = GoalCoverageWire


class CoverageReportWire(DeepRetrievalSchema):
    wire_version: Literal[COVERAGE_WIRE_VERSION] = COVERAGE_WIRE_VERSION
    plan_id: Identity
    goals: Annotated[tuple[GoalCoverageWire, ...], Field(min_length=1, max_length=MAX_GOALS)]
    plan_goal_ids: Annotated[tuple[GoalId, ...], Field(min_length=1, max_length=MAX_GOALS)]
    allowed_evidence_keys: Annotated[
        tuple[CanonicalEvidenceKeyWire, ...],
        Field(max_length=MAX_EVIDENCE_KEYS_PER_REPORT),
    ] = ()
    admitted_evidence_keys: Annotated[
        tuple[CanonicalEvidenceKeyWire, ...],
        Field(max_length=MAX_EVIDENCE_KEYS_PER_REPORT),
    ] = ()
    repair_count: Annotated[int, Field(ge=0, le=MAX_REPAIRS)] = 0
    usage: ModelUsageFactWire = ModelUsageFactWire()

    @model_validator(mode="after")
    def validate_allowlist(self) -> "CoverageReportWire":
        CoverageReport(
            plan_id=self.plan_id,
            goals=tuple(goal.to_domain() for goal in self.goals),
            plan_goal_ids=tuple(self.plan_goal_ids),
            allowed_evidence_keys=tuple(key.to_domain() for key in self.allowed_evidence_keys),
            admitted_evidence_keys=tuple(key.to_domain() for key in self.admitted_evidence_keys),
            repair_count=self.repair_count,
            usage=self.usage.to_domain(),
            wire_version=self.wire_version,
        )
        return self

    def to_domain(self) -> CoverageReport:
        return CoverageReport(
            plan_id=self.plan_id,
            goals=tuple(goal.to_domain() for goal in self.goals),
            plan_goal_ids=tuple(self.plan_goal_ids),
            allowed_evidence_keys=tuple(key.to_domain() for key in self.allowed_evidence_keys),
            admitted_evidence_keys=tuple(key.to_domain() for key in self.admitted_evidence_keys),
            repair_count=self.repair_count,
            usage=self.usage.to_domain(),
            wire_version=self.wire_version,
        )

    @classmethod
    def from_domain(cls, value: CoverageReport) -> "CoverageReportWire":
        return cls.model_validate(
            {
                "wire_version": value.wire_version,
                "plan_id": value.plan_id,
                "goals": tuple(GoalCoverageWire.from_domain(goal) for goal in value.goals),
                "plan_goal_ids": value.plan_goal_ids,
                "allowed_evidence_keys": tuple(CanonicalEvidenceKeyWire.from_domain(key) for key in value.allowed_evidence_keys),
                "admitted_evidence_keys": tuple(CanonicalEvidenceKeyWire.from_domain(key) for key in value.admitted_evidence_keys),
                "repair_count": value.repair_count,
                "usage": ModelUsageFactWire.model_validate(value.usage.__dict__ if hasattr(value.usage, "__dict__") else {
                    "input_tokens": value.usage.input_tokens,
                    "output_tokens": value.usage.output_tokens,
                }),
            }
        )

    @property
    def canonical_json(self) -> str:
        return self.to_domain().canonical_json

    @property
    def canonical_hash(self) -> str:
        return self.to_domain().canonical_hash

    @property
    def report_hash(self) -> str:
        return self.canonical_hash


CoverageWire = CoverageReportWire


class SingleGoalFallbackWire(DeepRetrievalSchema):
    wire_version: Literal[SINGLE_GOAL_FALLBACK_VERSION] = SINGLE_GOAL_FALLBACK_VERSION
    outcome: Literal[StructuredOutcome.SINGLE_GOAL_FALLBACK] = StructuredOutcome.SINGLE_GOAL_FALLBACK
    question_ref: Identity
    retrieval_mode: RetrievalMode = RetrievalMode.VECTOR
    reason: Literal[
        "plan_invalid", "plan_empty", "plan_budget_exceeded"
    ] = "plan_invalid"
    goal_count: Literal[1] = 1
    query_variant_count: Literal[1] = 1

    @field_validator("retrieval_mode", mode="before")
    @classmethod
    def parse_retrieval_mode(cls, value: object) -> RetrievalMode:
        return value if isinstance(value, RetrievalMode) else RetrievalMode(value)

    def to_domain(self) -> SingleGoalFallback:
        return SingleGoalFallback(
            question_ref=self.question_ref,
            retrieval_mode=self.retrieval_mode,
            reason=self.reason,
            wire_version=self.wire_version,
            outcome=self.outcome,
            goal_count=self.goal_count,
            query_variant_count=self.query_variant_count,
        )

    @classmethod
    def from_domain(cls, value: SingleGoalFallback) -> "SingleGoalFallbackWire":
        return cls.model_validate(
            {
                "wire_version": value.wire_version,
                "outcome": value.outcome,
                "question_ref": value.question_ref,
                "retrieval_mode": value.retrieval_mode,
                "reason": value.reason,
                "goal_count": value.goal_count,
                "query_variant_count": value.query_variant_count,
            }
        )


PlanFallbackWire = SingleGoalFallbackWire


class CoverageAssessmentFailureWire(DeepRetrievalSchema):
    wire_version: Literal[ASSESSMENT_FAILURE_VERSION] = ASSESSMENT_FAILURE_VERSION
    outcome: Literal[StructuredOutcome.ASSESSMENT_FAILURE] = StructuredOutcome.ASSESSMENT_FAILURE
    reason: Literal[
        "coverage_invalid", "coverage_budget_exceeded", "coverage_unavailable"
    ] = "coverage_invalid"
    repair_count: Annotated[int, Field(ge=0, le=MAX_REPAIRS)] = MAX_REPAIRS
    error_code: Literal["CHAT_ASSESSMENT_INVALID"] = "CHAT_ASSESSMENT_INVALID"

    def to_domain(self) -> CoverageAssessmentFailure:
        from rag_kb.domain.errors import ErrorCode

        return CoverageAssessmentFailure(
            reason=self.reason,
            repair_count=self.repair_count,
            wire_version=self.wire_version,
            outcome=self.outcome,
            error_code=ErrorCode.CHAT_ASSESSMENT_INVALID,
        )

    @classmethod
    def from_domain(cls, value: CoverageAssessmentFailure) -> "CoverageAssessmentFailureWire":
        return cls.model_validate(
            {
                "wire_version": value.wire_version,
                "outcome": value.outcome,
                "reason": value.reason,
                "repair_count": value.repair_count,
                "error_code": value.error_code.value,
            }
        )


AssessmentFailureWire = CoverageAssessmentFailureWire


__all__ = [
    "AssessmentFailureWire",
    "BudgetSnapshotWire",
    "CanonicalEvidenceKeyWire",
    "DeepRetrievalCapabilitySnapshotWire",
    "DeepRetrievalFailureDetailWire",
    "DeepRetrievalFailureWire",
    "CoverageAssessmentFailureWire",
    "CoverageItemWire",
    "CoverageReportWire",
    "CoverageWire",
    "DeepRetrievalBudgetSnapshotWire",
    "DeepRetrievalSchema",
    "EvidenceKeyWire",
    "GoalCoverageWire",
    "GoalWire",
    "ModelUsageFactWire",
    "MultiQueryPlanWire",
    "PlanEnvelopeWire",
    "PlanFallbackWire",
    "QueryGoalWire",
    "ServerBudgetSnapshotWire",
    "SingleGoalFallbackWire",
]
