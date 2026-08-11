"""Public chat session, history, and durable run DTOs."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from rag_kb.domain import (
    AnswerStyle,
    ChatWorkflowMode,
    InsufficiencyPolicy,
    RerankMode,
)
from rag_kb.schemas.common import OpaqueCursor, PublicSchema


ChatProgressStageValue = Literal[
    "understand_query",
    "select_workflow",
    "retrieve_evidence",
    "assess_evidence",
    "prepare_visual_evidence",
    "generate_answer",
    "validate_answer",
    "persist_result",
]


class ChatSessionCreate(PublicSchema):
    knowledge_base_id: UUID
    title: Annotated[str, Field(min_length=1, max_length=512)] | None = None

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("title must contain non-whitespace characters")
        return normalized


class ChatSessionResponse(PublicSchema):
    id: UUID
    knowledge_base_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class ChatSessionPage(PublicSchema):
    items: tuple[ChatSessionResponse, ...]
    next_cursor: OpaqueCursor | None = None


class ChatMessageResponse(PublicSchema):
    id: UUID
    session_id: UUID
    run_id: UUID | None
    role: Literal["user", "assistant"]
    assistant_status: Literal["generating", "completed", "failed"] | None
    content: str
    created_at: datetime


class ChatMessagePage(PublicSchema):
    items: tuple[ChatMessageResponse, ...]
    next_cursor: OpaqueCursor | None = None


class AnswerPolicyOverrides(PublicSchema):
    answer_style: AnswerStyle | None = None
    insufficiency_policy: InsufficiencyPolicy | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_non_override_dimensions(cls, value: Any) -> Any:
        if isinstance(value, dict):
            unsupported = set(value) - {"answer_style", "insufficiency_policy"}
            if unsupported:
                raise PydanticCustomError(
                    "answer_policy_not_supported",
                    "answer policy contains unsupported dimensions",
                )
        return value

    @field_validator("answer_style", mode="before")
    @classmethod
    def validate_answer_style(cls, value: Any) -> Any:
        if value is None or value in {item.value for item in AnswerStyle}:
            return value
        raise PydanticCustomError(
            "answer_policy_not_supported", "answer style is not supported"
        )

    @field_validator("insufficiency_policy", mode="before")
    @classmethod
    def validate_insufficiency_policy(cls, value: Any) -> Any:
        if value is None or value in {item.value for item in InsufficiencyPolicy}:
            return value
        raise PydanticCustomError(
            "answer_policy_not_supported",
            "insufficiency policy is not supported",
        )


class ChatRetrievalRequest(PublicSchema):
    mode: Literal["vector", "hybrid"] = "vector"
    top_k: Annotated[int, Field(ge=1, le=100)] = 10
    rerank_mode: RerankMode | None = None

    @model_validator(mode="after")
    def require_supported_rerank_combination(self) -> "ChatRetrievalRequest":
        if self.mode == "hybrid" and self.rerank_mode is RerankMode.NONE:
            raise ValueError("hybrid retrieval requires reranking")
        if (
            self.rerank_mode is RerankMode.LOCAL_MINILM_V1
            and self.top_k > 20
        ):
            raise ValueError("local reranking supports top_k up to 20")
        return self


class ChatWorkflowRequest(PublicSchema):
    mode: ChatWorkflowMode = ChatWorkflowMode.SIMPLE


class ChatWorkflowCapabilityResponse(PublicSchema):
    mode: ChatWorkflowMode
    enabled: bool


class ChatWorkflowCapabilitiesResponse(PublicSchema):
    version: Literal["chat_workflow_v1"]
    default_mode: Literal["simple"]
    modes: tuple[ChatWorkflowCapabilityResponse, ...]


class ChatResearchAspectResponse(PublicSchema):
    aspect: Annotated[str, Field(min_length=1, max_length=1024)]
    status: Literal["supported", "partial", "missing", "conflict"]
    evidence_keys: tuple[Annotated[str, Field(min_length=1, max_length=1024)], ...]


class ChatResearchResultResponse(PublicSchema):
    version: Literal["research_result_v1"]
    status: Literal[
        "sufficient", "partial", "no_evidence", "conflict", "premise_unsupported"
    ]
    selected_evidence_keys: tuple[str, ...]
    aspects: tuple[ChatResearchAspectResponse, ...]
    covered_aspects: tuple[str, ...]
    missing_aspects: tuple[str, ...]
    conflicts: tuple[str, ...]
    termination_reason: Literal[
        "sufficient",
        "partial",
        "no_evidence",
        "no_progress",
        "budget_exhausted",
        "conflict_unresolved",
        "premise_unsupported",
    ]
    scope_status: Literal["all", "resolved", "ambiguous", "unresolved"] = "all"
    resolved_document_count: Annotated[int, Field(ge=0, le=4)] = 0
    complete_scan_document_count: Annotated[int, Field(ge=0, le=4)] = 0
    scope_rejection_count: Annotated[int, Field(ge=0, le=4)] = 0
    scope_downgrade_reason: str | None = None
    calculation_call_count: Annotated[int, Field(ge=0, le=4)] = 0
    calculation_success_count: Annotated[int, Field(ge=0, le=4)] = 0
    calculation_rejection_reasons: tuple[
        Annotated[str, Field(min_length=1, max_length=64)], ...
    ] = ()
    calculation_elapsed_ms: Annotated[int, Field(ge=0, le=120_000)] = 0


class ChatSearchTraceStepResponse(PublicSchema):
    observation_id: str
    objective: str
    queries: tuple[str, ...]
    based_on_observation_ids: tuple[str, ...]
    result: Literal["evidence_found", "no_evidence", "verification_gap"]
    new_evidence_count: Annotated[int, Field(ge=0, le=100)]


class ChatSearchTraceResponse(PublicSchema):
    version: Literal["search_trace_v1"]
    steps: tuple[ChatSearchTraceStepResponse, ...]
    decision_rounds: Annotated[int, Field(ge=0, le=8)]
    retrieval_calls: Annotated[int, Field(ge=0, le=12)]
    verifier_calls: Annotated[int, Field(ge=0, le=4)]
    evidence_count: Annotated[int, Field(ge=0, le=100)]
    adjacency_loaded_count: Annotated[int, Field(ge=0, le=100)] = 0
    adjacency_selected_count: Annotated[int, Field(ge=0, le=100)] = 0
    scope_status: Literal["all", "resolved", "ambiguous", "unresolved"] = "all"
    resolved_document_count: Annotated[int, Field(ge=0, le=4)] = 0
    complete_scan_document_count: Annotated[int, Field(ge=0, le=4)] = 0
    scope_rejection_count: Annotated[int, Field(ge=0, le=4)] = 0
    scope_downgrade_reason: str | None = None
    calculation_call_count: Annotated[int, Field(ge=0, le=4)] = 0
    calculation_success_count: Annotated[int, Field(ge=0, le=4)] = 0
    calculation_rejection_reasons: tuple[
        Annotated[str, Field(min_length=1, max_length=64)], ...
    ] = ()
    calculation_elapsed_ms: Annotated[int, Field(ge=0, le=120_000)] = 0


class ChatWorkflowResponse(PublicSchema):
    version: Literal["chat_workflow_v1"]
    requested_mode: ChatWorkflowMode
    resolved_mode: Literal["pending", "simple", "agent"]
    route_status: Literal["not_applicable", "pending", "resolved", "fallback"]
    route_reason_codes: tuple[
        Literal[
            "single_lookup",
            "direct_summary",
            "multi_view_required",
            "multi_hop_required",
            "evidence_uncertain",
            "router_invalid",
            "router_unavailable",
        ],
        ...,
    ] = ()
    research_result: ChatResearchResultResponse | None = None
    search_trace: ChatSearchTraceResponse | None = None


class ChatRunCreate(PublicSchema):
    session_id: UUID
    knowledge_base_id: UUID
    message: Annotated[str, Field(min_length=1, max_length=32768)]
    answer_policy: AnswerPolicyOverrides = AnswerPolicyOverrides()
    workflow: ChatWorkflowRequest = ChatWorkflowRequest()
    retrieval: ChatRetrievalRequest = ChatRetrievalRequest()
    model_profile_revision_id: UUID | None = None

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message must contain non-whitespace characters")
        return normalized


class EffectiveAnswerPolicyResponse(PublicSchema):
    grounding_policy: Literal["evidence_only"]
    answer_style: AnswerStyle
    insufficiency_policy: InsufficiencyPolicy
    citation_required: Literal[True]
    citation_granularity: Literal["claim_level"]
    answer_task: Literal["answer"]
    policy_version: Literal["p1"]


class ChatRunErrorResponse(PublicSchema):
    code: str
    detail: dict[str, Any]
    retryable: bool


class ChatRunRetrievalResponse(PublicSchema):
    profile_version: Literal[
        "exact_vector_v1",
        "hybrid_fts_rrf_v1",
        "exact_vector_v2",
        "hybrid_fts_rrf_v2",
    ]
    strategy: Literal["exact_vector", "hybrid"]
    top_k: Annotated[int, Field(ge=1, le=100)]
    rerank_mode: RerankMode


class ChatRunQueryContextResponse(PublicSchema):
    strategy: Literal["recent_completed_turns_v1"]
    status: Literal["pending", "original", "contextualized"]
    history_turn_count: Annotated[int, Field(ge=0, le=6)]
    history_token_count: Annotated[int, Field(ge=0, le=4000)]
    history_truncated: bool
    standalone_query: str | None
    rewrite_source: Literal["original", "model", "repair", "fallback"] | None


class ChatRunModelResponse(PublicSchema):
    profile_revision_id: UUID | None
    profile_name: str | None
    provider_name: str
    model: str
    revision: int | None
    temperature: float
    top_p: float | None
    sampling_top_k: int | None
    max_output_tokens: int
    reasoning_effort: Literal["off", "low", "medium", "high"]


class ChatCitationAssetResponse(PublicSchema):
    id: UUID
    media_type: str
    checksum_sha256: str
    content_url: str
    width: int | None = None
    height: int | None = None
    visual_unit_id: UUID | None = None
    parent_citation_id: str | None = None
    relation_type: str | None = None
    selection_reason: str | None = None


class ChatCitationResponse(PublicSchema):
    ordinal: Annotated[int, Field(ge=0)]
    index_chunk_id: UUID | None
    document_id: UUID
    document_version_id: UUID
    document_display_name: str
    document_original_filename: str
    quoted_text: str
    source_location: dict[str, Any]
    score: float | None
    modality: str = "text"
    asset: ChatCitationAssetResponse | None = None
    matched_representations: tuple[str, ...] = ("text",)


class ChatFinalContextAssetResponse(PublicSchema):
    id: UUID
    media_type: str
    checksum_sha256: str
    content_url: str
    width: int | None = None
    height: int | None = None


class ChatFinalContextMediaResponse(PublicSchema):
    message_index: Annotated[int, Field(ge=0)]
    citation_ids: tuple[Annotated[str, Field(min_length=1, max_length=128)], ...]
    asset: ChatFinalContextAssetResponse


class ChatFinalContextMessageResponse(PublicSchema):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRunFinalContextResponse(PublicSchema):
    run_id: UUID
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    available: bool
    version: Literal["final_llm_context_v1"] | None = None
    operation: Literal["generate_answer", "repair_answer"] | None = None
    output_schema: Literal["answer_v1"] | None = None
    max_output_tokens: Annotated[int, Field(ge=1, le=8192)] | None = None
    messages: tuple[ChatFinalContextMessageResponse, ...] = ()
    media: tuple[ChatFinalContextMediaResponse, ...] = ()


class ChatRunResponse(PublicSchema):
    run_id: UUID
    knowledge_base_id: UUID
    session_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    index_revision_id: UUID
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    assistant_status: Literal["generating", "completed", "failed"]
    answer: str | None
    citations: tuple[ChatCitationResponse, ...]
    status_url: str
    events_url: str
    final_context_url: str
    effective_answer_policy: EffectiveAnswerPolicyResponse
    workflow: ChatWorkflowResponse
    retrieval: ChatRunRetrievalResponse
    model: ChatRunModelResponse
    query_context: ChatRunQueryContextResponse
    attempt: int
    error: ChatRunErrorResponse | None
    usage: dict[str, Any] | None
    timing: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class ChatAnswerCompletedEvent(PublicSchema):
    run_id: UUID
    message_id: UUID
    answer: str
    citations: tuple[ChatCitationResponse, ...]
    effective_answer_policy: EffectiveAnswerPolicyResponse
    status_url: str


class ChatRunFailedEvent(PublicSchema):
    run_id: UUID
    status: Literal["failed", "cancelled"]
    error: ChatRunErrorResponse
    effective_answer_policy: EffectiveAnswerPolicyResponse
    status_url: str


class ChatAnswerPreviewEvent(PublicSchema):
    run_id: UUID
    attempt: Annotated[int, Field(ge=1)]
    seq: Annotated[int, Field(ge=1)]
    delta: Annotated[str, Field(min_length=1)]


class ChatAnswerPreviewResetEvent(PublicSchema):
    run_id: UUID
    attempt: Annotated[int, Field(ge=1)]
    seq: Annotated[int, Field(ge=1)]
    reason: Literal[
        "generation_failed",
        "validation_repair",
        "preview_invalid",
    ]


class ChatWorkflowProgressFacts(PublicSchema):
    objective: Annotated[str, Field(max_length=160)] | None
    queries: Annotated[tuple[Annotated[str, Field(max_length=160)], ...], Field(max_length=3)]
    evidence_count: Annotated[int, Field(ge=0, le=1000)] | None
    new_evidence_count: Annotated[int, Field(ge=0, le=1000)] | None
    retrieval_calls: Annotated[int, Field(ge=0, le=1000)] | None
    route_status: Literal[
        "not_applicable", "pending", "resolved", "fallback"
    ] | None
    route_reason_codes: Annotated[
        tuple[
            Literal[
                "single_lookup",
                "direct_summary",
                "multi_view_required",
                "multi_hop_required",
                "evidence_uncertain",
                "router_invalid",
                "router_unavailable",
            ],
            ...,
        ],
        Field(max_length=6),
    ]
    research_status: Literal[
        "sufficient", "partial", "no_evidence", "conflict", "premise_unsupported"
    ] | None
    covered_aspects: Annotated[
        tuple[Annotated[str, Field(max_length=160)], ...], Field(max_length=6)
    ]
    missing_aspects: Annotated[
        tuple[Annotated[str, Field(max_length=160)], ...], Field(max_length=6)
    ]
    conflict_count: Annotated[int, Field(ge=0, le=1000)] | None
    decision: Literal[
        "select_simple",
        "select_agent",
        "search_evidence",
        "continue_search",
        "finish_research",
    ] | None


class ChatWorkflowProgressEvent(PublicSchema):
    run_id: UUID
    attempt: Annotated[int, Field(ge=1)]
    seq: Annotated[int, Field(ge=1)]
    active_stage: ChatProgressStageValue
    activity: Literal[
        "load_context",
        "contextualize_query",
        "route_decision",
        "simple_search",
        "agent_decision",
        "agent_search",
        "retrieval_complete",
        "verify_coverage",
        "research_complete",
        "assess_evidence",
        "prepare_visual_evidence",
        "generate_answer",
        "validate_answer",
        "persist_result",
    ]
    completed_stages: Annotated[
        tuple[ChatProgressStageValue, ...], Field(max_length=8)
    ]
    status: Literal["active", "completed"]
    requested_mode: Literal["simple", "agent", "auto"] | None
    resolved_mode: Literal["pending", "simple", "agent"]
    facts: ChatWorkflowProgressFacts
