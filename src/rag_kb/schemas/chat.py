"""Public chat session, history, and durable run DTOs."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from rag_kb.domain import (
    CHAT_AGENT_DEFAULT_EVIDENCE_ITEMS,
    CHAT_AGENT_DEFAULT_RETRIEVAL_CALLS,
    CHAT_AGENT_DEFAULT_TOTAL_TOKENS,
    CHAT_AGENT_MAX_EVIDENCE_ITEMS,
    CHAT_AGENT_MAX_GRAPH_CALLS,
    CHAT_AGENT_MAX_MODEL_ROUNDS,
    CHAT_AGENT_MAX_RETRIEVAL_CALLS,
    CHAT_AGENT_MAX_TOTAL_TOKENS,
    CHAT_AGENT_MIN_TOTAL_TOKENS,
    RerankMode,
)
from rag_kb.schemas.common import OpaqueCursor, PublicSchema


ChatProgressStageValue = Literal[
    "understand_query",
    "retrieve_evidence",
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


class ChatRetrievalRequest(PublicSchema):
    mode: Literal["vector", "hybrid", "graph", "auto"] = "vector"
    top_k: Annotated[int, Field(ge=1, le=100)] = 10
    rerank_mode: RerankMode | None = None

    @model_validator(mode="after")
    def require_supported_rerank_combination(self) -> "ChatRetrievalRequest":
        if self.mode == "graph":
            if not 4 <= self.top_k <= 20:
                raise ValueError("graph retrieval top_k must be between 4 and 20")
            if self.rerank_mode is not RerankMode.CLASSIC:
                raise ValueError("graph retrieval requires classic reranking")
            return self
        if self.mode == "hybrid" and self.rerank_mode is RerankMode.NONE:
            raise ValueError("hybrid retrieval requires reranking")
        if (
            self.rerank_mode is RerankMode.LOCAL_MINILM_V1
            and self.top_k > 20
        ):
            raise ValueError("local reranking supports top_k up to 20")
        return self


class ChatAgentBudgetResponse(PublicSchema):
    max_model_rounds: Annotated[int, Field(ge=1, le=CHAT_AGENT_MAX_MODEL_ROUNDS)]
    max_graph_calls: Annotated[int, Field(ge=1, le=CHAT_AGENT_MAX_GRAPH_CALLS)]
    max_total_tokens: Annotated[
        int,
        Field(ge=CHAT_AGENT_MIN_TOTAL_TOKENS, le=CHAT_AGENT_MAX_TOTAL_TOKENS),
    ] = CHAT_AGENT_DEFAULT_TOTAL_TOKENS
    max_evidence_items: Annotated[
        int, Field(ge=1, le=CHAT_AGENT_MAX_EVIDENCE_ITEMS)
    ] = CHAT_AGENT_DEFAULT_EVIDENCE_ITEMS
    max_retrieval_calls: Annotated[
        int, Field(ge=1, le=CHAT_AGENT_MAX_RETRIEVAL_CALLS)
    ] = CHAT_AGENT_DEFAULT_RETRIEVAL_CALLS


class ChatAgentTraceEventResponse(PublicSchema):
    tool: Literal[
        "search_knowledge_base",
        "search_graph_relations",
        "calculate",
        "submit_answer",
        "verifier",  # Historical traces only; current runs have no verifier.
        "protocol",
    ]
    status: Literal["ok", "rejected", "salvaged", "refused"]
    tool_call_id: Annotated[str, Field(min_length=1, max_length=128)]
    refs: tuple[Annotated[str, Field(min_length=1, max_length=128)], ...] = ()
    count: Annotated[int, Field(ge=0)] = 0
    retrieval_lane: Literal["simple", "graph_relations"] | None = None
    route_reason_code: Literal[
        "direct_relation",
        "relation_chain",
        "entity_alias",
        "cross_document_relation",
    ] | None = None
    route_result_code: Literal[
        "not_requested",
        "admitted",
        "no_evidence",
        "not_ready",
        "timeout",
        "unavailable",
        "rejected",
    ] | None = None
    new_evidence_count: Annotated[int, Field(ge=0, le=16)] | None = None
    call_index: Annotated[int, Field(ge=1, le=2)] | None = None
    invocation_source: Literal["agent", "legacy_guard"] | None = None
    duration_ms: Annotated[int, Field(ge=0)] | None = None
    candidate_count: Annotated[int, Field(ge=0)] | None = None
    path_count: Annotated[int, Field(ge=0)] | None = None
    hydrated_chunk_count: Annotated[int, Field(ge=0)] | None = None
    returned_chunk_count: Annotated[int, Field(ge=0)] | None = None
    hop1_count: Annotated[int, Field(ge=0)] | None = None
    hop2_count: Annotated[int, Field(ge=0)] | None = None
    hop3_count: Annotated[int, Field(ge=0)] | None = None


class ChatAgentTraceDiagnosticsResponse(PublicSchema):
    stop_reason: Literal[
        "submitted",
        "token_budget",
        "retrieval_query_budget",
        "evidence_budget",
        "no_new_evidence",
        "model_round_limit",
        "submit_protocol_invalid",
        "deadline_exceeded",
    ]
    forced_finalize: bool
    consecutive_no_new_evidence: Annotated[int, Field(ge=0)]
    elapsed_ms: Annotated[int, Field(ge=0)] | None = None
    deadline_ms: Annotated[int, Field(ge=0)] | None = None
    deadline_remaining_ms: Annotated[int, Field(ge=0)] | None = None
    deadline_exceeded: bool


class ChatAgentTraceResponse(PublicSchema):
    version: Literal["native_tool_calling_agent_v3"]
    events: tuple[ChatAgentTraceEventResponse, ...]
    budget: ChatAgentBudgetResponse
    usage: dict[str, Annotated[int, Field(ge=0)]]
    diagnostics: ChatAgentTraceDiagnosticsResponse | None = None
    outcome: Literal["answered", "partial", "refused", "clarify"]


class ChatAgentResponse(PublicSchema):
    version: Literal["native_tool_calling_agent_v3"]
    budget: ChatAgentBudgetResponse
    trace: ChatAgentTraceResponse | None = None


class ChatRunCreate(PublicSchema):
    session_id: UUID
    knowledge_base_id: UUID
    message: Annotated[str, Field(min_length=1, max_length=32768)]
    retrieval: ChatRetrievalRequest = ChatRetrievalRequest()
    model_profile_revision_id: UUID | None = None

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message must contain non-whitespace characters")
        return normalized


class ChatRunErrorResponse(PublicSchema):
    code: str
    detail: dict[str, Any]
    retryable: bool


class ChatRunRetrievalResponse(PublicSchema):
    profile_version: Annotated[str, Field(min_length=1, max_length=64)]
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
    effective_answer_policy: dict[str, Any]
    agent: ChatAgentResponse
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
    effective_answer_policy: dict[str, Any]
    status_url: str


class ChatRunFailedEvent(PublicSchema):
    run_id: UUID
    status: Literal["failed", "cancelled"]
    error: ChatRunErrorResponse
    effective_answer_policy: dict[str, Any]
    status_url: str


class ChatAgentProgressFacts(PublicSchema):
    objective: Annotated[str, Field(max_length=160)] | None
    queries: Annotated[tuple[Annotated[str, Field(max_length=160)], ...], Field(max_length=3)]
    evidence_count: Annotated[int, Field(ge=0, le=1000)] | None
    new_evidence_count: Annotated[int, Field(ge=0, le=1000)] | None
    retrieval_calls: Annotated[int, Field(ge=0, le=1000)] | None
    covered_aspects: Annotated[
        tuple[Annotated[str, Field(max_length=160)], ...], Field(max_length=6)
    ]
    missing_aspects: Annotated[
        tuple[Annotated[str, Field(max_length=160)], ...], Field(max_length=6)
    ]
    conflict_count: Annotated[int, Field(ge=0, le=1000)] | None


class ChatAgentProgressEvent(PublicSchema):
    run_id: UUID
    attempt: Annotated[int, Field(ge=1)]
    seq: Annotated[int, Field(ge=1)]
    active_stage: ChatProgressStageValue
    activity: Literal[
        "load_context",
        "tool_decision",
        "search_knowledge_base",
        "calculate",
        "submit_answer",
        "retrieval_complete",
        "prepare_visual_evidence",
        "generate_answer",
        "validate_answer",
        "persist_result",
    ]
    completed_stages: Annotated[
        tuple[ChatProgressStageValue, ...], Field(max_length=6)
    ]
    status: Literal["active", "completed"]
    facts: ChatAgentProgressFacts
