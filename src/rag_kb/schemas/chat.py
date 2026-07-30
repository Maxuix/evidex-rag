"""Public chat session, history, and durable run DTOs."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from rag_kb.domain import AnswerStyle, InsufficiencyPolicy
from rag_kb.schemas.common import OpaqueCursor, PublicSchema


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
    rerank: bool | None = None


class ChatRunCreate(PublicSchema):
    session_id: UUID
    knowledge_base_id: UUID
    message: Annotated[str, Field(min_length=1, max_length=32768)]
    answer_policy: AnswerPolicyOverrides = AnswerPolicyOverrides()
    retrieval: ChatRetrievalRequest = ChatRetrievalRequest()

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
    profile_version: Literal["exact_vector_v1", "hybrid_fts_rrf_v1"]
    strategy: Literal["exact_vector", "hybrid"]
    top_k: Annotated[int, Field(ge=1, le=100)]
    rerank: bool
    dense_candidate_count: int
    lexical_candidate_count: int
    cross_modal_candidate_count: int
    lexical_analyzer_version: str | None
    lexical_query_version: str | None
    rrf_k: int
    dense_weight_micros: int
    lexical_weight_micros: int
    cross_modal_weight_micros: int
    min_cosine_similarity: float
    min_rerank_score: float
    cross_modal_min_cosine_similarity: float
    rerank_vector_weight: float
    rerank_lexical_weight: float
    mmr_lambda: float


class ChatRunQueryContextResponse(PublicSchema):
    strategy: Literal["recent_completed_turns_v1"]
    status: Literal["pending", "original", "contextualized"]
    history_turn_count: Annotated[int, Field(ge=0, le=6)]
    history_token_count: Annotated[int, Field(ge=0, le=4000)]
    history_truncated: bool
    standalone_query: str | None
    rewrite_source: Literal["original", "model", "repair", "fallback"] | None


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
    max_output_tokens: Annotated[int, Field(ge=1, le=2048)] | None = None
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
    retrieval: ChatRunRetrievalResponse
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
