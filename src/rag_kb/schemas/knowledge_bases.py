"""Public knowledge-base API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from rag_kb.domain import (
    AnswerStyle,
    ChunkingPreset,
    InsufficiencyPolicy,
    ParsingPreset,
)
from rag_kb.schemas.common import OpaqueCursor, PublicSchema


KnowledgeBaseName = Annotated[str, Field(min_length=1, max_length=255)]


class RetrievalDefaults(PublicSchema):
    strategy: Literal["exact_vector"] = "exact_vector"
    top_k: Annotated[int, Field(ge=1, le=100)] = 10
    rerank: bool = True


class KnowledgeBaseAnswerPolicyDefaults(PublicSchema):
    answer_style: AnswerStyle = AnswerStyle.CONCISE
    insufficiency_policy: InsufficiencyPolicy = InsufficiencyPolicy.REFUSE


class KnowledgeBaseChunking(PublicSchema):
    preset: ChunkingPreset = ChunkingPreset.STRUCTURAL_BALANCED_V2


class KnowledgeBaseParsing(PublicSchema):
    preset: ParsingPreset = ParsingPreset.TEXT_LOCAL_V1


class KnowledgeBaseParsingResponse(PublicSchema):
    preset: ParsingPreset
    profile: Literal[
        "docling_text_local_v1",
        "docling_multimodal_local_v2",
    ]


class KnowledgeBaseChunkingResponse(PublicSchema):
    preset: ChunkingPreset
    profile: Literal[
        "structural_by_title_token_v3",
        "semantic_breakpoint_v2",
    ]


class KnowledgeBaseCreate(PublicSchema):
    name: KnowledgeBaseName
    parsing: KnowledgeBaseParsing = KnowledgeBaseParsing()
    chunking: KnowledgeBaseChunking = KnowledgeBaseChunking()
    retrieval_defaults: RetrievalDefaults = RetrievalDefaults()
    answer_policy_defaults: KnowledgeBaseAnswerPolicyDefaults = (
        KnowledgeBaseAnswerPolicyDefaults()
    )

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("name must contain non-whitespace characters")
        return normalized


class KnowledgeBaseUpdate(PublicSchema):
    name: KnowledgeBaseName | None = None
    retrieval_defaults: RetrievalDefaults | None = None
    answer_policy_defaults: KnowledgeBaseAnswerPolicyDefaults | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("name must contain non-whitespace characters")
        return normalized

    @model_validator(mode="after")
    def require_change(self) -> Self:
        if (
            self.name is None
            and self.retrieval_defaults is None
            and self.answer_policy_defaults is None
        ):
            raise ValueError("at least one knowledge-base field must be supplied")
        return self


class KnowledgeBaseResponse(PublicSchema):
    id: UUID
    name: str
    source_change_seq: int
    active_index_revision_id: UUID
    embedding_space_id: UUID
    parsing: KnowledgeBaseParsingResponse
    chunking: KnowledgeBaseChunkingResponse
    retrieval_defaults: RetrievalDefaults
    answer_policy_defaults: KnowledgeBaseAnswerPolicyDefaults = (
        KnowledgeBaseAnswerPolicyDefaults()
    )
    provisioned_at: datetime
    created_at: datetime
    updated_at: datetime


class KnowledgeBasePage(PublicSchema):
    items: tuple[KnowledgeBaseResponse, ...]
    next_cursor: OpaqueCursor | None = None
