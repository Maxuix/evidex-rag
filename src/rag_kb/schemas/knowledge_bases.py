"""Public knowledge-base API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Self, Union
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from rag_kb.domain import (
    ChunkingPreset,
    ParsingPreset,
    RerankMode,
)
from rag_kb.schemas.common import OpaqueCursor, PublicSchema


KnowledgeBaseName = Annotated[str, Field(min_length=1, max_length=255)]


class RetrievalDefaults(PublicSchema):
    strategy: Literal["exact_vector"] = "exact_vector"
    top_k: Annotated[int, Field(ge=1, le=100)] = 10
    rerank_mode: RerankMode = RerankMode.CLASSIC

    @model_validator(mode="after")
    def require_supported_rerank_combination(self) -> Self:
        if (
            self.rerank_mode is RerankMode.LOCAL_MINILM_V1
            and self.top_k > 20
        ):
            raise ValueError("local reranking supports top_k up to 20")
        return self


class KnowledgeBaseChunking(PublicSchema):
    preset: ChunkingPreset = ChunkingPreset.STRUCTURAL_BALANCED_V2


class KnowledgeBaseParsing(PublicSchema):
    preset: ParsingPreset = ParsingPreset.TEXT_LOCAL_V1


class KnowledgeBaseParsingResponse(PublicSchema):
    preset: ParsingPreset
    profile: Literal[
        "docling_text_local_v1",
        "docling_multimodal_local_v2",
        "docling_text_local_v2",
        "docling_multimodal_local_v3",
        "docling_text_local_v3",
        "docling_multimodal_local_v4",
        "docling_text_local_v4",
        "docling_multimodal_local_v5",
    ]


class KnowledgeBaseChunkingResponse(PublicSchema):
    preset: ChunkingPreset
    profile: Literal[
        "structural_by_title_token_v4",
        "structural_by_title_token_v5",
        "semantic_breakpoint_v3",
        "semantic_breakpoint_v4",
        "semantic_breakpoint_v5",
    ]


class TextOnlyEmbeddingSelection(PublicSchema):
    strategy: Literal["text_only"] = "text_only"
    text_profile_revision_id: UUID | None = None


class DualSpaceEmbeddingSelection(PublicSchema):
    strategy: Literal["dual_space"] = "dual_space"
    text_profile_revision_id: UUID | None = None
    multimodal_profile_revision_id: UUID | None = None


class UnifiedMultimodalEmbeddingSelection(PublicSchema):
    strategy: Literal["unified_multimodal"] = "unified_multimodal"
    profile_revision_id: UUID | None = None


KnowledgeBaseEmbeddingSelection = Annotated[
    Union[
        TextOnlyEmbeddingSelection,
        DualSpaceEmbeddingSelection,
        UnifiedMultimodalEmbeddingSelection,
    ],
    Field(discriminator="strategy"),
]


class KnowledgeBaseEmbeddingRoleResponse(PublicSchema):
    embedding_space_id: UUID
    profile_revision_id: UUID | None
    dimension: int


class KnowledgeBaseEmbeddingResponse(PublicSchema):
    strategy: Literal["text_only", "dual_space", "unified_multimodal"]
    text: KnowledgeBaseEmbeddingRoleResponse
    cross_modal: KnowledgeBaseEmbeddingRoleResponse | None = None


class KnowledgeBaseAutoQA(PublicSchema):
    enabled: bool = False
    model_profile_revision_id: UUID | None = None

    @model_validator(mode="after")
    def require_model_when_enabled(self) -> Self:
        if self.enabled and self.model_profile_revision_id is None:
            raise ValueError("auto_qa requires a chat model profile revision")
        if not self.enabled and self.model_profile_revision_id is not None:
            raise ValueError("auto_qa model is only allowed when enabled")
        return self


class KnowledgeBaseAutoQAResponse(PublicSchema):
    enabled: bool
    questions_per_chunk: int = 5
    model_profile_revision_id: UUID | None = None
    model_name: str | None = None
    model_revision: int | None = None


class KnowledgeBaseCreate(PublicSchema):
    name: KnowledgeBaseName
    parsing: KnowledgeBaseParsing = KnowledgeBaseParsing()
    chunking: KnowledgeBaseChunking = KnowledgeBaseChunking()
    retrieval_defaults: RetrievalDefaults = RetrievalDefaults()
    embedding: KnowledgeBaseEmbeddingSelection | None = None
    auto_qa: KnowledgeBaseAutoQA = KnowledgeBaseAutoQA()

    @model_validator(mode="after")
    def require_compatible_embedding_strategy(self) -> Self:
        if self.embedding is None:
            return self
        if self.parsing.preset is ParsingPreset.TEXT_LOCAL_V1:
            if self.embedding.strategy != "text_only":
                raise ValueError("text parsing requires text_only embedding")
        elif self.embedding.strategy == "text_only":
            raise ValueError("multimodal parsing requires dual or unified embedding")
        return self

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
        if self.name is None and self.retrieval_defaults is None:
            raise ValueError("at least one knowledge-base field must be supplied")
        return self


class KnowledgeBaseResponse(PublicSchema):
    id: UUID
    name: str
    source_change_seq: int
    active_index_revision_id: UUID
    embedding_space_id: UUID
    embedding: KnowledgeBaseEmbeddingResponse
    parsing: KnowledgeBaseParsingResponse
    chunking: KnowledgeBaseChunkingResponse
    retrieval_defaults: RetrievalDefaults
    answer_policy_defaults: dict[str, Any]
    auto_qa: KnowledgeBaseAutoQAResponse
    provisioned_at: datetime
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseDeleteResponse(PublicSchema):
    id: UUID
    name: str
    deleted_at: datetime


class KnowledgeBasePage(PublicSchema):
    items: tuple[KnowledgeBaseResponse, ...]
    next_cursor: OpaqueCursor | None = None
