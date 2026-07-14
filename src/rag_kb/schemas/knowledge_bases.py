"""Public knowledge-base API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from rag_kb.schemas.common import OpaqueCursor, PublicSchema


KnowledgeBaseName = Annotated[str, Field(min_length=1, max_length=255)]


class RetrievalDefaults(PublicSchema):
    strategy: Literal["exact_vector"] = "exact_vector"
    top_k: Annotated[int, Field(ge=1, le=100)] = 10


class KnowledgeBaseCreate(PublicSchema):
    name: KnowledgeBaseName
    retrieval_defaults: RetrievalDefaults = RetrievalDefaults()

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
    retrieval_defaults: RetrievalDefaults
    provisioned_at: datetime
    created_at: datetime
    updated_at: datetime


class KnowledgeBasePage(PublicSchema):
    items: tuple[KnowledgeBaseResponse, ...]
    next_cursor: OpaqueCursor | None = None
