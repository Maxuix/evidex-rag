"""Public Graph configuration transport schemas."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from rag_kb.schemas.common import PublicSchema


class GraphConfigResponse(PublicSchema):
    knowledge_base_id: UUID
    enabled: bool
    status: Literal["disabled", "building", "ready", "failed"]
    build_id: UUID
    chat_profile_revision_id: UUID | None
    profile_name: str | None
    provider_name: str | None
    model: str | None
    extractor_version: str
    last_error_code: str | None
    eligible_chunk_count: Annotated[int, Field(ge=0)]
    processed_chunk_count: Annotated[int, Field(ge=0)]
    extracted_chunk_count: Annotated[int, Field(ge=0)]
    empty_chunk_count: Annotated[int, Field(ge=0)]
    protocol_skipped_count: Annotated[int, Field(ge=0)]
    resource_skipped_count: Annotated[int, Field(ge=0)]
    allowed_skipped_count: Annotated[int, Field(ge=0)]
    requires_rebuild: bool


class GraphConfigUpdate(PublicSchema):
    enabled: bool = True
    chat_profile_revision_id: UUID | None = None
    retry: bool = False
    force_rebuild: bool = False

    @model_validator(mode="after")
    def require_profile_when_enabling(self) -> "GraphConfigUpdate":
        if self.enabled and not self.retry and self.chat_profile_revision_id is None:
            raise ValueError(
                "chat_profile_revision_id is required when enabling Graph"
            )
        if self.retry and self.chat_profile_revision_id is not None:
            raise ValueError("retry must not include a Chat Profile Revision")
        return self
