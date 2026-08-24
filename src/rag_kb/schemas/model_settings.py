"""Public DTOs for local user-managed model settings."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator

from rag_kb.domain import (
    EmbeddingDimensionRequestMode,
    EmbeddingDimensionSelectionSource,
    EmbeddingInputCapability,
    ModelKind,
    ModelProviderProtocol,
    ModelValidationStatus,
    ReasoningEffort,
)
from rag_kb.schemas.common import PublicSchema


DisplayName = Annotated[str, Field(min_length=1, max_length=255)]
ModelIdentifier = Annotated[str, Field(min_length=1, max_length=255)]


class ModelProviderCreate(PublicSchema):
    name: DisplayName
    protocol: ModelProviderProtocol
    base_url: AnyHttpUrl
    api_key: SecretStr
    timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 60.0
    max_retries: Annotated[int, Field(ge=0, le=10)] = 1
    max_concurrency: Annotated[int, Field(ge=1, le=32)] = 2

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _nonempty(value, "provider name")


class ModelProviderUpdate(PublicSchema):
    name: DisplayName | None = None
    protocol: ModelProviderProtocol | None = None
    base_url: AnyHttpUrl | None = None
    api_key: SecretStr | None = None
    timeout_seconds: Annotated[float, Field(gt=0, le=600)] | None = None
    max_retries: Annotated[int, Field(ge=0, le=10)] | None = None
    max_concurrency: Annotated[int, Field(ge=1, le=32)] | None = None
    enabled: bool | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        return _nonempty(value, "provider name") if value is not None else None

    @model_validator(mode="after")
    def require_change(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one provider field must be supplied")
        return self


class ChatModelParameters(PublicSchema):
    type: Literal["chat"] = "chat"
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.2
    top_p: Annotated[float, Field(gt=0, le=1)] | None = 0.9
    sampling_top_k: Annotated[int, Field(ge=1, le=1000)] | None = 40
    max_output_tokens: Annotated[int, Field(ge=1, le=8192)] = 4096
    reasoning_effort: ReasoningEffort = ReasoningEffort.OFF
    structured_output_mode: Literal["json_object", "json_schema"] = "json_object"
    vision_enabled: bool = False


class EmbeddingModelParameters(PublicSchema):
    type: Literal["embedding"] = "embedding"
    dimension: Literal["auto"] | Annotated[int, Field(ge=64, le=4096)] = "auto"
    max_batch_size: Annotated[int, Field(ge=1, le=100)] = 10
    shared_text_image_space_confirmed: bool = False


ModelParameters = ChatModelParameters | EmbeddingModelParameters


class ModelProfileCreate(PublicSchema):
    provider_id: UUID
    name: DisplayName
    kind: ModelKind
    model: ModelIdentifier
    parameters: ModelParameters

    @field_validator("name", "model")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _nonempty(value, "model field")

    @model_validator(mode="after")
    def require_kind_parameters(self) -> Self:
        if self.kind is ModelKind.CHAT and not isinstance(
            self.parameters, ChatModelParameters
        ):
            raise ValueError("chat models require chat parameters")
        if self.kind is not ModelKind.CHAT and not isinstance(
            self.parameters, EmbeddingModelParameters
        ):
            raise ValueError("embedding models require embedding parameters")
        if (
            self.kind is ModelKind.TEXT_EMBEDDING
            and isinstance(self.parameters, EmbeddingModelParameters)
            and self.parameters.shared_text_image_space_confirmed
        ):
            raise ValueError("text embedding models cannot confirm a shared image space")
        return self


class ModelProfileUpdate(PublicSchema):
    provider_id: UUID | None = None
    name: DisplayName | None = None
    model: ModelIdentifier | None = None
    parameters: ModelParameters | None = None
    enabled: bool | None = None

    @field_validator("name", "model")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _nonempty(value, "model field") if value is not None else None

    @model_validator(mode="after")
    def require_change(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one model field must be supplied")
        return self


class ModelProviderResponse(PublicSchema):
    id: UUID
    revision_id: UUID
    revision: int
    name: str
    protocol: ModelProviderProtocol
    base_url: str
    timeout_seconds: float
    max_retries: int
    max_concurrency: int
    enabled: bool
    api_key_configured: bool
    configuration_fingerprint: str
    created_at: datetime
    updated_at: datetime


class EmbeddingValidationResponse(PublicSchema):
    schema_version: Literal["embedding_validation_v1"]
    provider_supported_dimensions: tuple[int, ...] | None
    verified_dimensions: tuple[int, ...]
    provider_default_dimension: int | None
    recommended_dimension: int | None
    selected_dimension: Annotated[int, Field(ge=64, le=4096)]
    selection_source: EmbeddingDimensionSelectionSource
    dimension_request_mode: EmbeddingDimensionRequestMode
    input_capabilities: tuple[EmbeddingInputCapability, ...]
    shared_text_image_space_confirmed: bool
    distance_metric: Literal["cosine"]
    vector_data_type: Literal["float32"]
    normalization: Literal["l2", "client_l2_v1"]


class ModelProfileResponse(PublicSchema):
    id: UUID
    revision_id: UUID
    revision: int
    provider_id: UUID
    provider_revision_id: UUID
    name: str
    kind: ModelKind
    model: str
    parameters: ModelParameters
    enabled: bool
    provider_secret_available: bool
    validation_status: ModelValidationStatus
    validation_error_code: str | None
    validated_at: datetime | None
    configuration_fingerprint: str
    capability_fingerprint: str
    compatibility_fingerprint: str | None
    embedding_validation: EmbeddingValidationResponse | None = None
    created_at: datetime
    updated_at: datetime


class ModelSelectionUpdate(PublicSchema):
    chat_profile_revision_id: UUID | None = None
    text_embedding_profile_revision_id: UUID | None = None
    multimodal_embedding_profile_revision_id: UUID | None = None


class ModelSelectionResponse(PublicSchema):
    chat_profile_revision_id: UUID | None
    text_embedding_profile_revision_id: UUID | None
    multimodal_embedding_profile_revision_id: UUID | None
    updated_at: datetime


class ModelSettingsResponse(PublicSchema):
    providers: tuple[ModelProviderResponse, ...]
    profiles: tuple[ModelProfileResponse, ...]
    selection: ModelSelectionResponse


class ModelCatalogResponse(PublicSchema):
    models: tuple[str, ...]


def _nonempty(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must contain non-whitespace characters")
    return normalized
