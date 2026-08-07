"""User-managed local model-provider and model-profile facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import UUID


class ModelProviderProtocol(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"
    TONGYI_MULTIMODAL = "tongyi_multimodal"


class ModelKind(StrEnum):
    CHAT = "chat"
    TEXT_EMBEDDING = "text_embedding"
    MULTIMODAL_EMBEDDING = "multimodal_embedding"


class ModelValidationStatus(StrEnum):
    UNVERIFIED = "unverified"
    VALID = "valid"
    INVALID = "invalid"


MIN_EMBEDDING_DIMENSION = 64
MAX_EMBEDDING_DIMENSION = 4096
EMBEDDING_VALIDATION_SCHEMA_VERSION = "embedding_validation_v1"


class EmbeddingDimensionRequestMode(StrEnum):
    EXPLICIT = "explicit"
    OMITTED = "omitted"


class EmbeddingDimensionSelectionSource(StrEnum):
    PROVIDER_RECOMMENDED = "provider_recommended"
    PROVIDER_DEFAULT = "provider_default"
    AUTOMATIC_1024 = "automatic_1024"
    AUTOMATIC_ABOVE_1024 = "automatic_above_1024"
    AUTOMATIC_BELOW_1024 = "automatic_below_1024"
    PROVIDER_OBSERVED_DEFAULT = "provider_observed_default"
    USER_PROBE = "user_probe"
    LEGACY_EXPLICIT = "legacy_explicit"


class EmbeddingInputCapability(StrEnum):
    TEXT_DOCUMENT = "text_document"
    TEXT_QUERY = "text_query"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class EmbeddingValidationSnapshot:
    provider_supported_dimensions: tuple[int, ...] | None
    verified_dimensions: tuple[int, ...]
    provider_default_dimension: int | None
    recommended_dimension: int | None
    selected_dimension: int
    selection_source: EmbeddingDimensionSelectionSource
    dimension_request_mode: EmbeddingDimensionRequestMode
    input_capabilities: tuple[EmbeddingInputCapability, ...]
    shared_text_image_space_confirmed: bool
    distance_metric: str = "cosine"
    vector_data_type: str = "float32"
    normalization: str = "client_l2_v1"
    schema_version: str = EMBEDDING_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EMBEDDING_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported embedding validation snapshot")
        _require_embedding_dimension(self.selected_dimension)
        for value in self.verified_dimensions:
            _require_embedding_dimension(value)
        if self.selected_dimension not in self.verified_dimensions:
            raise ValueError("selected embedding dimension must be verified")
        if self.provider_supported_dimensions is not None:
            for value in self.provider_supported_dimensions:
                _require_embedding_dimension(value)
        for value in (self.provider_default_dimension, self.recommended_dimension):
            if value is not None:
                _require_embedding_dimension(value)
        if not self.input_capabilities:
            raise ValueError("embedding input capabilities must not be empty")
        if len(set(self.input_capabilities)) != len(self.input_capabilities):
            raise ValueError("embedding input capabilities must be unique")
        if self.distance_metric != "cosine":
            raise ValueError("only cosine embedding spaces are supported")
        if self.vector_data_type != "float32":
            raise ValueError("only float32 embedding spaces are supported")
        if self.normalization not in {"l2", "client_l2_v1"}:
            raise ValueError("unsupported embedding normalization")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_supported_dimensions": (
                list(self.provider_supported_dimensions)
                if self.provider_supported_dimensions is not None
                else None
            ),
            "verified_dimensions": list(self.verified_dimensions),
            "provider_default_dimension": self.provider_default_dimension,
            "recommended_dimension": self.recommended_dimension,
            "selected_dimension": self.selected_dimension,
            "selection_source": self.selection_source.value,
            "dimension_request_mode": self.dimension_request_mode.value,
            "input_capabilities": [value.value for value in self.input_capabilities],
            "shared_text_image_space_confirmed": (
                self.shared_text_image_space_confirmed
            ),
            "distance_metric": self.distance_metric,
            "vector_data_type": self.vector_data_type,
            "normalization": self.normalization,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EmbeddingValidationSnapshot:
        supported = value.get("provider_supported_dimensions")
        return cls(
            schema_version=str(value["schema_version"]),
            provider_supported_dimensions=(
                tuple(int(item) for item in supported)
                if isinstance(supported, list)
                else None
            ),
            verified_dimensions=tuple(
                int(item) for item in value["verified_dimensions"]
            ),
            provider_default_dimension=(
                int(value["provider_default_dimension"])
                if value.get("provider_default_dimension") is not None
                else None
            ),
            recommended_dimension=(
                int(value["recommended_dimension"])
                if value.get("recommended_dimension") is not None
                else None
            ),
            selected_dimension=int(value["selected_dimension"]),
            selection_source=EmbeddingDimensionSelectionSource(
                value["selection_source"]
            ),
            dimension_request_mode=EmbeddingDimensionRequestMode(
                value["dimension_request_mode"]
            ),
            input_capabilities=tuple(
                EmbeddingInputCapability(item)
                for item in value["input_capabilities"]
            ),
            shared_text_image_space_confirmed=bool(
                value["shared_text_image_space_confirmed"]
            ),
            distance_metric=str(value["distance_metric"]),
            vector_data_type=str(value["vector_data_type"]),
            normalization=str(value["normalization"]),
        )


def select_automatic_embedding_dimension(
    candidates: Sequence[int],
    *,
    provider_recommended_dimension: int | None = None,
    provider_default_dimension: int | None = None,
) -> tuple[int, EmbeddingDimensionSelectionSource]:
    allowed = tuple(sorted(set(candidates)))
    for value in allowed:
        _require_embedding_dimension(value)
    for value, source in (
        (
            provider_recommended_dimension,
            EmbeddingDimensionSelectionSource.PROVIDER_RECOMMENDED,
        ),
        (provider_default_dimension, EmbeddingDimensionSelectionSource.PROVIDER_DEFAULT),
    ):
        if value is not None and value in allowed:
            return value, source
    if 1024 in allowed:
        return 1024, EmbeddingDimensionSelectionSource.AUTOMATIC_1024
    above = tuple(value for value in allowed if value > 1024)
    if above:
        return min(above), EmbeddingDimensionSelectionSource.AUTOMATIC_ABOVE_1024
    below = tuple(value for value in allowed if value < 1024)
    if below:
        return max(below), EmbeddingDimensionSelectionSource.AUTOMATIC_BELOW_1024
    raise ValueError("embedding_dimension_required")


def _require_embedding_dimension(value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not MIN_EMBEDDING_DIMENSION <= value <= MAX_EMBEDDING_DIMENSION
    ):
        raise ValueError(
            f"embedding dimension must be between {MIN_EMBEDDING_DIMENSION} "
            f"and {MAX_EMBEDDING_DIMENSION}"
        )


class ReasoningEffort(StrEnum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ModelProvider:
    id: UUID
    workspace_id: UUID
    name: str
    enabled: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ModelProviderRevision:
    id: UUID
    workspace_id: UUID
    provider_id: UUID
    revision: int
    protocol: ModelProviderProtocol
    base_url: str
    secret_reference: str
    timeout_seconds: float
    max_retries: int
    max_concurrency: int
    configuration_fingerprint: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ModelProfile:
    id: UUID
    workspace_id: UUID
    provider_id: UUID
    name: str
    kind: ModelKind
    enabled: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ModelProfileRevision:
    id: UUID
    workspace_id: UUID
    profile_id: UUID
    provider_revision_id: UUID
    revision: int
    model: str
    configuration: Mapping[str, Any]
    configuration_fingerprint: str
    capability_fingerprint: str
    compatibility_fingerprint: str | None
    validation_status: ModelValidationStatus
    validation_error_code: str | None
    validation_snapshot: EmbeddingValidationSnapshot | None
    validated_at: datetime | None
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "configuration",
            MappingProxyType(dict(self.configuration)),
        )


@dataclass(frozen=True, slots=True)
class ModelSelection:
    workspace_id: UUID
    chat_profile_revision_id: UUID | None
    text_embedding_profile_revision_id: UUID | None
    multimodal_embedding_profile_revision_id: UUID | None
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ModelProviderBundle:
    provider: ModelProvider
    current_revision: ModelProviderRevision


@dataclass(frozen=True, slots=True)
class ModelProfileBundle:
    profile: ModelProfile
    current_revision: ModelProfileRevision
    provider: ModelProvider
    provider_revision: ModelProviderRevision
