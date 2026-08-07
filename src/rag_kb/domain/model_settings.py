"""User-managed local model-provider and model-profile facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
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
