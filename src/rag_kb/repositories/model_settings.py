"""Persistence contract for user-managed model settings."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain import (
    ModelKind,
    ModelProfile,
    ModelProfileBundle,
    ModelProfileRevision,
    ModelProvider,
    ModelProviderBundle,
    ModelProviderProtocol,
    ModelSelection,
    ModelValidationStatus,
    EmbeddingValidationSnapshot,
)


@runtime_checkable
class ModelSettingsRepository(Protocol):
    async def list_secret_references(self) -> tuple[str, ...]: ...

    async def list_providers(self) -> tuple[ModelProviderBundle, ...]: ...

    async def get_provider(self, provider_id: UUID) -> ModelProviderBundle | None: ...

    async def create_provider(
        self,
        *,
        name: str,
        protocol: ModelProviderProtocol,
        base_url: str,
        secret_reference: str,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        configuration_fingerprint: str,
    ) -> ModelProviderBundle: ...

    async def update_provider(
        self,
        provider_id: UUID,
        *,
        name: str | None,
        enabled: bool | None,
        protocol: ModelProviderProtocol | None,
        base_url: str | None,
        secret_reference: str | None,
        timeout_seconds: float | None,
        max_retries: int | None,
        max_concurrency: int | None,
        configuration_fingerprint: str | None,
    ) -> ModelProviderBundle | None: ...

    async def list_profiles(self) -> tuple[ModelProfileBundle, ...]: ...

    async def get_profile(self, profile_id: UUID) -> ModelProfileBundle | None: ...

    async def get_profile_revision(
        self, revision_id: UUID
    ) -> ModelProfileBundle | None: ...

    async def create_profile(
        self,
        *,
        provider: ModelProviderBundle,
        name: str,
        kind: ModelKind,
        model: str,
        configuration: dict[str, Any],
        configuration_fingerprint: str,
        capability_fingerprint: str,
        compatibility_fingerprint: str | None,
        validation_status: ModelValidationStatus,
    ) -> ModelProfileBundle: ...

    async def update_profile(
        self,
        profile_id: UUID,
        *,
        provider: ModelProviderBundle | None,
        name: str | None,
        enabled: bool | None,
        model: str | None,
        configuration: dict[str, Any] | None,
        configuration_fingerprint: str | None,
        capability_fingerprint: str | None,
        compatibility_fingerprint: str | None,
        validation_status: ModelValidationStatus | None,
    ) -> ModelProfileBundle | None: ...

    async def set_validation(
        self,
        revision_id: UUID,
        *,
        status: ModelValidationStatus,
        error_code: str | None,
        validation_snapshot: EmbeddingValidationSnapshot | None,
        capability_fingerprint: str | None,
        compatibility_fingerprint: str | None,
    ) -> ModelProfileBundle | None: ...

    async def get_selection(self) -> ModelSelection: ...

    async def update_selection(
        self,
        *,
        chat_profile_revision_id: UUID | None,
        text_embedding_profile_revision_id: UUID | None,
        multimodal_embedding_profile_revision_id: UUID | None,
    ) -> ModelSelection: ...
