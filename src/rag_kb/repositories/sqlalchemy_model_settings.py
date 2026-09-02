"""SQLAlchemy persistence for local model providers and profiles."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_kb.db.models import (
    ModelProfile as ModelProfileRow,
    ModelProfileRevision as ModelProfileRevisionRow,
    ModelProvider as ModelProviderRow,
    ModelProviderRevision as ModelProviderRevisionRow,
    ModelSelection as ModelSelectionRow,
)
from rag_kb.domain import (
    EmbeddingValidationSnapshot,
    ModelKind,
    ModelProfile,
    ModelProfileBundle,
    ModelProfileRevision,
    ModelProvider,
    ModelProviderBundle,
    ModelProviderProtocol,
    ModelProviderRevision,
    ModelSelection,
    ModelValidationStatus,
    ResourceNameConflictError,
)


class SqlAlchemyModelSettingsRepository:
    def __init__(
        self,
        session: AsyncSession,
        workspace_id: UUID,
    ) -> None:
        self._session = session
        self._workspace_id = workspace_id

    async def list_secret_references(self) -> tuple[str, ...]:
        values = await self._session.scalars(
            select(ModelProviderRevisionRow.secret_reference).where(
                ModelProviderRevisionRow.workspace_id == self._workspace_id
            )
        )
        return tuple(values)

    async def list_providers(self) -> tuple[ModelProviderBundle, ...]:
        rows = (
            await self._session.scalars(
                select(ModelProviderRow)
                .where(ModelProviderRow.workspace_id == self._workspace_id)
                .order_by(ModelProviderRow.name, ModelProviderRow.id)
            )
        ).all()
        return tuple([await self._provider_bundle(row) for row in rows])

    async def get_provider(self, provider_id: UUID) -> ModelProviderBundle | None:
        row = await self._session.scalar(
            select(ModelProviderRow).where(
                ModelProviderRow.workspace_id == self._workspace_id,
                ModelProviderRow.id == provider_id,
            )
        )
        return await self._provider_bundle(row) if row is not None else None

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
    ) -> ModelProviderBundle:
        duplicate = await self._session.scalar(
            select(ModelProviderRow.id).where(
                ModelProviderRow.workspace_id == self._workspace_id,
                ModelProviderRow.name == name,
            )
        )
        if duplicate is not None:
            raise ResourceNameConflictError("model-provider name already exists")
        row = ModelProviderRow(
            workspace_id=self._workspace_id,
            name=name,
            enabled=True,
        )
        self._session.add(row)
        await self._session.flush()
        revision = ModelProviderRevisionRow(
            workspace_id=self._workspace_id,
            provider_id=row.id,
            revision=1,
            protocol=protocol.value,
            base_url=base_url,
            secret_reference=secret_reference,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_concurrency=max_concurrency,
            configuration_fingerprint=configuration_fingerprint,
        )
        self._session.add(revision)
        await self._session.flush()
        return _provider_bundle(row, revision)

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
    ) -> ModelProviderBundle | None:
        row = await self._session.scalar(
            select(ModelProviderRow)
            .where(
                ModelProviderRow.workspace_id == self._workspace_id,
                ModelProviderRow.id == provider_id,
            )
            .with_for_update()
        )
        if row is None:
            return None
        current = await self._current_provider_revision(row.id)
        if name is not None and name != row.name:
            duplicate = await self._session.scalar(
                select(ModelProviderRow.id).where(
                    ModelProviderRow.workspace_id == self._workspace_id,
                    ModelProviderRow.name == name,
                    ModelProviderRow.id != provider_id,
                )
            )
            if duplicate is not None:
                raise ResourceNameConflictError("model-provider name already exists")
            row.name = name
        if enabled is not None:
            row.enabled = enabled
        if any(
            value is not None
            for value in (
                protocol,
                base_url,
                secret_reference,
                timeout_seconds,
                max_retries,
                max_concurrency,
                configuration_fingerprint,
            )
        ):
            current = ModelProviderRevisionRow(
                workspace_id=self._workspace_id,
                provider_id=row.id,
                revision=current.revision + 1,
                protocol=(protocol.value if protocol is not None else current.protocol),
                base_url=base_url if base_url is not None else current.base_url,
                secret_reference=(
                    secret_reference
                    if secret_reference is not None
                    else current.secret_reference
                ),
                timeout_seconds=(
                    timeout_seconds
                    if timeout_seconds is not None
                    else current.timeout_seconds
                ),
                max_retries=(
                    max_retries if max_retries is not None else current.max_retries
                ),
                max_concurrency=(
                    max_concurrency
                    if max_concurrency is not None
                    else current.max_concurrency
                ),
                configuration_fingerprint=(
                    configuration_fingerprint
                    if configuration_fingerprint is not None
                    else current.configuration_fingerprint
                ),
            )
            self._session.add(current)
        row.updated_at = datetime.now(UTC)
        await self._session.flush()
        return _provider_bundle(row, current)

    async def list_profiles(self) -> tuple[ModelProfileBundle, ...]:
        rows = (
            await self._session.scalars(
                select(ModelProfileRow)
                .where(ModelProfileRow.workspace_id == self._workspace_id)
                .order_by(ModelProfileRow.name, ModelProfileRow.id)
            )
        ).all()
        return tuple([await self._profile_bundle(row) for row in rows])

    async def get_profile(self, profile_id: UUID) -> ModelProfileBundle | None:
        row = await self._session.scalar(
            select(ModelProfileRow).where(
                ModelProfileRow.workspace_id == self._workspace_id,
                ModelProfileRow.id == profile_id,
            )
        )
        return await self._profile_bundle(row) if row is not None else None

    async def get_profile_revision(
        self, revision_id: UUID
    ) -> ModelProfileBundle | None:
        revision = await self._session.scalar(
            select(ModelProfileRevisionRow).where(
                ModelProfileRevisionRow.workspace_id == self._workspace_id,
                ModelProfileRevisionRow.id == revision_id,
            )
        )
        if revision is None:
            return None
        profile = await self._session.scalar(
            select(ModelProfileRow).where(
                ModelProfileRow.workspace_id == self._workspace_id,
                ModelProfileRow.id == revision.profile_id,
            )
        )
        assert profile is not None
        return await self._profile_bundle(profile, revision=revision)

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
    ) -> ModelProfileBundle:
        duplicate = await self._session.scalar(
            select(ModelProfileRow.id).where(
                ModelProfileRow.workspace_id == self._workspace_id,
                ModelProfileRow.name == name,
            )
        )
        if duplicate is not None:
            raise ResourceNameConflictError("model-profile name already exists")
        row = ModelProfileRow(
            workspace_id=self._workspace_id,
            provider_id=provider.provider.id,
            name=name,
            kind=kind.value,
            enabled=True,
        )
        self._session.add(row)
        await self._session.flush()
        revision = ModelProfileRevisionRow(
            workspace_id=self._workspace_id,
            profile_id=row.id,
            provider_revision_id=provider.current_revision.id,
            revision=1,
            model=model,
            configuration=configuration,
            configuration_fingerprint=configuration_fingerprint,
            capability_fingerprint=capability_fingerprint,
            compatibility_fingerprint=compatibility_fingerprint,
            validation_status=validation_status.value,
            validation_error_code=None,
            validation_snapshot=None,
            validated_at=None,
        )
        self._session.add(revision)
        await self._session.flush()
        return await self._profile_bundle(row, revision=revision)

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
    ) -> ModelProfileBundle | None:
        row = await self._session.scalar(
            select(ModelProfileRow)
            .where(
                ModelProfileRow.workspace_id == self._workspace_id,
                ModelProfileRow.id == profile_id,
            )
            .with_for_update()
        )
        if row is None:
            return None
        current = await self._current_profile_revision(row.id)
        if name is not None and name != row.name:
            duplicate = await self._session.scalar(
                select(ModelProfileRow.id).where(
                    ModelProfileRow.workspace_id == self._workspace_id,
                    ModelProfileRow.name == name,
                    ModelProfileRow.id != profile_id,
                )
            )
            if duplicate is not None:
                raise ResourceNameConflictError("model-profile name already exists")
            row.name = name
        if enabled is not None:
            row.enabled = enabled
        revision_values = (
            provider,
            model,
            configuration,
            configuration_fingerprint,
            capability_fingerprint,
            validation_status,
        )
        if any(value is not None for value in revision_values):
            provider_revision_id = (
                provider.current_revision.id
                if provider is not None
                else current.provider_revision_id
            )
            current = ModelProfileRevisionRow(
                workspace_id=self._workspace_id,
                profile_id=row.id,
                provider_revision_id=provider_revision_id,
                revision=current.revision + 1,
                model=model if model is not None else current.model,
                configuration=(
                    configuration if configuration is not None else current.configuration
                ),
                configuration_fingerprint=(
                    configuration_fingerprint
                    if configuration_fingerprint is not None
                    else current.configuration_fingerprint
                ),
                capability_fingerprint=(
                    capability_fingerprint
                    if capability_fingerprint is not None
                    else current.capability_fingerprint
                ),
                compatibility_fingerprint=compatibility_fingerprint,
                validation_status=(
                    validation_status.value
                    if validation_status is not None
                    else ModelValidationStatus.UNVERIFIED.value
                ),
                validation_error_code=None,
                validation_snapshot=None,
                validated_at=None,
            )
            self._session.add(current)
        row.updated_at = datetime.now(UTC)
        await self._session.flush()
        return await self._profile_bundle(row, revision=current)

    async def set_validation(
        self,
        revision_id: UUID,
        *,
        status: ModelValidationStatus,
        error_code: str | None,
        validation_snapshot: EmbeddingValidationSnapshot | None,
        capability_fingerprint: str | None,
        compatibility_fingerprint: str | None,
    ) -> ModelProfileBundle | None:
        revision = await self._session.scalar(
            select(ModelProfileRevisionRow)
            .where(
                ModelProfileRevisionRow.workspace_id == self._workspace_id,
                ModelProfileRevisionRow.id == revision_id,
            )
            .with_for_update()
        )
        if revision is None:
            return None
        revision.validation_status = status.value
        revision.validation_error_code = error_code
        revision.validation_snapshot = (
            validation_snapshot.as_dict() if validation_snapshot is not None else None
        )
        if capability_fingerprint is not None:
            revision.capability_fingerprint = capability_fingerprint
        revision.compatibility_fingerprint = compatibility_fingerprint
        revision.validated_at = datetime.now(UTC)
        await self._session.flush()
        profile = await self._session.scalar(
            select(ModelProfileRow).where(
                ModelProfileRow.workspace_id == self._workspace_id,
                ModelProfileRow.id == revision.profile_id,
            )
        )
        assert profile is not None
        return await self._profile_bundle(profile, revision=revision)

    async def get_selection(self) -> ModelSelection:
        row = await self._session.get(ModelSelectionRow, self._workspace_id)
        if row is None:
            return ModelSelection(
                workspace_id=self._workspace_id,
                chat_profile_revision_id=None,
                text_embedding_profile_revision_id=None,
                multimodal_embedding_profile_revision_id=None,
                updated_at=datetime.now(UTC),
            )
        return _selection(row)

    async def update_selection(
        self,
        *,
        chat_profile_revision_id: UUID | None,
        text_embedding_profile_revision_id: UUID | None,
        multimodal_embedding_profile_revision_id: UUID | None,
    ) -> ModelSelection:
        row = await self._session.scalar(
            select(ModelSelectionRow)
            .where(ModelSelectionRow.workspace_id == self._workspace_id)
            .with_for_update()
        )
        if row is None:
            row = ModelSelectionRow(workspace_id=self._workspace_id)
            self._session.add(row)
        row.chat_profile_revision_id = chat_profile_revision_id
        row.text_embedding_profile_revision_id = text_embedding_profile_revision_id
        row.multimodal_embedding_profile_revision_id = (
            multimodal_embedding_profile_revision_id
        )
        row.updated_at = datetime.now(UTC)
        await self._session.flush()
        return _selection(row)

    async def _provider_bundle(
        self, row: ModelProviderRow
    ) -> ModelProviderBundle:
        return _provider_bundle(row, await self._current_provider_revision(row.id))

    async def _profile_bundle(
        self,
        row: ModelProfileRow,
        *,
        revision: ModelProfileRevisionRow | None = None,
    ) -> ModelProfileBundle:
        current = revision or await self._current_profile_revision(row.id)
        provider_row = await self._session.scalar(
            select(ModelProviderRow).where(
                ModelProviderRow.workspace_id == self._workspace_id,
                ModelProviderRow.id == row.provider_id,
            )
        )
        provider_revision = await self._session.scalar(
            select(ModelProviderRevisionRow).where(
                ModelProviderRevisionRow.workspace_id == self._workspace_id,
                ModelProviderRevisionRow.id == current.provider_revision_id,
            )
        )
        assert provider_row is not None and provider_revision is not None
        return ModelProfileBundle(
            profile=_profile(row),
            current_revision=_profile_revision(current),
            provider=_provider(provider_row),
            provider_revision=_provider_revision(provider_revision),
        )

    async def _current_provider_revision(
        self, provider_id: UUID
    ) -> ModelProviderRevisionRow:
        row = await self._session.scalar(
            select(ModelProviderRevisionRow)
            .where(
                ModelProviderRevisionRow.workspace_id == self._workspace_id,
                ModelProviderRevisionRow.provider_id == provider_id,
            )
            .order_by(ModelProviderRevisionRow.revision.desc())
            .limit(1)
        )
        assert row is not None
        return row

    async def _current_profile_revision(
        self, profile_id: UUID
    ) -> ModelProfileRevisionRow:
        row = await self._session.scalar(
            select(ModelProfileRevisionRow)
            .where(
                ModelProfileRevisionRow.workspace_id == self._workspace_id,
                ModelProfileRevisionRow.profile_id == profile_id,
            )
            .order_by(ModelProfileRevisionRow.revision.desc())
            .limit(1)
        )
        assert row is not None
        return row


def _provider_bundle(
    provider: ModelProviderRow,
    revision: ModelProviderRevisionRow,
) -> ModelProviderBundle:
    return ModelProviderBundle(_provider(provider), _provider_revision(revision))


def _provider(row: ModelProviderRow) -> ModelProvider:
    return ModelProvider(
        id=row.id,
        workspace_id=row.workspace_id,
        name=row.name,
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _provider_revision(row: ModelProviderRevisionRow) -> ModelProviderRevision:
    return ModelProviderRevision(
        id=row.id,
        workspace_id=row.workspace_id,
        provider_id=row.provider_id,
        revision=row.revision,
        protocol=ModelProviderProtocol(row.protocol),
        base_url=row.base_url,
        secret_reference=row.secret_reference,
        timeout_seconds=row.timeout_seconds,
        max_retries=row.max_retries,
        max_concurrency=row.max_concurrency,
        configuration_fingerprint=row.configuration_fingerprint,
        created_at=row.created_at,
    )


def _profile(row: ModelProfileRow) -> ModelProfile:
    return ModelProfile(
        id=row.id,
        workspace_id=row.workspace_id,
        provider_id=row.provider_id,
        name=row.name,
        kind=ModelKind(row.kind),
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _profile_revision(row: ModelProfileRevisionRow) -> ModelProfileRevision:
    return ModelProfileRevision(
        id=row.id,
        workspace_id=row.workspace_id,
        profile_id=row.profile_id,
        provider_revision_id=row.provider_revision_id,
        revision=row.revision,
        model=row.model,
        configuration=dict(row.configuration),
        configuration_fingerprint=row.configuration_fingerprint,
        capability_fingerprint=row.capability_fingerprint,
        compatibility_fingerprint=row.compatibility_fingerprint,
        validation_status=ModelValidationStatus(row.validation_status),
        validation_error_code=row.validation_error_code,
        validation_snapshot=(
            EmbeddingValidationSnapshot.from_mapping(row.validation_snapshot)
            if row.validation_snapshot is not None
            else None
        ),
        validated_at=row.validated_at,
        created_at=row.created_at,
    )


def _selection(row: ModelSelectionRow) -> ModelSelection:
    return ModelSelection(
        workspace_id=row.workspace_id,
        chat_profile_revision_id=row.chat_profile_revision_id,
        text_embedding_profile_revision_id=row.text_embedding_profile_revision_id,
        multimodal_embedding_profile_revision_id=(
            row.multimodal_embedding_profile_revision_id
        ),
        updated_at=row.updated_at,
    )
