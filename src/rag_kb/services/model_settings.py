"""Application service for local model providers, profiles, and defaults."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from rag_kb.domain import (
    EmbeddingInputCapability,
    EmbeddingValidationSnapshot,
    MAX_EMBEDDING_DIMENSION,
    TEXT_DOCUMENT_TRANSFORMATION_VERSION,
    TEXT_QUERY_TRANSFORMATION_VERSION,
    MIN_EMBEDDING_DIMENSION,
    ModelKind,
    ModelProfileBundle,
    ModelProviderBundle,
    ModelProviderProtocol,
    ModelSelection,
    ModelValidationStatus,
    ResourceNotFoundError,
    ResourceStateConflictError,
)
from rag_kb.ports.model_secrets import ModelSecretStore
from rag_kb.uow import execute_in_transaction

if TYPE_CHECKING:
    from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWork, SqlAlchemyUnitOfWorkFactory


ModelProfileValidator = Callable[
    [ModelProfileBundle, str], Awaitable[EmbeddingValidationSnapshot | None]
]
ModelProviderCatalog = Callable[
    [ModelProviderBundle, str], Awaitable[tuple[str, ...]]
]


LOGGER = logging.getLogger("rag_kb.model_settings.secrets")


@dataclass(frozen=True, slots=True)
class ModelSettingsSnapshot:
    providers: tuple[ModelProviderBundle, ...]
    profiles: tuple[ModelProfileBundle, ...]
    selection: ModelSelection
    provider_secret_health: dict[UUID, bool]
    profile_secret_health: dict[UUID, bool]


class ModelSettingsService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWorkFactory,
        secret_store: ModelSecretStore,
        *,
        profile_validator: ModelProfileValidator | None = None,
        provider_catalog: ModelProviderCatalog | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._secret_store = secret_store
        self._profile_validator = profile_validator
        self._provider_catalog = provider_catalog

    async def snapshot(self) -> ModelSettingsSnapshot:
        async def load(uow: SqlAlchemyUnitOfWork):
            providers = await uow.model_settings.list_providers()
            profiles = await uow.model_settings.list_profiles()
            selection = await uow.model_settings.get_selection()
            return providers, profiles, selection

        providers, profiles, selection = await execute_in_transaction(
            self._unit_of_work, load
        )
        health: dict[UUID, bool] = {}
        for provider in providers:
            health[provider.current_revision.id] = await self._secret_available(
                provider.current_revision.secret_reference
            )
        for profile in profiles:
            health[profile.provider_revision.id] = await self._secret_available(
                profile.provider_revision.secret_reference
            )
        return ModelSettingsSnapshot(
            providers=providers,
            profiles=profiles,
            selection=selection,
            provider_secret_health={
                provider.current_revision.id: health[provider.current_revision.id]
                for provider in providers
            },
            profile_secret_health={
                profile.current_revision.id: health[profile.provider_revision.id]
                for profile in profiles
            },
        )

    async def provider_secret_available(
        self, value: ModelProviderBundle
    ) -> bool:
        return await self._secret_available(value.current_revision.secret_reference)

    async def profile_secret_available(
        self, value: ModelProfileBundle
    ) -> bool:
        return await self._secret_available(value.provider_revision.secret_reference)

    async def _secret_available(self, reference: str) -> bool:
        try:
            value = await asyncio.to_thread(self._secret_store.read, reference)
            return bool(value.strip())
        except (FileNotFoundError, PermissionError, OSError, UnicodeError, ValueError):
            LOGGER.warning(
                "model_secret_unavailable reason_code=%s",
                "SECRET_NOT_READABLE",
            )
            return False

    async def create_provider(
        self,
        *,
        name: str,
        protocol: ModelProviderProtocol,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
    ) -> ModelProviderBundle:
        secret_reference = await asyncio.to_thread(self._secret_store.write, api_key)
        fingerprint = provider_fingerprint(
            protocol=protocol,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_concurrency=max_concurrency,
        )

        async def persist(uow: SqlAlchemyUnitOfWork) -> ModelProviderBundle:
            return await uow.model_settings.create_provider(
                name=name,
                protocol=protocol,
                base_url=_normalize_url(base_url),
                secret_reference=secret_reference,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                max_concurrency=max_concurrency,
                configuration_fingerprint=fingerprint,
            )

        try:
            return await execute_in_transaction(self._unit_of_work, persist)
        except BaseException:
            await asyncio.to_thread(self._secret_store.delete, secret_reference)
            raise

    async def update_provider(
        self,
        provider_id: UUID,
        *,
        name: str | None,
        protocol: ModelProviderProtocol | None,
        base_url: str | None,
        api_key: str | None,
        timeout_seconds: float | None,
        max_retries: int | None,
        max_concurrency: int | None,
        enabled: bool | None,
    ) -> ModelProviderBundle:
        secret_reference = (
            await asyncio.to_thread(self._secret_store.write, api_key)
            if api_key is not None
            else None
        )

        async def persist(uow: SqlAlchemyUnitOfWork) -> ModelProviderBundle:
            current = await uow.model_settings.get_provider(provider_id)
            if current is None:
                raise ResourceNotFoundError("model provider was not found")
            revision = current.current_revision
            resolved_protocol = protocol or revision.protocol
            resolved_base_url = base_url or revision.base_url
            resolved_timeout = timeout_seconds or revision.timeout_seconds
            resolved_retries = (
                max_retries if max_retries is not None else revision.max_retries
            )
            resolved_concurrency = max_concurrency or revision.max_concurrency
            revision_changed = any(
                value is not None
                for value in (
                    protocol,
                    base_url,
                    api_key,
                    timeout_seconds,
                    max_retries,
                    max_concurrency,
                )
            )
            fingerprint = (
                provider_fingerprint(
                    protocol=resolved_protocol,
                    base_url=resolved_base_url,
                    timeout_seconds=resolved_timeout,
                    max_retries=resolved_retries,
                    max_concurrency=resolved_concurrency,
                )
                if revision_changed
                else None
            )
            updated = await uow.model_settings.update_provider(
                provider_id,
                name=name,
                enabled=enabled,
                protocol=resolved_protocol if revision_changed else None,
                base_url=(
                    _normalize_url(resolved_base_url) if revision_changed else None
                ),
                secret_reference=secret_reference,
                timeout_seconds=resolved_timeout if revision_changed else None,
                max_retries=resolved_retries if revision_changed else None,
                max_concurrency=(resolved_concurrency if revision_changed else None),
                configuration_fingerprint=fingerprint,
            )
            assert updated is not None
            return updated

        try:
            return await execute_in_transaction(self._unit_of_work, persist)
        except BaseException:
            if secret_reference is not None:
                await asyncio.to_thread(self._secret_store.delete, secret_reference)
            raise

    async def create_profile(
        self,
        *,
        provider_id: UUID,
        name: str,
        kind: ModelKind,
        model: str,
        parameters: dict[str, Any],
    ) -> ModelProfileBundle:
        configuration = dict(parameters)
        _require_parameters(kind, configuration)

        async def persist(uow: SqlAlchemyUnitOfWork) -> ModelProfileBundle:
            provider = await uow.model_settings.get_provider(provider_id)
            if provider is None:
                raise ResourceNotFoundError("model provider was not found")
            if not provider.provider.enabled:
                raise ResourceStateConflictError("model provider is disabled")
            _require_protocol(provider.current_revision.protocol, kind)
            fingerprints = model_fingerprints(
                kind,
                model,
                configuration,
                provider_configuration_fingerprint=(
                    provider.current_revision.configuration_fingerprint
                ),
            )
            return await uow.model_settings.create_profile(
                provider=provider,
                name=name,
                kind=kind,
                model=model,
                configuration=configuration,
                configuration_fingerprint=fingerprints[0],
                capability_fingerprint=fingerprints[1],
                compatibility_fingerprint=fingerprints[2],
                validation_status=ModelValidationStatus.UNVERIFIED,
            )

        return await execute_in_transaction(self._unit_of_work, persist)

    async def update_profile(
        self,
        profile_id: UUID,
        *,
        provider_id: UUID | None,
        name: str | None,
        model: str | None,
        parameters: dict[str, Any] | None,
        enabled: bool | None,
    ) -> ModelProfileBundle:
        async def persist(uow: SqlAlchemyUnitOfWork) -> ModelProfileBundle:
            current = await uow.model_settings.get_profile(profile_id)
            if current is None:
                raise ResourceNotFoundError("model profile was not found")
            provider = (
                await uow.model_settings.get_provider(provider_id)
                if provider_id is not None
                else None
            )
            if provider_id is not None and provider is None:
                raise ResourceNotFoundError("model provider was not found")
            resolved_provider = provider or ModelProviderBundle(
                current.provider,
                current.provider_revision,
            )
            if not resolved_provider.provider.enabled:
                raise ResourceStateConflictError("model provider is disabled")
            _require_protocol(resolved_provider.current_revision.protocol, current.profile.kind)
            if parameters is not None:
                _require_parameters(current.profile.kind, parameters)
            resolved_model = model or current.current_revision.model
            configuration = (
                dict(parameters)
                if parameters is not None
                else dict(current.current_revision.configuration)
            )
            revision_changed = provider is not None or model is not None or parameters is not None
            fingerprints = (
                model_fingerprints(
                    current.profile.kind,
                    resolved_model,
                    configuration,
                    provider_configuration_fingerprint=(
                        resolved_provider.current_revision.configuration_fingerprint
                    ),
                )
                if revision_changed
                else (None, None, None)
            )
            updated = await uow.model_settings.update_profile(
                profile_id,
                provider=provider,
                name=name,
                enabled=enabled,
                model=resolved_model if revision_changed else None,
                configuration=configuration if revision_changed else None,
                configuration_fingerprint=fingerprints[0],
                capability_fingerprint=fingerprints[1],
                compatibility_fingerprint=fingerprints[2],
                validation_status=(
                    ModelValidationStatus.UNVERIFIED if revision_changed else None
                ),
            )
            assert updated is not None
            return updated

        return await execute_in_transaction(self._unit_of_work, persist)

    async def update_selection(
        self,
        *,
        chat_profile_revision_id: UUID | None,
        text_embedding_profile_revision_id: UUID | None,
        multimodal_embedding_profile_revision_id: UUID | None,
    ) -> ModelSelection:
        async def persist(uow: SqlAlchemyUnitOfWork) -> ModelSelection:
            for revision_id, kind in (
                (chat_profile_revision_id, ModelKind.CHAT),
                (text_embedding_profile_revision_id, ModelKind.TEXT_EMBEDDING),
                (
                    multimodal_embedding_profile_revision_id,
                    ModelKind.MULTIMODAL_EMBEDDING,
                ),
            ):
                if revision_id is None:
                    continue
                bundle = await uow.model_settings.get_profile_revision(revision_id)
                if bundle is None:
                    raise ResourceNotFoundError("model profile revision was not found")
                if bundle.profile.kind is not kind:
                    raise ResourceStateConflictError("model profile kind is incompatible")
                if not bundle.profile.enabled or not bundle.provider.enabled:
                    raise ResourceStateConflictError("model profile is disabled")
                if (
                    bundle.current_revision.validation_status
                    is not ModelValidationStatus.VALID
                ):
                    raise ResourceStateConflictError("model profile is not validated")
            return await uow.model_settings.update_selection(
                chat_profile_revision_id=chat_profile_revision_id,
                text_embedding_profile_revision_id=(
                    text_embedding_profile_revision_id
                ),
                multimodal_embedding_profile_revision_id=(
                    multimodal_embedding_profile_revision_id
                ),
            )

        return await execute_in_transaction(self._unit_of_work, persist)

    async def list_provider_models(
        self,
        provider_id: UUID,
    ) -> tuple[str, ...]:
        if self._provider_catalog is None:
            raise ResourceStateConflictError("provider model discovery is unavailable")

        async def load(uow: SqlAlchemyUnitOfWork) -> ModelProviderBundle:
            provider = await uow.model_settings.get_provider(provider_id)
            if provider is None:
                raise ResourceNotFoundError("model provider was not found")
            if not provider.provider.enabled:
                raise ResourceStateConflictError("model provider is disabled")
            if (
                provider.current_revision.protocol
                is not ModelProviderProtocol.OPENAI_COMPATIBLE
            ):
                raise ResourceStateConflictError(
                    "model discovery is unsupported for this provider"
                )
            return provider

        provider = await execute_in_transaction(
            self._unit_of_work,
            load,
        )
        try:
            api_key = await asyncio.to_thread(
                self._secret_store.read,
                provider.current_revision.secret_reference,
            )
            return await self._provider_catalog(provider, api_key)
        except Exception as error:
            raise ResourceStateConflictError(
                "provider model catalog could not be loaded"
            ) from error

    async def validate_profile(
        self,
        profile_id: UUID,
    ) -> ModelProfileBundle:
        if self._profile_validator is None:
            raise ResourceStateConflictError("model validation is unavailable")

        async def load(uow: SqlAlchemyUnitOfWork) -> ModelProfileBundle:
            bundle = await uow.model_settings.get_profile(profile_id)
            if bundle is None:
                raise ResourceNotFoundError("model profile was not found")
            if not bundle.profile.enabled or not bundle.provider.enabled:
                raise ResourceStateConflictError("model profile is disabled")
            return bundle

        bundle = await execute_in_transaction(
            self._unit_of_work,
            load,
        )
        snapshot: EmbeddingValidationSnapshot | None = None
        try:
            api_key = await asyncio.to_thread(
                self._secret_store.read,
                bundle.provider_revision.secret_reference,
            )
            snapshot = await self._profile_validator(bundle, api_key)
            if bundle.profile.kind is ModelKind.CHAT and snapshot is not None:
                raise ModelProfileValidationError("provider_validation_failed")
            if bundle.profile.kind is not ModelKind.CHAT and snapshot is None:
                raise ModelProfileValidationError("embedding_response_invalid")
            status = ModelValidationStatus.VALID
            error_code = None
        except ModelProfileValidationError as error:
            if (
                bundle.current_revision.validation_status
                is ModelValidationStatus.VALID
            ):
                raise ResourceStateConflictError(
                    "validated model revision was preserved after revalidation conflict"
                ) from error
            status = ModelValidationStatus.INVALID
            error_code = error.error_code
        except Exception as error:
            if (
                bundle.current_revision.validation_status
                is ModelValidationStatus.VALID
            ):
                raise ResourceStateConflictError(
                    "validated model revision was preserved after revalidation failure"
                ) from error
            status = ModelValidationStatus.INVALID
            error_code = "provider_validation_failed"

        capability_fingerprint: str | None = None
        compatibility_fingerprint: str | None = None
        if snapshot is not None:
            capability_fingerprint = embedding_capability_fingerprint(snapshot)
            compatibility_fingerprint = embedding_compatibility_fingerprint(
                bundle, snapshot
            )
            frozen = bundle.current_revision
            if frozen.validation_status is ModelValidationStatus.VALID:
                if (
                    frozen.validation_snapshot != snapshot
                    or frozen.capability_fingerprint != capability_fingerprint
                    or frozen.compatibility_fingerprint != compatibility_fingerprint
                ):
                    raise ResourceStateConflictError(
                        "model validation facts changed; create a new revision"
                    )

        async def persist(uow: SqlAlchemyUnitOfWork) -> ModelProfileBundle:
            updated = await uow.model_settings.set_validation(
                bundle.current_revision.id,
                status=status,
                error_code=error_code,
                validation_snapshot=snapshot,
                capability_fingerprint=capability_fingerprint,
                compatibility_fingerprint=compatibility_fingerprint,
            )
            if updated is None:
                raise ResourceNotFoundError("model profile revision was not found")
            return updated

        return await execute_in_transaction(self._unit_of_work, persist)

def provider_fingerprint(
    *,
    protocol: ModelProviderProtocol,
    base_url: str,
    timeout_seconds: float,
    max_retries: int,
    max_concurrency: int,
) -> str:
    return _fingerprint(
        {
            "protocol": protocol.value,
            "base_url": _normalize_url(base_url),
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "max_concurrency": max_concurrency,
        }
    )


def model_fingerprints(
    kind: ModelKind,
    model: str,
    configuration: dict[str, Any],
    *,
    provider_configuration_fingerprint: str | None = None,
) -> tuple[str, str, str | None]:
    configuration_fingerprint = _fingerprint(
        {"kind": kind.value, "model": model, "configuration": configuration}
    )
    capability_fingerprint = _fingerprint(
        {
            "kind": kind.value,
            "validation": "unverified" if kind is not ModelKind.CHAT else None,
            "structured_output_mode": configuration.get("structured_output_mode"),
            "vision_enabled": configuration.get("vision_enabled", False),
            "reasoning": configuration.get("reasoning_effort", "off") != "off",
            "sampling_top_k": configuration.get("sampling_top_k") is not None,
        }
    )
    return (
        configuration_fingerprint,
        capability_fingerprint,
        None,
    )


class ModelProfileValidationError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def embedding_capability_fingerprint(
    snapshot: EmbeddingValidationSnapshot,
) -> str:
    return _fingerprint(
        {
            "schema": snapshot.schema_version,
            "provider_supported_dimensions": snapshot.provider_supported_dimensions,
            "verified_dimensions": snapshot.verified_dimensions,
            "selected_dimension": snapshot.selected_dimension,
            "input_capabilities": snapshot.input_capabilities,
            "shared_text_image_space_confirmed": (
                snapshot.shared_text_image_space_confirmed
            ),
        }
    )


def embedding_compatibility_fingerprint(
    bundle: ModelProfileBundle,
    snapshot: EmbeddingValidationSnapshot,
) -> str:
    image_preprocessing_version = (
        "tongyi_data_url_res1_v1"
        if EmbeddingInputCapability.IMAGE in snapshot.input_capabilities
        else None
    )
    return _fingerprint(
        {
            "fingerprint_schema": "embedding_space_v2",
            "model_profile_revision_id": str(bundle.current_revision.id),
            "provider_revision_id": str(bundle.provider_revision.id),
            "provider_protocol": bundle.provider_revision.protocol.value,
            "semantic_endpoint_identity": bundle.provider_revision.base_url,
            "profile_kind": bundle.profile.kind.value,
            "model": bundle.current_revision.model,
            "selected_dimension": snapshot.selected_dimension,
            "dimension_request_mode": snapshot.dimension_request_mode.value,
            "distance_metric": snapshot.distance_metric,
            "vector_data_type": snapshot.vector_data_type,
            "normalization": snapshot.normalization,
            "text_document_transformation_version": (
                TEXT_DOCUMENT_TRANSFORMATION_VERSION
            ),
            "text_query_transformation_version": (
                "tongyi_query_prefix_v1"
                if bundle.profile.kind is ModelKind.MULTIMODAL_EMBEDDING
                else TEXT_QUERY_TRANSFORMATION_VERSION
            ),
            "image_preprocessing_version": image_preprocessing_version,
        }
    )


def _fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normalize_url(value: str) -> str:
    return value.rstrip("/")


def _require_protocol(protocol: ModelProviderProtocol, kind: ModelKind) -> None:
    if kind is ModelKind.MULTIMODAL_EMBEDDING:
        compatible = protocol is ModelProviderProtocol.TONGYI_MULTIMODAL
    else:
        compatible = protocol is ModelProviderProtocol.OPENAI_COMPATIBLE
    if not compatible:
        raise ResourceStateConflictError("provider protocol does not support model kind")


def _require_parameters(kind: ModelKind, parameters: dict[str, Any]) -> None:
    parameter_type = parameters.get("type")
    if kind is ModelKind.CHAT and parameter_type != "chat":
        raise ResourceStateConflictError("chat model parameters are required")
    if kind is not ModelKind.CHAT and parameter_type != "embedding":
        raise ResourceStateConflictError("embedding model parameters are required")
    if kind is ModelKind.CHAT:
        return
    dimension = parameters.get("dimension", "auto")
    if dimension != "auto" and (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or not MIN_EMBEDDING_DIMENSION
        <= dimension
        <= MAX_EMBEDDING_DIMENSION
    ):
        raise ResourceStateConflictError(
            "embedding dimension must be auto or an integer between 64 and 4096"
        )
    if (
        kind is ModelKind.TEXT_EMBEDDING
        and parameters.get("shared_text_image_space_confirmed") is True
    ):
        raise ResourceStateConflictError(
            "text embedding models cannot confirm a shared image space"
        )
