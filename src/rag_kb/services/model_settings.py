"""Application service for local model providers, profiles, and defaults."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from rag_kb.auth import AccessPolicy, AuthContext
from rag_kb.domain import (
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
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


ModelProfileValidator = Callable[[ModelProfileBundle, str], Awaitable[None]]
ModelProviderCatalog = Callable[
    [ModelProviderBundle, str], Awaitable[tuple[str, ...]]
]


class ModelSettingsService:
    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        access_policy: AccessPolicy,
        secret_store: ModelSecretStore,
        *,
        profile_validator: ModelProfileValidator | None = None,
        provider_catalog: ModelProviderCatalog | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._access_policy = access_policy
        self._secret_store = secret_store
        self._profile_validator = profile_validator
        self._provider_catalog = provider_catalog

    async def snapshot(
        self, context: AuthContext
    ) -> tuple[
        tuple[ModelProviderBundle, ...],
        tuple[ModelProfileBundle, ...],
        ModelSelection,
    ]:
        self._authorize(context)

        async def load(uow: UnitOfWork):
            _require_scope(uow, context)
            providers = await uow.model_settings.list_providers()
            profiles = await uow.model_settings.list_profiles()
            selection = await uow.model_settings.get_selection()
            return providers, profiles, selection

        return await execute_in_transaction(
            self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST
        )

    async def create_provider(
        self,
        context: AuthContext,
        *,
        name: str,
        protocol: ModelProviderProtocol,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
    ) -> ModelProviderBundle:
        self._authorize(context)
        secret_reference = await asyncio.to_thread(self._secret_store.write, api_key)
        fingerprint = provider_fingerprint(
            protocol=protocol,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_concurrency=max_concurrency,
        )

        async def persist(uow: UnitOfWork) -> ModelProviderBundle:
            _require_scope(uow, context)
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
        context: AuthContext,
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
        self._authorize(context)
        secret_reference = (
            await asyncio.to_thread(self._secret_store.write, api_key)
            if api_key is not None
            else None
        )

        async def persist(uow: UnitOfWork) -> ModelProviderBundle:
            _require_scope(uow, context)
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
        context: AuthContext,
        *,
        provider_id: UUID,
        name: str,
        kind: ModelKind,
        model: str,
        parameters: dict[str, Any],
    ) -> ModelProfileBundle:
        self._authorize(context)
        configuration = dict(parameters)
        _require_parameters(kind, configuration)

        async def persist(uow: UnitOfWork) -> ModelProfileBundle:
            _require_scope(uow, context)
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
        context: AuthContext,
        profile_id: UUID,
        *,
        provider_id: UUID | None,
        name: str | None,
        model: str | None,
        parameters: dict[str, Any] | None,
        enabled: bool | None,
    ) -> ModelProfileBundle:
        self._authorize(context)

        async def persist(uow: UnitOfWork) -> ModelProfileBundle:
            _require_scope(uow, context)
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
        context: AuthContext,
        *,
        chat_profile_revision_id: UUID | None,
        text_embedding_profile_revision_id: UUID | None,
        multimodal_embedding_profile_revision_id: UUID | None,
    ) -> ModelSelection:
        self._authorize(context)

        async def persist(uow: UnitOfWork) -> ModelSelection:
            _require_scope(uow, context)
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
        context: AuthContext,
        provider_id: UUID,
    ) -> tuple[str, ...]:
        self._authorize(context)
        if self._provider_catalog is None:
            raise ResourceStateConflictError("provider model discovery is unavailable")

        async def load(uow: UnitOfWork) -> ModelProviderBundle:
            _require_scope(uow, context)
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
            purpose=UnitOfWorkPurpose.REQUEST,
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
        context: AuthContext,
        profile_id: UUID,
    ) -> ModelProfileBundle:
        self._authorize(context)
        if self._profile_validator is None:
            raise ResourceStateConflictError("model validation is unavailable")

        async def load(uow: UnitOfWork) -> ModelProfileBundle:
            _require_scope(uow, context)
            bundle = await uow.model_settings.get_profile(profile_id)
            if bundle is None:
                raise ResourceNotFoundError("model profile was not found")
            if not bundle.profile.enabled or not bundle.provider.enabled:
                raise ResourceStateConflictError("model profile is disabled")
            return bundle

        bundle = await execute_in_transaction(
            self._unit_of_work,
            load,
            purpose=UnitOfWorkPurpose.REQUEST,
        )
        try:
            api_key = await asyncio.to_thread(
                self._secret_store.read,
                bundle.provider_revision.secret_reference,
            )
            await self._profile_validator(bundle, api_key)
            status = ModelValidationStatus.VALID
            error_code = None
        except Exception:
            status = ModelValidationStatus.INVALID
            error_code = "provider_validation_failed"

        async def persist(uow: UnitOfWork) -> ModelProfileBundle:
            _require_scope(uow, context)
            updated = await uow.model_settings.set_validation(
                bundle.current_revision.id,
                status=status,
                error_code=error_code,
            )
            if updated is None:
                raise ResourceNotFoundError("model profile revision was not found")
            return updated

        return await execute_in_transaction(self._unit_of_work, persist)

    def _authorize(self, context: AuthContext) -> None:
        self._access_policy.metadata_filter(context)


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
            "structured_output_mode": configuration.get("structured_output_mode"),
            "vision_enabled": configuration.get("vision_enabled", False),
            "reasoning": configuration.get("reasoning_effort", "off") != "off",
            "sampling_top_k": configuration.get("sampling_top_k") is not None,
        }
    )
    compatibility_fingerprint = None
    if kind is not ModelKind.CHAT:
        compatibility_fingerprint = _fingerprint(
            {
                "kind": kind.value,
                "model": model,
                "dimension": configuration["dimension"],
                "distance_metric": configuration["distance_metric"],
                "vector_data_type": configuration["vector_data_type"],
                "normalization": configuration["normalization"],
                "provider_configuration_fingerprint": (
                    provider_configuration_fingerprint
                ),
            }
        )
    return (
        configuration_fingerprint,
        capability_fingerprint,
        compatibility_fingerprint,
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
    if (
        protocol is ModelProviderProtocol.TONGYI_MULTIMODAL
        and kind is not ModelKind.MULTIMODAL_EMBEDDING
    ):
        raise ResourceStateConflictError("provider protocol does not support model kind")


def _require_parameters(kind: ModelKind, parameters: dict[str, Any]) -> None:
    parameter_type = parameters.get("type")
    if kind is ModelKind.CHAT and parameter_type != "chat":
        raise ResourceStateConflictError("chat model parameters are required")
    if kind is not ModelKind.CHAT and parameter_type != "embedding":
        raise ResourceStateConflictError("embedding model parameters are required")


def _require_scope(uow: UnitOfWork, context: AuthContext) -> None:
    if uow.workspace_id != context.workspace_id:
        raise RuntimeError("Unit of Work workspace does not match AuthContext")
