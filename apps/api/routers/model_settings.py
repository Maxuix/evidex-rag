"""Local user-managed model-provider and model-profile HTTP transport."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status

from apps.api.security import get_auth_context
from rag_kb.auth import AuthContext
from rag_kb.domain import ModelProfileBundle, ModelProviderBundle, ModelSelection
from rag_kb.services.model_settings import ModelSettingsSnapshot
from rag_kb.schemas import (
    ChatModelParameters,
    EmbeddingModelParameters,
    ModelCatalogResponse,
    ModelProfileCreate,
    ModelProfileResponse,
    ModelProfileUpdate,
    ModelProviderCreate,
    ModelProviderResponse,
    ModelProviderUpdate,
    ModelSelectionResponse,
    ModelSelectionUpdate,
    ModelSettingsResponse,
)


router = APIRouter(tags=["model-settings"])


@router.get(
    "/model-settings",
    response_model=ModelSettingsResponse,
)
async def get_model_settings(
    request: Request,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> ModelSettingsResponse:
    return await _snapshot_response(request, context)


@router.post(
    "/model-providers",
    response_model=ModelProviderResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_model_provider(
    request: Request,
    payload: ModelProviderCreate,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> ModelProviderResponse:
    value = await request.app.state.dependencies.model_settings_service.create_provider(
        context,
        name=payload.name,
        protocol=payload.protocol,
        base_url=str(payload.base_url),
        api_key=payload.api_key.get_secret_value(),
        timeout_seconds=payload.timeout_seconds,
        max_retries=payload.max_retries,
        max_concurrency=payload.max_concurrency,
    )
    available = await request.app.state.dependencies.model_settings_service.provider_secret_available(
        context, value
    )
    return _provider_response(value, api_key_configured=available)


@router.patch(
    "/model-providers/{provider_id}",
    response_model=ModelProviderResponse,
)
async def update_model_provider(
    request: Request,
    provider_id: UUID,
    payload: ModelProviderUpdate,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> ModelProviderResponse:
    value = await request.app.state.dependencies.model_settings_service.update_provider(
        context,
        provider_id,
        name=payload.name,
        protocol=payload.protocol,
        base_url=str(payload.base_url) if payload.base_url is not None else None,
        api_key=(
            payload.api_key.get_secret_value() if payload.api_key is not None else None
        ),
        timeout_seconds=payload.timeout_seconds,
        max_retries=payload.max_retries,
        max_concurrency=payload.max_concurrency,
        enabled=payload.enabled,
    )
    available = await request.app.state.dependencies.model_settings_service.provider_secret_available(
        context, value
    )
    return _provider_response(value, api_key_configured=available)


@router.get(
    "/model-providers/{provider_id}/models",
    response_model=ModelCatalogResponse,
)
async def list_provider_models(
    request: Request,
    provider_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> ModelCatalogResponse:
    models = (
        await request.app.state.dependencies.model_settings_service.list_provider_models(
            context,
            provider_id,
        )
    )
    return ModelCatalogResponse(models=models)


@router.post(
    "/model-profiles",
    response_model=ModelProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_model_profile(
    request: Request,
    payload: ModelProfileCreate,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> ModelProfileResponse:
    value = await request.app.state.dependencies.model_settings_service.create_profile(
        context,
        provider_id=payload.provider_id,
        name=payload.name,
        kind=payload.kind,
        model=payload.model,
        parameters=payload.parameters.model_dump(mode="json"),
    )
    available = await request.app.state.dependencies.model_settings_service.profile_secret_available(
        context, value
    )
    return _profile_response(value, provider_secret_available=available)


@router.post(
    "/model-profiles/{profile_id}/validate",
    response_model=ModelProfileResponse,
)
async def validate_model_profile(
    request: Request,
    profile_id: UUID,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> ModelProfileResponse:
    value = await request.app.state.dependencies.model_settings_service.validate_profile(
        context,
        profile_id,
    )
    available = await request.app.state.dependencies.model_settings_service.profile_secret_available(
        context, value
    )
    return _profile_response(value, provider_secret_available=available)


@router.patch(
    "/model-profiles/{profile_id}",
    response_model=ModelProfileResponse,
)
async def update_model_profile(
    request: Request,
    profile_id: UUID,
    payload: ModelProfileUpdate,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> ModelProfileResponse:
    value = await request.app.state.dependencies.model_settings_service.update_profile(
        context,
        profile_id,
        provider_id=payload.provider_id,
        name=payload.name,
        model=payload.model,
        parameters=(
            payload.parameters.model_dump(mode="json")
            if payload.parameters is not None
            else None
        ),
        enabled=payload.enabled,
    )
    available = await request.app.state.dependencies.model_settings_service.profile_secret_available(
        context, value
    )
    return _profile_response(value, provider_secret_available=available)


@router.put(
    "/model-selection",
    response_model=ModelSelectionResponse,
)
async def update_model_selection(
    request: Request,
    payload: ModelSelectionUpdate,
    context: Annotated[AuthContext, Depends(get_auth_context)],
) -> ModelSelectionResponse:
    value = await request.app.state.dependencies.model_settings_service.update_selection(
        context,
        chat_profile_revision_id=payload.chat_profile_revision_id,
        text_embedding_profile_revision_id=(
            payload.text_embedding_profile_revision_id
        ),
        multimodal_embedding_profile_revision_id=(
            payload.multimodal_embedding_profile_revision_id
        ),
    )
    return _selection_response(value)


async def _snapshot_response(
    request: Request,
    context: AuthContext,
) -> ModelSettingsResponse:
    snapshot = (
        await request.app.state.dependencies.model_settings_service.snapshot(context)
    )
    return _settings_response(snapshot)


def _settings_response(snapshot: ModelSettingsSnapshot) -> ModelSettingsResponse:
    return ModelSettingsResponse(
        providers=tuple(
            _provider_response(
                value,
                api_key_configured=snapshot.provider_secret_health[value.current_revision.id],
            )
            for value in snapshot.providers
        ),
        profiles=tuple(
            _profile_response(
                value,
                provider_secret_available=snapshot.profile_secret_health[value.current_revision.id],
            )
            for value in snapshot.profiles
        ),
        selection=_selection_response(snapshot.selection),
    )


def _provider_response(
    value: ModelProviderBundle,
    *,
    api_key_configured: bool = True,
) -> ModelProviderResponse:
    provider = value.provider
    revision = value.current_revision
    return ModelProviderResponse(
        id=provider.id,
        revision_id=revision.id,
        revision=revision.revision,
        name=provider.name,
        protocol=revision.protocol,
        base_url=revision.base_url,
        timeout_seconds=revision.timeout_seconds,
        max_retries=revision.max_retries,
        max_concurrency=revision.max_concurrency,
        enabled=provider.enabled,
        api_key_configured=api_key_configured,
        configuration_fingerprint=revision.configuration_fingerprint,
        created_at=provider.created_at,
        updated_at=provider.updated_at,
    )


def _profile_response(
    value: ModelProfileBundle,
    *,
    provider_secret_available: bool = True,
) -> ModelProfileResponse:
    profile = value.profile
    revision = value.current_revision
    parameters = (
        ChatModelParameters.model_validate(
            {
                key: item
                for key, item in revision.configuration.items()
                if key in ChatModelParameters.model_fields
            }
        )
        if profile.kind.value == "chat"
        else EmbeddingModelParameters.model_validate(
            {
                "type": "embedding",
                "dimension": revision.configuration.get("dimension", "auto"),
                "max_batch_size": revision.configuration.get("max_batch_size", 10),
                "shared_text_image_space_confirmed": revision.configuration.get(
                    "shared_text_image_space_confirmed", False
                ),
            }
        )
    )
    return ModelProfileResponse(
        id=profile.id,
        revision_id=revision.id,
        revision=revision.revision,
        provider_id=profile.provider_id,
        provider_revision_id=revision.provider_revision_id,
        name=profile.name,
        kind=profile.kind,
        model=revision.model,
        parameters=parameters,
        enabled=profile.enabled,
        provider_secret_available=provider_secret_available,
        validation_status=revision.validation_status,
        validation_error_code=revision.validation_error_code,
        validated_at=revision.validated_at,
        configuration_fingerprint=revision.configuration_fingerprint,
        capability_fingerprint=revision.capability_fingerprint,
        compatibility_fingerprint=revision.compatibility_fingerprint,
        embedding_validation=(
            revision.validation_snapshot.as_dict()
            if revision.validation_snapshot is not None
            else None
        ),
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _selection_response(value: ModelSelection) -> ModelSelectionResponse:
    return ModelSelectionResponse(
        chat_profile_revision_id=value.chat_profile_revision_id,
        text_embedding_profile_revision_id=(
            value.text_embedding_profile_revision_id
        ),
        multimodal_embedding_profile_revision_id=(
            value.multimodal_embedding_profile_revision_id
        ),
        updated_at=value.updated_at,
    )
