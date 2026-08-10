"""Shared model and local-asset assembly used by the three process roots."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

from rag_kb.adapters.file_store.assets import LocalIndexAssetStore
from rag_kb.adapters.model_api.langchain_embeddings import (
    LangChainEmbeddingModelAdapter,
)
from rag_kb.adapters.model_api.multimodal_embeddings import (
    TongyiVisionEmbeddingAdapter,
)
from rag_kb.adapters.model_api.unconfigured import (
    UnconfiguredEmbeddingModelAdapter,
)
from rag_kb.config.settings import (
    EmbeddingProviderSettings,
    ModelProviderSettings,
    MultimodalEmbeddingProviderSettings,
    Settings,
)
from rag_kb.domain import (
    EmbeddingSpaceDefinition,
    ModelKind,
    ModelValidationStatus,
)
from rag_kb.ports.model_api import (
    EmbeddingModelAdapter,
    MultimodalEmbeddingAdapter,
)
from rag_kb.ports.model_secrets import ModelSecretStore
from rag_kb.services.content import (
    embedding_space_definition,
    unconfigured_embedding_space_definition,
)
from rag_kb.uow import (
    UnitOfWork,
    UnitOfWorkFactory,
    UnitOfWorkPurpose,
    execute_in_transaction,
)


EmbeddingModelLoader = Callable[
    [EmbeddingSpaceDefinition], Awaitable[EmbeddingModelAdapter]
]
MultimodalEmbeddingModelLoader = Callable[
    [EmbeddingSpaceDefinition], Awaitable[MultimodalEmbeddingAdapter]
]


@dataclass(frozen=True, slots=True)
class ModelAssetRuntime:
    """Provider-independent asset capability plus optional legacy model facts."""

    legacy_models: ModelProviderSettings | None
    embedding_settings: EmbeddingProviderSettings | None
    multimodal_settings: MultimodalEmbeddingProviderSettings | None
    embedding_space: EmbeddingSpaceDefinition
    asset_store: LocalIndexAssetStore


@dataclass(frozen=True, slots=True)
class LegacyEmbeddingAdapters:
    embedding: EmbeddingModelAdapter
    multimodal: MultimodalEmbeddingAdapter | None


@dataclass(frozen=True, slots=True)
class DynamicEmbeddingLoaders:
    embedding: EmbeddingModelLoader
    multimodal: MultimodalEmbeddingModelLoader


def assemble_model_asset_runtime(settings: Settings) -> ModelAssetRuntime:
    """Resolve optional legacy settings while always enabling local assets."""

    legacy_models = settings.model_provider
    embedding_settings = (
        legacy_models.embedding if legacy_models is not None else None
    )
    multimodal_settings = (
        legacy_models.multimodal_embedding if legacy_models is not None else None
    )
    embedding_space = (
        embedding_space_definition(embedding_settings)
        if embedding_settings is not None
        else unconfigured_embedding_space_definition()
    )
    asset_staging_path = settings.file_store.asset_staging_path
    asset_final_path = settings.file_store.asset_final_path
    assert asset_staging_path is not None
    assert asset_final_path is not None
    return ModelAssetRuntime(
        legacy_models=legacy_models,
        embedding_settings=embedding_settings,
        multimodal_settings=multimodal_settings,
        embedding_space=embedding_space,
        asset_store=LocalIndexAssetStore(
            asset_staging_path,
            asset_final_path,
        ),
    )


def build_legacy_embedding_adapters(
    runtime: ModelAssetRuntime,
) -> LegacyEmbeddingAdapters:
    """Construct only the optional legacy embedding transports."""

    embedding_settings = runtime.embedding_settings
    embedding: EmbeddingModelAdapter = (
        LangChainEmbeddingModelAdapter(
            base_url=str(embedding_settings.base_url),
            api_key=embedding_settings.api_key.get_secret_value(),
            embedding_space=runtime.embedding_space,
            max_batch_size=embedding_settings.max_batch_size,
            timeout_seconds=embedding_settings.timeout_seconds,
            max_retries=embedding_settings.max_retries,
            max_concurrency=embedding_settings.max_concurrency,
        )
        if embedding_settings is not None
        else UnconfiguredEmbeddingModelAdapter(runtime.embedding_space)
    )
    multimodal_settings = runtime.multimodal_settings
    multimodal = (
        TongyiVisionEmbeddingAdapter(
            endpoint=str(multimodal_settings.base_url),
            api_key=multimodal_settings.api_key.get_secret_value(),
            embedding_space=embedding_space_definition(multimodal_settings),
            max_batch_size=multimodal_settings.max_batch_size,
            timeout_seconds=multimodal_settings.timeout_seconds,
            max_retries=multimodal_settings.max_retries,
            max_concurrency=multimodal_settings.max_concurrency,
            text_query_template=multimodal_settings.text_query_template,
        )
        if multimodal_settings is not None
        else None
    )
    return LegacyEmbeddingAdapters(
        embedding=embedding,
        multimodal=multimodal,
    )


def build_dynamic_embedding_loaders(
    unit_of_work: UnitOfWorkFactory,
    secret_store: ModelSecretStore,
) -> DynamicEmbeddingLoaders:
    """Build revision-bound adapter loaders with one cache per model kind."""

    embedding_cache: dict[UUID, EmbeddingModelAdapter] = {}
    multimodal_cache: dict[UUID, MultimodalEmbeddingAdapter] = {}

    async def load_embedding(
        space: EmbeddingSpaceDefinition,
    ) -> EmbeddingModelAdapter:
        revision_id = _revision_id(space)
        cached = embedding_cache.get(revision_id)
        if cached is not None:
            return cached
        bundle = await _embedding_bundle(
            unit_of_work,
            space,
            ModelKind.TEXT_EMBEDDING,
        )
        key = await asyncio.to_thread(
            secret_store.read,
            bundle.provider_revision.secret_reference,
        )
        parameters = bundle.current_revision.configuration
        adapter = LangChainEmbeddingModelAdapter(
            base_url=bundle.provider_revision.base_url,
            api_key=key,
            embedding_space=space,
            max_batch_size=parameters["max_batch_size"],
            timeout_seconds=bundle.provider_revision.timeout_seconds,
            max_retries=bundle.provider_revision.max_retries,
            max_concurrency=bundle.provider_revision.max_concurrency,
        )
        embedding_cache[revision_id] = adapter
        return adapter

    async def load_multimodal(
        space: EmbeddingSpaceDefinition,
    ) -> MultimodalEmbeddingAdapter:
        revision_id = _revision_id(space)
        cached = multimodal_cache.get(revision_id)
        if cached is not None:
            return cached
        bundle = await _embedding_bundle(
            unit_of_work,
            space,
            ModelKind.MULTIMODAL_EMBEDDING,
        )
        key = await asyncio.to_thread(
            secret_store.read,
            bundle.provider_revision.secret_reference,
        )
        parameters = bundle.current_revision.configuration
        adapter = TongyiVisionEmbeddingAdapter(
            endpoint=bundle.provider_revision.base_url,
            api_key=key,
            embedding_space=space,
            max_batch_size=parameters["max_batch_size"],
            timeout_seconds=bundle.provider_revision.timeout_seconds,
            max_retries=bundle.provider_revision.max_retries,
            max_concurrency=bundle.provider_revision.max_concurrency,
        )
        multimodal_cache[revision_id] = adapter
        return adapter

    return DynamicEmbeddingLoaders(
        embedding=load_embedding,
        multimodal=load_multimodal,
    )


def _revision_id(space: EmbeddingSpaceDefinition) -> UUID:
    revision_id = space.model_profile_revision_id
    if revision_id is None:
        raise ValueError("embedding space has no model profile revision")
    return revision_id


async def _embedding_bundle(
    unit_of_work: UnitOfWorkFactory,
    space: EmbeddingSpaceDefinition,
    kind: ModelKind,
):
    revision_id = _revision_id(space)

    async def resolve(uow: UnitOfWork):
        bundle = await uow.model_settings.get_profile_revision(revision_id)
        if (
            bundle is None
            or bundle.profile.kind is not kind
            or not bundle.profile.enabled
            or not bundle.provider.enabled
            or bundle.current_revision.validation_status
            is not ModelValidationStatus.VALID
            or bundle.current_revision.compatibility_fingerprint
            != space.compatibility_fingerprint
        ):
            raise ValueError("embedding model profile is unavailable")
        return bundle

    return await execute_in_transaction(
        unit_of_work,
        resolve,
        purpose=UnitOfWorkPurpose.REQUEST,
    )
