"""API composition root for process-wide foundation dependencies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from rag_kb.adapters.file_store.assets import LocalIndexAssetStore
from rag_kb.adapters.file_store.local import LocalFileStore
from rag_kb.adapters.chat_preview.pg_notify import PgNotifyPreviewBroker
from rag_kb.adapters.lexical_store.postgres import PgLexicalStore
from rag_kb.adapters.markdown_media.http import PublicHttpImageFetcher
from rag_kb.adapters.model_api.langchain_embeddings import (
    LangChainEmbeddingModelAdapter,
    probe_openai_embedding_dimension,
)
from rag_kb.adapters.model_api.langchain_chat import LangChainChatModelAdapter
from rag_kb.adapters.model_api.model_catalog import (
    OpenAICompatibleModelCatalogAdapter,
)
from rag_kb.adapters.model_api.multimodal_embeddings import (
    TongyiVisionEmbeddingAdapter,
    probe_tongyi_embedding_dimension,
)
from rag_kb.adapters.model_api.unconfigured import UnconfiguredEmbeddingModelAdapter
from rag_kb.adapters.model_secrets.local import LocalModelSecretStore
from rag_kb.adapters.vector_store.pgvector import PgVectorStore
from rag_kb.answering.wire_schemas import WireRetrievalAgentAction
from rag_kb.auth import DevelopmentAuthProvider, SingleWorkspaceAccessPolicy
from rag_kb.config import (
    Settings,
    StartupValidation,
    load_settings,
    validate_startup_environment,
)
from rag_kb.db import (
    DatabaseProcess,
    DatabaseResources,
    check_database_ready,
    create_database_resources,
)
from rag_kb.domain import (
    AdmissionLimits,
    ChatModelMessage,
    ChatModelRequest,
    ChatOutputSchema,
    EmbeddingSpaceDefinition,
    EmbeddingDimensionRequestMode,
    EmbeddingDimensionSelectionSource,
    EmbeddingInputCapability,
    EmbeddingValidationSnapshot,
    ModelKind,
    ModelProfileBundle,
    ModelValidationStatus,
)
from rag_kb.ports.model_api import EmbeddingModelAdapter
from rag_kb.retrieval.service import RetrievalService
from rag_kb.services.admission import FileAdmissionService
from rag_kb.services.assets import IndexAssetService
from rag_kb.services.chat import ChatService, chat_model_configuration
from rag_kb.services.chat_delivery import (
    ChatEventWatcher,
    ChatSseConnectionLimiter,
    ChatTerminalWatcher,
)
from rag_kb.services.composite_evidence import CompositeEvidenceHydrationService
from rag_kb.services.content import (
    DocumentService,
    KnowledgeBaseService,
    build_content_services,
    embedding_space_definition,
    unconfigured_embedding_space_definition,
)
from rag_kb.services.files import SourceFileService
from rag_kb.services.indexing import IndexingJobService
from rag_kb.services.markdown_media import MarkdownMediaNormalizer
from rag_kb.services.model_settings import (
    ModelProfileValidationError,
    ModelSettingsService,
)
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory
from rag_kb.uow import UnitOfWork, UnitOfWorkPurpose, execute_in_transaction


@dataclass(frozen=True)
class ApiDependencies:
    """Dependencies currently safe to construct before API contracts exist."""

    settings: Settings
    startup: StartupValidation
    database: DatabaseResources
    unit_of_work: SqlAlchemyUnitOfWorkFactory
    auth_provider: DevelopmentAuthProvider
    access_policy: SingleWorkspaceAccessPolicy
    knowledge_base_service: KnowledgeBaseService
    document_service: DocumentService
    file_store: LocalFileStore
    asset_store: LocalIndexAssetStore | None
    index_asset_service: IndexAssetService | None
    source_file_service: SourceFileService
    file_admission_service: FileAdmissionService
    indexing_job_service: IndexingJobService
    embedding_provider: EmbeddingModelAdapter
    multimodal_embedding_provider: TongyiVisionEmbeddingAdapter | None
    vector_store: PgVectorStore
    retrieval_service: RetrievalService
    chat_service: ChatService
    chat_terminal_watcher: ChatTerminalWatcher
    chat_event_watcher: ChatEventWatcher
    chat_sse_connection_limiter: ChatSseConnectionLimiter
    chat_preview_broker: PgNotifyPreviewBroker | None
    model_secret_store: LocalModelSecretStore
    model_settings_service: ModelSettingsService

    async def close(self) -> None:
        """Release process-owned database resources during API shutdown."""

        if self.chat_preview_broker is not None:
            await self.chat_preview_broker.close()
        await self.database.close()

    async def start(self) -> None:
        """Fail startup when the local database is unavailable or stale."""

        await self.check_readiness()
        if self.chat_preview_broker is not None:
            await self.chat_preview_broker.start()

    async def check_readiness(self) -> None:
        await check_database_ready(self.database.engine)


def build_api_dependencies(
    settings: Settings | None = None,
    *,
    env_file: str | Path | None = ".env",
) -> ApiDependencies:
    """Load configuration explicitly and fail before constructing an API app."""

    resolved_settings = settings or load_settings(env_file=env_file)
    startup = validate_startup_environment(resolved_settings)
    database_settings = resolved_settings.database
    identity = resolved_settings.identity
    database = create_database_resources(
        database_settings.runtime_dsn.get_secret_value(),
        pool_size=database_settings.api_pool_size,
        max_overflow=database_settings.api_max_overflow,
        process=DatabaseProcess.API,
        statement_timeout_ms=database_settings.api_statement_timeout_ms,
        lock_timeout_ms=database_settings.lock_timeout_ms,
        idle_in_transaction_session_timeout_ms=(
            database_settings.idle_in_transaction_session_timeout_ms
        ),
    )
    unit_of_work = SqlAlchemyUnitOfWorkFactory(
        database.sessions,
        identity.workspace_id,
    )
    access_policy = SingleWorkspaceAccessPolicy(identity.workspace_id)
    model_secret_store = LocalModelSecretStore(
        resolved_settings.model_secrets.root_path
    )
    model_settings_service = ModelSettingsService(
        unit_of_work,
        access_policy,
        model_secret_store,
        profile_validator=_validate_model_profile,
        provider_catalog=OpenAICompatibleModelCatalogAdapter().list_models,
    )
    legacy_models = resolved_settings.model_provider
    embedding = legacy_models.embedding if legacy_models is not None else None
    embedding_space = (
        embedding_space_definition(embedding)
        if embedding is not None
        else unconfigured_embedding_space_definition()
    )
    embedding_provider: EmbeddingModelAdapter = (
        LangChainEmbeddingModelAdapter(
            base_url=str(embedding.base_url),
            api_key=embedding.api_key.get_secret_value(),
            embedding_space=embedding_space,
            max_batch_size=embedding.max_batch_size,
            timeout_seconds=embedding.timeout_seconds,
            max_retries=embedding.max_retries,
            max_concurrency=embedding.max_concurrency,
        )
        if embedding is not None
        else UnconfiguredEmbeddingModelAdapter(embedding_space)
    )
    multimodal_settings = (
        legacy_models.multimodal_embedding if legacy_models is not None else None
    )
    multimodal_embedding_provider = None
    asset_store = None
    if multimodal_settings is not None:
        assert resolved_settings.file_store.asset_staging_path is not None
        assert resolved_settings.file_store.asset_final_path is not None
        multimodal_embedding_provider = TongyiVisionEmbeddingAdapter(
            endpoint=str(multimodal_settings.base_url),
            api_key=multimodal_settings.api_key.get_secret_value(),
            embedding_space=embedding_space_definition(multimodal_settings),
            max_batch_size=multimodal_settings.max_batch_size,
            timeout_seconds=multimodal_settings.timeout_seconds,
            max_retries=multimodal_settings.max_retries,
            max_concurrency=multimodal_settings.max_concurrency,
            text_query_template=multimodal_settings.text_query_template,
        )
        asset_store = LocalIndexAssetStore(
            resolved_settings.file_store.asset_staging_path,
            resolved_settings.file_store.asset_final_path,
        )
    vector_store = PgVectorStore(
        database.sessions,
        embedding_space,
    )
    content_services = build_content_services(
        unit_of_work,
        access_policy,
        embedding,
        multimodal_settings,
    )
    file_store = LocalFileStore(
        resolved_settings.file_store.staging_path,
        resolved_settings.file_store.final_path,
    )
    retrieval_service = RetrievalService(
        access_policy,
        embedding_provider,
        vector_store,
        candidate_multiplier=resolved_settings.retrieval.candidate_multiplier,
        max_candidate_count=resolved_settings.retrieval.max_candidate_count,
        vector_weight=resolved_settings.retrieval.vector_weight,
        lexical_weight=resolved_settings.retrieval.lexical_weight,
        mmr_lambda=resolved_settings.retrieval.mmr_lambda,
        multimodal_embedding_provider=multimodal_embedding_provider,
        cross_modal_candidate_count=(
            resolved_settings.retrieval.cross_modal_candidate_count
        ),
        cross_modal_min_cosine_similarity=(
            resolved_settings.retrieval.cross_modal_min_cosine_similarity
        ),
        text_min_cosine_similarity=(
            resolved_settings.retrieval.min_cosine_similarity
        ),
        rrf_k=resolved_settings.retrieval.rrf_k,
        cross_modal_weight_micros=(
            resolved_settings.retrieval.cross_modal_weight_micros
        ),
        lexical_store=PgLexicalStore(database.sessions),
        hybrid_enabled=resolved_settings.retrieval.hybrid_enabled,
        lexical_analyzer_version=(
            resolved_settings.retrieval.lexical_analyzer_version
        ),
        lexical_query_version=(
            resolved_settings.retrieval.lexical_query_version
        ),
        lexical_candidate_count=(
            resolved_settings.retrieval.lexical_candidate_count
        ),
        dense_weight_micros=resolved_settings.retrieval.dense_weight_micros,
        lexical_weight_micros=(
            resolved_settings.retrieval.lexical_weight_micros
        ),
        min_rerank_score=resolved_settings.retrieval.min_rerank_score,
        relation_hydrator=CompositeEvidenceHydrationService(unit_of_work),
        deadline_seconds=resolved_settings.retrieval.deadline_seconds,
        embedding_model_resolver=_embedding_model_loader(
            unit_of_work, model_secret_store
        ),
        multimodal_embedding_model_resolver=_multimodal_model_loader(
            unit_of_work, model_secret_store
        ),
    )
    chat_service = ChatService(
        unit_of_work,
        access_policy,
        model_configuration=(
            chat_model_configuration(legacy_models.chat)
            if legacy_models is not None
            else {}
        ),
        default_rerank=resolved_settings.retrieval.rerank_enabled,
        hybrid_enabled=retrieval_service.hybrid_request_enabled(),
        agent_enabled=resolved_settings.chat_workflow.agent_enabled,
        auto_enabled=resolved_settings.chat_workflow.auto_enabled,
        retrieval_profile_factory=lambda strategy, top_k, rerank: (
            retrieval_service.execution_profile(
                strategy=strategy,
                top_k=top_k,
                rerank=rerank,
            )
        ),
        allow_legacy_model_configuration=False,
    )
    chat_delivery = resolved_settings.chat_delivery
    chat_preview_broker = (
        PgNotifyPreviewBroker(
            database_settings.runtime_dsn.get_secret_value(),
            subscriber_queue_size=chat_delivery.preview_queue_size,
        )
        if chat_delivery.preview_enabled
        else None
    )
    chat_terminal_watcher = ChatTerminalWatcher(
        chat_service,
        poll_interval_seconds=chat_delivery.poll_interval_seconds,
        jitter_ratio=chat_delivery.jitter_ratio,
        max_duration_seconds=(
            chat_delivery.max_connection_duration_seconds
        ),
    )
    return ApiDependencies(
        settings=resolved_settings,
        startup=startup,
        database=database,
        unit_of_work=unit_of_work,
        auth_provider=DevelopmentAuthProvider(
            deployment_profile=resolved_settings.app.deployment_profile.value,
            principal_id=identity.principal_id,
            client_id=identity.client_id,
            workspace_id=identity.workspace_id,
        ),
        access_policy=access_policy,
        knowledge_base_service=content_services.knowledge_bases,
        document_service=content_services.documents,
        file_store=file_store,
        asset_store=asset_store,
        index_asset_service=(
            IndexAssetService(unit_of_work, access_policy, asset_store)
            if asset_store is not None
            else None
        ),
        source_file_service=SourceFileService(
            content_services.documents,
            file_store,
            MarkdownMediaNormalizer(PublicHttpImageFetcher()),
        ),
        file_admission_service=FileAdmissionService(AdmissionLimits()),
        indexing_job_service=IndexingJobService(unit_of_work, access_policy),
        embedding_provider=embedding_provider,
        multimodal_embedding_provider=multimodal_embedding_provider,
        vector_store=vector_store,
        retrieval_service=retrieval_service,
        chat_service=chat_service,
        chat_terminal_watcher=chat_terminal_watcher,
        chat_event_watcher=ChatEventWatcher(chat_terminal_watcher),
        chat_sse_connection_limiter=ChatSseConnectionLimiter(
            chat_delivery.max_connections_per_principal_run
        ),
        chat_preview_broker=chat_preview_broker,
        model_secret_store=model_secret_store,
        model_settings_service=model_settings_service,
    )


async def _validate_model_profile(
    bundle: ModelProfileBundle,
    api_key: str,
) -> EmbeddingValidationSnapshot | None:
    provider = bundle.provider_revision
    revision = bundle.current_revision
    parameters = dict(revision.configuration)
    if bundle.profile.kind is ModelKind.CHAT:
        adapter = LangChainChatModelAdapter(
            base_url=provider.base_url,
            api_key=api_key,
            model=revision.model,
            timeout_seconds=provider.timeout_seconds,
            max_retries=provider.max_retries,
            max_concurrency=provider.max_concurrency,
            temperature=parameters.get("temperature", 0.2),
            top_p=parameters.get("top_p", 0.9),
            sampling_top_k=parameters.get("sampling_top_k", 40),
            max_tokens=parameters.get("max_output_tokens", 4096),
            structured_output_mode=parameters.get(
                "structured_output_mode", "json_object"
            ),
            reasoning_effort=parameters.get("reasoning_effort", "off"),
        )
        response = await adapter.complete(
            ChatModelRequest(
                messages=(
                    ChatModelMessage(
                        "system",
                        "Return only the exact JSON object supplied by the user, "
                        "with no markdown or additional fields.",
                    ),
                    ChatModelMessage(
                        "user",
                        '{"version":"retrieval_agent_action_v1",'
                        '"action":"search","objective":"validation probe",'
                        '"queries":[{"query":"validation",'
                        '"based_on_observation_ids":[]}],'
                        '"proposed_reason":null,'
                        '"selected_evidence_keys":[]}',
                    ),
                ),
                output_schema=ChatOutputSchema.RETRIEVAL_AGENT_ACTION_V1,
                max_output_tokens=768,
                thinking_enabled=False,
            )
        )
        try:
            WireRetrievalAgentAction.model_validate_json(response.content)
        except ValueError as error:
            raise ModelProfileValidationError(
                "provider_validation_failed"
            ) from error
        if response.model != revision.model:
            raise ModelProfileValidationError("provider_validation_failed")
        return None

    configured_dimension = parameters.get("dimension", "auto")
    requested_dimension = (
        configured_dimension if isinstance(configured_dimension, int) else None
    )
    if bundle.profile.kind is ModelKind.TEXT_EMBEDDING:
        actual_dimension = await probe_openai_embedding_dimension(
            base_url=provider.base_url,
            api_key=api_key,
            model=revision.model,
            requested_dimension=requested_dimension,
            timeout_seconds=provider.timeout_seconds,
            max_retries=provider.max_retries,
        )
        capabilities = (
            EmbeddingInputCapability.TEXT_DOCUMENT,
            EmbeddingInputCapability.TEXT_QUERY,
        )
    else:
        actual_dimension = await probe_tongyi_embedding_dimension(
            endpoint=provider.base_url,
            api_key=api_key,
            model=revision.model,
            requested_dimension=requested_dimension,
            timeout_seconds=provider.timeout_seconds,
            max_retries=provider.max_retries,
        )
        capabilities = (
            EmbeddingInputCapability.TEXT_DOCUMENT,
            EmbeddingInputCapability.TEXT_QUERY,
            EmbeddingInputCapability.IMAGE,
        )
    if requested_dimension is not None and actual_dimension != requested_dimension:
        raise ModelProfileValidationError("embedding_dimension_mismatch")
    automatic = requested_dimension is None
    return EmbeddingValidationSnapshot(
        provider_supported_dimensions=None,
        verified_dimensions=(actual_dimension,),
        provider_default_dimension=actual_dimension if automatic else None,
        recommended_dimension=None,
        selected_dimension=actual_dimension,
        selection_source=(
            EmbeddingDimensionSelectionSource.PROVIDER_OBSERVED_DEFAULT
            if automatic
            else EmbeddingDimensionSelectionSource.USER_PROBE
        ),
        dimension_request_mode=(
            EmbeddingDimensionRequestMode.OMITTED
            if automatic
            else EmbeddingDimensionRequestMode.EXPLICIT
        ),
        input_capabilities=capabilities,
        shared_text_image_space_confirmed=bool(
            parameters.get("shared_text_image_space_confirmed", False)
        ),
    )


async def _embedding_bundle(unit_of_work, space, kind):
    revision_id = space.model_profile_revision_id
    if revision_id is None:
        raise ValueError("embedding space has no model profile revision")

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
        unit_of_work, resolve, purpose=UnitOfWorkPurpose.REQUEST
    )


def _embedding_model_loader(unit_of_work, secret_store):
    cache = {}

    async def load(space):
        revision_id = space.model_profile_revision_id
        if revision_id in cache:
            return cache[revision_id]
        bundle = await _embedding_bundle(
            unit_of_work, space, ModelKind.TEXT_EMBEDDING
        )
        key = await asyncio.to_thread(
            secret_store.read, bundle.provider_revision.secret_reference
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
        cache[revision_id] = adapter
        return adapter

    return load


def _multimodal_model_loader(unit_of_work, secret_store):
    cache = {}

    async def load(space):
        revision_id = space.model_profile_revision_id
        if revision_id in cache:
            return cache[revision_id]
        bundle = await _embedding_bundle(
            unit_of_work, space, ModelKind.MULTIMODAL_EMBEDDING
        )
        key = await asyncio.to_thread(
            secret_store.read, bundle.provider_revision.secret_reference
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
        cache[revision_id] = adapter
        return adapter

    return load
