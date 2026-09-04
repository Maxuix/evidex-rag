"""API composition root for process-wide foundation dependencies."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from pathlib import Path
from uuid import uuid4

from apps.model_asset_runtime import (
    assemble_model_asset_runtime,
    build_dynamic_embedding_loaders,
    build_graphiti_runtime,
    build_legacy_embedding_adapters,
)
from rag_kb.adapters.graphiti.client import GraphitiRuntime
from rag_kb.adapters.file_store.local import LocalFileStore
from rag_kb.adapters.graph_store.postgres import PgGraphStore
from rag_kb.adapters.chat_preview.pg_notify import PgNotifyPreviewBroker
from rag_kb.adapters.lexical_store.postgres import PgLexicalStore
from rag_kb.adapters.local_reranker import LocalMiniLmReranker
from rag_kb.adapters.model_api.langchain_embeddings import (
    probe_openai_embedding_dimension,
)
from rag_kb.adapters.model_api.langchain_chat import LangChainChatModelAdapter
from rag_kb.adapters.model_api.model_catalog import (
    OpenAICompatibleModelCatalogAdapter,
)
from rag_kb.adapters.model_api.multimodal_embeddings import (
    probe_tongyi_embedding_dimension,
)
from rag_kb.adapters.model_secrets.local import LocalModelSecretStore
from rag_kb.adapters.vector_store.pgvector import PgVectorStore
from rag_kb.config import (
    Settings,
    StartupValidation,
    load_settings,
    validate_startup_environment,
)
from rag_kb.config.settings import ChatProviderSettings
from rag_kb.db import (
    DatabaseProcess,
    DatabaseResources,
    check_database_ready,
    create_database_resources,
    ensure_local_workspace,
)
from rag_kb.domain import (
    AdmissionLimits,
    ChatModelMessage,
    ChatModelRequest,
    ChatModelVisualContent,
    ChatToolDefinition,
    EmbeddingDimensionRequestMode,
    EmbeddingDimensionSelectionSource,
    EmbeddingInputCapability,
    EmbeddingValidationSnapshot,
    ModelKind,
    ModelProfileBundle,
)
from rag_kb.ports.files import IndexAssetStore
from rag_kb.ports.model_api import (
    EmbeddingModelAdapter,
    MultimodalEmbeddingAdapter,
)
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
)
from rag_kb.services.files import SourceFileService
from rag_kb.graph import GraphConfigurationService
from rag_kb.services.indexing import IndexingJobService
from rag_kb.services.markdown_media import MarkdownMediaNormalizer
from rag_kb.services.model_settings import (
    ModelProfileValidationError,
    ModelSettingsService,
)
from rag_kb.tokenizer import preflight_tokenizer
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


@dataclass(frozen=True)
class ApiDependencies:
    """Dependencies currently safe to construct before API contracts exist."""

    settings: Settings
    startup: StartupValidation
    database: DatabaseResources
    unit_of_work: SqlAlchemyUnitOfWorkFactory
    knowledge_base_service: KnowledgeBaseService
    document_service: DocumentService
    file_store: LocalFileStore
    asset_store: IndexAssetStore
    index_asset_service: IndexAssetService
    source_file_service: SourceFileService
    file_admission_service: FileAdmissionService
    indexing_job_service: IndexingJobService
    embedding_provider: EmbeddingModelAdapter
    multimodal_embedding_provider: MultimodalEmbeddingAdapter | None
    vector_store: PgVectorStore
    retrieval_service: RetrievalService
    graph_configuration_service: GraphConfigurationService
    chat_service: ChatService
    chat_terminal_watcher: ChatTerminalWatcher
    chat_event_watcher: ChatEventWatcher
    chat_sse_connection_limiter: ChatSseConnectionLimiter
    chat_preview_broker: PgNotifyPreviewBroker | None
    model_secret_store: LocalModelSecretStore
    model_settings_service: ModelSettingsService
    graphiti_runtime: GraphitiRuntime

    async def close(self) -> None:
        """Release process-owned database resources during API shutdown."""

        if self.chat_preview_broker is not None:
            await self.chat_preview_broker.close()
        await self.graphiti_runtime.close()
        await self.database.close()

    async def start(self) -> None:
        """Fail startup when the local database is unavailable or stale."""

        await self.check_readiness()
        if self.chat_preview_broker is not None:
            await self.chat_preview_broker.start()

    async def check_readiness(self) -> None:
        await check_database_ready(self.database.engine)
        await ensure_local_workspace(
            self.database.engine,
            self.settings.identity.workspace_id,
        )


def build_api_dependencies(
    settings: Settings | None = None,
    *,
    env_file: str | Path | None = ".env.local",
) -> ApiDependencies:
    """Load configuration explicitly and fail before constructing an API app."""

    preflight_tokenizer()
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
    model_secret_store = LocalModelSecretStore(
        resolved_settings.model_secrets.root_path
    )
    graphiti_runtime = build_graphiti_runtime(
        unit_of_work, model_secret_store, resolved_settings
    )
    model_settings_service = ModelSettingsService(
        unit_of_work,
        model_secret_store,
        profile_validator=_validate_model_profile,
        provider_catalog=OpenAICompatibleModelCatalogAdapter().list_models,
    )
    model_assets = assemble_model_asset_runtime(resolved_settings)
    legacy_embeddings = build_legacy_embedding_adapters(model_assets)
    dynamic_embeddings = build_dynamic_embedding_loaders(
        unit_of_work,
        model_secret_store,
    )
    legacy_models = model_assets.legacy_models
    embedding = model_assets.embedding_settings
    multimodal_settings = model_assets.multimodal_settings
    embedding_space = model_assets.embedding_space
    embedding_provider = legacy_embeddings.embedding
    multimodal_embedding_provider = legacy_embeddings.multimodal
    asset_store = model_assets.asset_store
    vector_store = PgVectorStore(
        database.sessions,
        embedding_space,
    )
    graph_store = PgGraphStore(database.sessions)
    content_services = build_content_services(
        unit_of_work,
        embedding,
        multimodal_settings,
    )
    file_store = LocalFileStore(
        resolved_settings.file_store.staging_path,
        resolved_settings.file_store.final_path,
    )
    retrieval_service = RetrievalService(
        identity.workspace_id,
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
        embedding_model_resolver=dynamic_embeddings.embedding,
        multimodal_embedding_model_resolver=dynamic_embeddings.multimodal,
        text_reranker=LocalMiniLmReranker(),
        graph_store=graph_store,
        graphiti_graph=graphiti_runtime,
    )
    chat_service = ChatService(
        unit_of_work,
        model_configuration=(
            chat_model_configuration(legacy_models.chat)
            if legacy_models is not None
            else {}
        ),
        default_rerank=resolved_settings.retrieval.rerank_enabled,
        retrieval_profile_factory=lambda strategy, top_k, rerank_mode: (
            retrieval_service.execution_profile(
                strategy=strategy,
                top_k=top_k,
                rerank_mode=rerank_mode,
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
        knowledge_base_service=content_services.knowledge_bases,
        document_service=content_services.documents,
        file_store=file_store,
        asset_store=asset_store,
        index_asset_service=IndexAssetService(
            unit_of_work,
            asset_store,
        ),
        source_file_service=SourceFileService(
            content_services.documents,
            file_store,
            identity.workspace_id,
            MarkdownMediaNormalizer(),
        ),
        file_admission_service=FileAdmissionService(AdmissionLimits()),
        indexing_job_service=IndexingJobService(unit_of_work),
        embedding_provider=embedding_provider,
        multimodal_embedding_provider=multimodal_embedding_provider,
        vector_store=vector_store,
        retrieval_service=retrieval_service,
        graph_configuration_service=GraphConfigurationService(
            unit_of_work, graphiti_runtime
        ),
        chat_service=chat_service,
        chat_terminal_watcher=chat_terminal_watcher,
        chat_event_watcher=ChatEventWatcher(chat_terminal_watcher),
        chat_sse_connection_limiter=ChatSseConnectionLimiter(
            chat_delivery.max_connections_per_run
        ),
        chat_preview_broker=chat_preview_broker,
        model_secret_store=model_secret_store,
        model_settings_service=model_settings_service,
        graphiti_runtime=graphiti_runtime,
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
            reasoning_effort=parameters.get("reasoning_effort", "off"),
            max_visual_images=ChatProviderSettings.max_visual_images,
            max_visual_image_bytes=ChatProviderSettings.max_visual_image_bytes,
            max_visual_total_bytes=ChatProviderSettings.max_visual_total_bytes,
        )
        search_tool = ChatToolDefinition(
            name="search_knowledge_base",
            description="Return bounded knowledge-base evidence.",
            input_schema={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3,
                    }
                },
                "required": ["queries"],
                "additionalProperties": False,
            },
        )
        first = await adapter.complete(
            ChatModelRequest(
                messages=(
                    ChatModelMessage(
                        "system",
                        "Call search_knowledge_base exactly once with the query validation.",
                    ),
                    ChatModelMessage("user", "Validate native tool calling."),
                ),
                tools=(search_tool,),
                tool_choice="search_knowledge_base",
                max_output_tokens=768,
                thinking_enabled=False,
            )
        )
        if (
            first.model != revision.model
            or len(first.tool_calls) != 1
            or first.tool_calls[0].name != "search_knowledge_base"
        ):
            raise ModelProfileValidationError("provider_validation_failed")
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
            "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        call = first.tool_calls[0]
        second = await adapter.complete(
            ChatModelRequest(
                messages=(
                    ChatModelMessage(
                        "system",
                        "Write a plain-text final response and cite the supplied "
                        "evidence inline with [ev_validation]. Do not call tools.",
                    ),
                    ChatModelMessage("user", "Validate native tool calling."),
                    ChatModelMessage("assistant", first.content, tool_calls=(call,)),
                    ChatModelMessage(
                        "tool",
                        '{"status":"ok","evidence_refs":["ev_validation"]}',
                        tool_call_id=call.id,
                    ),
                    ChatModelMessage(
                        "evidence",
                        "The attached server evidence is bound to ev_validation.",
                        visual_content=(
                            ChatModelVisualContent(
                                citation_ids=("cite_1",),
                                asset_id=uuid4(),
                                media_type="image/png",
                                checksum_sha256=hashlib.sha256(png).hexdigest(),
                                content=png,
                                width=1,
                                height=1,
                            ),
                        ),
                    ),
                ),
                tools=(),
                tool_choice="none",
                max_output_tokens=768,
                thinking_enabled=False,
            )
        )
        if (
            second.model != revision.model
            or second.tool_calls
            or "[ev_validation]" not in second.content.lower()
        ):
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
