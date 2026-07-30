"""API composition root for process-wide foundation dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_kb.adapters.file_store.assets import LocalIndexAssetStore
from rag_kb.adapters.file_store.local import LocalFileStore
from rag_kb.adapters.chat_preview.pg_notify import PgNotifyPreviewBroker
from rag_kb.adapters.lexical_store.postgres import PgLexicalStore
from rag_kb.adapters.markdown_media.http import PublicHttpImageFetcher
from rag_kb.adapters.model_api.langchain_embeddings import (
    LangChainEmbeddingModelAdapter,
)
from rag_kb.adapters.model_api.multimodal_embeddings import (
    TongyiVisionEmbeddingAdapter,
)
from rag_kb.adapters.vector_store.pgvector import PgVectorStore
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
    RuntimeReadiness,
    create_database_resources,
    validate_runtime_readiness,
)
from rag_kb.domain import AdmissionLimits
from rag_kb.ports.model_api import EmbeddingModelAdapter
from rag_kb.retrieval.service import RetrievalService
from rag_kb.services.admission import FileAdmissionService
from rag_kb.services.assets import IndexAssetService
from rag_kb.services.chat import ChatService, chat_model_configuration
from rag_kb.services.chat_delivery import (
    ChatSseConnectionLimiter,
    ChatTerminalWatcher,
)
from rag_kb.services.composite_evidence import CompositeEvidenceHydrationService
from rag_kb.services.content import (
    DocumentService,
    KnowledgeBaseService,
    build_content_services,
    embedding_space_definition,
)
from rag_kb.services.files import SourceFileService
from rag_kb.services.indexing import IndexingJobService
from rag_kb.services.markdown_media import MarkdownMediaNormalizer
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


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
    chat_sse_connection_limiter: ChatSseConnectionLimiter
    chat_preview_broker: PgNotifyPreviewBroker | None

    async def close(self) -> None:
        """Release process-owned database resources during API shutdown."""

        if self.chat_preview_broker is not None:
            await self.chat_preview_broker.close()
        await self.database.close()

    async def start(self) -> RuntimeReadiness:
        """Fail startup when the migration-created runtime is incompatible."""

        readiness = await self.check_readiness()
        if self.chat_preview_broker is not None:
            await self.chat_preview_broker.start()
        return readiness

    async def check_readiness(self) -> RuntimeReadiness:
        return await validate_runtime_readiness(self.database.engine)


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
    embedding = resolved_settings.model_provider.embedding
    embedding_space = embedding_space_definition(embedding)
    embedding_provider = LangChainEmbeddingModelAdapter(
        base_url=str(embedding.base_url),
        api_key=embedding.api_key.get_secret_value(),
        embedding_space=embedding_space,
        max_batch_size=embedding.max_batch_size,
        timeout_seconds=embedding.timeout_seconds,
        max_retries=embedding.max_retries,
        max_concurrency=embedding.max_concurrency,
    )
    multimodal_settings = resolved_settings.model_provider.multimodal_embedding
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
        resolved_settings.model_provider.multimodal_embedding,
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
    )
    chat_service = ChatService(
        unit_of_work,
        access_policy,
        model_configuration=chat_model_configuration(
            resolved_settings.model_provider.chat
        ),
        default_rerank=resolved_settings.retrieval.rerank_enabled,
        hybrid_enabled=resolved_settings.retrieval.hybrid_enabled,
        retrieval_profile_factory=lambda strategy, top_k, rerank: (
            retrieval_service.execution_profile(
                strategy=strategy,
                top_k=top_k,
                rerank=rerank,
            )
        ),
        context_strategy=resolved_settings.session_context.strategy,
        context_max_turns=resolved_settings.session_context.max_turns,
        context_max_tokens=(
            resolved_settings.session_context.max_context_tokens
        ),
        context_tokenizer=resolved_settings.session_context.tokenizer,
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
        file_admission_service=FileAdmissionService(
            AdmissionLimits(
                max_bytes=resolved_settings.file_admission.max_bytes,
                max_markdown_bundle_bytes=(
                    resolved_settings.file_admission.max_markdown_bundle_bytes
                ),
                max_lines=resolved_settings.file_admission.max_lines,
                max_csv_columns=resolved_settings.parser.max_csv_columns,
                max_csv_cells=resolved_settings.parser.max_csv_cells,
                max_archive_entries=(
                    resolved_settings.file_admission.max_archive_entries
                ),
                max_expanded_bytes=(
                    resolved_settings.file_admission.max_expanded_bytes
                ),
                max_assets=resolved_settings.parser.max_assets,
                max_image_width=resolved_settings.parser.max_image_width,
                max_image_height=resolved_settings.parser.max_image_height,
                max_image_pixels=resolved_settings.parser.max_image_pixels,
                max_total_image_pixels=(
                    resolved_settings.parser.max_total_image_pixels
                ),
            )
        ),
        indexing_job_service=IndexingJobService(unit_of_work, access_policy),
        embedding_provider=embedding_provider,
        multimodal_embedding_provider=multimodal_embedding_provider,
        vector_store=vector_store,
        retrieval_service=retrieval_service,
        chat_service=chat_service,
        chat_terminal_watcher=ChatTerminalWatcher(
            chat_service,
            poll_interval_seconds=chat_delivery.poll_interval_seconds,
            jitter_ratio=chat_delivery.jitter_ratio,
            max_duration_seconds=(
                chat_delivery.max_connection_duration_seconds
            ),
        ),
        chat_sse_connection_limiter=ChatSseConnectionLimiter(
            chat_delivery.max_connections_per_principal_run
        ),
        chat_preview_broker=chat_preview_broker,
    )
