"""Worker composition root for process-wide foundation dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
from uuid import uuid4

from rag_kb.adapters.file_store.assets import LocalIndexAssetStore
from rag_kb.adapters.file_store.local import LocalFileStore
from rag_kb.adapters.chat_preview.pg_notify import PgNotifyPreviewSink
from rag_kb.adapters.lexical_store.postgres import PgLexicalStore
from rag_kb.adapters.model_api.langchain_chat import LangChainChatModelAdapter
from rag_kb.adapters.model_api.langchain_embeddings import (
    LangChainEmbeddingModelAdapter,
)
from rag_kb.adapters.model_api.multimodal_embeddings import (
    TongyiVisionEmbeddingAdapter,
)
from rag_kb.adapters.parser.docling.parser import DoclingParser
from rag_kb.adapters.vector_store.pgvector import PgVectorStore
from rag_kb.answering.pipeline_steps import (
    AnswerGenerationStep,
    CosineEvidenceAssessmentStep,
)
from rag_kb.answering.structure_validator import AnswerStructureValidationStep
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
from rag_kb.domain import ParserLimits
from rag_kb.indexing.pipeline import IndexingPipeline
from rag_kb.memory import ConversationContextSelector, SessionQueryContextualizer
from rag_kb.ports.model_api import ChatModelAdapter, EmbeddingModelAdapter
from rag_kb.retrieval.service import RetrievalService
from rag_kb.scheduling.chat import ChatRunScheduler
from rag_kb.scheduling.indexing import IndexingJobScheduler, RetryPolicy
from rag_kb.services.assets import IndexAssetService
from rag_kb.services.chat_execution import (
    ChatEvidenceRetriever,
    ChatContextualizedQueryStore,
    ChatExecutionContextLoader,
    ChatRunCoordinator,
)
from rag_kb.services.chat_terminal import (
    ChatFailureSettlementService,
    ChatResultPersistenceStep,
)
from rag_kb.services.chat_visuals import VisualEvidencePreparationStep
from rag_kb.services.composite_evidence import CompositeEvidenceHydrationService
from rag_kb.services.content import (
    build_content_services,
    embedding_space_definition,
)
from rag_kb.services.files import FileReconciliationService
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory
from rag_kb.workflows.contracts import GraphRunner
from rag_kb.workflows.langgraph_runner import LangGraphRunner


@dataclass(frozen=True)
class WorkerDependencies:
    """Dependencies currently safe to construct before task contracts exist."""

    settings: Settings
    startup: StartupValidation
    database: DatabaseResources
    unit_of_work: SqlAlchemyUnitOfWorkFactory
    auth_provider: DevelopmentAuthProvider
    access_policy: SingleWorkspaceAccessPolicy
    file_store: LocalFileStore
    asset_store: LocalIndexAssetStore | None
    index_asset_service: IndexAssetService | None
    reconciliation_service: FileReconciliationService
    document_parser: DoclingParser
    embedding_provider: EmbeddingModelAdapter
    multimodal_embedding_provider: TongyiVisionEmbeddingAdapter | None
    vector_store: PgVectorStore
    retrieval_service: RetrievalService
    chat_model_adapter: ChatModelAdapter
    query_contextualizer: SessionQueryContextualizer
    session_context_selector: ConversationContextSelector
    evidence_assessor: CosineEvidenceAssessmentStep
    visual_evidence_preparer: VisualEvidencePreparationStep
    answer_generator: AnswerGenerationStep
    structure_validator: AnswerStructureValidationStep
    result_persister: ChatResultPersistenceStep
    failure_settler: ChatFailureSettlementService
    chat_runner: GraphRunner
    chat_scheduler: ChatRunScheduler
    indexing_pipeline: IndexingPipeline
    indexing_scheduler: IndexingJobScheduler
    chat_preview_sink: PgNotifyPreviewSink | None

    async def close(self) -> None:
        """Release process-owned resources during Worker shutdown."""

        # The parser owns a killable conversion child; reap it before the
        # database goes away so no conversion outlives its job.
        self.document_parser.close()
        if self.chat_preview_sink is not None:
            await self.chat_preview_sink.close()
        await self.database.close()

    async def start(self) -> RuntimeReadiness:
        """Fail startup when the migration-created runtime is incompatible."""

        readiness = await self.check_readiness()
        if self.chat_preview_sink is not None:
            await self.chat_preview_sink.start()
        return readiness

    async def check_readiness(self) -> RuntimeReadiness:
        return await validate_runtime_readiness(self.database.engine)


def build_worker_dependencies(
    settings: Settings | None = None,
    *,
    env_file: str | Path | None = ".env",
    worker_id: str | None = None,
) -> WorkerDependencies:
    """Load configuration explicitly and fail before starting task polling."""

    resolved_settings = settings or load_settings(env_file=env_file)
    startup = validate_startup_environment(resolved_settings)
    database_settings = resolved_settings.database
    identity = resolved_settings.identity
    access_policy = SingleWorkspaceAccessPolicy(identity.workspace_id)
    database = create_database_resources(
        database_settings.runtime_dsn.get_secret_value(),
        pool_size=database_settings.worker_pool_size,
        max_overflow=database_settings.worker_max_overflow,
        process=DatabaseProcess.WORKER,
        statement_timeout_ms=database_settings.worker_statement_timeout_ms,
        lock_timeout_ms=database_settings.lock_timeout_ms,
        idle_in_transaction_session_timeout_ms=(
            database_settings.idle_in_transaction_session_timeout_ms
        ),
    )
    unit_of_work = SqlAlchemyUnitOfWorkFactory(
        database.sessions,
        identity.workspace_id,
    )
    content_services = build_content_services(
        unit_of_work,
        access_policy,
        resolved_settings.model_provider.embedding,
        resolved_settings.model_provider.multimodal_embedding,
    )
    file_store = LocalFileStore(
        resolved_settings.file_store.staging_path,
        resolved_settings.file_store.final_path,
    )
    parser_limits = ParserLimits()
    document_parser = DoclingParser(
        parser_limits,
        artifacts_path=resolved_settings.parser.docling_artifacts_path,
        artifact_manifest_path=(
            resolved_settings.parser.docling_artifact_manifest_path
        ),
    )
    embedding_settings = resolved_settings.model_provider.embedding
    embedding_space = embedding_space_definition(embedding_settings)
    embedding_provider = LangChainEmbeddingModelAdapter(
        base_url=str(embedding_settings.base_url),
        api_key=embedding_settings.api_key.get_secret_value(),
        embedding_space=embedding_space,
        max_batch_size=embedding_settings.max_batch_size,
        timeout_seconds=embedding_settings.timeout_seconds,
        max_retries=embedding_settings.max_retries,
        max_concurrency=embedding_settings.max_concurrency,
    )
    multimodal_settings = resolved_settings.model_provider.multimodal_embedding
    multimodal_embedding_provider = None
    asset_store = None
    if multimodal_settings is not None:
        assert resolved_settings.file_store.asset_staging_path is not None
        assert resolved_settings.file_store.asset_final_path is not None
        multimodal_space = embedding_space_definition(multimodal_settings)
        multimodal_embedding_provider = TongyiVisionEmbeddingAdapter(
            endpoint=str(multimodal_settings.base_url),
            api_key=multimodal_settings.api_key.get_secret_value(),
            embedding_space=multimodal_space,
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
    chat_settings = resolved_settings.model_provider.chat
    chat_adapter_arguments = {
        "base_url": str(chat_settings.base_url),
        "api_key": chat_settings.api_key.get_secret_value(),
        "model": chat_settings.model,
        "timeout_seconds": chat_settings.timeout_seconds,
        "max_retries": chat_settings.max_retries,
        "max_concurrency": chat_settings.max_concurrency,
        "temperature": chat_settings.temperature,
        "max_tokens": chat_settings.max_tokens,
        "thinking_enabled": chat_settings.thinking_enabled,
        "max_visual_images": chat_settings.max_visual_images,
        "max_visual_image_bytes": chat_settings.max_visual_image_bytes,
        "max_visual_total_bytes": chat_settings.max_visual_total_bytes,
    }
    chat_model_adapter = LangChainChatModelAdapter(
        **chat_adapter_arguments,
        structured_output_mode=chat_settings.structured_output_mode,
    )
    indexing_pipeline = IndexingPipeline(
        unit_of_work,
        file_store,
        document_parser,
        embedding_provider,
        embedding_space,
        asset_store=asset_store,
        multimodal_embedding_provider=multimodal_embedding_provider,
        parser_limits=parser_limits,
    )
    poller = resolved_settings.job_poller
    resolved_worker_id = worker_id or _worker_id()
    retry_policy = RetryPolicy(
        max_attempts=poller.max_attempts,
        base_delay_seconds=poller.retry_base_delay_seconds,
        max_delay_seconds=poller.retry_max_delay_seconds,
    )
    vector_store = PgVectorStore(
        database.sessions,
        embedding_space,
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
        # This is the raw process setting; RetrievalService's request gate
        # always combines it with lexical adapter availability through
        # hybrid_request_enabled(). Worker ChatEvidenceRetriever therefore
        # shares the same effective predicate as API retrieval.
        hybrid_enabled=resolved_settings.retrieval.hybrid_enabled,
        lexical_analyzer_version=(
            resolved_settings.retrieval.lexical_analyzer_version
        ),
        lexical_query_version=resolved_settings.retrieval.lexical_query_version,
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
    evidence_assessor = CosineEvidenceAssessmentStep(
        resolved_settings.retrieval.min_cosine_similarity,
        resolved_settings.retrieval.min_rerank_score,
        resolved_settings.retrieval.cross_modal_min_cosine_similarity,
    )
    index_asset_service = (
        IndexAssetService(unit_of_work, access_policy, asset_store)
        if asset_store is not None
        else None
    )
    visual_evidence_preparer = VisualEvidencePreparationStep(
        index_asset_service,
        max_images=chat_settings.max_visual_images,
        max_image_bytes=chat_settings.max_visual_image_bytes,
        max_total_bytes=chat_settings.max_visual_total_bytes,
        max_pixels=chat_settings.max_visual_pixels,
    )
    chat_delivery = resolved_settings.chat_delivery
    chat_preview_sink = (
        PgNotifyPreviewSink(
            database_settings.runtime_dsn.get_secret_value(),
            flush_interval_ms=chat_delivery.preview_flush_interval_ms,
            max_total_bytes=chat_delivery.preview_max_total_bytes,
        )
        if chat_delivery.preview_enabled
        else None
    )
    answer_generator = AnswerGenerationStep(
        chat_model_adapter,
        preview_sink=chat_preview_sink,
        preview_max_visible_bytes=chat_delivery.preview_max_total_bytes,
    )
    structure_validator = AnswerStructureValidationStep(
        chat_model_adapter,
        preview_sink=chat_preview_sink,
    )
    result_persister = ChatResultPersistenceStep(unit_of_work)
    failure_settler = ChatFailureSettlementService(
        unit_of_work,
        max_attempts=poller.max_attempts,
        base_delay_seconds=poller.retry_base_delay_seconds,
        max_delay_seconds=poller.retry_max_delay_seconds,
    )
    chat_coordinator = ChatRunCoordinator(unit_of_work)
    context_loader = ChatExecutionContextLoader(unit_of_work)
    query_contextualizer = SessionQueryContextualizer(
        chat_model_adapter,
        ChatContextualizedQueryStore(unit_of_work),
    )
    session_context_selector = ConversationContextSelector()
    evidence_retriever = ChatEvidenceRetriever(retrieval_service)
    chat_runner = LangGraphRunner(
        context_loader,
        evidence_retriever,
        evidence_assessor,
        answer_generator,
        structure_validator,
        result_persister,
        visual_evidence_preparer=visual_evidence_preparer,
        query_contextualizer=query_contextualizer,
        deadline_seconds=poller.chat_deadline_seconds,
    )
    chat_scheduler = ChatRunScheduler(
        chat_coordinator,
        chat_runner,
        failure_settler,
        worker_id=resolved_worker_id,
        heartbeat_interval_seconds=poller.heartbeat_interval_seconds,
        stale_after_seconds=poller.stale_after_seconds,
        retry_policy=retry_policy,
        reconciliation_batch_size=poller.reconciliation_batch_size,
    )
    indexing_scheduler = IndexingJobScheduler(
        unit_of_work,
        indexing_pipeline,
        worker_id=resolved_worker_id,
        heartbeat_interval_seconds=poller.heartbeat_interval_seconds,
        stale_after_seconds=poller.stale_after_seconds,
        deadline_seconds=poller.indexing_deadline_seconds,
        retry_policy=retry_policy,
        reconciliation_batch_size=poller.reconciliation_batch_size,
    )
    return WorkerDependencies(
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
        file_store=file_store,
        asset_store=asset_store,
        index_asset_service=index_asset_service,
        reconciliation_service=FileReconciliationService(
            unit_of_work,
            content_services.documents,
            file_store,
            batch_size=resolved_settings.file_store.reconciliation_batch_size,
            orphan_grace_seconds=resolved_settings.file_store.orphan_grace_seconds,
            cleanup_max_attempts=resolved_settings.file_store.cleanup_max_attempts,
            cleanup_base_delay_seconds=(
                resolved_settings.file_store.cleanup_base_delay_seconds
            ),
        ),
        document_parser=document_parser,
        embedding_provider=embedding_provider,
        multimodal_embedding_provider=multimodal_embedding_provider,
        vector_store=vector_store,
        retrieval_service=retrieval_service,
        chat_model_adapter=chat_model_adapter,
        query_contextualizer=query_contextualizer,
        session_context_selector=session_context_selector,
        evidence_assessor=evidence_assessor,
        visual_evidence_preparer=visual_evidence_preparer,
        answer_generator=answer_generator,
        structure_validator=structure_validator,
        result_persister=result_persister,
        failure_settler=failure_settler,
        chat_runner=chat_runner,
        chat_scheduler=chat_scheduler,
        indexing_pipeline=indexing_pipeline,
        indexing_scheduler=indexing_scheduler,
        chat_preview_sink=chat_preview_sink,
    )


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
