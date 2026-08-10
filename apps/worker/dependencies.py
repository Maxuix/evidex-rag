"""Worker composition root for process-wide foundation dependencies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import socket
from uuid import uuid4

from apps.model_asset_runtime import (
    assemble_model_asset_runtime,
    build_dynamic_embedding_loaders,
    build_legacy_embedding_adapters,
)
from rag_kb.adapters.file_store.local import LocalFileStore
from rag_kb.adapters.chat_preview.pg_notify import PgNotifyPreviewSink
from rag_kb.adapters.lexical_store.postgres import PgLexicalStore
from rag_kb.adapters.model_api.langchain_chat import LangChainChatModelAdapter
from rag_kb.adapters.model_api.routing_chat import RoutingChatModelAdapter
from rag_kb.adapters.model_api.unconfigured import (
    UnconfiguredChatModelAdapter,
)
from rag_kb.adapters.model_secrets.local import LocalModelSecretStore
from rag_kb.adapters.parser.docling.parser import DoclingParser
from rag_kb.adapters.vector_store.pgvector import PgVectorStore
from rag_kb.answering.pipeline_steps import (
    AdaptiveEvidenceAssessmentStep,
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
    check_database_ready,
    create_database_resources,
)
from rag_kb.domain import (
    ChatModelExecutionError,
    ErrorCode,
    ModelKind,
    ModelValidationStatus,
    ParserLimits,
)
from rag_kb.indexing.pipeline import IndexingPipeline
from rag_kb.memory import ConversationContextSelector, SessionQueryContextualizer
from rag_kb.ports.files import IndexAssetStore
from rag_kb.ports.model_api import (
    ChatModelAdapter,
    EmbeddingModelAdapter,
    MultimodalEmbeddingAdapter,
)
from rag_kb.retrieval.service import RetrievalService
from rag_kb.retrieval.agent import RetrievalAgentService
from rag_kb.retrieval.router import AutoWorkflowRouter
from rag_kb.scheduling.chat import ChatRunScheduler
from rag_kb.scheduling.indexing import IndexingJobScheduler, RetryPolicy
from rag_kb.services.assets import IndexAssetService
from rag_kb.services.chat_execution import (
    ChatEvidenceRetriever,
    ChatContextualizedQueryStore,
    ChatExecutionContextLoader,
    ChatRunCoordinator,
    ChatWorkflowStateStore,
)
from rag_kb.services.chat_terminal import (
    ChatFailureSettlementService,
    ChatResultPersistenceStep,
)
from rag_kb.services.chat_visuals import VisualEvidencePreparationStep
from rag_kb.services.composite_evidence import CompositeEvidenceHydrationService
from rag_kb.services.content import (
    build_content_services,
)
from rag_kb.services.files import FileReconciliationService
from rag_kb.uow import UnitOfWork, UnitOfWorkPurpose, execute_in_transaction
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
    asset_store: IndexAssetStore
    index_asset_service: IndexAssetService
    reconciliation_service: FileReconciliationService
    document_parser: DoclingParser
    embedding_provider: EmbeddingModelAdapter
    multimodal_embedding_provider: MultimodalEmbeddingAdapter | None
    vector_store: PgVectorStore
    retrieval_service: RetrievalService
    chat_model_adapter: ChatModelAdapter
    query_contextualizer: SessionQueryContextualizer
    session_context_selector: ConversationContextSelector
    evidence_assessor: CosineEvidenceAssessmentStep
    adaptive_evidence_assessor: AdaptiveEvidenceAssessmentStep
    retrieval_agent: RetrievalAgentService
    workflow_router: AutoWorkflowRouter
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

    async def start(self) -> None:
        """Fail startup when the local database is unavailable or stale."""

        await self.check_readiness()
        if self.chat_preview_sink is not None:
            await self.chat_preview_sink.start()

    async def check_readiness(self) -> None:
        await check_database_ready(self.database.engine)


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
    model_assets = assemble_model_asset_runtime(resolved_settings)
    legacy_embeddings = build_legacy_embedding_adapters(model_assets)
    legacy_models = model_assets.legacy_models
    embedding_settings = model_assets.embedding_settings
    multimodal_settings = model_assets.multimodal_settings
    content_services = build_content_services(
        unit_of_work,
        access_policy,
        embedding_settings,
        multimodal_settings,
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
        checkpoint_root=resolved_settings.file_store.parser_temp_path,
    )
    embedding_space = model_assets.embedding_space
    embedding_provider = legacy_embeddings.embedding
    multimodal_embedding_provider = legacy_embeddings.multimodal
    asset_store = model_assets.asset_store
    chat_settings = legacy_models.chat if legacy_models is not None else None
    legacy_chat_model_adapter: ChatModelAdapter = (
        LangChainChatModelAdapter(
            base_url=str(chat_settings.base_url),
            api_key=chat_settings.api_key.get_secret_value(),
            model=chat_settings.model,
            timeout_seconds=chat_settings.timeout_seconds,
            max_retries=chat_settings.max_retries,
            max_concurrency=chat_settings.max_concurrency,
            temperature=chat_settings.temperature,
            max_tokens=chat_settings.max_tokens,
            thinking_enabled=chat_settings.thinking_enabled,
            max_visual_images=chat_settings.max_visual_images,
            max_visual_image_bytes=chat_settings.max_visual_image_bytes,
            max_visual_total_bytes=chat_settings.max_visual_total_bytes,
            structured_output_mode=chat_settings.structured_output_mode,
        )
        if chat_settings is not None
        else UnconfiguredChatModelAdapter()
    )
    model_secret_store = LocalModelSecretStore(
        resolved_settings.model_secrets.root_path
    )
    dynamic_embeddings = build_dynamic_embedding_loaders(
        unit_of_work,
        model_secret_store,
    )
    chat_model_adapter = RoutingChatModelAdapter(
        _chat_model_loader(unit_of_work, model_secret_store),
        legacy_fallback=legacy_chat_model_adapter,
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
        embedding_model_resolver=dynamic_embeddings.embedding,
        multimodal_embedding_model_resolver=dynamic_embeddings.multimodal,
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
        embedding_model_resolver=dynamic_embeddings.embedding,
        multimodal_embedding_model_resolver=dynamic_embeddings.multimodal,
    )
    evidence_assessor = CosineEvidenceAssessmentStep(
        resolved_settings.retrieval.min_cosine_similarity,
        resolved_settings.retrieval.min_rerank_score,
        resolved_settings.retrieval.cross_modal_min_cosine_similarity,
    )
    index_asset_service = IndexAssetService(
        unit_of_work,
        access_policy,
        asset_store,
    )
    visual_evidence_preparer = VisualEvidencePreparationStep(
        index_asset_service,
        max_images=(chat_settings.max_visual_images if chat_settings else 2),
        max_image_bytes=(chat_settings.max_visual_image_bytes if chat_settings else 5_242_880),
        max_total_bytes=(chat_settings.max_visual_total_bytes if chat_settings else 12_582_912),
        max_pixels=(chat_settings.max_visual_pixels if chat_settings else 16_000_000),
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
    adaptive_evidence_assessor = AdaptiveEvidenceAssessmentStep(
        resolved_settings.retrieval.min_cosine_similarity,
        resolved_settings.retrieval.min_rerank_score,
        resolved_settings.retrieval.cross_modal_min_cosine_similarity,
    )
    retrieval_agent = RetrievalAgentService(
        chat_model_adapter,
        evidence_retriever,
        min_cosine_similarity=resolved_settings.retrieval.min_cosine_similarity,
        min_rerank_score=resolved_settings.retrieval.min_rerank_score,
        cross_modal_min_cosine_similarity=(
            resolved_settings.retrieval.cross_modal_min_cosine_similarity
        ),
    )
    workflow_router = AutoWorkflowRouter(
        chat_model_adapter,
        evidence_retriever,
        ChatWorkflowStateStore(unit_of_work),
    )
    chat_runner = LangGraphRunner(
        context_loader,
        evidence_retriever,
        evidence_assessor,
        answer_generator,
        structure_validator,
        result_persister,
        visual_evidence_preparer=visual_evidence_preparer,
        query_contextualizer=query_contextualizer,
        retrieval_agent=retrieval_agent,
        workflow_router=workflow_router,
        adaptive_evidence_assessor=adaptive_evidence_assessor,
        progress_sink=chat_preview_sink,
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
        adaptive_evidence_assessor=adaptive_evidence_assessor,
        retrieval_agent=retrieval_agent,
        workflow_router=workflow_router,
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


def _chat_model_loader(
    unit_of_work: SqlAlchemyUnitOfWorkFactory,
    secret_store: LocalModelSecretStore,
):
    async def load(revision_id):
        async def resolve(uow: UnitOfWork):
            bundle = await uow.model_settings.get_profile_revision(revision_id)
            if bundle is None:
                raise ChatModelExecutionError(
                    ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                    diagnostic={"check": "model_profile_revision"},
                )
            if (
                bundle.profile.kind is not ModelKind.CHAT
                or not bundle.profile.enabled
                or not bundle.provider.enabled
                or bundle.current_revision.validation_status
                is not ModelValidationStatus.VALID
            ):
                raise ChatModelExecutionError(
                    ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                    diagnostic={"check": "model_profile_state"},
                )
            return bundle

        bundle = await execute_in_transaction(
            unit_of_work,
            resolve,
            purpose=UnitOfWorkPurpose.REQUEST,
        )
        try:
            api_key = await asyncio.to_thread(
                secret_store.read,
                bundle.provider_revision.secret_reference,
            )
        except (OSError, ValueError) as error:
            raise ChatModelExecutionError(
                ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                diagnostic={"check": "model_provider_secret"},
            ) from error
        parameters = dict(bundle.current_revision.configuration)
        return LangChainChatModelAdapter(
            base_url=bundle.provider_revision.base_url,
            api_key=api_key,
            model=bundle.current_revision.model,
            timeout_seconds=bundle.provider_revision.timeout_seconds,
            max_retries=bundle.provider_revision.max_retries,
            max_concurrency=bundle.provider_revision.max_concurrency,
            temperature=parameters.get("temperature", 0.2),
            top_p=parameters.get("top_p", 0.9),
            sampling_top_k=parameters.get("sampling_top_k", 40),
            max_tokens=parameters.get("max_output_tokens", 4096),
            structured_output_mode=parameters.get(
                "structured_output_mode", "json_object"
            ),
            reasoning_effort=parameters.get("reasoning_effort", "off"),
            thinking_enabled=parameters.get("reasoning_effort", "off") != "off",
        )

    return load
