"""Worker composition root for process-wide foundation dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
from uuid import uuid4

from rag_kb.auth import DevelopmentAuthProvider, SingleWorkspaceAccessPolicy
from rag_kb.adapters import (
    ChatModelAdapter,
    FixedPgVectorSpace,
    IsolatedPlainTextProcessor,
    LangChainChatModelAdapter,
    LocalFileStore,
    OpenAICompatibleChatModelAdapter,
    OpenAICompatibleEmbeddingProvider,
    PgVectorStore,
)
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
from rag_kb.indexing import IndexingPipeline
from rag_kb.scheduling import (
    ChatRunScheduler,
    FairWorkerScheduler,
    IndexingJobScheduler,
    RetryPolicy,
    WeightedLaneSelector,
)
from rag_kb.services import (
    AnswerGenerationStep,
    AnswerStructureValidationStep,
    ChatEvidenceRetriever,
    ChatExecutionContextLoader,
    ChatFailureSettlementService,
    ChatResultPersistenceStep,
    ChatRunCoordinator,
    DirectChatPipeline,
    EvidenceAssessmentStep,
    FileReconciliationService,
    ParserLimits,
    RetrievalService,
    build_content_services,
    embedding_space_definition,
)
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory
from rag_kb.workflows import DirectGraphRunner, GraphRunner, LangGraphRunner


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
    reconciliation_service: FileReconciliationService
    document_processor: IsolatedPlainTextProcessor
    embedding_provider: OpenAICompatibleEmbeddingProvider
    vector_store: PgVectorStore
    retrieval_service: RetrievalService
    chat_model_adapter: ChatModelAdapter
    evidence_assessor: EvidenceAssessmentStep
    answer_generator: AnswerGenerationStep
    structure_validator: AnswerStructureValidationStep
    result_persister: ChatResultPersistenceStep
    failure_settler: ChatFailureSettlementService
    chat_pipeline: DirectChatPipeline
    chat_runner: GraphRunner
    chat_scheduler: ChatRunScheduler
    indexing_pipeline: IndexingPipeline
    indexing_scheduler: IndexingJobScheduler
    lane_selector: WeightedLaneSelector
    worker_scheduler: FairWorkerScheduler

    async def close(self) -> None:
        """Release process-owned database resources during Worker shutdown."""

        await self.database.close()

    async def start(self) -> RuntimeReadiness:
        """Fail startup when the migration-created runtime is incompatible."""

        return await self.check_readiness()

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
    )
    unit_of_work = SqlAlchemyUnitOfWorkFactory(
        database.sessions,
        identity.workspace_id,
    )
    content_services = build_content_services(
        unit_of_work,
        access_policy,
        resolved_settings.model_provider.embedding,
    )
    file_store = LocalFileStore(
        resolved_settings.file_store.staging_path,
        resolved_settings.file_store.final_path,
    )
    document_processor = IsolatedPlainTextProcessor(
        ParserLimits(
            max_chunks=resolved_settings.parser.max_chunks,
            wall_seconds=resolved_settings.parser.wall_seconds,
            cpu_seconds=resolved_settings.parser.cpu_seconds,
            memory_bytes=resolved_settings.parser.memory_bytes,
        )
    )
    embedding_settings = resolved_settings.model_provider.embedding
    embedding_space = embedding_space_definition(embedding_settings)
    embedding_provider = OpenAICompatibleEmbeddingProvider(
        base_url=str(embedding_settings.base_url),
        api_key=embedding_settings.api_key.get_secret_value(),
        embedding_space=embedding_space,
        max_batch_size=embedding_settings.max_batch_size,
        timeout_seconds=embedding_settings.timeout_seconds,
        max_retries=embedding_settings.max_retries,
        max_concurrency=embedding_settings.max_concurrency,
    )
    chat_settings = resolved_settings.model_provider.chat
    chat_adapter_arguments = {
        "base_url": str(chat_settings.base_url),
        "api_key": chat_settings.api_key.get_secret_value(),
        "model": chat_settings.model,
        "timeout_seconds": chat_settings.timeout_seconds,
        "max_retries": chat_settings.max_retries,
        "max_concurrency": chat_settings.max_concurrency,
    }
    if resolved_settings.model_adapter_backend == "langchain":
        chat_model_adapter = LangChainChatModelAdapter(
            **chat_adapter_arguments,
            structured_output_mode=chat_settings.structured_output_mode,
        )
    else:
        chat_model_adapter = OpenAICompatibleChatModelAdapter(
            **chat_adapter_arguments,
        )
    indexing_pipeline = IndexingPipeline(
        unit_of_work,
        file_store,
        document_processor,
        embedding_provider,
        FixedPgVectorSpace(embedding_space),
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
        FixedPgVectorSpace(embedding_space),
    )
    retrieval_service = RetrievalService(
        access_policy,
        embedding_provider,
        vector_store,
    )
    evidence_assessor = EvidenceAssessmentStep(chat_model_adapter)
    answer_generator = AnswerGenerationStep(chat_model_adapter)
    structure_validator = AnswerStructureValidationStep(chat_model_adapter)
    result_persister = ChatResultPersistenceStep(unit_of_work)
    failure_settler = ChatFailureSettlementService(
        unit_of_work,
        max_attempts=poller.max_attempts,
        base_delay_seconds=poller.retry_base_delay_seconds,
        max_delay_seconds=poller.retry_max_delay_seconds,
    )
    chat_coordinator = ChatRunCoordinator(unit_of_work)
    context_loader = ChatExecutionContextLoader(unit_of_work)
    evidence_retriever = ChatEvidenceRetriever(retrieval_service)
    chat_pipeline = DirectChatPipeline(
        context_loader,
        evidence_retriever,
        evidence_assessor,
        answer_generator,
        structure_validator,
        result_persister,
        deadline_seconds=poller.chat_deadline_seconds,
    )
    if resolved_settings.chat_workflow_backend == "langgraph":
        chat_runner = LangGraphRunner(
            context_loader,
            evidence_retriever,
            evidence_assessor,
            answer_generator,
            structure_validator,
            result_persister,
            deadline_seconds=poller.chat_deadline_seconds,
        )
    else:
        chat_runner = DirectGraphRunner(chat_pipeline)
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
        concurrency=poller.indexing_concurrency,
        poll_interval_seconds=poller.poll_interval_seconds,
        heartbeat_interval_seconds=poller.heartbeat_interval_seconds,
        stale_after_seconds=poller.stale_after_seconds,
        deadline_seconds=poller.indexing_deadline_seconds,
        retry_policy=retry_policy,
        reconciliation_batch_size=poller.reconciliation_batch_size,
    )
    lane_selector = WeightedLaneSelector(
        chat_weight=poller.chat_weight,
        indexing_weight=poller.indexing_weight,
        aging_seconds=poller.aging_seconds,
    )
    worker_scheduler = FairWorkerScheduler(
        chat_scheduler,
        indexing_scheduler,
        lane_selector,
        chat_concurrency=poller.chat_concurrency,
        indexing_concurrency=poller.indexing_concurrency,
        poll_interval_seconds=poller.poll_interval_seconds,
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
        document_processor=document_processor,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        retrieval_service=retrieval_service,
        chat_model_adapter=chat_model_adapter,
        evidence_assessor=evidence_assessor,
        answer_generator=answer_generator,
        structure_validator=structure_validator,
        result_persister=result_persister,
        failure_settler=failure_settler,
        chat_pipeline=chat_pipeline,
        chat_runner=chat_runner,
        chat_scheduler=chat_scheduler,
        indexing_pipeline=indexing_pipeline,
        indexing_scheduler=indexing_scheduler,
        lane_selector=lane_selector,
        worker_scheduler=worker_scheduler,
    )


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
