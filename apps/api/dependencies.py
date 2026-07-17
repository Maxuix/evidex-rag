"""API composition root for process-wide foundation dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_kb.auth import DevelopmentAuthProvider, SingleWorkspaceAccessPolicy
from rag_kb.adapters import (
    EmbeddingModelAdapter,
    FixedPgVectorSpace,
    LangChainEmbeddingModelAdapter,
    LocalFileStore,
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
from rag_kb.services import (
    AdmissionLimits,
    ChatSseConnectionLimiter,
    ChatService,
    ChatTerminalWatcher,
    DocumentService,
    FileAdmissionService,
    IndexingJobService,
    KnowledgeBaseService,
    SourceFileService,
    build_content_services,
    chat_model_configuration,
    embedding_space_definition,
)
from rag_kb.retrieval import RetrievalService
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
    source_file_service: SourceFileService
    file_admission_service: FileAdmissionService
    indexing_job_service: IndexingJobService
    embedding_provider: EmbeddingModelAdapter
    vector_store: PgVectorStore
    retrieval_service: RetrievalService
    chat_service: ChatService
    chat_terminal_watcher: ChatTerminalWatcher
    chat_sse_connection_limiter: ChatSseConnectionLimiter

    async def close(self) -> None:
        """Release process-owned database resources during API shutdown."""

        await self.database.close()

    async def start(self) -> RuntimeReadiness:
        """Fail startup when the migration-created runtime is incompatible."""

        return await self.check_readiness()

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
    vector_store = PgVectorStore(
        database.sessions,
        FixedPgVectorSpace(embedding_space),
    )
    content_services = build_content_services(unit_of_work, access_policy, embedding)
    file_store = LocalFileStore(
        resolved_settings.file_store.staging_path,
        resolved_settings.file_store.final_path,
    )
    chat_service = ChatService(
        unit_of_work,
        access_policy,
        model_configuration=chat_model_configuration(
            resolved_settings.model_provider.chat
        ),
    )
    chat_delivery = resolved_settings.chat_delivery
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
        source_file_service=SourceFileService(
            content_services.documents,
            file_store,
        ),
        file_admission_service=FileAdmissionService(
            AdmissionLimits(
                max_bytes=resolved_settings.file_admission.max_bytes,
                max_lines=resolved_settings.file_admission.max_lines,
            )
        ),
        indexing_job_service=IndexingJobService(unit_of_work, access_policy),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        retrieval_service=RetrievalService(
            access_policy,
            embedding_provider,
            vector_store,
        ),
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
    )
