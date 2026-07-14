"""Worker composition root for process-wide foundation dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_kb.auth import DevelopmentAuthProvider, SingleWorkspaceAccessPolicy
from rag_kb.adapters import (
    FixedPgVectorSpace,
    IsolatedPlainTextProcessor,
    LocalFileStore,
    OpenAICompatibleEmbeddingProvider,
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
from rag_kb.services import (
    FileReconciliationService,
    ParserLimits,
    build_content_services,
    embedding_space_definition,
)
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


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
    indexing_pipeline: IndexingPipeline

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
        indexing_pipeline=IndexingPipeline(
            unit_of_work,
            file_store,
            document_processor,
            embedding_provider,
            FixedPgVectorSpace(embedding_space),
        ),
    )
