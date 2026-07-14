"""SQLAlchemy implementation of the asynchronous Unit of Work."""

from __future__ import annotations

import asyncio
from enum import StrEnum
from types import TracebackType
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, AsyncSessionTransaction, async_sessionmaker

from rag_kb.repositories import (
    ContentMutationRepository,
    DocumentRepository,
    FileConsistencyRepository,
    KnowledgeBaseRepository,
    WorkspaceRepository,
)
from rag_kb.repositories.sqlalchemy_content import (
    SqlAlchemyContentMutationRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyFileConsistencyRepository,
    SqlAlchemyKnowledgeBaseRepository,
)
from rag_kb.repositories.sqlalchemy import SqlAlchemyWorkspaceRepository
from rag_kb.uow.contracts import (
    TransactionMode,
    UnitOfWorkConcurrencyError,
    UnitOfWorkPurpose,
    UnitOfWorkStateError,
)


class _State(StrEnum):
    NEW = "new"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    CLOSED = "closed"


class SqlAlchemyUnitOfWork:
    """One session and one transaction, owned by one asyncio task."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        workspace_id: UUID,
        purpose: UnitOfWorkPurpose,
        mode: TransactionMode,
    ) -> None:
        self.workspace_id = workspace_id
        self.purpose = purpose
        self.mode = mode
        self._sessions = sessions
        self._session: AsyncSession | None = None
        self._transaction: AsyncSessionTransaction | None = None
        self._workspace_repository: WorkspaceRepository | None = None
        self._knowledge_base_repository: KnowledgeBaseRepository | None = None
        self._document_repository: DocumentRepository | None = None
        self._content_mutation_repository: ContentMutationRepository | None = None
        self._file_consistency_repository: FileConsistencyRepository | None = None
        self._owner_task: asyncio.Task[object] | None = None
        self._state = _State.NEW

    @property
    def workspaces(self) -> WorkspaceRepository:
        self._ensure_active()
        assert self._workspace_repository is not None
        return self._workspace_repository

    @property
    def knowledge_bases(self) -> KnowledgeBaseRepository:
        self._ensure_active()
        assert self._knowledge_base_repository is not None
        return self._knowledge_base_repository

    @property
    def documents(self) -> DocumentRepository:
        self._ensure_active()
        assert self._document_repository is not None
        return self._document_repository

    @property
    def content_mutations(self) -> ContentMutationRepository:
        self._ensure_active()
        assert self._content_mutation_repository is not None
        return self._content_mutation_repository

    @property
    def file_consistency(self) -> FileConsistencyRepository:
        self._ensure_active()
        assert self._file_consistency_repository is not None
        return self._file_consistency_repository

    async def __aenter__(self) -> SqlAlchemyUnitOfWork:
        if self._state is not _State.NEW:
            raise UnitOfWorkStateError("a Unit of Work instance is single-use")

        owner_task = asyncio.current_task()
        if owner_task is None:
            raise UnitOfWorkStateError("Unit of Work requires an asyncio task")

        session = self._sessions()
        self._owner_task = owner_task
        self._session = session
        self._state = _State.ACTIVE
        self._workspace_repository = SqlAlchemyWorkspaceRepository(
            session,
            self.workspace_id,
            self._ensure_active,
        )
        self._knowledge_base_repository = SqlAlchemyKnowledgeBaseRepository(
            session, self.workspace_id, self._ensure_active
        )
        self._document_repository = SqlAlchemyDocumentRepository(
            session, self.workspace_id, self._ensure_active
        )
        self._content_mutation_repository = SqlAlchemyContentMutationRepository(
            session, self.workspace_id, self._ensure_active
        )
        self._file_consistency_repository = SqlAlchemyFileConsistencyRepository(
            session, self.workspace_id, self._ensure_active
        )
        try:
            self._transaction = await session.begin()
            if self.mode is TransactionMode.REPEATABLE_READ_ONLY:
                await session.execute(
                    text(
                        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                    )
                )
        except BaseException:
            await session.close()
            self._state = _State.CLOSED
            raise
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._ensure_owner()
        session = self._require_session()
        try:
            if self._state is _State.ACTIVE:
                await self._rollback_active()
        finally:
            await session.close()
            self._state = _State.CLOSED

    async def commit(self) -> None:
        self._ensure_active()
        transaction = self._require_transaction()
        try:
            await transaction.commit()
        except BaseException:
            await self._require_session().rollback()
            self._state = _State.ROLLED_BACK
            raise
        self._state = _State.COMMITTED

    async def rollback(self) -> None:
        self._ensure_active()
        await self._rollback_active()

    async def _rollback_active(self) -> None:
        transaction = self._require_transaction()
        if transaction.is_active:
            await transaction.rollback()
        self._state = _State.ROLLED_BACK

    def _ensure_active(self) -> None:
        self._ensure_owner()
        if self._state is not _State.ACTIVE:
            raise UnitOfWorkStateError(
                f"Unit of Work transaction is not active: {self._state.value}"
            )

    def _ensure_owner(self) -> None:
        if self._owner_task is not None and asyncio.current_task() is not self._owner_task:
            raise UnitOfWorkConcurrencyError(
                "an AsyncSession cannot be shared with another asyncio task"
            )

    def _require_session(self) -> AsyncSession:
        if self._session is None:
            raise UnitOfWorkStateError("Unit of Work has not been entered")
        return self._session

    def _require_transaction(self) -> AsyncSessionTransaction:
        if self._transaction is None:
            raise UnitOfWorkStateError("Unit of Work transaction has not started")
        return self._transaction


class SqlAlchemyUnitOfWorkFactory:
    """Create a fresh Unit of Work for every operation boundary."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        workspace_id: UUID,
    ) -> None:
        self._sessions = sessions
        self._workspace_id = workspace_id

    def __call__(
        self,
        *,
        purpose: UnitOfWorkPurpose = UnitOfWorkPurpose.COMMAND,
        mode: TransactionMode = TransactionMode.READ_WRITE,
    ) -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(
            self._sessions,
            workspace_id=self._workspace_id,
            purpose=purpose,
            mode=mode,
        )
