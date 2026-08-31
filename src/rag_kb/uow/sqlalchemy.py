"""SQLAlchemy implementation of the asynchronous transaction boundary."""

from __future__ import annotations

from types import TracebackType
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_kb.repositories.sqlalchemy import SqlAlchemyWorkspaceRepository
from rag_kb.repositories.sqlalchemy_chat import SqlAlchemyChatRepository
from rag_kb.repositories.sqlalchemy_content import (
    SqlAlchemyContentMutationRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyFileConsistencyRepository,
    SqlAlchemyKnowledgeBaseRepository,
)
from rag_kb.repositories.sqlalchemy_graph import SqlAlchemyGraphRepository
from rag_kb.repositories.sqlalchemy_indexing import SqlAlchemyIndexingRepository
from rag_kb.repositories.sqlalchemy_model_settings import (
    SqlAlchemyModelSettingsRepository,
)
from rag_kb.uow.mode import TransactionMode


class SqlAlchemyUnitOfWork:
    """One SQLAlchemy session with one explicit transaction."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        workspace_id: UUID,
        mode: TransactionMode,
    ) -> None:
        self.workspace_id = workspace_id
        self.mode = mode
        self._sessions = sessions
        self._session: AsyncSession | None = None
        self._workspace_repository: SqlAlchemyWorkspaceRepository | None = None
        self._knowledge_base_repository: SqlAlchemyKnowledgeBaseRepository | None = None
        self._document_repository: SqlAlchemyDocumentRepository | None = None
        self._chat_repository: SqlAlchemyChatRepository | None = None
        self._content_mutation_repository: SqlAlchemyContentMutationRepository | None = None
        self._file_consistency_repository: SqlAlchemyFileConsistencyRepository | None = None
        self._indexing_repository: SqlAlchemyIndexingRepository | None = None
        self._model_settings_repository: SqlAlchemyModelSettingsRepository | None = None
        self._graph_repository: SqlAlchemyGraphRepository | None = None

    @property
    def workspaces(self) -> SqlAlchemyWorkspaceRepository:
        assert self._workspace_repository is not None
        return self._workspace_repository

    @property
    def knowledge_bases(self) -> SqlAlchemyKnowledgeBaseRepository:
        assert self._knowledge_base_repository is not None
        return self._knowledge_base_repository

    @property
    def documents(self) -> SqlAlchemyDocumentRepository:
        assert self._document_repository is not None
        return self._document_repository

    @property
    def chat(self) -> SqlAlchemyChatRepository:
        assert self._chat_repository is not None
        return self._chat_repository

    @property
    def content_mutations(self) -> SqlAlchemyContentMutationRepository:
        assert self._content_mutation_repository is not None
        return self._content_mutation_repository

    @property
    def file_consistency(self) -> SqlAlchemyFileConsistencyRepository:
        assert self._file_consistency_repository is not None
        return self._file_consistency_repository

    @property
    def indexing(self) -> SqlAlchemyIndexingRepository:
        assert self._indexing_repository is not None
        return self._indexing_repository

    @property
    def model_settings(self) -> SqlAlchemyModelSettingsRepository:
        assert self._model_settings_repository is not None
        return self._model_settings_repository

    @property
    def graph(self) -> SqlAlchemyGraphRepository:
        assert self._graph_repository is not None
        return self._graph_repository

    async def __aenter__(self) -> SqlAlchemyUnitOfWork:
        session = self._sessions()
        # Repositories only run inside this explicit boundary.  Disabling
        # autobegin also prevents a post-commit repository call from silently
        # opening a second transaction on the same session.
        session.sync_session.autobegin = False
        self._session = session
        try:
            await session.begin()
            if self.mode is TransactionMode.REPEATABLE_READ_ONLY:
                await session.execute(
                    text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                )
            self._workspace_repository = SqlAlchemyWorkspaceRepository(
                session, self.workspace_id
            )
            self._knowledge_base_repository = SqlAlchemyKnowledgeBaseRepository(
                session, self.workspace_id
            )
            self._document_repository = SqlAlchemyDocumentRepository(
                session, self.workspace_id
            )
            self._chat_repository = SqlAlchemyChatRepository(session, self.workspace_id)
            self._content_mutation_repository = SqlAlchemyContentMutationRepository(
                session, self.workspace_id
            )
            self._file_consistency_repository = SqlAlchemyFileConsistencyRepository(
                session, self.workspace_id
            )
            self._indexing_repository = SqlAlchemyIndexingRepository(
                session, self.workspace_id
            )
            self._model_settings_repository = SqlAlchemyModelSettingsRepository(
                session, self.workspace_id
            )
            self._graph_repository = SqlAlchemyGraphRepository(
                session, self.workspace_id
            )
        except BaseException:
            await session.rollback()
            await session.close()
            self._session = None
            raise
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        session = self._session
        if session is None:
            return
        try:
            if session.in_transaction():
                await session.rollback()
        finally:
            await session.close()

    async def commit(self) -> None:
        session = self._session
        assert session is not None
        try:
            await session.commit()
        except BaseException:
            await session.rollback()
            raise

    async def rollback(self) -> None:
        session = self._session
        assert session is not None
        await session.rollback()


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
        mode: TransactionMode = TransactionMode.READ_WRITE,
    ) -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(
            self._sessions,
            workspace_id=self._workspace_id,
            mode=mode,
        )
