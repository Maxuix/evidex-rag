"""Public asynchronous Unit of Work contracts."""

from __future__ import annotations

from enum import StrEnum
from types import TracebackType
from typing import Protocol
from uuid import UUID

from rag_kb.repositories import (
    ChatRepository,
    ContentMutationRepository,
    DocumentRepository,
    FileConsistencyRepository,
    GraphRepository,
    IndexingRepository,
    KnowledgeBaseRepository,
    ModelSettingsRepository,
    WorkspaceRepository,
)


class TransactionMode(StrEnum):
    READ_WRITE = "read_write"
    REPEATABLE_READ_ONLY = "repeatable_read_only"


class UnitOfWorkStateError(RuntimeError):
    """The Unit of Work is being used outside its one transaction."""


class UnitOfWorkConcurrencyError(UnitOfWorkStateError):
    """A child or concurrent task attempted to share the Unit of Work."""


class UnitOfWork(Protocol):
    mode: TransactionMode
    workspace_id: UUID

    @property
    def workspaces(self) -> WorkspaceRepository: ...

    @property
    def knowledge_bases(self) -> KnowledgeBaseRepository: ...

    @property
    def documents(self) -> DocumentRepository: ...

    @property
    def chat(self) -> ChatRepository: ...

    @property
    def content_mutations(self) -> ContentMutationRepository: ...

    @property
    def file_consistency(self) -> FileConsistencyRepository: ...

    @property
    def indexing(self) -> IndexingRepository: ...

    @property
    def model_settings(self) -> ModelSettingsRepository: ...

    @property
    def graph(self) -> GraphRepository: ...

    async def __aenter__(self) -> UnitOfWork: ...

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


class UnitOfWorkFactory(Protocol):
    def __call__(
        self,
        *,
        mode: TransactionMode = TransactionMode.READ_WRITE,
    ) -> UnitOfWork: ...
