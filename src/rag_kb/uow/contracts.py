"""Public asynchronous Unit of Work contracts."""

from __future__ import annotations

from enum import StrEnum
from types import TracebackType
from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_kb.repositories import (
    ContentMutationRepository,
    DocumentRepository,
    EvaluationRepository,
    FileConsistencyRepository,
    IndexingRepository,
    KnowledgeBaseRepository,
    WorkspaceRepository,
)


class UnitOfWorkPurpose(StrEnum):
    REQUEST = "request"
    COMMAND = "command"
    POLL = "poll"
    CLAIM = "claim"
    HEARTBEAT = "heartbeat"
    RECONCILIATION = "reconciliation"
    READ_SNAPSHOT = "read_snapshot"
    INDEXING = "indexing"


class TransactionMode(StrEnum):
    READ_WRITE = "read_write"
    REPEATABLE_READ_ONLY = "repeatable_read_only"


class UnitOfWorkStateError(RuntimeError):
    """The Unit of Work is being used outside its one transaction."""


class UnitOfWorkConcurrencyError(UnitOfWorkStateError):
    """A child or concurrent task attempted to share the Unit of Work."""


@runtime_checkable
class UnitOfWork(Protocol):
    purpose: UnitOfWorkPurpose
    mode: TransactionMode
    workspace_id: UUID

    @property
    def workspaces(self) -> WorkspaceRepository: ...

    @property
    def knowledge_bases(self) -> KnowledgeBaseRepository: ...

    @property
    def documents(self) -> DocumentRepository: ...

    @property
    def content_mutations(self) -> ContentMutationRepository: ...

    @property
    def file_consistency(self) -> FileConsistencyRepository: ...

    @property
    def indexing(self) -> IndexingRepository: ...

    @property
    def evaluations(self) -> EvaluationRepository: ...

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
        purpose: UnitOfWorkPurpose = UnitOfWorkPurpose.COMMAND,
        mode: TransactionMode = TransactionMode.READ_WRITE,
    ) -> UnitOfWork: ...
