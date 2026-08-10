"""Application services for claimed LangGraph chat execution."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from rag_kb.auth import AuthContext
from rag_kb.domain import (
    ChatExecutionCommand,
    ChatExecutionContext,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ChatRunLease,
    ChatWorkflowState,
    ContextualizedQuery,
    ErrorCode,
    Evidence,
    EvidencePack,
    ResourceNotFoundError,
    RetrievalRequest,
    RetrievalExecutionError,
    RetrievalStrategy,
    ReconciliationResult,
)
from rag_kb.retrieval import RetrievalService
from rag_kb.retrieval.profile import parse_retrieval_snapshot
from rag_kb.uow import (
    UnitOfWork,
    UnitOfWorkFactory,
    UnitOfWorkPurpose,
    execute_in_transaction,
)


class ChatPipelineStep(Protocol):
    async def run(self, state: ChatPipelineState) -> ChatPipelineState: ...


class ChatRunCoordinator:
    """Keep claim and lease-CAS operations inside short database transactions."""

    def __init__(self, unit_of_work: UnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def claim(
        self, *, worker_id: str, observed_at: datetime, max_attempts: int
    ) -> ChatRunLease | None:
        async def persist(uow: UnitOfWork) -> ChatRunLease | None:
            return await uow.chat.claim_run(
                worker_id=worker_id,
                observed_at=observed_at,
                max_attempts=max_attempts,
            )

        return await execute_in_transaction(
            self._unit_of_work, persist, purpose=UnitOfWorkPurpose.CLAIM
        )

    async def heartbeat(
        self, lease: ChatRunLease, *, observed_at: datetime
    ) -> bool:
        async def persist(uow: UnitOfWork) -> bool:
            return await uow.chat.heartbeat_run(lease, observed_at=observed_at)

        return await execute_in_transaction(
            self._unit_of_work, persist, purpose=UnitOfWorkPurpose.HEARTBEAT
        )

    async def reconcile_stale(
        self,
        *,
        stale_before: datetime,
        observed_at: datetime,
        max_attempts: int,
        retry_at_by_attempt: tuple[datetime, ...],
        limit: int,
    ) -> ReconciliationResult:
        async def persist(uow: UnitOfWork) -> ReconciliationResult:
            return await uow.chat.reconcile_stale_runs(
                stale_before=stale_before,
                observed_at=observed_at,
                max_attempts=max_attempts,
                retry_at_by_attempt=retry_at_by_attempt,
                limit=limit,
            )

        return await execute_in_transaction(
            self._unit_of_work,
            persist,
            purpose=UnitOfWorkPurpose.RECONCILIATION,
        )


class ChatExecutionContextLoader:
    def __init__(self, unit_of_work: UnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def load(self, command: ChatExecutionCommand) -> ChatExecutionContext:
        async def load(uow: UnitOfWork) -> ChatExecutionContext | None:
            return await uow.chat.load_execution_context(command.lease)

        context = await execute_in_transaction(
            self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST
        )
        if context is None:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.LOAD_CONTEXT,
                diagnostic={"check": "active_lease"},
            )
        return context


class ChatContextualizedQueryStore:
    """Persist a query artifact using the active ChatRun lease."""

    def __init__(self, unit_of_work: UnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def persist(
        self,
        context: ChatExecutionContext,
        value: ContextualizedQuery,
    ) -> ContextualizedQuery | None:
        async def persist(uow: UnitOfWork) -> ContextualizedQuery | None:
            return await uow.chat.save_contextualized_query(
                context.lease, value
            )

        return await execute_in_transaction(self._unit_of_work, persist)


class ChatWorkflowStateStore:
    """Persist an Auto resolution using the active lease/attempt CAS."""

    def __init__(self, unit_of_work: UnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def persist_resolution(
        self,
        context: ChatExecutionContext,
        value: ChatWorkflowState,
    ) -> ChatWorkflowState | None:
        async def persist(uow: UnitOfWork) -> ChatWorkflowState | None:
            return await uow.chat.save_workflow_resolution(
                context.lease, value
            )

        return await execute_in_transaction(self._unit_of_work, persist)


class ChatEvidenceRetriever:
    def __init__(self, retrieval: RetrievalService) -> None:
        self._retrieval = retrieval

    async def retrieve(
        self,
        context: ChatExecutionContext,
        query_context: ContextualizedQuery | None = None,
    ) -> EvidencePack:
        try:
            if query_context is None:
                query = context.query
            elif query_context.standalone_query is None:
                raise ValueError
            else:
                query = query_context.standalone_query
            return await self.retrieve_query(context, query)
        except ChatPipelineExecutionError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "retrieval_snapshot"},
            ) from error

    async def retrieve_query(
        self,
        context: ChatExecutionContext,
        query: str,
        *,
        top_k_override: int | None = None,
    ) -> EvidencePack:
        try:
            strategy, top_k, rerank_mode = parse_retrieval_snapshot(
                context.retrieval_strategy,
            )
            if top_k_override is not None:
                if not 1 <= top_k_override <= top_k:
                    raise ValueError
                top_k = top_k_override
            request = RetrievalRequest(
                knowledge_base_id=context.knowledge_base_id,
                query=query,
                top_k=top_k,
                strategy=strategy,
                rerank_mode=rerank_mode,
                include_debug=True,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "retrieval_snapshot"},
            ) from error
        try:
            pack = await self._retrieval.retrieve(
                AuthContext(
                    principal_id=context.principal_id,
                    client_id=context.client_id,
                    workspace_id=context.workspace_id,
                ),
                request,
            )
        except RetrievalExecutionError as error:
            raise ChatPipelineExecutionError(
                error.code,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic=error.diagnostic,
            ) from error
        if (
            pack.knowledge_base_id != context.knowledge_base_id
            or pack.index_revision_id != context.index_revision_id
        ):
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_REVISION_MISMATCH,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "frozen_revision"},
            )
        return EvidencePack(
            knowledge_base_id=pack.knowledge_base_id,
            index_revision_id=pack.index_revision_id,
            strategy=pack.strategy,
            evidence=pack.evidence,
            debug=pack.debug,
        )

    async def retrieve_adjacent(
        self,
        context: ChatExecutionContext,
        anchors: tuple[Evidence, ...],
    ) -> tuple[Evidence, ...]:
        if not anchors:
            return ()
        try:
            evidence = await self._retrieval.retrieve_adjacent_evidence(
                AuthContext(
                    principal_id=context.principal_id,
                    client_id=context.client_id,
                    workspace_id=context.workspace_id,
                ),
                knowledge_base_id=context.knowledge_base_id,
                index_revision_id=context.index_revision_id,
                anchors=anchors,
            )
        except ResourceNotFoundError as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_REVISION_MISMATCH,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "adjacency_frozen_revision"},
            ) from error
        except RetrievalExecutionError as error:
            code = (
                ErrorCode.CHAT_REVISION_MISMATCH
                if error.diagnostic.get("check")
                == "adjacency_frozen_revision"
                else error.code
            )
            raise ChatPipelineExecutionError(
                code,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic=error.diagnostic,
            ) from error
        anchor_ids = {item.index_chunk_id for item in anchors}
        if any(
            item.index_revision_id != context.index_revision_id
            or item.adjacency_anchor_index_chunk_id not in anchor_ids
            or item.adjacency_offset not in {-1, 1}
            for item in evidence
        ):
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_REVISION_MISMATCH,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "adjacency_frozen_scope"},
            )
        return evidence
