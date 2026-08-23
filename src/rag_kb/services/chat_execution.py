"""Application services for claimed native Agent execution."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from rag_kb.auth import AuthContext
from rag_kb.domain import (
    ChatExecutionCommand,
    ChatExecutionContext,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ChatRunLease,
    ErrorCode,
    Evidence,
    EvidencePack,
    GraphitiSupplementResult,
    GraphRetrievalRequest,
    RetrievalRequest,
    RetrievalExecutionError,
    RetrievalStrategy,
    ReconciliationResult,
)
from rag_kb.retrieval import RetrievalService
from rag_kb.retrieval.profile import parse_chat_retrieval_snapshot
from rag_kb.uow import (
    UnitOfWork,
    UnitOfWorkFactory,
    UnitOfWorkPurpose,
    execute_in_transaction,
)


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


class ChatEvidenceRetriever:
    def __init__(self, retrieval: RetrievalService) -> None:
        self._retrieval = retrieval

    async def retrieve(self, context: ChatExecutionContext) -> EvidencePack:
        return await self.retrieve_query(context, context.query)

    async def retrieve_query(
        self,
        context: ChatExecutionContext,
        query: str,
        *,
        top_k_override: int | None = None,
    ) -> EvidencePack:
        try:
            strategy, top_k, rerank_mode, execution_type = parse_chat_retrieval_snapshot(
                context.retrieval_strategy,
            )
            if top_k_override is not None:
                if not 1 <= top_k_override <= top_k:
                    raise ValueError
                top_k = top_k_override
            request = (
                GraphRetrievalRequest(
                    knowledge_base_id=context.knowledge_base_id,
                    query=query,
                    top_k=top_k,
                    rerank_mode=rerank_mode,
                    include_debug=True,
                )
                if execution_type == "manual_graph"
                else RetrievalRequest(
                    knowledge_base_id=context.knowledge_base_id,
                    query=query,
                    top_k=top_k,
                    strategy=RetrievalStrategy.EXACT_VECTOR
                    if execution_type == "adaptive_graphiti"
                    else strategy,
                    rerank_mode=rerank_mode,
                    include_debug=True,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "retrieval_snapshot"},
            ) from error
        try:
            auth_context = AuthContext(
                principal_id=context.principal_id,
                client_id=context.client_id,
                workspace_id=context.workspace_id,
            )
            pack = (
                await self._retrieval.retrieve_graph(auth_context, request)
                if execution_type == "manual_graph"
                else await self._retrieval.retrieve(auth_context, request)
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

    async def retrieve_graphiti_supplement(
        self,
        context: ChatExecutionContext,
        query: str,
        *,
        excluded_index_chunk_ids: tuple[UUID, ...],
    ) -> GraphitiSupplementResult:
        try:
            _, _, rerank_mode, execution_type = parse_chat_retrieval_snapshot(
                context.retrieval_strategy,
            )
            if execution_type != "adaptive_graphiti":
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "adaptive_retrieval_snapshot"},
            ) from error
        try:
            result = await self._retrieval.retrieve_graphiti_supplement(
                AuthContext(
                    principal_id=context.principal_id,
                    client_id=context.client_id,
                    workspace_id=context.workspace_id,
                ),
                knowledge_base_id=context.knowledge_base_id,
                index_revision_id=context.index_revision_id,
                query=query,
                rerank_mode=rerank_mode,
                excluded_index_chunk_ids=excluded_index_chunk_ids,
            )
        except RetrievalExecutionError as error:
            raise ChatPipelineExecutionError(
                error.code,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic=error.diagnostic,
            ) from error
        evidence_ids = tuple(item.index_chunk_id for item in result.evidence)
        new_evidence_ids = tuple(result.new_index_chunk_ids or ())
        excluded_ids = set(excluded_index_chunk_ids)
        if (
            len(result.evidence) > 4
            or result.route_result_code == "admitted" and not result.evidence
            or any(
                item.index_revision_id != context.index_revision_id
                for item in result.evidence
            )
            or len(evidence_ids) != len(set(evidence_ids))
            or any(item not in evidence_ids for item in new_evidence_ids)
            or any(item in excluded_ids for item in new_evidence_ids)
        ):
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_REVISION_MISMATCH,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "adaptive_supplement_evidence"},
            )
        return result
