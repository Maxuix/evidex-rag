"""Application services for claimed native Agent execution."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from rag_kb.domain import (
    ChatExecutionCommand,
    ChatExecutionContext,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatRunLease,
    ErrorCode,
    Evidence,
    EvidencePack,
    GraphRetrievalRequest,
    GraphSearchResult,
    ResourceNotFoundError,
    RetrievalRequest,
    RetrievalExecutionError,
    RetrievalStrategy,
    ReconciliationResult,
    ServingDocumentList,
)
from rag_kb.retrieval import RetrievalService
from rag_kb.retrieval.profile import (
    parse_adaptive_graphiti_snapshot,
    parse_chat_retrieval_snapshot,
)
from rag_kb.uow import execute_in_transaction

if TYPE_CHECKING:
    from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWork, SqlAlchemyUnitOfWorkFactory


class ChatRunCoordinator:
    """Keep claim and lease-CAS operations inside short database transactions."""

    def __init__(self, unit_of_work: SqlAlchemyUnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def claim(
        self, *, observed_at: datetime, max_attempts: int
    ) -> ChatRunLease | None:
        async def persist(uow: SqlAlchemyUnitOfWork) -> ChatRunLease | None:
            return await uow.chat.claim_run(
                observed_at=observed_at,
                max_attempts=max_attempts,
            )

        return await execute_in_transaction(
            self._unit_of_work, persist
        )

    async def heartbeat(
        self, lease: ChatRunLease, *, observed_at: datetime
    ) -> bool:
        async def persist(uow: SqlAlchemyUnitOfWork) -> bool:
            return await uow.chat.heartbeat_run(lease, observed_at=observed_at)

        return await execute_in_transaction(
            self._unit_of_work, persist
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
        async def persist(uow: SqlAlchemyUnitOfWork) -> ReconciliationResult:
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
        )


class ChatExecutionContextLoader:
    def __init__(self, unit_of_work: SqlAlchemyUnitOfWorkFactory) -> None:
        self._unit_of_work = unit_of_work

    async def load(self, command: ChatExecutionCommand) -> ChatExecutionContext:
        async def load(uow: SqlAlchemyUnitOfWork) -> ChatExecutionContext | None:
            return await uow.chat.load_execution_context(command.lease)

        context = await execute_in_transaction(
            self._unit_of_work, load
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

    async def semantic_search(
        self,
        context: ChatExecutionContext,
        query: str,
        *,
        top_k_override: int | None = None,
    ) -> EvidencePack:
        try:
            _strategy, top_k, rerank_mode, execution_type = parse_chat_retrieval_snapshot(
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
                    strategy=RetrievalStrategy.EXACT_VECTOR,
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
            pack = (
                await self._retrieval.retrieve_graph(request)
                if execution_type == "manual_graph"
                else await self._retrieval.retrieve(request)
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

    async def keyword_search(
        self,
        context: ChatExecutionContext,
        query: str,
        *,
        top_k_override: int | None = None,
    ) -> EvidencePack:
        try:
            _strategy, top_k, _rerank_mode, _execution_type = (
                parse_chat_retrieval_snapshot(context.retrieval_strategy)
            )
            if top_k_override is not None:
                if not 1 <= top_k_override <= top_k:
                    raise ValueError
                top_k = top_k_override
            request = RetrievalRequest(
                knowledge_base_id=context.knowledge_base_id,
                query=query,
                top_k=top_k,
                include_debug=True,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "retrieval_snapshot"},
            ) from error
        try:
            pack = await self._retrieval.retrieve_lexical_only(request)
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

    async def keyword_search_capable(
        self,
        context: ChatExecutionContext,
    ) -> bool:
        try:
            if not self._retrieval.hybrid_request_enabled():
                return False
            status = await self._retrieval.lexical_manifest_status(
                context.knowledge_base_id
            )
        except RetrievalExecutionError:
            return False
        return (
            status is not None
            and status.complete
            and status.resolved_active_revision_id == context.index_revision_id
        )

    async def read_chunk_context(
        self,
        context: ChatExecutionContext,
        anchors: tuple[Evidence, ...],
    ) -> tuple[Evidence, ...]:
        try:
            return await self._retrieval.retrieve_adjacent_evidence(
                knowledge_base_id=context.knowledge_base_id,
                index_revision_id=context.index_revision_id,
                anchors=anchors,
            )
        except RetrievalExecutionError as error:
            raise ChatPipelineExecutionError(
                error.code,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic=error.diagnostic,
            ) from error

    async def list_documents(
        self,
        context: ChatExecutionContext,
    ) -> ServingDocumentList:
        listed = await self._retrieval.list_serving_documents(
            context.knowledge_base_id
        )
        if listed is None:
            raise ResourceNotFoundError(
                "knowledge base or active revision was not found"
            )
        if listed.resolved_active_revision_id != context.index_revision_id:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_REVISION_MISMATCH,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "frozen_revision"},
            )
        return listed

    async def search_graph_relations(
        self,
        context: ChatExecutionContext,
        query: str,
        *,
        excluded_index_chunk_ids: tuple[UUID, ...],
    ) -> GraphSearchResult:
        try:
            profile = parse_adaptive_graphiti_snapshot(
                context.retrieval_strategy,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "adaptive_retrieval_snapshot"},
            ) from error
        try:
            result = await self._retrieval.search_graph_relations(
                knowledge_base_id=context.knowledge_base_id,
                index_revision_id=context.index_revision_id,
                query=query,
                rerank_mode=profile.rerank_mode,
                excluded_index_chunk_ids=excluded_index_chunk_ids,
                edge_limit=profile.graph_edge_limit,
                source_chunk_target=profile.graph_source_chunk_target,
                source_chunk_limit=profile.graph_source_chunk_limit,
                call_timeout_seconds=profile.graph_call_timeout_seconds,
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
            len(result.evidence) > profile.graph_source_chunk_limit
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
                diagnostic={"check": "graph_search_evidence"},
            )
        return result

    async def graph_relations_capable(
        self,
        context: ChatExecutionContext,
    ) -> bool:
        """Read active READY build capability without calling any model."""

        try:
            parse_adaptive_graphiti_snapshot(context.retrieval_strategy)
        except (KeyError, TypeError, ValueError) as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "adaptive_retrieval_snapshot"},
            ) from error
        try:
            return await self._retrieval.search_graph_relations_capable(
                knowledge_base_id=context.knowledge_base_id,
                index_revision_id=context.index_revision_id,
            )
        except RetrievalExecutionError:
            return False
