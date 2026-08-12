"""Application services for claimed native Agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from collections.abc import Mapping
from typing import Protocol
from uuid import UUID

from rag_kb.auth import AuthContext
from rag_kb.domain import (
    ChatExecutionCommand,
    ChatExecutionContext,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ChatRunLease,
    ContextualizedQuery,
    ErrorCode,
    Evidence,
    EvidencePack,
    EvidenceScoreKind,
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


@dataclass(frozen=True, slots=True)
class RuntimeDocumentScope:
    status: str
    document_ids: tuple[UUID, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    complete_scan_document_count: int = 0
    scope_rejection_count: int = 0
    downgrade_reason: str | None = None

    @property
    def rejected(self) -> bool:
        return self.status in {"ambiguous", "unresolved"}


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


class ChatEvidenceRetriever:
    def __init__(
        self,
        retrieval: RetrievalService,
        unit_of_work: UnitOfWorkFactory | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._unit_of_work = unit_of_work

    async def load_document_scope(
        self, context: ChatExecutionContext
    ) -> RuntimeDocumentScope:
        raw_scope = context.retrieval_strategy.get("document_scope")
        if not isinstance(raw_scope, Mapping):
            return RuntimeDocumentScope(status="all")
        status = str(raw_scope.get("status", "unresolved"))
        if status != "resolved":
            return RuntimeDocumentScope(
                status=status,
                scope_rejection_count=1,
                downgrade_reason=(
                    "explicit_document_scope_" + status
                ),
            )
        raw_resolved = raw_scope.get("resolved")
        if not isinstance(raw_resolved, (list, tuple)) or not raw_resolved:
            return RuntimeDocumentScope(
                status="unresolved",
                scope_rejection_count=1,
                downgrade_reason="resolved_scope_without_targets",
            )
        try:
            document_ids = tuple(
                dict.fromkeys(UUID(str(item["document_id"])) for item in raw_resolved)
            )
        except (KeyError, TypeError, ValueError):
            return RuntimeDocumentScope(
                status="unresolved",
                scope_rejection_count=1,
                downgrade_reason="invalid_frozen_document_scope",
            )
        if not 1 <= len(document_ids) <= 4:
            return RuntimeDocumentScope(
                status="unresolved",
                scope_rejection_count=1,
                downgrade_reason="document_scope_bound",
            )
        if self._unit_of_work is None:
            return RuntimeDocumentScope(
                status="resolved",
                document_ids=document_ids,
                downgrade_reason="complete_scan_unavailable",
            )

        resolved_by_id = {
            UUID(str(item["document_id"])): item
            for item in raw_resolved
            if isinstance(item, Mapping) and item.get("document_id") is not None
        }

        async def inspect(uow: UnitOfWork) -> tuple[Evidence, ...]:
            all_evidence: list[Evidence] = []
            for document_id in document_ids:
                facts = resolved_by_id.get(document_id)
                if facts is None:
                    continue
                inspection = await uow.documents.inspect_chunks(
                    document_id,
                    limit=8,
                    after=None,
                )
                if inspection is None:
                    continue
                if (
                    inspection.document_id != document_id
                    or inspection.index_revision_id != context.index_revision_id
                    or str(inspection.document_version_id)
                    != str(facts.get("document_version_id"))
                    or str(inspection.indexed_document_version_id)
                    != str(facts.get("indexed_document_version_id"))
                    or inspection.total_chunks > 8
                    or inspection.next_values is not None
                    or any(
                        item.modality not in {"text", "table"}
                        or item.excluded_at is not None
                        for item in inspection.items
                    )
                ):
                    continue
                for item in inspection.items:
                    all_evidence.append(
                        Evidence(
                            rank=len(all_evidence) + 1,
                            index_chunk_id=item.id,
                            indexed_document_version_id=(
                                inspection.indexed_document_version_id
                            ),
                            document_id=inspection.document_id,
                            document_version_id=inspection.document_version_id,
                            index_revision_id=inspection.index_revision_id,
                            ordinal=item.ordinal,
                            text=item.content,
                            source_location=item.source_location,
                            hierarchy=item.hierarchy,
                            source_metadata={
                                **item.source_metadata,
                                "evidence_type": "complete_scan",
                            },
                            score=1.0,
                            vector_similarity=1.0,
                            modality=item.modality,
                            evidence_group_key=item.evidence_group_key,
                            matched_representations=("complete_scan",),
                            document_display_name=str(
                                facts.get("display_name") or "document"
                            ),
                            document_original_filename=str(
                                facts.get("original_filename") or "document"
                            ),
                        )
                    )
            if sum(len(item.text) for item in all_evidence) > 32_000:
                return ()
            return tuple(all_evidence)

        evidence = await execute_in_transaction(
            self._unit_of_work,
            inspect,
            purpose=UnitOfWorkPurpose.READ_SNAPSHOT,
        )
        complete_documents = len(
            {item.document_id for item in evidence}
        )
        return RuntimeDocumentScope(
            status="resolved",
            document_ids=document_ids,
            evidence=evidence,
            complete_scan_document_count=complete_documents,
            downgrade_reason=(
                None if complete_documents == len(document_ids) else "complete_scan_incomplete"
            ),
        )

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
        document_ids: tuple[UUID, ...] = (),
    ) -> EvidencePack:
        try:
            scope = _runtime_document_scope(context)
            strategy, top_k, rerank_mode = parse_retrieval_snapshot(
                context.retrieval_strategy,
            )
            if top_k_override is not None:
                if not 1 <= top_k_override <= top_k:
                    raise ValueError
                top_k = top_k_override
            requested_document_ids = tuple(dict.fromkeys(document_ids))
            if requested_document_ids:
                if scope.status != "resolved" or not set(
                    requested_document_ids
                ) <= set(scope.document_ids):
                    raise ValueError
            else:
                requested_document_ids = scope.document_ids
            request = RetrievalRequest(
                knowledge_base_id=context.knowledge_base_id,
                query=query,
                top_k=top_k,
                strategy=strategy,
                rerank_mode=rerank_mode,
                include_debug=True,
                document_ids=requested_document_ids,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "retrieval_snapshot"},
            ) from error
        try:
            if scope.rejected:
                return EvidencePack(
                    knowledge_base_id=context.knowledge_base_id,
                    index_revision_id=context.index_revision_id,
                    strategy=strategy,
                    evidence=(),
                )
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


def _runtime_document_scope(context: ChatExecutionContext) -> RuntimeDocumentScope:
    raw_scope = context.retrieval_strategy.get("document_scope")
    if not isinstance(raw_scope, Mapping):
        return RuntimeDocumentScope(status="all")
    status = str(raw_scope.get("status", "unresolved"))
    if status != "resolved":
        return RuntimeDocumentScope(
            status=status,
            scope_rejection_count=1,
            downgrade_reason="explicit_document_scope_" + status,
        )
    raw_resolved = raw_scope.get("resolved")
    if not isinstance(raw_resolved, (list, tuple)):
        return RuntimeDocumentScope(
            status="unresolved",
            scope_rejection_count=1,
            downgrade_reason="invalid_frozen_document_scope",
        )
    try:
        document_ids = tuple(
            dict.fromkeys(UUID(str(item["document_id"])) for item in raw_resolved)
        )
    except (KeyError, TypeError, ValueError):
        return RuntimeDocumentScope(
            status="unresolved",
            scope_rejection_count=1,
            downgrade_reason="invalid_frozen_document_scope",
        )
    if not 1 <= len(document_ids) <= 4:
        return RuntimeDocumentScope(
            status="unresolved",
            scope_rejection_count=1,
            downgrade_reason="document_scope_bound",
        )
    return RuntimeDocumentScope(status="resolved", document_ids=document_ids)
