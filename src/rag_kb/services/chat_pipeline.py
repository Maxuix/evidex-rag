"""Application services for claimed direct chat execution."""

from __future__ import annotations

import asyncio
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
    ErrorCode,
    EvidencePack,
    RetrievalRequest,
    RetrievalStrategy,
)
from rag_kb.retrieval import RetrievalService
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
        try:
            strategy = RetrievalStrategy(context.retrieval_strategy["strategy"])
            top_k = int(context.retrieval_strategy["top_k"])
            rerank = context.retrieval_strategy["rerank"]
            if not isinstance(rerank, bool):
                raise ValueError
            request = RetrievalRequest(
                knowledge_base_id=context.knowledge_base_id,
                query=context.query,
                top_k=top_k,
                strategy=strategy,
                rerank=rerank,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "retrieval_snapshot"},
            ) from error
        pack = await self._retrieval.retrieve(
            AuthContext(
                principal_id=context.principal_id,
                client_id=context.client_id,
                workspace_id=context.workspace_id,
            ),
            request,
        )
        if (
            pack.knowledge_base_id != context.knowledge_base_id
            or pack.index_revision_id != context.index_revision_id
        ):
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_REVISION_MISMATCH,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "frozen_revision"},
            )
        return pack


class DirectChatPipeline:
    """Execute the frozen P1A sequence without checkpoint or graph recovery."""

    def __init__(
        self,
        context_loader: ChatExecutionContextLoader,
        evidence_retriever: ChatEvidenceRetriever,
        evidence_assessor: ChatPipelineStep,
        answer_generator: ChatPipelineStep,
        structure_validator: ChatPipelineStep,
        result_persister: ChatPipelineStep,
        *,
        deadline_seconds: float,
    ) -> None:
        if deadline_seconds <= 0:
            raise ValueError("chat pipeline deadline must be positive")
        self._context_loader = context_loader
        self._evidence_retriever = evidence_retriever
        self._steps = (
            (ChatPipelinePhase.ASSESS_EVIDENCE, evidence_assessor),
            (ChatPipelinePhase.GENERATE_OR_REFUSE, answer_generator),
            (ChatPipelinePhase.VALIDATE_STRUCTURE, structure_validator),
            (ChatPipelinePhase.PERSIST_RESULT, result_persister),
        )
        self._deadline_seconds = deadline_seconds

    async def execute(self, command: ChatExecutionCommand) -> ChatPipelineState:
        phase = ChatPipelinePhase.LOAD_CONTEXT
        state: ChatPipelineState | None = None
        try:
            async with asyncio.timeout(self._deadline_seconds):
                context = await self._context_loader.load(command)
                if context.lease != command.lease:
                    raise ChatPipelineExecutionError(
                        ErrorCode.CHAT_CONTEXT_INVALID,
                        phase=ChatPipelinePhase.LOAD_CONTEXT,
                        diagnostic={"check": "claimed_lease"},
                    )
                phase = ChatPipelinePhase.RETRIEVE_EVIDENCE
                pack = await self._evidence_retriever.retrieve(context)
                state = ChatPipelineState(context=context, evidence_pack=pack)
                for phase, step in self._steps:
                    state = await step.run(state)
                    if (
                        not isinstance(state, ChatPipelineState)
                        or state.context != context
                        or state.evidence_pack != pack
                    ):
                        raise TypeError("chat pipeline step changed frozen inputs")
                return state
        except TimeoutError as error:
            failure = ChatPipelineExecutionError(
                ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED,
                phase=phase,
                diagnostic={"check": "task_deadline"},
            )
            if state is not None and state.answering is not None:
                failure.retain_model_calls(state.answering.model_calls)
            raise failure from error
        except ChatPipelineExecutionError as error:
            if state is not None and state.answering is not None:
                error.retain_model_calls(state.answering.model_calls)
            raise
        except Exception as error:
            failure = ChatPipelineExecutionError(
                ErrorCode.CHAT_PIPELINE_STEP_FAILED,
                phase=phase,
                diagnostic={"check": "step_contract"},
            )
            if state is not None and state.answering is not None:
                failure.retain_model_calls(state.answering.model_calls)
            raise failure from error
