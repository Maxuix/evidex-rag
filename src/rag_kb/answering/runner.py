"""Plain async runner for one claimed native-agent ChatRun."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
from typing import TYPE_CHECKING
from uuid import UUID

from rag_kb.answering.agent import ChatAgentProgress, NativeToolCallingAgent
from rag_kb.domain import (
    ChatExecutionCommand,
    ChatProgressActivity,
    ChatProgressFacts,
    ChatProgressStage,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ErrorCode,
)
from rag_kb.ports.chat_preview import ChatPreviewSink
from rag_kb.observability import get_logger, log_exception

if TYPE_CHECKING:
    from rag_kb.services.chat_execution import ChatExecutionContextLoader
    from rag_kb.services.chat_progress import ChatProgressReporter
    from rag_kb.services.chat_terminal import ChatResultPersistenceStep


LOGGER = get_logger("rag_kb.answering.runner")


ProgressReporterFactory = Callable[
    [UUID, int, ChatPreviewSink | None],
    "ChatProgressReporter",
]


class NativeAgentRunner:
    def __init__(
        self,
        context_loader: ChatExecutionContextLoader,
        agent: NativeToolCallingAgent,
        result_persister: ChatResultPersistenceStep,
        *,
        deadline_seconds: float,
        progress_sink: ChatPreviewSink | None = None,
        progress_reporter_factory: ProgressReporterFactory,
    ) -> None:
        if deadline_seconds <= 0:
            raise ValueError("chat agent deadline must be positive")
        self._context_loader = context_loader
        self._agent = agent
        self._result_persister = result_persister
        self._deadline_seconds = deadline_seconds
        self._progress_sink = progress_sink
        self._progress_reporter_factory = progress_reporter_factory

    async def execute(self, command: ChatExecutionCommand) -> ChatPipelineState:
        state: ChatPipelineState | None = None
        progress = ChatAgentProgress(deadline_seconds=self._deadline_seconds)
        phase = ChatPipelinePhase.LOAD_CONTEXT
        reporter = self._progress_reporter_factory(
            command.lease.run_id,
            command.lease.attempt,
            self._progress_sink,
        )
        try:
            async with asyncio.timeout(self._deadline_seconds):
                await reporter.show(
                    ChatProgressStage.UNDERSTAND_QUERY,
                    ChatProgressActivity.LOAD_CONTEXT,
                )
                context = await self._context_loader.load(command)
                if context.lease != command.lease:
                    raise ChatPipelineExecutionError(
                        ErrorCode.CHAT_CONTEXT_INVALID,
                        phase=phase,
                        diagnostic={"check": "claimed_lease"},
                    )
                phase = ChatPipelinePhase.GENERATE_OR_REFUSE
                await reporter.show(
                    ChatProgressStage.RETRIEVE_EVIDENCE,
                    ChatProgressActivity.TOOL_DECISION,
                    completed=(ChatProgressStage.UNDERSTAND_QUERY,),
                )
                state = await self._agent.run(
                    context,
                    deadline_seconds=self._deadline_seconds,
                    progress=progress,
                )
                phase = ChatPipelinePhase.PERSIST_RESULT
                trace = state.artifacts.get("chat_agent_trace")
                await reporter.show(
                    ChatProgressStage.PERSIST_RESULT,
                    ChatProgressActivity.PERSIST_RESULT,
                    facts=ChatProgressFacts(
                        evidence_count=len(state.evidence_pack.evidence)
                        if state.evidence_pack is not None
                        else 0,
                        retrieval_calls=getattr(trace, "retrieval_calls", None),
                    ),
                    completed=(
                        ChatProgressStage.RETRIEVE_EVIDENCE,
                        ChatProgressStage.GENERATE_ANSWER,
                    ),
                )
                result = await self._result_persister.run(state)
                await reporter.finish(ChatProgressActivity.PERSIST_RESULT)
                return result
        except TimeoutError as error:
            failure = ChatPipelineExecutionError(
                ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED,
                phase=phase,
                diagnostic={"check": "task_deadline"},
            )
            if state is not None and state.answering is not None:
                failure.retain_model_calls(state.answering.model_calls)
            else:
                failure.retain_model_calls(tuple(progress.model_calls))
                failure.retain_agent_trace(progress.partial_trace())
            raise failure from error
        except ChatPipelineExecutionError:
            raise
        except Exception as error:
            log_exception(
                LOGGER,
                "chat_agent_step_failed",
                error,
                level=logging.ERROR,
                run_id=command.lease.run_id,
                attempt=command.lease.attempt,
                phase=phase.value,
            )
            failure = ChatPipelineExecutionError(
                ErrorCode.CHAT_PIPELINE_STEP_FAILED,
                phase=phase,
                diagnostic={"check": "step_contract"},
            )
            if state is not None and state.answering is not None:
                failure.retain_model_calls(state.answering.model_calls)
            raise failure from error
