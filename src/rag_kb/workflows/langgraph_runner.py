"""LangGraph runner for the existing deterministic evidence-only chat flow."""

from __future__ import annotations

import asyncio

from rag_kb.domain import (
    ChatExecutionCommand,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ErrorCode,
)
from rag_kb.services.chat_execution import (
    ChatEvidenceRetriever,
    ChatExecutionContextLoader,
    ChatPipelineStep,
)
from rag_kb.workflows.chat_graph import compile_chat_graph
from rag_kb.workflows.state_mapping import (
    ChatGraphProgress,
    ChatGraphState,
    retain_progress,
)


class LangGraphRunner:
    """Compile once and execute the fixed graph without persistence or retries."""

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
            raise ValueError("chat graph deadline must be positive")
        self._context_loader = context_loader
        self._evidence_retriever = evidence_retriever
        self._steps = {
            "assess_evidence": (
                ChatPipelinePhase.ASSESS_EVIDENCE,
                evidence_assessor,
            ),
            "generate_or_refuse": (
                ChatPipelinePhase.GENERATE_OR_REFUSE,
                answer_generator,
            ),
            "validate_structure": (
                ChatPipelinePhase.VALIDATE_STRUCTURE,
                structure_validator,
            ),
            "persist_result": (
                ChatPipelinePhase.PERSIST_RESULT,
                result_persister,
            ),
        }
        self._deadline_seconds = deadline_seconds
        self._graph = compile_chat_graph(
            {
                "load_context": self._load_context,
                "retrieve_evidence": self._retrieve_evidence,
                "assess_evidence": self._assess_evidence,
                "generate_or_refuse": self._generate_or_refuse,
                "validate_structure": self._validate_structure,
                "persist_result": self._persist_result,
            }
        )

    async def execute(self, command: ChatExecutionCommand) -> ChatPipelineState:
        progress = ChatGraphProgress()
        try:
            async with asyncio.timeout(self._deadline_seconds):
                result = await self._graph.ainvoke(
                    {"command": command, "progress": progress}
                )
            state = result.get("pipeline_state")
            if not isinstance(state, ChatPipelineState):
                raise TypeError("chat graph did not return a pipeline state")
            return state
        except TimeoutError as error:
            failure = ChatPipelineExecutionError(
                ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED,
                phase=progress.phase,
                diagnostic={"check": "task_deadline"},
            )
            failure.retain_model_calls(progress.model_calls)
            raise failure from error
        except ChatPipelineExecutionError as error:
            error.retain_model_calls(progress.model_calls)
            raise
        except Exception as error:
            failure = ChatPipelineExecutionError(
                ErrorCode.CHAT_PIPELINE_STEP_FAILED,
                phase=progress.phase,
                diagnostic={"check": "step_contract"},
            )
            failure.retain_model_calls(progress.model_calls)
            raise failure from error

    async def _load_context(self, graph: ChatGraphState) -> dict[str, object]:
        progress = graph["progress"]
        progress.phase = ChatPipelinePhase.LOAD_CONTEXT
        context = await self._context_loader.load(graph["command"])
        if context.lease != graph["command"].lease:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.LOAD_CONTEXT,
                diagnostic={"check": "claimed_lease"},
            )
        return {"context": context}

    async def _retrieve_evidence(self, graph: ChatGraphState) -> dict[str, object]:
        progress = graph["progress"]
        progress.phase = ChatPipelinePhase.RETRIEVE_EVIDENCE
        context = graph["context"]
        pack = await self._evidence_retriever.retrieve(context)
        return {
            "evidence_pack": pack,
            "pipeline_state": ChatPipelineState(
                context=context,
                evidence_pack=pack,
            ),
        }

    async def _assess_evidence(self, graph: ChatGraphState) -> dict[str, object]:
        return await self._run_step(graph, "assess_evidence")

    async def _generate_or_refuse(self, graph: ChatGraphState) -> dict[str, object]:
        return await self._run_step(graph, "generate_or_refuse")

    async def _validate_structure(self, graph: ChatGraphState) -> dict[str, object]:
        return await self._run_step(graph, "validate_structure")

    async def _persist_result(self, graph: ChatGraphState) -> dict[str, object]:
        return await self._run_step(graph, "persist_result")

    async def _run_step(
        self,
        graph: ChatGraphState,
        name: str,
    ) -> dict[str, object]:
        phase, step = self._steps[name]
        progress = graph["progress"]
        progress.phase = phase
        state = await step.run(graph["pipeline_state"])
        if (
            not isinstance(state, ChatPipelineState)
            or state.context is not graph["context"]
            or state.evidence_pack is not graph["evidence_pack"]
        ):
            raise TypeError("chat graph step changed frozen inputs")
        retain_progress(progress, state)
        return {"pipeline_state": state}
