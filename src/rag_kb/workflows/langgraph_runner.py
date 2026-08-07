"""LangGraph runner for the existing deterministic evidence-only chat flow."""

from __future__ import annotations

import asyncio
from typing import Protocol

from rag_kb.domain import (
    CONTEXTUAL_QUERY_VERSION,
    ChatExecutionCommand,
    ChatExecutionContext,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ErrorCode,
    ContextualizedQuery,
    QueryContextStatus,
    QueryRewriteSource,
    ChatResolvedMode,
    ChatProgressActivity,
    ChatProgressDecision,
    ChatProgressFacts,
    ChatProgressStage,
    ChatWorkflowMode,
    ChatWorkflowState,
    ChatModelCallRecord,
    hydrate_chat_workflow_configuration,
    hydrate_chat_workflow_state,
)
from rag_kb.ports.chat_preview import ChatPreviewSink
from rag_kb.retrieval.agent import (
    WORKFLOW_MODEL_CALLS_ARTIFACT,
    WORKFLOW_STATE_ARTIFACT,
    AgentResearchOutcome,
    RetrievalAgentService,
)
from rag_kb.services.chat_execution import (
    ChatEvidenceRetriever,
    ChatExecutionContextLoader,
    ChatPipelineStep,
)
from rag_kb.services.chat_progress import (
    ChatProgressReporter,
    bounded_progress_values,
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
        visual_evidence_preparer: ChatPipelineStep | None = None,
        query_contextualizer: QueryContextualizer | None = None,
        workflow_router: WorkflowRouter | None = None,
        retrieval_agent: RetrievalAgentService | None = None,
        adaptive_evidence_assessor: ChatPipelineStep | None = None,
        progress_sink: ChatPreviewSink | None = None,
        deadline_seconds: float,
    ) -> None:
        if deadline_seconds <= 0:
            raise ValueError("chat graph deadline must be positive")
        self._context_loader = context_loader
        self._query_contextualizer = (
            query_contextualizer or _OriginalOnlyContextualizer()
        )
        self._evidence_retriever = evidence_retriever
        self._workflow_router = workflow_router or _PersistedWorkflowRouter()
        self._retrieval_agent = retrieval_agent
        self._adaptive_evidence_assessor = adaptive_evidence_assessor
        self._progress_sink = progress_sink
        self._steps = {
            "assess_evidence": (
                ChatPipelinePhase.ASSESS_EVIDENCE,
                evidence_assessor,
            ),
            "generate_or_refuse": (
                ChatPipelinePhase.GENERATE_OR_REFUSE,
                answer_generator,
            ),
            "prepare_visual_evidence": (
                ChatPipelinePhase.PREPARE_VISUAL_EVIDENCE,
                visual_evidence_preparer or _PassThroughStep(),
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
                "contextualize_query": self._contextualize_query,
                "route_workflow": self._route_workflow,
                "retrieve_evidence": self._retrieve_evidence,
                "research_evidence": self._research_evidence,
                "assess_evidence": self._assess_evidence,
                "prepare_visual_evidence": self._prepare_visual_evidence,
                "generate_or_refuse": self._generate_or_refuse,
                "validate_structure": self._validate_structure,
                "persist_result": self._persist_result,
            },
        )

    async def execute(self, command: ChatExecutionCommand) -> ChatPipelineState:
        reporter = ChatProgressReporter(
            command.lease.run_id,
            command.lease.attempt,
            self._progress_sink,
        )
        progress = ChatGraphProgress(reporter=reporter)
        try:
            await reporter.show(
                ChatProgressStage.UNDERSTAND_QUERY,
                ChatProgressActivity.LOAD_CONTEXT,
            )
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
        configuration = hydrate_chat_workflow_configuration(
            context.workflow_configuration
        )
        if progress.reporter is not None:
            progress.reporter.configure(configuration.requested_mode)
            await progress.reporter.show(
                ChatProgressStage.UNDERSTAND_QUERY,
                ChatProgressActivity.CONTEXTUALIZE_QUERY,
            )
        return {"context": context}

    async def _retrieve_evidence(self, graph: ChatGraphState) -> dict[str, object]:
        progress = graph["progress"]
        progress.phase = ChatPipelinePhase.RETRIEVE_EVIDENCE
        context = graph["context"]
        query_context = graph["query_context"]
        pack = await self._evidence_retriever.retrieve(context, query_context)
        if progress.reporter is not None:
            await progress.reporter.show(
                ChatProgressStage.RETRIEVE_EVIDENCE,
                ChatProgressActivity.RETRIEVAL_COMPLETE,
                facts=ChatProgressFacts(evidence_count=len(pack.evidence)),
            )
            await progress.reporter.show(
                ChatProgressStage.ASSESS_EVIDENCE,
                ChatProgressActivity.ASSESS_EVIDENCE,
                facts=ChatProgressFacts(evidence_count=len(pack.evidence)),
                completed=(ChatProgressStage.RETRIEVE_EVIDENCE,),
            )
        return {
            "evidence_pack": pack,
            "pipeline_state": ChatPipelineState(
                context=context,
                query_context=query_context,
                evidence_pack=pack,
                artifacts={
                    WORKFLOW_STATE_ARTIFACT: graph["workflow_state"],
                    WORKFLOW_MODEL_CALLS_ARTIFACT: graph.get(
                        "workflow_model_calls", ()
                    ),
                },
            ),
        }

    async def _research_evidence(
        self, graph: ChatGraphState
    ) -> dict[str, object]:
        progress = graph["progress"]
        progress.phase = ChatPipelinePhase.RETRIEVE_EVIDENCE
        if self._retrieval_agent is None:
            raise ChatPipelineExecutionError(
                ErrorCode.CAPABILITY_NOT_ENABLED,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"capability": "chat_agent"},
            )
        outcome = await self._retrieval_agent.research(
            graph["context"],
            graph["query_context"],
            prior_model_calls=graph.get("workflow_model_calls", ()),
            resolved_workflow_state=graph["workflow_state"],
            progress=progress.reporter,
        )
        progress.model_calls = (
            graph["query_context"].model_calls_for_attempt(
                graph["context"].attempt
            )
            + outcome.model_calls
        )
        if progress.reporter is not None:
            result = outcome.workflow_state.research_result
            trace = outcome.workflow_state.search_trace
            await progress.reporter.show(
                ChatProgressStage.RETRIEVE_EVIDENCE,
                ChatProgressActivity.RESEARCH_COMPLETE,
                facts=ChatProgressFacts(
                    evidence_count=len(outcome.evidence_pack.evidence),
                    retrieval_calls=(trace.retrieval_calls if trace else None),
                    research_status=(result.status if result else None),
                    covered_aspects=(
                        bounded_progress_values(result.covered_aspects)
                        if result
                        else ()
                    ),
                    missing_aspects=(
                        bounded_progress_values(result.missing_aspects)
                        if result
                        else ()
                    ),
                    conflict_count=(len(result.conflicts) if result else None),
                    decision=ChatProgressDecision.FINISH_RESEARCH,
                ),
            )
            await progress.reporter.show(
                ChatProgressStage.ASSESS_EVIDENCE,
                ChatProgressActivity.ASSESS_EVIDENCE,
                facts=ChatProgressFacts(
                    evidence_count=len(outcome.evidence_pack.evidence),
                    research_status=(result.status if result else None),
                ),
                completed=(ChatProgressStage.RETRIEVE_EVIDENCE,),
            )
        return {
            "workflow_state": outcome.workflow_state,
            "workflow_model_calls": outcome.model_calls,
            "evidence_pack": outcome.evidence_pack,
            "pipeline_state": ChatPipelineState(
                context=graph["context"],
                query_context=graph["query_context"],
                evidence_pack=outcome.evidence_pack,
                artifacts={
                    WORKFLOW_STATE_ARTIFACT: outcome.workflow_state,
                    WORKFLOW_MODEL_CALLS_ARTIFACT: outcome.model_calls,
                },
            ),
        }

    async def _contextualize_query(
        self, graph: ChatGraphState
    ) -> dict[str, object]:
        progress = graph["progress"]
        progress.phase = ChatPipelinePhase.CONTEXTUALIZE_QUERY
        value = await self._query_contextualizer.contextualize(graph["context"])
        progress.model_calls = value.model_calls_for_attempt(
            graph["context"].attempt
        )
        if progress.reporter is not None:
            await progress.reporter.show(
                ChatProgressStage.SELECT_WORKFLOW,
                ChatProgressActivity.ROUTE_DECISION,
                completed=(ChatProgressStage.UNDERSTAND_QUERY,),
            )
        return {"query_context": value}

    async def _route_workflow(
        self, graph: ChatGraphState
    ) -> dict[str, object]:
        progress = graph["progress"]
        progress.phase = ChatPipelinePhase.RETRIEVE_EVIDENCE
        state, calls = await self._workflow_router.resolve(
            graph["context"], graph["query_context"]
        )
        progress.model_calls = (
            graph["query_context"].model_calls_for_attempt(
                graph["context"].attempt
            )
            + calls
        )
        if progress.reporter is not None:
            configuration = hydrate_chat_workflow_configuration(
                graph["context"].workflow_configuration
            )
            progress.reporter.configure(
                configuration.requested_mode,
                state.resolved_mode,
            )
            decision = (
                ChatProgressDecision.SELECT_AGENT
                if state.resolved_mode is ChatResolvedMode.AGENT
                else ChatProgressDecision.SELECT_SIMPLE
            )
            await progress.reporter.show(
                ChatProgressStage.RETRIEVE_EVIDENCE,
                (
                    ChatProgressActivity.AGENT_DECISION
                    if state.resolved_mode is ChatResolvedMode.AGENT
                    else ChatProgressActivity.SIMPLE_SEARCH
                ),
                facts=ChatProgressFacts(
                    route_status=state.route_status,
                    route_reason_codes=state.route_reason_codes[:6],
                    decision=decision,
                ),
                completed=(ChatProgressStage.SELECT_WORKFLOW,),
            )
        return {
            "workflow_state": state,
            "workflow_model_calls": calls,
        }

    async def _assess_evidence(self, graph: ChatGraphState) -> dict[str, object]:
        workflow_state = graph["workflow_state"]
        if workflow_state.research_result is None:
            return await self._run_step(graph, "assess_evidence")
        if self._adaptive_evidence_assessor is None:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.ASSESS_EVIDENCE,
                diagnostic={"check": "adaptive_assessor_dependency"},
            )
        return await self._run_step(
            graph,
            "assess_evidence",
            step_override=self._adaptive_evidence_assessor,
        )

    async def _generate_or_refuse(self, graph: ChatGraphState) -> dict[str, object]:
        return await self._run_step(graph, "generate_or_refuse")

    async def _prepare_visual_evidence(
        self, graph: ChatGraphState
    ) -> dict[str, object]:
        return await self._run_step(graph, "prepare_visual_evidence")

    async def _validate_structure(self, graph: ChatGraphState) -> dict[str, object]:
        return await self._run_step(graph, "validate_structure")

    async def _persist_result(self, graph: ChatGraphState) -> dict[str, object]:
        return await self._run_step(graph, "persist_result")

    async def _run_step(
        self,
        graph: ChatGraphState,
        name: str,
        *,
        step_override: ChatPipelineStep | None = None,
    ) -> dict[str, object]:
        phase, step = self._steps[name]
        step = step_override or step
        progress = graph["progress"]
        progress.phase = phase
        stage, activity = _progress_step(name)
        if progress.reporter is not None:
            await progress.reporter.show(stage, activity)
        state = await step.run(graph["pipeline_state"])
        if (
            not isinstance(state, ChatPipelineState)
            or state.context is not graph["context"]
            or state.evidence_pack is not graph["evidence_pack"]
            or state.query_context is not graph["query_context"]
        ):
            raise TypeError("chat graph step changed frozen inputs")
        retain_progress(progress, state)
        if progress.reporter is not None:
            next_step = _next_progress_step(name)
            if next_step is None:
                await progress.reporter.finish(activity)
            else:
                next_stage, next_activity = next_step
                await progress.reporter.show(
                    next_stage,
                    next_activity,
                    completed=(stage,),
                )
        return {"pipeline_state": state}


class QueryContextualizer(Protocol):
    async def contextualize(
        self, context: ChatExecutionContext
    ) -> ContextualizedQuery: ...


class WorkflowRouter(Protocol):
    async def resolve(
        self,
        context: ChatExecutionContext,
        query_context: ContextualizedQuery,
    ) -> tuple[ChatWorkflowState, tuple[ChatModelCallRecord, ...]]: ...


class _PersistedWorkflowRouter:
    async def resolve(
        self,
        context: ChatExecutionContext,
        query_context: ContextualizedQuery,
    ) -> tuple[ChatWorkflowState, tuple[ChatModelCallRecord, ...]]:
        del query_context
        try:
            configuration = hydrate_chat_workflow_configuration(
                context.workflow_configuration
            )
            state = hydrate_chat_workflow_state(context.workflow_state)
        except (TypeError, ValueError) as error:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "workflow_snapshot"},
            ) from error
        expected = {
            ChatWorkflowMode.SIMPLE: ChatResolvedMode.SIMPLE,
            ChatWorkflowMode.AGENT: ChatResolvedMode.AGENT,
        }.get(configuration.requested_mode)
        if expected is not None and state.resolved_mode is not expected:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"check": "workflow_resolution"},
            )
        if state.resolved_mode is ChatResolvedMode.PENDING:
            raise ChatPipelineExecutionError(
                ErrorCode.CAPABILITY_NOT_ENABLED,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"capability": "chat_auto"},
            )
        return state, ()


class _OriginalOnlyContextualizer:
    async def contextualize(
        self, context: ChatExecutionContext
    ) -> ContextualizedQuery:
        if context.contextualized_query is not None:
            return context.contextualized_query
        if context.conversation_context.turns:
            raise ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.CONTEXTUALIZE_QUERY,
                diagnostic={"check": "contextualizer_dependency"},
            )
        return ContextualizedQuery(
            version=CONTEXTUAL_QUERY_VERSION,
            status=QueryContextStatus.ORIGINAL,
            original_query=context.query,
            standalone_query=context.query,
            context_hash=context.conversation_context.content_hash,
            rewrite_source=QueryRewriteSource.ORIGINAL,
        )


class _PassThroughStep:
    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        return state


def _progress_step(
    name: str,
) -> tuple[ChatProgressStage, ChatProgressActivity]:
    return {
        "assess_evidence": (
            ChatProgressStage.ASSESS_EVIDENCE,
            ChatProgressActivity.ASSESS_EVIDENCE,
        ),
        "prepare_visual_evidence": (
            ChatProgressStage.PREPARE_VISUAL_EVIDENCE,
            ChatProgressActivity.PREPARE_VISUAL_EVIDENCE,
        ),
        "generate_or_refuse": (
            ChatProgressStage.GENERATE_ANSWER,
            ChatProgressActivity.GENERATE_ANSWER,
        ),
        "validate_structure": (
            ChatProgressStage.VALIDATE_ANSWER,
            ChatProgressActivity.VALIDATE_ANSWER,
        ),
        "persist_result": (
            ChatProgressStage.PERSIST_RESULT,
            ChatProgressActivity.PERSIST_RESULT,
        ),
    }[name]


def _next_progress_step(
    name: str,
) -> tuple[ChatProgressStage, ChatProgressActivity] | None:
    order = (
        "assess_evidence",
        "prepare_visual_evidence",
        "generate_or_refuse",
        "validate_structure",
        "persist_result",
    )
    index = order.index(name)
    return _progress_step(order[index + 1]) if index + 1 < len(order) else None
