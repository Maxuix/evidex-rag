from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime

from rag_kb.answering import (
    AnswerGenerationStep,
    AnswerStructureValidationStep,
    AdaptiveEvidenceAssessmentStep,
    CosineEvidenceAssessmentStep,
)
from rag_kb.domain import (
    AnswerOutcome,
    ChatExecutionCommand,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ErrorCode,
    EvidenceCoverage,
    EvidencePack,
    CONTEXTUAL_QUERY_VERSION,
    ChatModelCallRecord,
    ChatModelOperation,
    ChatResolvedMode,
    ChatRouteStatus,
    ChatWorkflowMode,
    ChatWorkflowState,
    ContextualizedQuery,
    QueryContextStatus,
    QueryRewriteSource,
    ResearchAspect,
    ResearchAspectStatus,
    ResearchResult,
    ResearchStatus,
    ResearchTerminationReason,
    SearchTrace,
    SearchTraceStep,
    initial_chat_workflow,
)
from rag_kb.retrieval.agent import AgentResearchOutcome, evidence_key
from rag_kb.workflows.langgraph_runner import LangGraphRunner
from rag_kb.workflows.chat_graph import CHAT_GRAPH_NODES
from tests.unit.test_answering import (
    _Model,
    _context,
    _pack,
    _response,
)


class _Loader:
    def __init__(self, context, *, delay: float = 0) -> None:
        self.context = context
        self.delay = delay

    async def load(self, command):
        del command
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.context


class _Retriever:
    def __init__(self, pack: EvidencePack, *, error: Exception | None = None) -> None:
        self.pack = pack
        self.error = error
        self.calls = 0

    async def retrieve(self, context, query_context=None):
        del context, query_context
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.pack


class _Persister:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        self.calls += 1
        return ChatPipelineState(
            context=state.context,
            evidence_pack=state.evidence_pack,
            answering=state.answering,
            query_context=state.query_context,
            artifacts={**state.artifacts, "persisted": True},
        )


class _FailStep:
    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        del state
        raise RuntimeError("sensitive step content")


class _ProgressSink:
    enabled = True

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[object] = []

    async def emit_progress(self, *, run_id, attempt, update) -> None:
        del run_id, attempt
        if self.fail:
            raise OSError("progress transport unavailable")
        self.events.append(update)


def _answer(outcome: AnswerOutcome) -> str:
    if outcome is AnswerOutcome.ANSWERED:
        return json.dumps(
            {
                "outcome": "answered",
                "claims": [{"text": "Policy applies.", "citation_ids": ["cite_1"]}],
                "missing_aspects": [],
            }
        )
    return json.dumps(
        {
            "outcome": "partial",
            "claims": [{"text": "Policy applies.", "citation_ids": ["cite_1"]}],
            "missing_aspects": ["deadline"],
        }
    )


def _build(
    *,
    context,
    pack: EvidencePack,
    responses: tuple = (),
    deadline: float = 1,
    loader=None,
    retriever=None,
    assessor=None,
    contextualizer=None,
    workflow_router=None,
    retrieval_agent=None,
    adaptive_assessor=None,
    progress_sink=None,
):
    model = _Model(*responses)
    context_loader = loader or _Loader(context)
    evidence_retriever = retriever or _Retriever(pack)
    evidence_assessor = assessor or CosineEvidenceAssessmentStep(0.6)
    answer_generator = AnswerGenerationStep(model)
    validator = AnswerStructureValidationStep(model)
    persister = _Persister()
    values = (
        context_loader,
        evidence_retriever,
        evidence_assessor,
        answer_generator,
        validator,
        persister,
    )
    return (
        LangGraphRunner(
            *values,
            query_contextualizer=contextualizer,
            workflow_router=workflow_router,
            retrieval_agent=retrieval_agent,
            adaptive_evidence_assessor=adaptive_assessor,
            progress_sink=progress_sink,
            deadline_seconds=deadline,
        ),  # type: ignore[arg-type]
        model,
        persister,
    )


class LangGraphRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_safe_agent_degradation_completes_with_refusal(self) -> None:
        base = _context(insufficiency="partial_answer")
        configuration, initial = initial_chat_workflow(ChatWorkflowMode.AGENT)
        context = replace(
            base,
            workflow_configuration=configuration.as_dict(),
            workflow_state=initial.as_dict(),
        )
        pack = _pack(context)
        research_result = ResearchResult(
            status=ResearchStatus.NO_EVIDENCE,
            selected_evidence_keys=(),
            aspects=(
                ResearchAspect(
                    aspect="reliable_research_result",
                    status=ResearchAspectStatus.MISSING,
                ),
            ),
            covered_aspects=(),
            missing_aspects=("reliable_research_result",),
            conflicts=(),
            termination_reason=ResearchTerminationReason.NO_EVIDENCE,
            degradation_reason="research_result_verification_wire_schema_invalid",
        )
        workflow_state = ChatWorkflowState(
            resolved_mode=ChatResolvedMode.AGENT,
            route_status=ChatRouteStatus.NOT_APPLICABLE,
            research_result=research_result,
            search_trace=SearchTrace(
                steps=(),
                decision_rounds=1,
                retrieval_calls=0,
                verifier_calls=1,
                evidence_count=0,
            ),
        )

        class Router:
            async def resolve(self, value, query_context):
                del value, query_context
                return initial, ()

        class Agent:
            async def research(self, value, query_context, **kwargs):
                del value, query_context, kwargs
                return AgentResearchOutcome(pack, workflow_state, ())

        runner, model, persister = _build(
            context=context,
            pack=pack,
            workflow_router=Router(),
            retrieval_agent=Agent(),
            adaptive_assessor=AdaptiveEvidenceAssessmentStep(0.6),
        )

        output = await runner.execute(ChatExecutionCommand(context.lease))

        self.assertEqual(persister.calls, 1)
        self.assertTrue(output.artifacts["persisted"])
        self.assertEqual(model.requests, [])
        assert output.answering is not None
        assert output.answering.draft is not None
        self.assertEqual(
            output.answering.draft.control_reason.value,
            "no_usable_evidence",
        )

    async def test_simple_run_reports_full_stage_progress_and_delivery_is_best_effort(
        self,
    ) -> None:
        context = _context(insufficiency="partial_answer")
        pack = _pack(context)
        sink = _ProgressSink()
        runner, _, persister = _build(
            context=context,
            pack=pack,
            progress_sink=sink,
        )

        await runner.execute(ChatExecutionCommand(context.lease))

        self.assertEqual(persister.calls, 1)
        self.assertGreaterEqual(len(sink.events), 8)
        final = sink.events[-1]
        self.assertEqual(final.status.value, "completed")
        self.assertEqual(final.active_stage.value, "persist_result")
        self.assertEqual(len(final.completed_stages), 8)
        self.assertEqual(final.facts.route_status.value, "not_applicable")

        failing = _ProgressSink(fail=True)
        runner, _, persister = _build(
            context=context,
            pack=pack,
            progress_sink=failing,
        )
        await runner.execute(ChatExecutionCommand(context.lease))
        self.assertEqual(persister.calls, 1)

    async def test_agent_branch_skips_simple_retrieval_and_uses_verified_assessment(
        self,
    ) -> None:
        base = _context(insufficiency="partial_answer")
        configuration, initial = initial_chat_workflow(ChatWorkflowMode.AGENT)
        context = replace(
            base,
            workflow_configuration=configuration.as_dict(),
            workflow_state=initial.as_dict(),
        )
        pack = _pack(context, "Policy applies.")
        key = evidence_key(pack.evidence[0])
        result = ResearchResult(
            status=ResearchStatus.SUFFICIENT,
            selected_evidence_keys=(key,),
            aspects=(
                ResearchAspect(
                    aspect="question",
                    status=ResearchAspectStatus.SUPPORTED,
                    evidence_keys=(key,),
                ),
            ),
            covered_aspects=("question",),
            missing_aspects=(),
            conflicts=(),
            termination_reason=ResearchTerminationReason.SUFFICIENT,
        )
        workflow_state = ChatWorkflowState(
            resolved_mode=ChatResolvedMode.AGENT,
            route_status=ChatRouteStatus.NOT_APPLICABLE,
            research_result=result,
            search_trace=SearchTrace(
                steps=(
                    SearchTraceStep(
                        observation_id="obs_1",
                        objective="find policy",
                        queries=("policy",),
                        based_on_observation_ids=(),
                        result="evidence_found",
                        new_evidence_count=1,
                    ),
                ),
                decision_rounds=2,
                retrieval_calls=1,
                verifier_calls=1,
                evidence_count=1,
            ),
        )
        workflow_call = ChatModelCallRecord(
            operation=ChatModelOperation.VERIFY_RESEARCH_RESULT,
            model="fixed-model",
            provider_request_id="verify-agent",
            usage={"prompt_tokens": 3},
        )

        class Router:
            calls = 0

            async def resolve(self, value, query_context):
                del value, query_context
                self.calls += 1
                return initial, ()

        class Agent:
            calls = 0

            async def research(self, value, query_context, **kwargs):
                del value, query_context, kwargs
                self.calls += 1
                return AgentResearchOutcome(pack, workflow_state, (workflow_call,))

        router = Router()
        agent = Agent()
        simple_retriever = _Retriever(pack)
        runner, _, persister = _build(
            context=context,
            pack=pack,
            responses=(_response(_answer(AnswerOutcome.ANSWERED)),),
            retriever=simple_retriever,
            workflow_router=router,
            retrieval_agent=agent,
            adaptive_assessor=AdaptiveEvidenceAssessmentStep(0.6),
        )

        output = await runner.execute(ChatExecutionCommand(context.lease))

        self.assertEqual(router.calls, 1)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(simple_retriever.calls, 0)
        self.assertEqual(persister.calls, 1)
        assert output.answering is not None
        self.assertIn(workflow_call, output.answering.model_calls)
        self.assertIs(
            output.answering.assessment.coverage, EvidenceCoverage.SUFFICIENT
        )

    async def test_fallback_query_continues_through_retrieval(self) -> None:
        context = _context()
        query_context = ContextualizedQuery(
            version=CONTEXTUAL_QUERY_VERSION,
            status=QueryContextStatus.CONTEXTUALIZED,
            original_query=context.query,
            standalone_query="The prior topic; explain it more clearly",
            context_hash=context.conversation_context.content_hash,
            model_calls=(
                ChatModelCallRecord(
                    operation=ChatModelOperation.CONTEXTUALIZE_QUERY,
                    model="fixed-model",
                    provider_request_id="context-call",
                    usage={"prompt_tokens": 4},
                ),
                ChatModelCallRecord(
                    operation=ChatModelOperation.CONTEXTUALIZE_QUERY,
                    model="fixed-model",
                    provider_request_id="repair-call",
                    usage={"prompt_tokens": 4},
                ),
            ),
            created_at=datetime.now(UTC),
            origin_attempt=1,
            rewrite_source=QueryRewriteSource.FALLBACK,
        )

        class Contextualizer:
            async def contextualize(self, value):
                del value
                return query_context

        retriever = _Retriever(_pack(context))
        runner, model, persister = _build(
            context=context,
            pack=retriever.pack,
            retriever=retriever,
            contextualizer=Contextualizer(),
        )

        result = await runner.execute(ChatExecutionCommand(context.lease))

        self.assertEqual(retriever.calls, 1)
        self.assertEqual(model.requests, [])
        self.assertEqual(persister.calls, 1)
        assert result.answering is not None
        assert result.answering.draft is not None
        self.assertEqual(
            result.answering.draft.control_reason.value, "no_usable_evidence"
        )

    async def test_answer_routes_cover_generation_refusal_and_repair(self) -> None:
        cases = (
            (
                "sufficient",
                ("evidence",),
                (_response(_answer(AnswerOutcome.ANSWERED)),),
            ),
            ("no_evidence", (), ()),
            (
                "repair_success",
                ("evidence",),
                (
                    _response("not-json"),
                    _response(_answer(AnswerOutcome.ANSWERED)),
                ),
            ),
            (
                "repair_fallback",
                ("evidence",),
                (
                    _response("not-json"),
                    _response('{"still":"invalid"}'),
                ),
            ),
        )
        for name, texts, responses in cases:
            with self.subTest(name=name):
                context = _context(insufficiency="partial_answer")
                pack = _pack(context, *texts)
                runner, model, persister = _build(
                    context=context,
                    pack=pack,
                    responses=responses,
                )
                result = await runner.execute(ChatExecutionCommand(context.lease))

                self.assertEqual(persister.calls, 1)
                self.assertTrue(result.artifacts["persisted"])
                self.assertEqual(len(model.requests), len(responses))

    async def test_errors_and_deadline_map_to_the_active_phase(self) -> None:
        context = _context()
        pack = _pack(context, "evidence")
        cases = (
            (
                "revision",
                {},
                _Retriever(
                    pack,
                    error=ChatPipelineExecutionError(
                        ErrorCode.CHAT_REVISION_MISMATCH,
                        phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                        diagnostic={"check": "frozen_revision"},
                    ),
                ),
                None,
                ErrorCode.CHAT_REVISION_MISMATCH,
                ChatPipelinePhase.RETRIEVE_EVIDENCE,
            ),
            (
                "deadline",
                {"deadline": 0.001, "loader": _Loader(context, delay=0.02)},
                None,
                None,
                ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED,
                ChatPipelinePhase.LOAD_CONTEXT,
            ),
            (
                "unexpected",
                {},
                None,
                _FailStep(),
                ErrorCode.CHAT_PIPELINE_STEP_FAILED,
                ChatPipelinePhase.ASSESS_EVIDENCE,
            ),
        )
        for name, overrides, retriever, assessor, code, phase in cases:
            with self.subTest(name=name):
                runner, _, _ = _build(
                    context=context,
                    pack=pack,
                    responses=(),
                    retriever=retriever,
                    assessor=assessor,
                    **overrides,
                )
                with self.assertRaises(ChatPipelineExecutionError) as raised:
                    await runner.execute(ChatExecutionCommand(context.lease))

                self.assertEqual(
                    (raised.exception.code, raised.exception.phase),
                    (code, phase),
                )
                self.assertNotIn("sensitive", str(raised.exception.diagnostic))

    async def test_graph_is_fixed_compiled_once_and_has_no_checkpointer(self) -> None:
        context = _context()
        runner, _, _ = _build(
            context=context,
            pack=_pack(context),
        )
        compiled_id = id(runner._graph)

        await runner.execute(ChatExecutionCommand(context.lease))

        self.assertEqual(id(runner._graph), compiled_id)
        self.assertIsNone(runner._graph.checkpointer)
        graph = runner._graph.get_graph()
        self.assertEqual(
            tuple(name for name in CHAT_GRAPH_NODES if name in graph.nodes),
            CHAT_GRAPH_NODES,
        )


if __name__ == "__main__":
    unittest.main()
