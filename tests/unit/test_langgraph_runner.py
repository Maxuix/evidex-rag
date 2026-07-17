from __future__ import annotations

import asyncio
import json
import unittest

from rag_kb.answering import (
    AnswerGenerationStep,
    AnswerStructureValidationStep,
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
)
from rag_kb.workflows import LangGraphRunner
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

    async def retrieve(self, context):
        del context
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
            artifacts={**state.artifacts, "persisted": True},
        )


class _FailStep:
    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        del state
        raise RuntimeError("sensitive step content")


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
        LangGraphRunner(*values, deadline_seconds=deadline),  # type: ignore[arg-type]
        model,
        persister,
    )


class LangGraphRunnerTests(unittest.IsolatedAsyncioTestCase):
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
