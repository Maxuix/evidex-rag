from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import replace
import unittest
from uuid import uuid4

from rag_kb.answering.agent import AGENT_TRACE_ARTIFACT, NativeToolCallingAgent
from rag_kb.domain import (
    AnswerOutcome,
    ChatAgentBudget,
    ChatExecutionContext,
    ChatModelResponse,
    ChatRunLease,
    ChatToolCall,
    Evidence,
    EvidencePack,
    EvidenceScoreKind,
    RetrievalStrategy,
)
from rag_kb.services.chat_visuals import VisualEvidencePreparationStep


class _Model:
    def __init__(self, *calls: ChatToolCall | None) -> None:
        self.calls = list(calls)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        value = self.calls.pop(0)
        return ChatModelResponse(
            content="" if value is not None else "plain assistant text",
            model="fixed-model",
            finish_reason="tool_calls" if value is not None else "stop",
            provider_request_id=f"request-{len(self.requests)}",
            usage={"total_tokens": 5},
            tool_calls=((value,) if value is not None else ()),
        )


class _Retriever:
    def __init__(self, pack: EvidencePack) -> None:
        self.pack = pack
        self.queries = []

    async def retrieve_query(self, context, query, *, top_k_override=None):
        del context, top_k_override
        self.queries.append(query)
        return self.pack


def _context() -> ChatExecutionContext:
    run_id = uuid4()
    workspace_id = uuid4()
    retrieval = {
        "profile_version": "exact_vector_v2",
        "strategy": "exact_vector",
        "top_k": 3,
        "rerank_mode": "none",
    }
    return ChatExecutionContext(
        lease=ChatRunLease(run_id, workspace_id, "worker", 1, datetime.now(UTC)),
        run_id=run_id,
        workspace_id=workspace_id,
        knowledge_base_id=uuid4(),
        session_id=uuid4(),
        user_message_id=uuid4(),
        assistant_message_id=uuid4(),
        index_revision_id=uuid4(),
        principal_id="principal",
        client_id="client",
        query="What was the revenue and change?",
        effective_policy={"insufficiency_policy": "partial_answer"},
        retrieval_strategy=retrieval,
        model_configuration={"resolved_model": "fixed-model", "max_output_tokens": 2048},
        attempt=1,
    )


def _pack(context: ChatExecutionContext, *, text: str = "Revenue was 10 in 2025 and 5 in 2024.") -> EvidencePack:
    return EvidencePack(
        knowledge_base_id=context.knowledge_base_id,
        index_revision_id=context.index_revision_id,
        strategy=RetrievalStrategy.EXACT_VECTOR,
        evidence=(
            Evidence(
                rank=1,
                index_chunk_id=uuid4(),
                indexed_document_version_id=uuid4(),
                document_id=uuid4(),
                document_version_id=uuid4(),
                index_revision_id=context.index_revision_id,
                ordinal=0,
                text=text,
                source_location={"page": 1},
                hierarchy={},
                source_metadata={},
                score=0.9,
                score_kind=EvidenceScoreKind.COSINE_SIMILARITY,
                vector_similarity=0.9,
                document_display_name="Report",
                document_original_filename="report.pdf",
            ),
        ),
    )


def _agent(model, retriever, *, budget=ChatAgentBudget()):
    return NativeToolCallingAgent(
        model,
        retriever,
        VisualEvidencePreparationStep(None),
        min_cosine_similarity=0.2,
        min_rerank_score=0.45,
        cross_modal_min_cosine_similarity=0.25,
        budget=budget,
    )


class NativeToolCallingAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_issues_stable_ref_and_submit_answer_completes(self) -> None:
        context = _context()
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["revenue"]}),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "Revenue was 10 in 2025.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(len(state.answering.rendered.citations), 1)
        self.assertEqual([tool.name for tool in model.requests[0].tools], [
            "search_knowledge_base", "calculate", "submit_answer"
        ])
        self.assertIn('"evidence_ref":"ev_1"', model.requests[1].messages[-1].content)
        self.assertIn('"groups":[{"query":"revenue"', model.requests[1].messages[-1].content)
        self.assertEqual(state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls, 1)

    async def test_invalid_claim_is_removed_and_valid_claim_is_salvaged_as_partial(self) -> None:
        context = _context()
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["revenue"]}),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {"text": "Revenue was 10.", "kind": "fact", "evidence_refs": ["ev_1"], "calculation_refs": []},
                        {"text": "Unsupported.", "kind": "fact", "evidence_refs": ["ev_other_run"], "calculation_refs": []},
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.PARTIAL)
        self.assertEqual(len(state.answering.validated.claims), 1)
        self.assertIn("invalid evidence", state.answering.rendered.content)
        self.assertEqual(state.artifacts[AGENT_TRACE_ARTIFACT].events[-1].status, "salvaged")

    async def test_calculation_ref_expands_to_original_evidence_citation(self) -> None:
        context = _context()
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["revenue"]}),
            ChatToolCall("calc-1", "calculate", {"expression": "10-5", "evidence_refs": ["ev_1"]}),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {"text": "Revenue increased by 5.", "kind": "fact", "evidence_refs": [], "calculation_refs": ["calc_1"]}
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(len(state.answering.rendered.citations), 1)
        self.assertIn('"calculation_ref":"calc_1"', model.requests[2].messages[-1].content)

    async def test_last_round_forces_submit_and_refuses_when_protocol_is_ignored(self) -> None:
        context = _context()
        model = _Model(
            None,
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["revenue"]}),
        )
        budget = ChatAgentBudget(model_rounds=2, retrieval_calls=1, calculation_calls=0)
        context = replace(
            context,
            agent_configuration={
                "version": "native_tool_calling_agent_v1",
                "budget": budget.as_dict(),
            },
        )

        state = await _agent(model, _Retriever(_pack(context)), budget=budget).run(context)

        self.assertEqual(model.requests[-1].tool_choice, "submit_answer")
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.answering.rendered.citations, ())

    async def test_query_budget_rejects_a_batch_before_any_retrieval(self) -> None:
        context = _context()
        budget = ChatAgentBudget(
            model_rounds=2,
            retrieval_calls=2,
            calculation_calls=0,
        )
        context = replace(
            context,
            agent_configuration={
                "version": "native_tool_calling_agent_v1",
                "budget": budget.as_dict(),
            },
        )
        retriever = _Retriever(_pack(context))
        model = _Model(
            ChatToolCall(
                "search-1",
                "search_knowledge_base",
                {"queries": ["one", "two", "three"]},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever, budget=budget).run(context)

        self.assertEqual(retriever.queries, [])
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls, 0)


if __name__ == "__main__":
    unittest.main()
