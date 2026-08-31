from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from rag_kb.domain import (
    ChatExecutionContext,
    ChatAgentTraceEvent,
    ChatPipelineExecutionError,
    ChatRunLease,
    ErrorCode,
    Evidence,
    EvidencePack,
    EvidenceScoreKind,
    GraphSearchResult,
    RetrievalDebug,
    RetrievalQueryPlan,
    RerankMode,
    RetrievalStrategy,
)
from rag_kb.services.chat_execution import ChatEvidenceRetriever
from rag_kb.retrieval.profile import exact_profile
from rag_kb.retrieval.profile import adaptive_graphiti_profile


def _context() -> ChatExecutionContext:
    run_id = uuid4()
    workspace_id = uuid4()
    return ChatExecutionContext(
        lease=ChatRunLease(
            run_id=run_id,
            workspace_id=workspace_id,
            claimed_by="worker",
            attempt=1,
            claimed_at=datetime.now(UTC),
        ),
        run_id=run_id,
        workspace_id=workspace_id,
        knowledge_base_id=uuid4(),
        session_id=uuid4(),
        user_message_id=uuid4(),
        assistant_message_id=uuid4(),
        index_revision_id=uuid4(),
        principal_id="principal",
        client_id="client",
        query="What is frozen?",
        effective_policy={"grounding_policy": "evidence_only"},
        retrieval_strategy=exact_profile(
            top_k=3, rerank_mode=RerankMode.NONE
        ).as_dict(),
        model_configuration={"requested_model": "fixed"},
        attempt=1,
    )


def _graph_evidence(
    context: ChatExecutionContext,
    *,
    chunk_id,
    rank: int = 1,
    graph_hop_count: int = 1,
) -> Evidence:
    return Evidence(
        rank=rank,
        index_chunk_id=chunk_id,
        indexed_document_version_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        index_revision_id=context.index_revision_id,
        ordinal=0,
        text="graph path evidence",
        source_location={},
        hierarchy={},
        source_metadata={},
        score=1.0,
        score_kind=EvidenceScoreKind.GRAPH_PATH,
        graph_path_id="path",
        graph_anchor_index_chunk_id=uuid4(),
        graph_hop_count=graph_hop_count,
        graph_path_rank=1,
    )


class ChatExecutionServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_admitted_graph_trace_counts_only_new_path_chunks(self) -> None:
        event = ChatAgentTraceEvent(
            tool="search_graph_relations",
            status="ok",
            tool_call_id="graph-overlap",
            refs=("ev_1", "ev_2"),
            count=2,
            retrieval_lane="graph_relations",
            route_reason_code="direct_relation",
            route_result_code="admitted",
            new_evidence_count=1,
            call_index=1,
            invocation_source="agent",
            duration_ms=42,
            candidate_count=16,
            path_count=1,
            returned_chunk_count=2,
            hop1_count=2,
            hop2_count=0,
            hop3_count=0,
        )

        self.assertEqual(event.new_evidence_count, 1)
        self.assertEqual(event.call_index, 1)

    def test_admitted_graph_trace_still_requires_new_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "must carry new evidence"):
            ChatAgentTraceEvent(
                tool="search_graph_relations",
                status="ok",
                tool_call_id="graph-empty",
                refs=(),
                count=0,
                retrieval_lane="graph_relations",
                route_reason_code="direct_relation",
                route_result_code="admitted",
                new_evidence_count=0,
                call_index=1,
                invocation_source="agent",
                duration_ms=10,
            )

    def test_graph_trace_requires_call_index_and_duration_for_agent_calls(self) -> None:
        with self.assertRaisesRegex(ValueError, "Graph trace route fields are incomplete"):
            ChatAgentTraceEvent(
                tool="search_graph_relations",
                status="ok",
                tool_call_id="graph-call",
                retrieval_lane="graph_relations",
                route_reason_code="relation_chain",
                route_result_code="no_evidence",
                new_evidence_count=0,
                duration_ms=10,
            )
        with self.assertRaisesRegex(ValueError, "requires a duration"):
            ChatAgentTraceEvent(
                tool="search_graph_relations",
                status="ok",
                tool_call_id="graph-call",
                retrieval_lane="graph_relations",
                route_reason_code="relation_chain",
                route_result_code="no_evidence",
                new_evidence_count=0,
                call_index=1,
                invocation_source="agent",
            )
        with self.assertRaisesRegex(ValueError, "cannot fake a duration"):
            ChatAgentTraceEvent(
                tool="search_graph_relations",
                status="ok",
                tool_call_id="guard_3",
                retrieval_lane="graph_relations",
                route_reason_code="relation_chain",
                route_result_code="no_evidence",
                new_evidence_count=0,
                call_index=1,
                invocation_source="legacy_guard",
                duration_ms=5,
            )

    async def test_adaptive_snapshot_uses_exact_simple_and_graph_method(self) -> None:
        context = _context()
        context = replace(
            context,
            retrieval_strategy=adaptive_graphiti_profile(
                top_k=3,
            ).as_dict(),
        )
        chunk_id = uuid4()
        evidence = _graph_evidence(context, chunk_id=chunk_id)

        class Retrieval:
            request = None
            graph_call = None

            async def retrieve(self, auth, request):
                del auth
                self.request = request
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=context.index_revision_id,
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                )

            async def search_graph_relations(self, auth, **kwargs):
                del auth
                self.graph_call = kwargs
                return GraphSearchResult(
                    "admitted",
                    (evidence,),
                    new_index_chunk_ids=(chunk_id,),
                    candidate_count=16,
                    path_count=1,
                    hydrated_chunk_count=2,
                    hop1_count=1,
                    hop2_count=0,
                    hop3_count=0,
                )

            async def search_graph_relations_capable(self, auth, **kwargs):
                del auth, kwargs
                return True

        retrieval = Retrieval()
        retriever = ChatEvidenceRetriever(retrieval)  # type: ignore[arg-type]
        await retriever.retrieve_query(context, "query")
        result = await retriever.search_graph_relations(
            context,
            "relation",
            excluded_index_chunk_ids=(),
        )

        self.assertIsNotNone(retrieval.request)
        self.assertIs(retrieval.request.strategy, RetrievalStrategy.EXACT_VECTOR)
        self.assertEqual(result.evidence, (evidence,))
        assert retrieval.graph_call is not None
        self.assertEqual(
            retrieval.graph_call["index_revision_id"], context.index_revision_id
        )
        self.assertEqual(retrieval.graph_call["edge_limit"], 16)
        self.assertEqual(retrieval.graph_call["source_chunk_target"], 12)
        self.assertEqual(retrieval.graph_call["source_chunk_limit"], 16)
        self.assertEqual(retrieval.graph_call["call_timeout_seconds"], 90)
        self.assertEqual(
            retrieval.graph_call["rerank_mode"],
            RerankMode.NONE,
        )

    async def test_retrieval_fails_closed_when_active_revision_moved(self) -> None:
        context = _context()

        class Retrieval:
            async def retrieve(self, auth, request):
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=uuid4(),
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                )

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            retriever = ChatEvidenceRetriever(Retrieval())  # type: ignore[arg-type]
            await retriever.retrieve(context)

        self.assertEqual(raised.exception.code, ErrorCode.CHAT_REVISION_MISMATCH)

    async def test_graph_result_accepts_complete_path_overlap_with_zero_new(
        self,
    ) -> None:
        context = replace(
            _context(),
            retrieval_strategy=adaptive_graphiti_profile(top_k=3).as_dict(),
        )
        chunk_id = uuid4()
        evidence = _graph_evidence(context, chunk_id=chunk_id)

        class Retrieval:
            async def search_graph_relations(self, auth, **kwargs):
                del auth, kwargs
                return GraphSearchResult(
                    "no_evidence",
                    (evidence,),
                    new_index_chunk_ids=(),
                    candidate_count=1,
                    path_count=1,
                    hydrated_chunk_count=1,
                    hop1_count=1,
                    hop2_count=0,
                    hop3_count=0,
                )

        result = await ChatEvidenceRetriever(  # type: ignore[arg-type]
            Retrieval()
        ).search_graph_relations(
            context,
            "relation",
            excluded_index_chunk_ids=(chunk_id,),
        )

        self.assertEqual(result.evidence, (evidence,))
        self.assertEqual(result.new_evidence_count, 0)
        self.assertEqual(result.route_result_code, "no_evidence")

    async def test_graph_admission_cannot_report_excluded_chunks_as_new(self) -> None:
        context = replace(
            _context(),
            retrieval_strategy=adaptive_graphiti_profile(top_k=3).as_dict(),
        )
        chunk_id = uuid4()

        class Retrieval:
            async def search_graph_relations(self, auth, **kwargs):
                del auth, kwargs
                return GraphSearchResult(
                    "admitted",
                    (_graph_evidence(context, chunk_id=chunk_id),),
                    new_index_chunk_ids=(chunk_id,),
                )

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await ChatEvidenceRetriever(Retrieval()).search_graph_relations(  # type: ignore[arg-type]
                context,
                "relation",
                excluded_index_chunk_ids=(chunk_id,),
            )

        self.assertEqual(raised.exception.code, ErrorCode.CHAT_REVISION_MISMATCH)

    async def test_graph_relations_capable_requires_adaptive_profile(self) -> None:
        context = _context()

        class Retrieval:
            async def search_graph_relations_capable(self, auth, **kwargs):
                del auth, kwargs
                return True

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await ChatEvidenceRetriever(Retrieval()).graph_relations_capable(  # type: ignore[arg-type]
                context,
            )

        self.assertEqual(raised.exception.code, ErrorCode.CHAT_CONTEXT_INVALID)

    async def test_native_image_only_evidence_is_retained_for_visual_preparation(
        self,
    ) -> None:
        context = _context()
        visual = Evidence(
            rank=1,
            index_chunk_id=uuid4(),
            indexed_document_version_id=uuid4(),
            document_id=uuid4(),
            document_version_id=uuid4(),
            index_revision_id=context.index_revision_id,
            ordinal=0,
            text="",
            source_location={"page_number": 1},
            hierarchy={},
            source_metadata={},
            score=0.5,
            document_display_name="Guide",
            document_original_filename="guide.png",
            modality="image",
            matched_representations=("native_image",),
        )

        class Retrieval:
            async def retrieve(self, auth, request):
                del auth
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=context.index_revision_id,
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                    evidence=(visual,),
                )

        retriever = ChatEvidenceRetriever(Retrieval())  # type: ignore[arg-type]
        result = await retriever.retrieve(context)

        self.assertEqual(result.evidence, (visual,))

    async def test_retrieval_preserves_internal_debug_for_terminal_diagnostics(
        self,
    ) -> None:
        context = _context()
        debug = RetrievalDebug(
            query_plan=RetrievalQueryPlan(
                workspace_id=context.workspace_id,
                knowledge_base_id=context.knowledge_base_id,
                strategy=RetrievalStrategy.EXACT_VECTOR,
                top_k=3,
            ),
            resolved_active_revision_id=context.index_revision_id,
            result_count=0,
            text_candidate_count=4,
            cross_modal_candidate_count=2,
            hydrated_relation_count=1,
            evidence_group_count=1,
        )

        class Retrieval:
            async def retrieve(self, auth, request):
                del auth
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=context.index_revision_id,
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                    debug=debug,
                )

        result = await ChatEvidenceRetriever(Retrieval()).retrieve(context)  # type: ignore[arg-type]

        self.assertIs(result.debug, debug)


if __name__ == "__main__":
    unittest.main()