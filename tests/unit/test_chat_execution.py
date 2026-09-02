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
    LexicalManifestStatus,
    ResourceNotFoundError,
    RetrievalDebug,
    RetrievalExecutionError,
    RetrievalQueryPlan,
    RerankMode,
    RetrievalStrategy,
    ServingDocumentEntry,
    ServingDocumentList,
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
        query="What is frozen?",
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

            async def retrieve(self, request):
                self.request = request
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=context.index_revision_id,
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                )

            async def search_graph_relations(self, **kwargs):
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

            async def search_graph_relations_capable(self, **kwargs):
                del kwargs
                return True

        retrieval = Retrieval()
        retriever = ChatEvidenceRetriever(retrieval)  # type: ignore[arg-type]
        await retriever.semantic_search(context, "query")
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
            async def retrieve(self, request):
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=uuid4(),
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                )

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            retriever = ChatEvidenceRetriever(Retrieval())  # type: ignore[arg-type]
            await retriever.semantic_search(context, context.query)

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
            async def search_graph_relations(self, **kwargs):
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
            async def search_graph_relations(self, **kwargs):
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
            async def search_graph_relations_capable(self, **kwargs):
                del kwargs
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
            async def retrieve(self, request):
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=context.index_revision_id,
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                    evidence=(visual,),
                )

        retriever = ChatEvidenceRetriever(Retrieval())  # type: ignore[arg-type]
        result = await retriever.semantic_search(context, context.query)

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
            async def retrieve(self, request):
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=context.index_revision_id,
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                    debug=debug,
                )

        result = await ChatEvidenceRetriever(Retrieval()).semantic_search(  # type: ignore[arg-type]
            context, context.query
        )

        self.assertIs(result.debug, debug)

    async def test_semantic_search_rejects_top_k_above_frozen_limit(self) -> None:
        context = _context()

        class Retrieval:
            async def retrieve(self, request):
                raise AssertionError("invalid override must fail before retrieve")

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await ChatEvidenceRetriever(Retrieval()).semantic_search(  # type: ignore[arg-type]
                context, "query", top_k_override=4
            )
        self.assertEqual(raised.exception.code, ErrorCode.CHAT_CONTEXT_INVALID)

    async def test_keyword_search_uses_frozen_top_k_and_translates_errors(
        self,
    ) -> None:
        context = _context()
        captured = {}

        class Retrieval:
            async def retrieve_lexical_only(self, request):
                captured["request"] = request
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=context.index_revision_id,
                    strategy=RetrievalStrategy.HYBRID,
                )

        pack = await ChatEvidenceRetriever(Retrieval()).keyword_search(  # type: ignore[arg-type]
            context, "ABC-42", top_k_override=2
        )
        self.assertEqual(captured["request"].query, "ABC-42")
        self.assertEqual(captured["request"].top_k, 2)
        self.assertEqual(pack.index_revision_id, context.index_revision_id)

        class Mismatch:
            async def retrieve_lexical_only(self, request):
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=uuid4(),
                    strategy=RetrievalStrategy.HYBRID,
                )

        with self.assertRaises(ChatPipelineExecutionError) as mismatch:
            await ChatEvidenceRetriever(Mismatch()).keyword_search(  # type: ignore[arg-type]
                context, "ABC-42"
            )
        self.assertEqual(mismatch.exception.code, ErrorCode.CHAT_REVISION_MISMATCH)

        class Failing:
            async def retrieve_lexical_only(self, request):
                del request
                raise RetrievalExecutionError(
                    ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                    diagnostic={"check": "lexical_manifest_hash"},
                )

        with self.assertRaises(ChatPipelineExecutionError) as failed:
            await ChatEvidenceRetriever(Failing()).keyword_search(  # type: ignore[arg-type]
                context, "ABC-42"
            )
        self.assertEqual(failed.exception.code, ErrorCode.INDEX_REVISION_INCOMPATIBLE)
        self.assertEqual(
            failed.exception.diagnostic,
            {"check": "lexical_manifest_hash"},
        )

    async def test_keyword_search_capable_true_false_and_swallowed_error(
        self,
    ) -> None:
        context = _context()

        class Ready:
            def hybrid_request_enabled(self):
                return True

            async def lexical_manifest_status(self, knowledge_base_id):
                del knowledge_base_id
                return LexicalManifestStatus(
                    resolved_active_revision_id=context.index_revision_id,
                    serving_target_count=1,
                    manifested_target_count=1,
                )

        self.assertTrue(
            await ChatEvidenceRetriever(Ready()).keyword_search_capable(  # type: ignore[arg-type]
                context
            )
        )

        class Disabled:
            def hybrid_request_enabled(self):
                return False

            async def lexical_manifest_status(self, knowledge_base_id):
                raise AssertionError("disabled hybrid must not probe manifests")

        self.assertFalse(
            await ChatEvidenceRetriever(Disabled()).keyword_search_capable(  # type: ignore[arg-type]
                context
            )
        )

        class Incomplete:
            def hybrid_request_enabled(self):
                return True

            async def lexical_manifest_status(self, knowledge_base_id):
                del knowledge_base_id
                return LexicalManifestStatus(
                    resolved_active_revision_id=context.index_revision_id,
                    serving_target_count=2,
                    manifested_target_count=1,
                )

        self.assertFalse(
            await ChatEvidenceRetriever(Incomplete()).keyword_search_capable(  # type: ignore[arg-type]
                context
            )
        )

        class Failing:
            def hybrid_request_enabled(self):
                return True

            async def lexical_manifest_status(self, knowledge_base_id):
                del knowledge_base_id
                raise RetrievalExecutionError(ErrorCode.INTERNAL_SERVER_ERROR)

        self.assertFalse(
            await ChatEvidenceRetriever(Failing()).keyword_search_capable(  # type: ignore[arg-type]
                context
            )
        )

    async def test_read_chunk_context_delegates_and_translates_errors(self) -> None:
        context = _context()
        anchor = Evidence(
            rank=1,
            index_chunk_id=uuid4(),
            indexed_document_version_id=uuid4(),
            document_id=uuid4(),
            document_version_id=uuid4(),
            index_revision_id=context.index_revision_id,
            ordinal=0,
            text="anchor",
            source_location={},
            hierarchy={},
            source_metadata={},
            score=0.9,
        )
        neighbor = replace(
            anchor,
            rank=1,
            index_chunk_id=uuid4(),
            ordinal=1,
            text="neighbor",
            score=0.0,
            score_kind=EvidenceScoreKind.ADJACENCY,
            vector_similarity=None,
            adjacency_anchor_index_chunk_id=anchor.index_chunk_id,
            adjacency_offset=1,
        )

        class Retrieval:
            async def retrieve_adjacent_evidence(self, **kwargs):
                self.kwargs = kwargs
                return (neighbor,)

        retrieval = Retrieval()
        result = await ChatEvidenceRetriever(retrieval).read_chunk_context(  # type: ignore[arg-type]
            context, (anchor,)
        )
        self.assertEqual(result, (neighbor,))
        self.assertEqual(
            retrieval.kwargs["index_revision_id"], context.index_revision_id
        )

        class Failing:
            async def retrieve_adjacent_evidence(self, **kwargs):
                del kwargs
                raise RetrievalExecutionError(
                    ErrorCode.INTERNAL_SERVER_ERROR,
                    diagnostic={"check": "adjacency_anchor_scope"},
                )

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await ChatEvidenceRetriever(Failing()).read_chunk_context(  # type: ignore[arg-type]
                context, (anchor,)
            )
        self.assertEqual(raised.exception.code, ErrorCode.INTERNAL_SERVER_ERROR)

    async def test_list_documents_none_and_revision_guards(self) -> None:
        context = _context()
        listed = ServingDocumentList(
            resolved_active_revision_id=context.index_revision_id,
            entries=(
                ServingDocumentEntry(
                    document_id=uuid4(),
                    document_version_id=uuid4(),
                    indexed_document_version_id=uuid4(),
                    display_name="Report",
                    original_filename="report.pdf",
                    version_number=1,
                    chunk_count=2,
                ),
            ),
        )

        class Retrieval:
            async def list_serving_documents(self, knowledge_base_id):
                del knowledge_base_id
                return listed

        result = await ChatEvidenceRetriever(Retrieval()).list_documents(  # type: ignore[arg-type]
            context
        )
        self.assertEqual(result, listed)

        class Missing:
            async def list_serving_documents(self, knowledge_base_id):
                del knowledge_base_id
                return None

        with self.assertRaises(ResourceNotFoundError):
            await ChatEvidenceRetriever(Missing()).list_documents(context)  # type: ignore[arg-type]

        class Moved:
            async def list_serving_documents(self, knowledge_base_id):
                del knowledge_base_id
                return replace(listed, resolved_active_revision_id=uuid4())

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await ChatEvidenceRetriever(Moved()).list_documents(context)  # type: ignore[arg-type]
        self.assertEqual(raised.exception.code, ErrorCode.CHAT_REVISION_MISMATCH)


if __name__ == "__main__":
    unittest.main()

