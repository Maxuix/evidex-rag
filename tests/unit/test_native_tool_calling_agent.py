from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import unittest
from uuid import uuid4

from rag_kb.answering.agent import (
    AGENT_TRACE_ARTIFACT,
    NativeToolCallingAgent,
    _has_graph_relation_signal,
    _initial_messages,
    _search_arguments,
    _supplement_arguments,
    _tools,
)
from rag_kb.domain import (
    AnswerOutcome,
    CHAT_GRAPHITI_ROUTE_REASONS,
    ChatAgentBudget,
    ChatExecutionContext,
    ChatModelExecutionError,
    ChatModelResponse,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatRunLease,
    ChatToolCall,
    ConversationTurn,
    Evidence,
    EvidenceAsset,
    EvidencePack,
    EvidenceScoreKind,
    ErrorCode,
    GraphitiSupplementResult,
    IndexAssetContent,
    IndexAssetSnapshot,
    RelatedVisualEvidence,
    RetrievalStrategy,
)
from rag_kb.memory import select_conversation_context
from rag_kb.services.chat_visuals import VisualEvidencePreparationStep


class _Model:
    def __init__(
        self,
        *calls: ChatToolCall | tuple[ChatToolCall, ...] | None | BaseException,
    ) -> None:
        self.calls = list(calls)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        value = self.calls.pop(0)
        if isinstance(value, BaseException):
            raise value
        tool_calls = value if isinstance(value, tuple) else ((value,) if value else ())
        return ChatModelResponse(
            content="" if tool_calls else "plain assistant text",
            model="fixed-model",
            finish_reason="tool_calls" if tool_calls else "stop",
            provider_request_id=f"request-{len(self.requests)}",
            usage={"total_tokens": 5},
            tool_calls=tool_calls,
        )


class _Retriever:
    def __init__(self, pack: EvidencePack) -> None:
        self.pack = pack
        self.queries = []

    async def retrieve_query(self, context, query, *, top_k_override=None):
        del context, top_k_override
        self.queries.append(query)
        return self.pack


class _QueryRetriever:
    def __init__(self, packs_by_query: dict[str, EvidencePack]) -> None:
        self.packs_by_query = packs_by_query
        self.queries = []

    async def retrieve_query(self, context, query, *, top_k_override=None):
        del context, top_k_override
        self.queries.append(query)
        return self.packs_by_query[query]


class _AdaptiveRetriever(_Retriever):
    def __init__(
        self,
        pack: EvidencePack,
        supplement: GraphitiSupplementResult,
    ) -> None:
        super().__init__(pack)
        self.supplement = supplement
        self.supplement_queries = []
        self.supplement_exclusions = []

    async def retrieve_graphiti_supplement(
        self,
        context,
        query,
        *,
        excluded_index_chunk_ids,
    ):
        del context
        self.supplement_queries.append(query)
        self.supplement_exclusions.append(tuple(excluded_index_chunk_ids))
        return self.supplement


class _AssetReader:
    def __init__(self, content: IndexAssetContent) -> None:
        self.content = content
        self.asset_ids = []

    async def read(self, context, asset_id):
        del context
        self.asset_ids.append(asset_id)
        return self.content


class _AssetMapReader:
    def __init__(self, *contents: IndexAssetContent) -> None:
        self.contents = {item.snapshot.id: item for item in contents}
        self.asset_ids = []

    async def read(self, context, asset_id):
        del context
        self.asset_ids.append(asset_id)
        return self.contents[asset_id]


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
        model_configuration={"resolved_model": "fixed-model", "max_tokens": 2048},
        attempt=1,
    )


def _adaptive_context() -> ChatExecutionContext:
    return replace(
        _context(),
        retrieval_strategy={
            "profile_version": "adaptive_graphiti_v2",
            "strategy": "exact_vector",
            "top_k": 3,
            "rerank_mode": "none",
            "router": "native_agent_path_guard_v2",
            "augmentation": "graphiti_path_v2",
        },
    )


def _pack(
    context: ChatExecutionContext,
    *,
    text: str = "Revenue was 10 in 2025 and 5 in 2024.",
    count: int = 1,
) -> EvidencePack:
    return EvidencePack(
        knowledge_base_id=context.knowledge_base_id,
        index_revision_id=context.index_revision_id,
        strategy=RetrievalStrategy.EXACT_VECTOR,
        evidence=tuple(
            Evidence(
                rank=index,
                index_chunk_id=uuid4(),
                indexed_document_version_id=uuid4(),
                document_id=uuid4(),
                document_version_id=uuid4(),
                index_revision_id=context.index_revision_id,
                ordinal=index - 1,
                text=text,
                source_location={"page": 1},
                hierarchy={},
                source_metadata={},
                score=0.9,
                score_kind=EvidenceScoreKind.COSINE_SIMILARITY,
                vector_similarity=0.9,
                document_display_name="Report",
                document_original_filename="report.pdf",
            )
            for index in range(1, count + 1)
        ),
    )


def _graphiti_pack(
    context: ChatExecutionContext,
    *,
    text: str = "Graphiti source",
    hop_count: int = 2,
) -> EvidencePack:
    base = _pack(context, text=text)
    source = base.evidence[0]
    return replace(
        base,
        evidence=(
            replace(
                source,
                score=1.0,
                score_kind=EvidenceScoreKind.GRAPH_PATH,
                vector_similarity=None,
                graph_path_id="supplement-path",
                graph_anchor_index_chunk_id=source.index_chunk_id,
                graph_hop_count=hop_count,
                graph_path_rank=1,
                matched_representations=("graph_path", "text"),
            ),
        ),
    )


def _agent(model, retriever, *, visual_preparer=None):
    return NativeToolCallingAgent(
        model,
        retriever,
        (
            visual_preparer
            if visual_preparer is not None
            else VisualEvidencePreparationStep(None)
        ),
        min_cosine_similarity=0.2,
        min_rerank_score=0.45,
        cross_modal_min_cosine_similarity=0.25,
    )


def _tool_payload(request, tool_call_id: str) -> dict:
    for message in request.messages:
        if message.role == "tool" and message.tool_call_id == tool_call_id:
            return json.loads(message.content)
    raise AssertionError(f"tool result not found: {tool_call_id}")


def _native_visual_pack(
    context: ChatExecutionContext,
) -> tuple[EvidencePack, IndexAssetContent, bytes]:
    image_bytes = b"native-image-evidence"
    checksum = hashlib.sha256(image_bytes).hexdigest()
    asset_id = uuid4()
    asset = EvidenceAsset(
        id=asset_id,
        media_type="image/png",
        checksum_sha256=checksum,
        content_url=f"/api/v1/index-assets/{asset_id}/content",
        width=2,
        height=2,
    )
    base_pack = _pack(context)
    base_evidence = base_pack.evidence[0]
    visual_evidence = replace(
        base_evidence,
        text="",
        modality="image",
        asset=asset,
        evidence_group_key="native-image-1",
        matched_representations=("native_image",),
        cross_modal_rank=1,
    )
    pack = replace(base_pack, evidence=(visual_evidence,))
    content = IndexAssetContent(
        snapshot=IndexAssetSnapshot(
            id=asset_id,
            workspace_id=context.workspace_id,
            kb_id=context.knowledge_base_id,
            document_id=visual_evidence.document_id,
            document_version_id=visual_evidence.document_version_id,
            indexed_document_version_id=(
                visual_evidence.indexed_document_version_id
            ),
            storage_uri=f"memory://{asset_id}",
            media_type=asset.media_type,
            checksum_sha256=checksum,
            size_bytes=len(image_bytes),
        ),
        content=image_bytes,
    )
    return pack, content, image_bytes


def _same_unit_table_visual_pack(
    context: ChatExecutionContext,
    table_text: str,
    *,
    image_bytes: bytes = b"same-unit-table-image",
) -> tuple[EvidencePack, IndexAssetContent, EvidenceAsset, bytes]:
    checksum = hashlib.sha256(image_bytes).hexdigest()
    asset_id = uuid4()
    asset = EvidenceAsset(
        id=asset_id,
        media_type="image/png",
        checksum_sha256=checksum,
        content_url=f"/api/v1/index-assets/{asset_id}/content",
        width=4,
        height=3,
    )
    base_pack = _pack(context, text=table_text)
    base_evidence = base_pack.evidence[0]
    related = RelatedVisualEvidence(
        visual_unit_id=base_evidence.index_chunk_id,
        asset=asset,
        relation_type="table_of",
        relation_confidence_micros=1_000_000,
        relation_provenance="table_identity_v2",
        evidence_group_key="table:1",
        figure_label="Table 1",
        parent_chunk_id=base_evidence.index_chunk_id,
        modality="table",
        source_location={"page": 1},
        text_space_rank=1,
        cross_modal_rank=1,
    )
    table_evidence = replace(
        base_evidence,
        modality="table",
        asset=asset,
        evidence_group_key="table:1",
        matched_representations=("table_text",),
        text_space_rank=1,
        cross_modal_rank=1,
        related_visuals=(related,),
    )
    pack = replace(base_pack, evidence=(table_evidence,))
    content = IndexAssetContent(
        snapshot=IndexAssetSnapshot(
            id=asset_id,
            workspace_id=context.workspace_id,
            kb_id=context.knowledge_base_id,
            document_id=table_evidence.document_id,
            document_version_id=table_evidence.document_version_id,
            indexed_document_version_id=table_evidence.indexed_document_version_id,
            storage_uri=f"memory://{asset_id}",
            media_type=asset.media_type,
            checksum_sha256=checksum,
            size_bytes=len(image_bytes),
        ),
        content=image_bytes,
    )
    return pack, content, asset, image_bytes


class NativeToolCallingAgentTests(unittest.IsolatedAsyncioTestCase):
    def test_adaptive_prompt_does_not_claim_a_fixed_tool_count(self) -> None:
        prompt = _initial_messages(
            _adaptive_context(),
            ChatAgentBudget(),
            adaptive=True,
        )[0].content

        self.assertNotIn("three supplied tools", prompt)
        self.assertIn("currently supplied tools", prompt)
        self.assertIn("one-to-three-hop chains", prompt)

    def test_route_reason_contract_has_no_unreachable_reason(self) -> None:
        self.assertEqual(
            CHAT_GRAPHITI_ROUTE_REASONS,
            frozenset(
                {
                    "cross_document_relation_gap",
                    "entity_alias_gap",
                    "relation_chain_gap",
                }
            ),
        )
        self.assertEqual(
            _supplement_arguments(
                {
                    "query": "relation",
                    "route_reason_code": "relation_chain_gap",
                }
            ),
            ("relation", "relation_chain_gap"),
        )
        self.assertIsNone(
            _supplement_arguments(
                {
                    "query": "relation",
                    "route_reason_code": "relational_query_without_simple_evidence",
                }
            )
        )

    def test_adaptive_schema_parser_and_late_visibility_are_consistent(self) -> None:
        initial = _tools(
            adaptive=True,
            graphiti_enabled=False,
            round_number=1,
            max_model_rounds=8,
        )
        eligible = _tools(
            adaptive=True,
            graphiti_enabled=True,
            round_number=6,
            max_model_rounds=8,
        )
        late = _tools(
            adaptive=True,
            graphiti_enabled=True,
            round_number=7,
            max_model_rounds=8,
        )
        self.assertNotIn("graphiti_supplement", [tool.name for tool in initial])
        self.assertIn("graphiti_supplement", [tool.name for tool in eligible])
        self.assertIn("graphiti_supplement", [tool.name for tool in late])
        search_schema = next(
            tool.input_schema for tool in eligible if tool.name == "search_knowledge_base"
        )
        supplement_schema = next(
            tool.input_schema for tool in eligible if tool.name == "graphiti_supplement"
        )
        self.assertEqual(tuple(search_schema["required"]), ("queries",))
        self.assertEqual(
            set(supplement_schema["required"]), {"query", "route_reason_code"}
        )
        self.assertEqual(_search_arguments({"queries": ["one", "two"]}), ("one", "two"))
        self.assertIsNone(
            _search_arguments(
                {
                    "retrieval_lane": "simple",
                    "queries": ["one"],
                }
            )
        )
        self.assertEqual(
            _supplement_arguments(
                {
                    "query": "relation",
                    "route_reason_code": "cross_document_relation_gap",
                }
            ),
            ("relation", "cross_document_relation_gap"),
        )
        self.assertEqual(
            _supplement_arguments(
                {
                    "query": "relation",
                    "route_reason_code": "relation_chain_gap",
                }
            ),
            ("relation", "relation_chain_gap"),
        )

    def test_only_model_rounds_are_configured_as_a_loop_guard(self) -> None:
        self.assertEqual(ChatAgentBudget().as_dict(), {"max_model_rounds": 8})
        for value in (True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ChatAgentBudget(max_model_rounds=value)

    async def test_agent_configuration_is_strictly_current_v2(self) -> None:
        invalid_configurations = (
            {
                "version": "native_tool_calling_agent_v1",
                "budget": {"max_model_rounds": 8},
            },
            {
                "version": "native_tool_calling_agent_v2",
                "budget": {"max_model_rounds": 8, "retrieval_calls": 6},
            },
            {
                "version": "native_tool_calling_agent_v2",
                "budget": {"max_model_rounds": True},
            },
            {
                "version": "native_tool_calling_agent_v2",
                "budget": {"max_model_rounds": "8"},
            },
            {
                "version": "native_tool_calling_agent_v2",
                "budget": {"max_model_rounds": 8.0},
            },
        )
        for configuration in invalid_configurations:
            with self.subTest(configuration=configuration):
                context = replace(_context(), agent_configuration=configuration)
                model = _Model()
                with self.assertRaises(ChatPipelineExecutionError) as raised:
                    await _agent(model, _Retriever(_pack(context))).run(context)
                self.assertEqual(raised.exception.code, ErrorCode.CHAT_CONTEXT_INVALID)
                self.assertEqual(model.requests, [])

    async def test_initial_request_includes_session_history_in_chronological_order(
        self,
    ) -> None:
        chronological = (
            ConversationTurn(
                uuid4(),
                "What is retrieval-augmented generation?",
                uuid4(),
                "It combines retrieval with generation.",
            ),
            ConversationTurn(
                uuid4(),
                "What is its main benefit?",
                uuid4(),
                "It can ground answers in retrieved sources.",
            ),
        )
        context = replace(
            _context(),
            query="What about its limitations?",
            conversation_context=select_conversation_context(
                tuple(reversed(chronological))
            ),
        )
        model = _Model(
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            )
        )

        await _agent(model, _Retriever(_pack(context))).run(context)

        messages = model.requests[0].messages
        self.assertIn("conversation history", messages[0].content)
        self.assertIn("Prior assistant messages are never evidence", messages[0].content)
        self.assertEqual(
            [(message.role, message.content) for message in messages[1:]],
            [
                ("user", chronological[0].user_content),
                ("assistant", chronological[0].assistant_content),
                ("user", chronological[1].user_content),
                ("assistant", chronological[1].assistant_content),
                ("user", context.query),
            ],
        )

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
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].budget.as_dict(),
            {"max_model_rounds": 8},
        )

    async def test_agent_keeps_lexical_rrf_evidence_without_cosine_gate(self) -> None:
        context = _context()
        base = _pack(context).evidence[0]
        lexical = replace(
            base,
            score=0.02,
            score_kind=EvidenceScoreKind.RECIPROCAL_RANK_FUSION,
            vector_similarity=0.10,
            lexical_score=0.9,
            lexical_rank=1,
            fusion_score=0.02,
        )
        pack = replace(_pack(context), evidence=(lexical,))
        model = _Model(
            ChatToolCall(
                "search-lexical",
                "search_knowledge_base",
                {"queries": ["revenue"]},
            ),
            ChatToolCall(
                "submit-lexical",
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

        state = await _agent(model, _Retriever(pack)).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertIn('"evidence_ref":"ev_1"', model.requests[1].messages[-1].content)

    async def test_adaptive_graphiti_is_simple_first_and_appends_one_supplement(self) -> None:
        context = _adaptive_context()
        simple_pack = _pack(context, text="Simple source", count=1)
        supplement_pack = _graphiti_pack(context, text="Graphiti source")
        retriever = _AdaptiveRetriever(
            simple_pack,
            GraphitiSupplementResult("admitted", supplement_pack.evidence),
        )
        model = _Model(
            ChatToolCall(
                "simple-1",
                "search_knowledge_base",
                {"queries": ["revenue"]},
            ),
            ChatToolCall(
                "graph-1",
                "graphiti_supplement",
                {
                    "query": "revenue relation",
                    "route_reason_code": "cross_document_relation_gap",
                },
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The relation is supported by Graphiti.",
                            "kind": "fact",
                            "evidence_refs": ["ev_2"],
                            "calculation_refs": [],
                        },
                        {
                            "text": "Simple and Graphiti evidence agree.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1", "ev_2"],
                            "calculation_refs": [],
                        },
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.queries, ["revenue"])
        self.assertEqual(retriever.supplement_queries, ["revenue relation"])
        self.assertEqual(state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls, 2)
        graph_event = state.artifacts[AGENT_TRACE_ARTIFACT].events[1]
        self.assertEqual(graph_event.tool, "graphiti_supplement")
        self.assertEqual(graph_event.retrieval_lane, "graphiti_supplement")
        self.assertEqual(graph_event.route_result_code, "admitted")
        self.assertEqual(graph_event.new_evidence_count, 1)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(len(state.answering.rendered.citations), 2)
        self.assertIn(
            "graphiti_supplement",
            [tool.name for tool in model.requests[1].tools],
        )
        self.assertNotIn(
            "retrieval_lane",
            model.requests[0].tools[0].input_schema["properties"],
        )
        self.assertNotIn("edge-fact", model.requests[2].messages[-1].content)

    async def test_adaptive_submit_guard_uses_original_question_and_rechecks_draft(self) -> None:
        context = replace(
            _adaptive_context(),
            query="WTC-7 最终属于哪个集团？",
        )
        retriever = _AdaptiveRetriever(
            _pack(context, text="Simple source", count=1),
            GraphitiSupplementResult(
                "admitted",
                _graphiti_pack(context, text="Missing path source").evidence,
            ),
        )
        first_submission = {
            "outcome": "answered",
            "claims": [
                {
                    "text": "The Simple source appears sufficient.",
                    "kind": "fact",
                    "evidence_refs": ["ev_1"],
                    "calculation_refs": [],
                }
            ],
            "unanswered": [],
        }
        model = _Model(
            ChatToolCall("simple", "search_knowledge_base", {"queries": ["short query"]}),
            ChatToolCall("early-submit", "submit_answer", first_submission),
            ChatToolCall(
                "rechecked-submit",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The complete path uses both sources.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1", "ev_2"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.supplement_queries, [context.query])
        self.assertEqual(len(model.requests), 3)
        self.assertIn("Missing path source", model.requests[2].messages[-1].content)
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        graph_events = [
            event
            for event in trace.events
            if event.retrieval_lane == "graphiti_supplement"
        ]
        self.assertEqual(len(graph_events), 1)
        self.assertEqual(graph_events[0].route_result_code, "admitted")
        self.assertEqual(trace.retrieval_calls, 2)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)

    async def test_adaptive_submit_guard_does_not_route_without_new_path_evidence(self) -> None:
        context = replace(
            _adaptive_context(),
            query="甲公司最终属于哪个集团？",
        )
        retriever = _AdaptiveRetriever(
            _pack(context, text="Direct source", count=1),
            GraphitiSupplementResult("no_new_evidence"),
        )
        model = _Model(
            ChatToolCall("simple", "search_knowledge_base", {"queries": ["direct fact"]}),
            ChatToolCall(
                "submit",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The direct fact is supported.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.supplement_queries, [context.query])
        self.assertEqual(len(model.requests), 2)
        graph_events = tuple(
            event
            for event in state.artifacts[AGENT_TRACE_ARTIFACT].events
            if event.retrieval_lane == "graphiti_supplement"
        )
        self.assertEqual(len(graph_events), 1)
        self.assertEqual(graph_events[0].route_result_code, "no_new_evidence")
        self.assertEqual(graph_events[0].new_evidence_count, 0)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)

    async def test_adaptive_submit_guard_does_not_surface_a_single_hop(self) -> None:
        context = replace(
            _adaptive_context(),
            query="甲公司控股的乙公司的母公司是谁？",
        )
        retriever = _AdaptiveRetriever(
            _pack(context, text="Direct source", count=1),
            GraphitiSupplementResult(
                "admitted",
                _graphiti_pack(
                    context,
                    text="Unrelated one-hop source",
                    hop_count=1,
                ).evidence,
            ),
        )
        model = _Model(
            ChatToolCall("simple", "search_knowledge_base", {"queries": ["direct fact"]}),
            ChatToolCall(
                "submit",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The direct fact is supported.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(len(model.requests), 2)
        graph_events = tuple(
            event
            for event in state.artifacts[AGENT_TRACE_ARTIFACT].events
            if event.retrieval_lane == "graphiti_supplement"
        )
        self.assertEqual(len(graph_events), 1)
        self.assertEqual(graph_events[0].route_result_code, "no_new_evidence")
        self.assertEqual(graph_events[0].new_evidence_count, 0)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)

    async def test_adaptive_submit_guard_skips_a_plain_direct_fact(self) -> None:
        context = _adaptive_context()
        retriever = _AdaptiveRetriever(
            _pack(context, text="Revenue was 10.", count=1),
            GraphitiSupplementResult("no_new_evidence"),
        )
        model = _Model(
            ChatToolCall("simple", "search_knowledge_base", {"queries": ["revenue"]}),
            ChatToolCall(
                "submit",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "Revenue was 10.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(
            retriever.supplement_queries, ["What was the revenue and change?"]
        )
        self.assertEqual(state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls, 2)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)

    def test_graph_relation_signal_covers_generic_relation_families(self) -> None:
        questions = (
            "某设备所属企业的总部在哪里？",
            "某公司合作的服务商是谁？",
            "某基金参股的企业承建了什么项目？",
            "Which company developed and deployed this product?",
        )

        self.assertTrue(all(_has_graph_relation_signal(item) for item in questions))
        self.assertFalse(_has_graph_relation_signal("Which team manages this product?"))
        self.assertFalse(_has_graph_relation_signal("What was revenue in 2025?"))

    async def test_adaptive_graphiti_rejects_graph_before_simple_and_repeated_graph(self) -> None:
        context = _adaptive_context()
        retriever = _AdaptiveRetriever(
            _pack(context),
            GraphitiSupplementResult("no_new_evidence"),
        )
        model = _Model(
            ChatToolCall(
                "graph-before-simple",
                "graphiti_supplement",
                {
                    "query": "relation",
                    "route_reason_code": "relation_chain_gap",
                },
            ),
            ChatToolCall(
                "simple-1",
                "search_knowledge_base",
                {"queries": ["relation"]},
            ),
            ChatToolCall(
                "graph-1",
                "graphiti_supplement",
                {
                    "query": "relation",
                    "route_reason_code": "cross_document_relation_gap",
                },
            ),
            ChatToolCall(
                "graph-2",
                "graphiti_supplement",
                {
                    "query": "relation again",
                    "route_reason_code": "entity_alias_gap",
                },
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.supplement_queries, ["relation"])
        self.assertEqual(state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls, 2)
        route_events = [
            event
            for event in state.artifacts[AGENT_TRACE_ARTIFACT].events
            if event.retrieval_lane == "graphiti_supplement"
        ]
        self.assertEqual(
            [event.route_result_code for event in route_events],
            ["rejected", "no_new_evidence", "rejected"],
        )
        self.assertTrue(
            all(event.tool == "graphiti_supplement" for event in route_events)
        )

    async def test_adaptive_empty_simple_can_use_graphiti_once(self) -> None:
        context = _adaptive_context()
        retriever = _AdaptiveRetriever(
            _pack(context),
            GraphitiSupplementResult(
                "admitted",
                _graphiti_pack(context, text="Graph-only source").evidence,
            ),
        )
        model = _Model(
            ChatToolCall(
                "simple-empty",
                "search_knowledge_base",
                {"queries": ["missing relation"]},
            ),
            ChatToolCall(
                "graph-after-empty",
                "graphiti_supplement",
                {
                    "query": "entity relation",
                    "route_reason_code": "cross_document_relation_gap",
                },
            ),
            ChatToolCall(
                "submit-graph-only",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The relation is supported.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.supplement_queries, ["entity relation"])
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)

    async def test_profile_output_limit_is_forwarded_without_agent_clamping(self) -> None:
        context = replace(
            _context(),
            model_configuration={"resolved_model": "fixed-model", "max_tokens": 4096},
        )
        model = _Model(
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            )
        )

        await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(model.requests[0].max_output_tokens, 4096)

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
        self.assertNotIn("invalid evidence", state.answering.rendered.content.lower())
        self.assertNotIn("evidence references", state.answering.rendered.content.lower())
        self.assertEqual(
            state.answering.validated.missing_aspects,
            ("Some requested parts remain unanswered",),
        )
        self.assertEqual(state.artifacts[AGENT_TRACE_ARTIFACT].events[-1].status, "salvaged")

    async def test_yes_no_relation_submission_gets_open_world_support_review(self) -> None:
        context = replace(
            _context(),
            query="甲公司是否控股乙公司？",
        )
        model = _Model(
            ChatToolCall(
                "search-1",
                "search_knowledge_base",
                {"queries": ["甲公司 乙公司 控股"]},
            ),
            ChatToolCall(
                "submit-unreviewed",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "另一家公司控股乙公司。",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
            ChatToolCall(
                "submit-reviewed",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(len(model.requests), 3)
        self.assertEqual(
            tuple(tool.name for tool in model.requests[2].tools),
            ("submit_answer",),
        )
        self.assertIn("exact proposition", model.requests[2].messages[-1].content)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.answering.rendered.citations, ())

    async def test_all_invalid_claims_get_one_submit_only_repair_with_usable_pool(self) -> None:
        context = _context()
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["revenue"]}),
            ChatToolCall(
                "submit-invalid",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "Revenue was 10.",
                            "kind": "fact",
                            "evidence_refs": ["ev_other_run"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
            ChatToolCall(
                "submit-repair",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "Revenue was 10.",
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
        self.assertEqual(len(model.requests), 3)
        self.assertEqual(
            tuple(tool.name for tool in model.requests[2].tools),
            ("submit_answer",),
        )
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls,
            1,
        )
        self.assertNotIn("ev_other_run", model.requests[2].messages[-1].content)

    async def test_yes_no_support_review_still_runs_after_submission_repair(self) -> None:
        context = replace(_context(), query="甲公司是否控股乙公司？")
        model = _Model(
            ChatToolCall(
                "search-1",
                "search_knowledge_base",
                {"queries": ["甲公司 乙公司 控股"]},
            ),
            ChatToolCall(
                "submit-invalid",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "甲公司控股乙公司。",
                            "kind": "fact",
                            "evidence_refs": ["ev_other_run"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
            ChatToolCall(
                "submit-repaired",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "另一家公司控股乙公司。",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
            ChatToolCall(
                "submit-reviewed",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(len(model.requests), 4)
        self.assertEqual(
            tuple(tool.name for tool in model.requests[3].tools),
            ("submit_answer",),
        )
        self.assertIn("exact proposition", model.requests[3].messages[-1].content)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)

    async def test_active_partial_submission_preserves_unanswered_aspects(self) -> None:
        context = _context()
        model = _Model(
            ChatToolCall(
                "search-1", "search_knowledge_base", {"queries": ["revenue"]}
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "partial",
                    "claims": [
                        {
                            "text": "Revenue was 10 in 2025.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": ["The cause of the change is not supported."],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.PARTIAL)
        self.assertEqual(
            state.answering.validated.missing_aspects,
            ("The cause of the change is not supported.",),
        )

    async def test_submit_normalizes_optional_fact_fields_and_unanswered(self) -> None:
        context = _context()
        model = _Model(
            ChatToolCall(
                "search-normalized-submit",
                "search_knowledge_base",
                {"queries": ["revenue"]},
            ),
            ChatToolCall(
                "normalized-submit",
                "submit_answer",
                {
                    "outcome": "partial",
                    "claims": [
                        {
                            "text": "Revenue was 10.",
                            "evidence_refs": ["ev_1"],
                        }
                    ],
                    "unanswered": ["  Cause unknown.  ", "", "Cause unknown."],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.PARTIAL)
        self.assertEqual(
            state.answering.validated.missing_aspects,
            ("Cause unknown.",),
        )

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

    async def test_five_calculations_are_not_rejected_by_a_cumulative_budget(self) -> None:
        context = _context()
        model = _Model(
            ChatToolCall(
                "search-1", "search_knowledge_base", {"queries": ["revenue"]}
            ),
            *(
                ChatToolCall(
                    f"calc-{index}",
                    "calculate",
                    {"expression": "10-5", "evidence_refs": ["ev_1"]},
                )
                for index in range(1, 6)
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The repeated calculation produced 5.",
                            "kind": "fact",
                            "evidence_refs": [],
                            "calculation_refs": ["calc_5"],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].calculation_calls,
            5,
        )

    async def test_round_limit_adds_one_submit_only_finalize_call(self) -> None:
        context = _context()
        model = _Model(
            None,
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["revenue"]}),
            ChatToolCall(
                "submit-final",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )
        budget = ChatAgentBudget(max_model_rounds=2)
        context = replace(
            context,
            agent_configuration={
                "version": "native_tool_calling_agent_v2",
                "budget": budget.as_dict(),
            },
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(len(model.requests), 3)
        self.assertEqual(model.requests[0].tool_choice, "required")
        self.assertEqual(model.requests[1].tool_choice, "required")
        self.assertTrue(
            all(
                [tool.name for tool in request.tools]
                == ["search_knowledge_base", "calculate", "submit_answer"]
                for request in model.requests[:2]
            )
        )
        self.assertEqual(model.requests[-1].tool_choice, "submit_answer")
        self.assertEqual(
            [tool.name for tool in model.requests[-1].tools], ["submit_answer"]
        )
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.answering.rendered.citations, ())

    async def test_forced_finalize_can_keep_a_complete_answer(self) -> None:
        context = _context()
        budget = ChatAgentBudget(max_model_rounds=2)
        context = replace(
            context,
            agent_configuration={
                "version": "native_tool_calling_agent_v2",
                "budget": budget.as_dict(),
            },
        )
        model = _Model(
            ChatToolCall(
                "search-1", "search_knowledge_base", {"queries": ["revenue"]}
            ),
            None,
            ChatToolCall(
                "submit-final",
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

    async def test_forced_finalize_cannot_bypass_the_graph_path_guard(self) -> None:
        context = replace(
            _adaptive_context(),
            query="甲公司控股的企业最终属于哪个集团？",
            agent_configuration={
                "version": "native_tool_calling_agent_v2",
                "budget": ChatAgentBudget(max_model_rounds=1).as_dict(),
            },
        )
        retriever = _AdaptiveRetriever(
            _pack(context, text="Simple relation source"),
            GraphitiSupplementResult("no_new_evidence"),
        )
        model = _Model(
            ChatToolCall(
                "search-1",
                "search_knowledge_base",
                {"queries": ["甲公司 集团"]},
            ),
            ChatToolCall(
                "submit-final",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "该集团可由 Simple 证据确定。",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.answering.rendered.citations, ())
        self.assertEqual(retriever.supplement_queries, [])

    async def test_forced_refusal_with_a_valid_claim_is_salvaged_as_partial(self) -> None:
        context = _context()
        budget = ChatAgentBudget(max_model_rounds=1)
        context = replace(
            context,
            agent_configuration={
                "version": "native_tool_calling_agent_v2",
                "budget": budget.as_dict(),
            },
        )
        model = _Model(
            ChatToolCall(
                "search-1", "search_knowledge_base", {"queries": ["revenue"]}
            ),
            ChatToolCall(
                "submit-final",
                "submit_answer",
                {
                    "outcome": "refused",
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

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.PARTIAL)
        self.assertEqual(len(state.answering.validated.claims), 1)
        self.assertTrue(state.answering.validated.missing_aspects)

    async def test_forced_invalid_protocol_variants_complete_as_refused(self) -> None:
        variants = {
            "no_tool_call": None,
            "multiple_tool_calls": (
                ChatToolCall(
                    "submit-a",
                    "submit_answer",
                    {"outcome": "refused", "claims": [], "unanswered": []},
                ),
                ChatToolCall(
                    "submit-b",
                    "submit_answer",
                    {"outcome": "refused", "claims": [], "unanswered": []},
                ),
            ),
            "wrong_tool": ChatToolCall(
                "search-final", "search_knowledge_base", {"queries": ["revenue"]}
            ),
            "malformed_payload": ChatToolCall(
                "submit-malformed",
                "submit_answer",
                {"outcome": "answered", "claims": []},
            ),
        }
        for name, forced_response in variants.items():
            with self.subTest(name=name):
                context = _context()
                budget = ChatAgentBudget(max_model_rounds=1)
                context = replace(
                    context,
                    agent_configuration={
                        "version": "native_tool_calling_agent_v2",
                        "budget": budget.as_dict(),
                    },
                )
                model = _Model(None, forced_response)
                retriever = _Retriever(_pack(context))

                state = await _agent(model, retriever).run(context)

                self.assertEqual(
                    state.answering.rendered.outcome,
                    AnswerOutcome.REFUSED,
                )
                self.assertEqual(state.answering.rendered.citations, ())
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(model.requests[-1].tool_choice, "submit_answer")
                self.assertEqual(retriever.queries, [])

    async def test_search_queries_are_not_rejected_by_a_query_budget(self) -> None:
        context = _context()
        budget = ChatAgentBudget(max_model_rounds=2)
        context = replace(
            context,
            agent_configuration={
                "version": "native_tool_calling_agent_v2",
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

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.queries, ["one", "two", "three"])
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls, 3)

    async def test_forced_finalize_with_invalid_claims_completes_as_refused(self) -> None:
        context = _context()
        budget = ChatAgentBudget(max_model_rounds=2)
        context = replace(
            context,
            agent_configuration={
                "version": "native_tool_calling_agent_v2",
                "budget": budget.as_dict(),
            },
        )
        model = _Model(
            ChatToolCall(
                "search-1", "search_knowledge_base", {"queries": ["revenue"]}
            ),
            None,
            ChatToolCall(
                "submit-final",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "Unsupported.",
                            "kind": "fact",
                            "evidence_refs": ["ev_other_run"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.answering.rendered.citations, ())
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].as_dict()["usage"]["model_rounds"],
            3,
        )

    async def test_forced_finalize_provider_error_remains_a_technical_failure(self) -> None:
        context = _context()
        budget = ChatAgentBudget(max_model_rounds=2)
        context = replace(
            context,
            agent_configuration={
                "version": "native_tool_calling_agent_v2",
                "budget": budget.as_dict(),
            },
        )
        model = _Model(
            None,
            None,
            ChatModelExecutionError(
                ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                diagnostic={"retryable": True},
            ),
        )

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(raised.exception.code, ErrorCode.CHAT_PROVIDER_UNAVAILABLE)
        self.assertEqual(
            raised.exception.phase, ChatPipelinePhase.GENERATE_OR_REFUSE
        )
        self.assertEqual(len(raised.exception.model_calls), 2)

    async def test_repeated_queries_are_executed_without_a_policy_gate(self) -> None:
        context = _context()
        retriever = _Retriever(_pack(context))
        model = _Model(
            ChatToolCall(
                "search-1",
                "search_knowledge_base",
                {"queries": ["revenue", "revenue", "revenue"]},
            ),
            ChatToolCall(
                "search-2",
                "search_knowledge_base",
                {"queries": ["revenue", "revenue", "revenue"]},
            ),
            ChatToolCall(
                "search-3",
                "search_knowledge_base",
                {"queries": ["revenue", "revenue", "revenue"]},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.queries, ["revenue"] * 9)
        first = _tool_payload(model.requests[1], "search-1")
        self.assertEqual(
            first["groups"][0]["results"][0]["content"],
            retriever.pack.evidence[0].text,
        )
        for group in first["groups"][1:]:
            self.assertEqual(
                group["results"],
                [
                    {
                        "evidence_ref": "ev_1",
                        "content_already_provided": True,
                    }
                ],
            )
        self.assertEqual(state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls, 9)

    async def test_multi_query_long_evidence_sends_every_full_chunk(self) -> None:
        context = _context()
        queries = ("revenue", "margin", "guidance")
        packs_by_query = {
            query: _pack(
                context,
                text="x" * 3_200,
                count=4,
            )
            for query in queries
        }
        retriever = _QueryRetriever(packs_by_query)
        model = _Model(
            ChatToolCall(
                "search-long",
                "search_knowledge_base",
                {"queries": list(queries)},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever).run(context)

        payload = _tool_payload(model.requests[1], "search-long")
        self.assertEqual(
            [group["query"] for group in payload["groups"]],
            list(queries),
        )
        sent_characters = 0
        for group in payload["groups"]:
            expected = [
                item.text for item in packs_by_query[group["query"]].evidence
            ]
            actual = [item["content"] for item in group["results"]]
            self.assertEqual(actual, expected)
            self.assertTrue(
                all("content_already_provided" not in item for item in group["results"])
            )
            sent_characters += sum(len(value) for value in actual)
        self.assertGreater(sent_characters, 24_000)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)

    async def test_repeated_table_ref_sends_full_content_only_once(self) -> None:
        context = _context()
        table_text = "metric|2025|2024\n" + "gross margin|50|40\n" * 180
        base_pack = _pack(context, text=table_text)
        table_pack = replace(
            base_pack,
            evidence=(
                replace(
                    base_pack.evidence[0],
                    modality="table",
                    matched_representations=("table_text",),
                ),
            ),
        )
        model = _Model(
            ChatToolCall(
                "search-table-1",
                "search_knowledge_base",
                {"queries": ["gross margin table"]},
            ),
            ChatToolCall(
                "search-table-2",
                "search_knowledge_base",
                {"queries": ["same gross margin table"]},
            ),
            ChatToolCall(
                "submit-table",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "Gross margin was 50 in 2025.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, _Retriever(table_pack)).run(context)

        first = _tool_payload(model.requests[1], "search-table-1")
        first_item = first["groups"][0]["results"][0]
        self.assertGreater(len(table_text), 2_400)
        self.assertEqual(first_item["content"], table_text)
        self.assertEqual(first_item["document"], "Report")
        self.assertEqual(
            first_item["document_id"],
            str(table_pack.evidence[0].document_id),
        )
        self.assertEqual(
            first_item["document_version_id"],
            str(table_pack.evidence[0].document_version_id),
        )
        self.assertEqual(first_item["location"], {"page": 1})
        self.assertEqual(first_item["rank"], 1)
        second = _tool_payload(model.requests[2], "search-table-2")
        self.assertEqual(
            second["groups"][0]["results"],
            [
                {
                    "evidence_ref": "ev_1",
                    "content_already_provided": True,
                }
            ],
        )
        self.assertFalse(
            any(message.role == "evidence" for message in model.requests[2].messages)
        )
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(state.answering.rendered.citations[0].modality, "table")
        self.assertEqual(
            state.answering.rendered.citations[0].matched_representations,
            ("table_text",),
        )
        self.assertEqual(state.answering.rendered.citations[0].quoted_text, table_text)

    async def test_same_unit_table_visual_preserves_table_body_and_asset(self) -> None:
        context = _context()
        table_text = (
            "Metric | 2025 | 2024\n"
            "Revenue | 10 | 5\n"
            "Gross margin | 50% | 40%"
        )
        pack, content, asset, image_bytes = _same_unit_table_visual_pack(
            context,
            table_text,
        )
        model = _Model(
            ChatToolCall(
                "search-same-unit-1",
                "search_knowledge_base",
                {"queries": ["revenue table"]},
            ),
            ChatToolCall(
                "search-same-unit-2",
                "search_knowledge_base",
                {"queries": ["same revenue table"]},
            ),
            ChatToolCall(
                "submit-same-unit",
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

        state = await _agent(
            model,
            _Retriever(pack),
            visual_preparer=VisualEvidencePreparationStep(_AssetReader(content)),
        ).run(context)

        first = _tool_payload(model.requests[1], "search-same-unit-1")
        first_item = first["groups"][0]["results"][0]
        self.assertEqual(first_item["evidence_ref"], "ev_1")
        self.assertEqual(first_item["content"], table_text)
        self.assertNotIn("visual evidence", first_item["content"])
        self.assertTrue(first_item["visual_attached"])
        second = _tool_payload(model.requests[2], "search-same-unit-2")
        self.assertEqual(
            second["groups"][0]["results"],
            [
                {
                    "evidence_ref": "ev_1",
                    "content_already_provided": True,
                }
            ],
        )
        visual_messages = [
            message
            for message in model.requests[2].messages
            if message.role == "evidence"
        ]
        self.assertEqual(len(visual_messages), 1)
        self.assertEqual(
            json.loads(visual_messages[0].content),
            {"visual_evidence_refs": {"cite_1": "ev_1"}},
        )
        self.assertEqual(len(visual_messages[0].visual_content), 1)
        self.assertEqual(
            visual_messages[0].visual_content[0].content,
            image_bytes,
        )

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(len(state.answering.rendered.citations), 1)
        citation = state.answering.rendered.citations[0]
        self.assertEqual(citation.citation_id, "cite_1")
        self.assertEqual(citation.modality, "table")
        self.assertEqual(citation.quoted_text, table_text)
        self.assertIsNotNone(citation.asset_snapshot)
        assert citation.asset_snapshot is not None
        self.assertEqual(citation.asset_snapshot["id"], str(asset.id))
        self.assertEqual(citation.asset_snapshot["media_type"], asset.media_type)
        self.assertEqual(
            citation.asset_snapshot["checksum_sha256"],
            asset.checksum_sha256,
        )
        self.assertEqual(
            citation.asset_snapshot["visual_unit_id"],
            str(pack.evidence[0].index_chunk_id),
        )
        self.assertEqual(citation.asset_snapshot["relation_type"], "table_of")
        self.assertEqual(len(state.answering.visual_content), 1)
        self.assertEqual(state.answering.visual_content[0].asset_id, asset.id)
        self.assertEqual(
            state.answering.visual_content[0].citation_ids,
            ("cite_1",),
        )
        self.assertEqual(state.answering.visual_content[0].content, image_bytes)

    async def test_later_same_unit_table_uses_stable_parent_citation(self) -> None:
        context = _context()
        first_seed, first_content, _, _ = _same_unit_table_visual_pack(
            context,
            "unused table seed",
        )
        first_parent = first_seed.evidence[0]
        distinct_visual_id = uuid4()
        distinct_related = replace(
            first_parent.related_visuals[0],
            visual_unit_id=distinct_visual_id,
            relation_type="explicit_figure_reference",
            relation_provenance="author_reference_v2",
            evidence_group_key="figure:1",
            figure_label="Figure 1",
            modality="image",
        )
        first_parent = replace(
            first_parent,
            text="Narrative parent with a distinct related figure.",
            modality="text",
            asset=None,
            evidence_group_key="text:1",
            matched_representations=("text",),
            related_visuals=(distinct_related,),
        )
        first_pack = replace(first_seed, evidence=(first_parent,))

        table_text = "Metric | 2025 | 2024\nRevenue | 10 | 5"
        table_pack, table_content, table_asset, table_image_bytes = (
            _same_unit_table_visual_pack(
                context,
                table_text,
                image_bytes=b"later-same-unit-table-image",
            )
        )
        retriever = _QueryRetriever(
            {"narrative figure": first_pack, "revenue table": table_pack}
        )
        model = _Model(
            ChatToolCall(
                "search-parent",
                "search_knowledge_base",
                {"queries": ["narrative figure"]},
            ),
            ChatToolCall(
                "search-table",
                "search_knowledge_base",
                {"queries": ["revenue table"]},
            ),
            ChatToolCall(
                "submit-table",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "Revenue was 10 in 2025.",
                            "kind": "fact",
                            "evidence_refs": ["ev_3"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(
            model,
            retriever,
            visual_preparer=VisualEvidencePreparationStep(
                _AssetMapReader(first_content, table_content)
            ),
        ).run(context)

        first_visual_messages = [
            message
            for message in model.requests[1].messages
            if message.role == "evidence"
        ]
        self.assertEqual(len(first_visual_messages), 1)
        self.assertEqual(
            json.loads(first_visual_messages[0].content),
            {"visual_evidence_refs": {"cite_2": "ev_2"}},
        )
        second = _tool_payload(model.requests[2], "search-table")
        second_item = second["groups"][0]["results"][0]
        self.assertEqual(second_item["evidence_ref"], "ev_3")
        self.assertEqual(second_item["rank"], 3)
        self.assertEqual(second_item["content"], table_text)
        self.assertTrue(second_item["visual_attached"])

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(len(state.answering.rendered.citations), 1)
        citation = state.answering.rendered.citations[0]
        self.assertEqual(citation.citation_id, "cite_3")
        self.assertEqual(citation.quoted_text, table_text)
        self.assertIsNotNone(citation.asset_snapshot)
        assert citation.asset_snapshot is not None
        self.assertEqual(citation.asset_snapshot["id"], str(table_asset.id))
        self.assertEqual(
            citation.asset_snapshot["parent_citation_id"],
            "cite_3",
        )
        self.assertNotEqual(
            citation.asset_snapshot["parent_citation_id"],
            "cite_2",
        )
        self.assertEqual(len(state.answering.visual_content), 1)
        self.assertEqual(state.answering.visual_content[0].asset_id, table_asset.id)
        self.assertEqual(
            state.answering.visual_content[0].citation_ids,
            ("cite_3",),
        )
        self.assertEqual(
            state.answering.visual_content[0].content,
            table_image_bytes,
        )

    async def test_repeated_visual_ref_does_not_resend_body_or_image(self) -> None:
        context = _context()
        pack, content, image_bytes = _native_visual_pack(context)
        reader = _AssetReader(content)
        model = _Model(
            ChatToolCall(
                "search-visual-1",
                "search_knowledge_base",
                {"queries": ["chart"]},
            ),
            ChatToolCall(
                "search-visual-2",
                "search_knowledge_base",
                {"queries": ["same chart"]},
            ),
            ChatToolCall(
                "submit-visual",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The chart supports the answer.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(
            model,
            _Retriever(pack),
            visual_preparer=VisualEvidencePreparationStep(reader),
        ).run(context)

        first = _tool_payload(model.requests[1], "search-visual-1")
        first_item = first["groups"][0]["results"][0]
        self.assertEqual(first_item["content"], "[image visual evidence]")
        self.assertTrue(first_item["visual_attached"])
        first_visual_messages = [
            message
            for message in model.requests[1].messages
            if message.role == "evidence"
        ]
        after_repeat_visual_messages = [
            message
            for message in model.requests[2].messages
            if message.role == "evidence"
        ]
        self.assertEqual(len(first_visual_messages), 1)
        self.assertEqual(len(after_repeat_visual_messages), 1)
        self.assertEqual(
            json.loads(first_visual_messages[0].content),
            {"visual_evidence_refs": {"cite_1": "ev_1"}},
        )
        self.assertEqual(len(first_visual_messages[0].visual_content), 1)
        self.assertEqual(
            first_visual_messages[0].visual_content[0].content,
            image_bytes,
        )
        second = _tool_payload(model.requests[2], "search-visual-2")
        self.assertEqual(
            second["groups"][0]["results"],
            [
                {
                    "evidence_ref": "ev_1",
                    "content_already_provided": True,
                }
            ],
        )
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(state.answering.rendered.citations[0].modality, "image")
        self.assertEqual(len(state.answering.visual_content), 1)
        self.assertEqual(state.answering.visual_content[0].content, image_bytes)

    async def test_visual_ref_without_sent_image_remains_uncitable(self) -> None:
        context = _context()
        pack, _, _ = _native_visual_pack(context)
        model = _Model(
            ChatToolCall(
                "search-visual",
                "search_knowledge_base",
                {"queries": ["chart"]},
            ),
            ChatToolCall(
                "submit-visual",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The chart supports the answer.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, _Retriever(pack)).run(context)

        payload = _tool_payload(model.requests[1], "search-visual")
        self.assertFalse(payload["groups"][0]["results"][0]["visual_attached"])
        self.assertFalse(
            any(message.role == "evidence" for message in model.requests[1].messages)
        )
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.answering.rendered.citations, ())
        self.assertEqual(state.answering.visual_content, ())

    async def test_caption_text_is_citable_when_frozen_vision_is_disabled(self) -> None:
        context = replace(
            _context(),
            model_configuration={
                "resolved_model": "fixed-model",
                "max_tokens": 2048,
                "vision_enabled": False,
            },
        )
        native_pack, content, _ = _native_visual_pack(context)
        caption_evidence = replace(
            native_pack.evidence[0],
            text="The caption states revenue reached 10.",
            matched_representations=("caption_text",),
        )
        pack = replace(native_pack, evidence=(caption_evidence,))
        reader = _AssetReader(content)
        model = _Model(
            ChatToolCall(
                "search-caption",
                "search_knowledge_base",
                {"queries": ["revenue caption"]},
            ),
            ChatToolCall(
                "submit-caption",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "Revenue reached 10.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(
            model,
            _Retriever(pack),
            visual_preparer=VisualEvidencePreparationStep(reader),
        ).run(context)

        self.assertEqual(reader.asset_ids, [])
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(state.answering.visual_content, ())
        self.assertIsNone(state.answering.rendered.citations[0].asset_snapshot)

    async def test_visual_budget_is_shared_across_distinct_searches(self) -> None:
        context = replace(
            _context(),
            model_configuration={
                "resolved_model": "fixed-model",
                "max_tokens": 2048,
                "vision_enabled": True,
                "max_visual_images": 1,
                "max_visual_image_bytes": 5_242_880,
                "max_visual_total_bytes": 12_582_912,
                "max_visual_pixels": 16_000_000,
            },
        )
        first_pack, first_content, _ = _native_visual_pack(context)
        second_pack, second_content, _ = _native_visual_pack(context)
        reader = _AssetMapReader(first_content, second_content)
        model = _Model(
            ChatToolCall(
                "search-first-visual",
                "search_knowledge_base",
                {"queries": ["first chart"]},
            ),
            ChatToolCall(
                "search-second-visual",
                "search_knowledge_base",
                {"queries": ["second chart"]},
            ),
            ChatToolCall(
                "submit-first-visual",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The first chart supports the answer.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(
            model,
            _QueryRetriever(
                {
                    "first chart": first_pack,
                    "second chart": second_pack,
                }
            ),
            visual_preparer=VisualEvidencePreparationStep(reader),
        ).run(context)

        self.assertEqual(reader.asset_ids, [first_content.snapshot.id])
        self.assertEqual(len(state.answering.visual_content), 1)
        self.assertEqual(
            [item.asset_id for item in state.answering.visual_decisions if item.selected],
            [first_content.snapshot.id],
        )

    async def test_more_than_legacy_evidence_limit_does_not_fail_the_answer(self) -> None:
        context = _context()
        model = _Model(
            ChatToolCall(
                "search-1", "search_knowledge_base", {"queries": ["revenue"]}
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The final item is supported.",
                            "kind": "fact",
                            "evidence_refs": ["ev_105"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(
            model, _Retriever(_pack(context, count=105))
        ).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        self.assertEqual(trace.evidence_ref_count, 105)
        self.assertEqual(trace.events[0].count, 105)
        self.assertEqual(len(trace.events[0].refs), 100)

    async def test_one_claim_can_reference_more_than_one_hundred_evidence_refs(self) -> None:
        context = _context()
        evidence_refs = [f"ev_{index}" for index in range(1, 106)]
        model = _Model(
            ChatToolCall(
                "search-1", "search_knowledge_base", {"queries": ["revenue"]}
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "All retrieved items support the combined claim.",
                            "kind": "fact",
                            "evidence_refs": evidence_refs,
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(
            model, _Retriever(_pack(context, count=105))
        ).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(len(state.answering.validated.claims[0].citation_ids), 105)
        self.assertEqual(len(state.answering.rendered.citations), 105)
        self.assertEqual(
            len(state.artifacts[AGENT_TRACE_ARTIFACT].events[-1].refs),
            100,
        )


if __name__ == "__main__":
    unittest.main()
