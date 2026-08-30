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
    _graph_arguments,
    _initial_messages,
    _search_arguments,
    _tools,
)
from rag_kb.domain import (
    AnswerConflictAdjudication,
    AnswerConflictType,
    AnswerOutcome,
    CHAT_GRAPH_SEARCH_REASONS,
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
    GraphSearchResult,
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
        verdicts: list[str | BaseException] | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.calls = list(calls)
        self.verdicts = list(verdicts or [])
        self.usage = usage or {"total_tokens": 5}
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if not request.tools:
            # Verification-stage call: use a scripted verdict when provided,
            # otherwise auto-approve every submitted claim.
            if self.verdicts:
                value = self.verdicts.pop(0)
                if isinstance(value, BaseException):
                    raise value
                content = value
            else:
                payload = json.loads(request.messages[-1].content)
                content = json.dumps(
                    {
                        "premise": "none",
                        "claims": [
                            {"index": index, "support": "supported"}
                            for index in range(len(payload["claims"]))
                        ],
                    }
                )
            return ChatModelResponse(
                content=content,
                model="fixed-model",
                finish_reason="stop",
                provider_request_id=f"request-{len(self.requests)}",
                usage=dict(self.usage),
                tool_calls=(),
            )
        value = self.calls.pop(0)
        if isinstance(value, BaseException):
            raise value
        tool_calls = value if isinstance(value, tuple) else ((value,) if value else ())
        return ChatModelResponse(
            content="" if tool_calls else "plain assistant text",
            model="fixed-model",
            finish_reason="tool_calls" if tool_calls else "stop",
            provider_request_id=f"request-{len(self.requests)}",
            usage=dict(self.usage),
            tool_calls=tool_calls,
        )


class _Retriever:
    def __init__(self, pack: EvidencePack, *, graph_ready: bool = False) -> None:
        self.pack = pack
        self.queries = []
        self.graph_ready = graph_ready
        self.capability_calls = 0

    async def retrieve_query(self, context, query, *, top_k_override=None):
        del context, top_k_override
        self.queries.append(query)
        return self.pack

    async def graph_relations_capable(self, context):
        del context
        self.capability_calls += 1
        return self.graph_ready

    async def search_graph_relations(
        self, context, query, *, excluded_index_chunk_ids
    ):
        raise AssertionError("unexpected Graph search on a plain retriever")


class _QueryRetriever:
    def __init__(self, packs_by_query: dict[str, EvidencePack]) -> None:
        self.packs_by_query = packs_by_query
        self.queries = []

    async def retrieve_query(self, context, query, *, top_k_override=None):
        del context, top_k_override
        self.queries.append(query)
        return self.packs_by_query[query]

    async def graph_relations_capable(self, context):
        del context
        return False


class _GraphRetriever:
    def __init__(
        self,
        pack: EvidencePack,
        graph_results: list[GraphSearchResult],
        *,
        graph_ready: bool = True,
    ) -> None:
        self.pack = pack
        self.graph_results = list(graph_results)
        self.queries = []
        self.graph_queries = []
        self.graph_exclusions = []
        self.capability_calls = 0
        self.graph_ready = graph_ready

    async def retrieve_query(self, context, query, *, top_k_override=None):
        del context, top_k_override
        self.queries.append(query)
        return self.pack

    async def graph_relations_capable(self, context):
        del context
        self.capability_calls += 1
        return self.graph_ready

    async def search_graph_relations(
        self, context, query, *, excluded_index_chunk_ids
    ):
        del context
        self.graph_queries.append(query)
        self.graph_exclusions.append(tuple(excluded_index_chunk_ids))
        result = self.graph_results.pop(0)
        return result


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
            "profile_version": "adaptive_graphiti_v3",
            "strategy": "exact_vector",
            "top_k": 3,
            "rerank_mode": "none",
            "router": "native_agent_graph_tool_v1",
            "augmentation": "graphiti_path_v3",
            "graph_edge_limit": 16,
            "graph_source_chunk_target": 12,
            "graph_source_chunk_limit": 16,
            "graph_call_timeout_seconds": 90,
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


def _graph_pack(
    context: ChatExecutionContext,
    *,
    text: str = "Graph source",
    hop_count: int = 2,
    rank: int = 1,
    chunk_id=None,
) -> EvidencePack:
    base = _pack(context, text=text)
    source = base.evidence[0]
    return replace(
        base,
        evidence=(
            replace(
                source,
                score=1.0 / rank,
                score_kind=EvidenceScoreKind.GRAPH_PATH,
                vector_similarity=None,
                index_chunk_id=chunk_id or source.index_chunk_id,
                graph_path_id=f"path-{rank}",
                graph_anchor_index_chunk_id=source.index_chunk_id,
                graph_hop_count=hop_count,
                graph_path_rank=rank,
                matched_representations=("graph_path", "text"),
            ),
        ),
    )


def _graph_result(
    pack: EvidencePack,
    *,
    new_ids: tuple | None = None,
    result_code: str = "admitted",
) -> GraphSearchResult:
    return GraphSearchResult(
        result_code,
        pack.evidence,
        new_index_chunk_ids=new_ids,
        candidate_count=16,
        path_count=1,
        hydrated_chunk_count=len(pack.evidence),
        hop1_count=sum(item.graph_hop_count == 1 for item in pack.evidence),
        hop2_count=sum(item.graph_hop_count == 2 for item in pack.evidence),
        hop3_count=sum(item.graph_hop_count == 3 for item in pack.evidence),
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
    def test_system_prompt_requires_structured_conflict_claims(self) -> None:
        prompt = _initial_messages(_context(), ChatAgentBudget())[0].content
        self.assertIn('kind="conflict"', prompt)
        self.assertIn("supporting_refs", prompt)
        self.assertIn("Do not silently merge a conflict into a one-sided fact claim", prompt)

    def test_submit_answer_schema_accepts_conflict_claims(self) -> None:
        submit = next(tool for tool in _tools() if tool.name == "submit_answer")
        claim = submit.input_schema["properties"]["claims"]["items"]
        self.assertEqual(claim["properties"]["kind"]["enum"], ("fact", "conflict"))
        conflict = claim["properties"]["conflict"]
        self.assertEqual(
            set(conflict["required"]),
            {"supporting_refs", "conflicting_refs", "type", "adjudication"},
        )
        self.assertFalse(conflict["additionalProperties"])
        self.assertEqual(
            conflict["properties"]["type"]["enum"],
            tuple(item.value for item in AnswerConflictType),
        )
        self.assertEqual(
            conflict["properties"]["adjudication"]["enum"],
            tuple(item.value for item in AnswerConflictAdjudication),
        )
        self.assertIn("kind='conflict'", submit.description)

    def test_adaptive_prompt_describes_first_class_graph_without_commands(self) -> None:
        prompt = _initial_messages(
            _adaptive_context(),
            ChatAgentBudget(),
            adaptive=True,
        )[0].content

        self.assertNotIn("three supplied tools", prompt)
        self.assertIn("currently supplied tools", prompt)
        self.assertIn("one-to-three-hop chains", prompt)
        self.assertIn("up to twice", prompt)
        self.assertIn("never a prerequisite", prompt)
        self.assertIn("A single hop is already a complete path", prompt)

    def test_graph_reason_contract_has_exactly_the_observable_reasons(self) -> None:
        self.assertEqual(
            CHAT_GRAPH_SEARCH_REASONS,
            frozenset(
                {
                    "direct_relation",
                    "relation_chain",
                    "entity_alias",
                    "cross_document_relation",
                }
            ),
        )
        self.assertEqual(
            _graph_arguments({"query": "relation", "reason": "relation_chain"}),
            ("relation", "relation_chain"),
        )
        self.assertIsNone(
            _graph_arguments({"query": "relation", "reason": "relational_query"})
        )
        self.assertIsNone(
            _graph_arguments({"query": "relation", "route_reason_code": "relation_chain"})
        )

    def test_graph_tool_visibility_follows_capability_and_call_count(self) -> None:
        hidden = _tools(adaptive=True, graph_ready=False, graph_calls_remaining=2)
        ready = _tools(adaptive=True, graph_ready=True, graph_calls_remaining=2)
        exhausted = _tools(adaptive=True, graph_ready=True, graph_calls_remaining=0)
        plain = _tools()

        self.assertNotIn("search_graph_relations", [tool.name for tool in hidden])
        self.assertIn("search_graph_relations", [tool.name for tool in ready])
        self.assertNotIn("search_graph_relations", [tool.name for tool in exhausted])
        self.assertNotIn("search_graph_relations", [tool.name for tool in plain])
        self.assertEqual(
            [tool.name for tool in ready],
            ["search_knowledge_base", "search_graph_relations", "calculate", "submit_answer"],
        )
        graph_schema = next(
            tool.input_schema for tool in ready if tool.name == "search_graph_relations"
        )
        self.assertEqual(tuple(graph_schema["required"]), ("query", "reason"))
        self.assertEqual(_search_arguments({"queries": ["one", "two"]}), ("one", "two"))
        self.assertIsNone(_search_arguments({"retrieval_lane": "simple", "queries": ["one"]}))

    def test_agent_budget_configures_rounds_and_graph_calls(self) -> None:
        self.assertEqual(
            ChatAgentBudget().as_dict(),
            {
                "max_model_rounds": 8,
                "max_graph_calls": 2,
                "max_total_tokens": 150000,
                "max_evidence_items": 64,
                "max_retrieval_calls": 16,
                "soft_deadline_reserve_seconds": 60.0,
            },
        )
        for kwargs in (
            {"max_model_rounds": True},
            {"max_model_rounds": 1.5},
            {"max_graph_calls": True},
            {"max_graph_calls": 0},
            {"max_graph_calls": 3},
            {"max_graph_calls": 1.5},
            {"max_total_tokens": True},
            {"max_total_tokens": 0},
            {"max_total_tokens": 10_000_001},
            {"max_evidence_items": 0},
            {"max_evidence_items": 513},
            {"max_retrieval_calls": 0},
            {"max_retrieval_calls": 65},
            {"soft_deadline_reserve_seconds": -1.0},
            {"soft_deadline_reserve_seconds": 601.0},
            {"soft_deadline_reserve_seconds": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ChatAgentBudget(**kwargs)

    def test_agent_budget_defaults_fill_legacy_two_key_configuration(self) -> None:
        context = replace(
            _context(),
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
                "budget": {"max_model_rounds": 8, "max_graph_calls": 2},
            },
        )
        from rag_kb.answering.agent import _budget_from_context

        budget = _budget_from_context(context)
        self.assertEqual(budget, ChatAgentBudget())

    async def test_agent_configuration_is_strictly_current_v3(self) -> None:
        invalid_configurations = (
            {
                "version": "native_tool_calling_agent_v2",
                "budget": {"max_model_rounds": 8, "max_graph_calls": 2},
            },
            {
                "version": "native_tool_calling_agent_v3",
                "budget": {"max_model_rounds": 8, "max_graph_calls": 3},
            },
            {
                "version": "native_tool_calling_agent_v3",
                "budget": {"max_model_rounds": 8},
            },
            {
                "version": "native_tool_calling_agent_v3",
                "budget": {"max_model_rounds": True, "max_graph_calls": 2},
            },
            {
                "version": "native_tool_calling_agent_v3",
                "budget": {"max_model_rounds": "8", "max_graph_calls": 2},
            },
            {
                "version": "native_tool_calling_agent_v3",
                "budget": {"max_model_rounds": 8.0, "max_graph_calls": 2},
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

    async def test_capability_is_checked_before_rounds_without_a_model_call(self) -> None:
        context = _adaptive_context()
        retriever = _GraphRetriever(
            _pack(context),
            [_graph_result(_graph_pack(context))],
        )
        model = _Model(
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            )
        )

        await _agent(model, retriever).run(context)

        self.assertEqual(retriever.capability_calls, 1)
        self.assertEqual(len(model.requests), 1)
        self.assertIn("search_graph_relations", [tool.name for tool in model.requests[0].tools])

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
        self.assertEqual(len(model.requests), 3)
        self.assertEqual(model.requests[2].tools, ())
        self.assertEqual(model.requests[2].tool_choice, "none")
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].budget.as_dict(),
            ChatAgentBudget().as_dict(),
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

    async def test_graph_first_round_exposes_graph_and_agent_may_start_with_it(
        self,
    ) -> None:
        context = _adaptive_context()
        graph_pack = _graph_pack(context, text="Graph-only source", hop_count=1)
        retriever = _GraphRetriever(
            _pack(context, text="Simple source", count=1),
            [_graph_result(graph_pack)],
        )
        model = _Model(
            ChatToolCall(
                "graph-1",
                "search_graph_relations",
                {"query": "revenue relation", "reason": "direct_relation"},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The relation is supported by Graph.",
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

        self.assertEqual(retriever.queries, [])
        self.assertEqual(retriever.graph_queries, ["revenue relation"])
        self.assertEqual(state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls, 1)
        graph_event = state.artifacts[AGENT_TRACE_ARTIFACT].events[0]
        self.assertEqual(graph_event.tool, "search_graph_relations")
        self.assertEqual(graph_event.retrieval_lane, "graph_relations")
        self.assertEqual(graph_event.route_result_code, "admitted")
        self.assertEqual(graph_event.new_evidence_count, 1)
        self.assertEqual(graph_event.call_index, 1)
        self.assertEqual(graph_event.invocation_source, "agent")
        self.assertIsNotNone(graph_event.duration_ms)
        self.assertEqual(graph_event.candidate_count, 16)
        self.assertEqual(graph_event.hop1_count, 1)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(len(state.answering.rendered.citations), 1)
        self.assertIn(
            "search_graph_relations",
            [tool.name for tool in model.requests[0].tools],
        )
        self.assertNotIn("edge-fact", model.requests[1].messages[-1].content)
        self.assertEqual(retriever.capability_calls, 1)

    async def test_graph_may_follow_simple_and_a_second_graph_call_is_allowed(
        self,
    ) -> None:
        context = _adaptive_context()
        first_graph = _graph_pack(
            context,
            text="First hop source",
            hop_count=1,
            rank=1,
        )
        second_graph = _graph_pack(
            context,
            text="Second chain source",
            hop_count=2,
            rank=2,
        )
        retriever = _GraphRetriever(
            _pack(context, text="Simple source", count=1),
            [
                _graph_result(first_graph),
                _graph_result(second_graph),
            ],
        )
        model = _Model(
            ChatToolCall(
                "simple-1", "search_knowledge_base", {"queries": ["revenue"]}
            ),
            ChatToolCall(
                "graph-1",
                "search_graph_relations",
                {"query": "revenue relation", "reason": "relation_chain"},
            ),
            ChatToolCall(
                "graph-2",
                "search_graph_relations",
                {"query": "deeper chain", "reason": "relation_chain"},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The chain is complete.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1", "ev_2", "ev_3"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.queries, ["revenue"])
        self.assertEqual(retriever.graph_queries, ["revenue relation", "deeper chain"])
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        self.assertEqual(trace.retrieval_calls, 3)
        self.assertEqual(trace.retrieval_tool_calls, 3)
        self.assertEqual(trace.simple_tool_calls, 1)
        self.assertEqual(trace.graph_tool_calls, 2)
        graph_events = [
            event for event in trace.events if event.retrieval_lane == "graph_relations"
        ]
        self.assertEqual([event.call_index for event in graph_events], [1, 2])
        self.assertEqual([event.route_result_code for event in graph_events], ["admitted", "admitted"])
        self.assertEqual([event.new_evidence_count for event in graph_events], [1, 1])
        self.assertEqual(graph_events[0].invocation_source, "agent")
        self.assertEqual(len(model.requests), 5)
        # The third model request no longer exposes the exhausted Graph tool.
        self.assertNotIn(
            "search_graph_relations",
            [tool.name for tool in model.requests[3].tools],
        )
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)

    async def test_third_graph_call_is_rejected_without_external_query(self) -> None:
        context = replace(
            _adaptive_context(),
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
                "budget": {"max_model_rounds": 6, "max_graph_calls": 2},
            },
        )
        graph_pack = _graph_pack(context, text="Graph source", hop_count=1)
        retriever = _GraphRetriever(
            _pack(context),
            [
                _graph_result(graph_pack),
                _graph_result(graph_pack),
            ],
        )
        model = _Model(
            ChatToolCall(
                "graph-1",
                "search_graph_relations",
                {"query": "relation", "reason": "direct_relation"},
            ),
            ChatToolCall(
                "graph-2",
                "search_graph_relations",
                {"query": "relation again", "reason": "entity_alias"},
            ),
            ChatToolCall(
                "graph-3",
                "search_graph_relations",
                {"query": "third relation", "reason": "direct_relation"},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.graph_queries, ["relation", "relation again"])
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        graph_events = [
            event for event in trace.events if event.tool == "search_graph_relations"
        ]
        self.assertEqual(len(graph_events), 3)
        self.assertEqual([event.status for event in graph_events], ["ok", "ok", "rejected"])
        self.assertEqual([event.call_index for event in graph_events[:2]], [1, 2])
        self.assertIsNone(graph_events[2].call_index)
        self.assertEqual(trace.retrieval_calls, 2)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)

    async def test_graph_unavailable_hides_tool_and_rejects_stray_calls(self) -> None:
        context = _adaptive_context()
        retriever = _GraphRetriever(
            _pack(context),
            [],
            graph_ready=False,
        )
        model = _Model(
            ChatToolCall(
                "graph-early",
                "search_graph_relations",
                {"query": "relation", "reason": "direct_relation"},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.graph_queries, [])
        self.assertNotIn(
            "search_graph_relations",
            [tool.name for tool in model.requests[0].tools],
        )
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        rejected = [event for event in trace.events if event.tool == "search_graph_relations"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].status, "rejected")
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)

    async def test_submit_never_triggers_an_implicit_graph_call(self) -> None:
        context = replace(
            _adaptive_context(),
            query="WTC-7 最终属于哪个集团？",
        )
        retriever = _GraphRetriever(
            _pack(context, text="Simple source", count=1),
            [],
        )
        model = _Model(
            ChatToolCall("simple", "search_knowledge_base", {"queries": ["short query"]}),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
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
                },
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.graph_queries, [])
        self.assertEqual(len(model.requests), 3)
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        graph_events = [
            event for event in trace.events if event.retrieval_lane == "graph_relations"
        ]
        self.assertEqual(graph_events, [])
        self.assertEqual(trace.retrieval_calls, 1)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)

    async def test_second_graph_call_with_no_new_evidence_returns_path_status(
        self,
    ) -> None:
        context = _adaptive_context()
        first_graph = _graph_pack(context, text="First source", hop_count=1, rank=1)
        retriever = _GraphRetriever(
            _pack(context),
            [
                _graph_result(first_graph),
                GraphSearchResult(
                    "no_evidence",
                    first_graph.evidence,
                    new_index_chunk_ids=(),
                    candidate_count=16,
                    path_count=1,
                    hydrated_chunk_count=1,
                    hop1_count=1,
                    hop2_count=0,
                    hop3_count=0,
                ),
            ],
        )
        model = _Model(
            ChatToolCall(
                "graph-1",
                "search_graph_relations",
                {"query": "relation", "reason": "direct_relation"},
            ),
            ChatToolCall(
                "graph-2",
                "search_graph_relations",
                {"query": "relation again", "reason": "relation_chain"},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The direct relation is supported.",
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

        self.assertEqual(retriever.graph_queries, ["relation", "relation again"])
        payload = _tool_payload(model.requests[2], "graph-2")
        self.assertEqual(payload["status"], "graph_relations")
        self.assertEqual(payload["route_result_code"], "no_evidence")
        self.assertEqual(payload["new_evidence_count"], 0)
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        graph_events = [
            event for event in trace.events if event.retrieval_lane == "graph_relations"
        ]
        self.assertEqual([event.route_result_code for event in graph_events], ["admitted", "no_evidence"])
        self.assertEqual([event.new_evidence_count for event in graph_events], [1, 0])
        self.assertEqual(graph_events[1].call_index, 2)
        self.assertIsNotNone(graph_events[1].duration_ms)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)

    async def test_simple_and_two_graph_calls_dedupe_and_merge_provenance(self) -> None:
        context = _adaptive_context()
        shared_chunk = replace(
            _graph_pack(context, text="Shared path source", hop_count=1, rank=3).evidence[0],
            graph_path_id="path-shared",
        )
        graph_two = _graph_pack(
            context,
            text="First graph source",
            hop_count=1,
            rank=2,
        )
        first_result_pack = replace(
            _pack(context, text="Graph source"),
            evidence=(
                replace(shared_chunk, rank=1),
                replace(graph_two.evidence[0], rank=2),
            ),
        )
        retriever = _GraphRetriever(
            _pack(context),
            [
                GraphSearchResult(
                    "admitted",
                    first_result_pack.evidence,
                    new_index_chunk_ids=tuple(
                        item.index_chunk_id for item in first_result_pack.evidence
                    ),
                    candidate_count=16,
                    path_count=2,
                    hydrated_chunk_count=2,
                    hop1_count=2,
                    hop2_count=0,
                    hop3_count=0,
                ),
                GraphSearchResult(
                    "no_evidence",
                    first_result_pack.evidence,
                    new_index_chunk_ids=(),
                    candidate_count=16,
                    path_count=1,
                    hydrated_chunk_count=2,
                    hop1_count=2,
                    hop2_count=0,
                    hop3_count=0,
                ),
            ],
        )
        model = _Model(
            ChatToolCall(
                "graph-1",
                "search_graph_relations",
                {"query": "relation", "reason": "direct_relation"},
            ),
            ChatToolCall(
                "graph-2",
                "search_graph_relations",
                {"query": "relation again", "reason": "relation_chain"},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "The relation is supported by both paths.",
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

        self.assertEqual(len(state.answering.validated.claims[0].citation_ids), 2)
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        graph_events = [
            event for event in trace.events if event.retrieval_lane == "graph_relations"
        ]
        self.assertEqual([event.route_result_code for event in graph_events], ["admitted", "no_evidence"])
        self.assertEqual([event.new_evidence_count for event in graph_events], [2, 0])
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
        submit_events = [
            event
            for event in state.artifacts[AGENT_TRACE_ARTIFACT].events
            if event.tool == "submit_answer"
        ]
        self.assertEqual(submit_events[-1].status, "salvaged")

    async def test_valid_conflict_claim_is_retained_with_structure(self) -> None:
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
                            "text": (
                                "A later report says revenue was 12; an older memo "
                                "says 10. The later version is reliable."
                            ),
                            "kind": "conflict",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                            "conflict": {
                                "supporting_refs": ["ev_1"],
                                "conflicting_refs": ["ev_2"],
                                "type": "version",
                                "adjudication": "resolvable",
                            },
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context, count=2))).run(context)

        claim = state.answering.validated.claims[0]
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(claim.citation_ids, ("cite_1", "cite_2"))
        self.assertIsNotNone(claim.conflict)
        self.assertEqual(claim.conflict.supporting_citation_ids, ("cite_1",))
        self.assertEqual(claim.conflict.conflicting_citation_ids, ("cite_2",))
        self.assertEqual(claim.conflict.conflict_type, AnswerConflictType.VERSION)
        self.assertEqual(
            claim.conflict.adjudication, AnswerConflictAdjudication.RESOLVABLE
        )
        self.assertIn("[1]", state.answering.rendered.content)
        self.assertIn("[2]", state.answering.rendered.content)

    async def test_conflict_claim_rejects_empty_intersecting_and_unissued_refs(self) -> None:
        cases = (
            (
                "empty",
                {
                    "supporting_refs": [],
                    "conflicting_refs": ["ev_2"],
                    "type": "version",
                    "adjudication": "unresolvable",
                },
                "conflict_ref",
            ),
            (
                "intersecting",
                {
                    "supporting_refs": ["ev_1"],
                    "conflicting_refs": ["ev_1"],
                    "type": "opinion",
                    "adjudication": "unresolvable",
                },
                "conflict_ref",
            ),
            (
                "unissued",
                {
                    "supporting_refs": ["ev_1"],
                    "conflicting_refs": ["ev_other_run"],
                    "type": "temporal",
                    "adjudication": "resolvable",
                },
                "conflict_ref",
            ),
        )
        for name, conflict, reason in cases:
            with self.subTest(name=name):
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
                                    "text": "Sources disagree on revenue.",
                                    "kind": "conflict",
                                    "evidence_refs": ["ev_1"],
                                    "calculation_refs": [],
                                    "conflict": conflict,
                                }
                            ],
                            "unanswered": [],
                        },
                    ),
                )

                state = await _agent(
                    model, _Retriever(_pack(context, count=2))
                ).run(context)

                self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
                event = state.artifacts[AGENT_TRACE_ARTIFACT].events[-1]
                self.assertEqual(event.status, "salvaged")
                self.assertEqual(event.rejection_reasons, (reason,))
                self.assertEqual(len(model.requests), 2)

    async def test_fact_claim_cannot_carry_a_conflict_object(self) -> None:
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
                            "text": "Revenue was 10.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                            "conflict": {
                                "supporting_refs": ["ev_1"],
                                "conflicting_refs": ["ev_2"],
                                "type": "version",
                                "adjudication": "resolvable",
                            },
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context, count=2))).run(context)

        event = state.artifacts[AGENT_TRACE_ARTIFACT].events[-1]
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(event.rejection_reasons, ("conflict_shape",))

    async def test_conflict_kind_without_conflict_object_is_rejected(self) -> None:
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
                            "text": "Sources disagree on revenue.",
                            "kind": "conflict",
                            "evidence_refs": ["ev_1", "ev_2"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context, count=2))).run(context)

        event = state.artifacts[AGENT_TRACE_ARTIFACT].events[-1]
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(event.rejection_reasons, ("conflict_shape",))

    async def test_invalid_conflict_claim_is_salvaged_when_a_fact_claim_remains(self) -> None:
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
                            "text": "Revenue was 10.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        },
                        {
                            "text": "Sources disagree.",
                            "kind": "conflict",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                            "conflict": {
                                "supporting_refs": ["ev_1"],
                                "conflicting_refs": ["ev_1"],
                                "type": "opinion",
                                "adjudication": "unresolvable",
                            },
                        },
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context, count=2))).run(context)

        submit_events = [
            event
            for event in state.artifacts[AGENT_TRACE_ARTIFACT].events
            if event.tool == "submit_answer"
        ]
        event = submit_events[-1]
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.PARTIAL)
        self.assertEqual(len(state.answering.validated.claims), 1)
        self.assertIsNone(state.answering.validated.claims[0].conflict)
        self.assertEqual(event.status, "salvaged")
        self.assertEqual(event.rejection_reasons, ("conflict_ref",))

    async def test_false_premise_verdict_refuses_the_submission(self) -> None:
        context = replace(
            _context(),
            query="甲公司为什么收购了乙公司？",
        )
        model = _Model(
            ChatToolCall(
                "search-1",
                "search_knowledge_base",
                {"queries": ["甲公司 乙公司 收购"]},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "甲公司收购了乙公司，因为双方业务互补。",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
            verdicts=[
                json.dumps(
                    {
                        "premise": "unsupported",
                        "claims": [{"index": 0, "support": "unsupported"}],
                    }
                )
            ],
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(len(model.requests), 3)
        verifier_request = model.requests[2]
        self.assertEqual(verifier_request.tools, ())
        payload = json.loads(verifier_request.messages[-1].content)
        self.assertEqual(payload["query"], "甲公司为什么收购了乙公司？")
        self.assertEqual(len(payload["claims"]), 1)
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.answering.rendered.citations, ())
        verifier_events = [
            event
            for event in state.artifacts[AGENT_TRACE_ARTIFACT].events
            if event.tool == "verifier"
        ]
        self.assertEqual(len(verifier_events), 1)
        self.assertEqual(verifier_events[0].status, "refused")
        self.assertEqual(verifier_events[0].rejection_reasons, ("false_premise",))

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
        self.assertEqual(len(model.requests), 4)
        self.assertEqual(
            tuple(tool.name for tool in model.requests[2].tools),
            ("submit_answer",),
        )
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls,
            1,
        )
        self.assertNotIn("ev_other_run", model.requests[2].messages[-1].content)

    async def test_verifier_still_runs_after_submission_repair(self) -> None:
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
            verdicts=[
                json.dumps(
                    {
                        "premise": "unsupported",
                        "claims": [{"index": 0, "support": "unsupported"}],
                    }
                )
            ],
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(len(model.requests), 4)
        self.assertEqual(
            tuple(tool.name for tool in model.requests[2].tools),
            ("submit_answer",),
        )
        self.assertEqual(model.requests[3].tools, ())
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.answering.rendered.citations, ())

    async def test_unsupported_claim_is_dropped_and_downgraded_to_partial(self) -> None:
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
                            "text": "Revenue was 10 in 2025.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        },
                        {
                            "text": "The board resigned over the result.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        },
                    ],
                    "unanswered": [],
                },
            ),
            verdicts=[
                json.dumps(
                    {
                        "premise": "none",
                        "claims": [
                            {"index": 0, "support": "supported"},
                            {"index": 1, "support": "unsupported"},
                        ],
                    }
                )
            ],
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.PARTIAL)
        self.assertEqual(len(state.answering.validated.claims), 1)
        self.assertEqual(
            state.answering.validated.claims[0].text, "Revenue was 10 in 2025."
        )
        self.assertEqual(
            state.answering.validated.missing_aspects,
            ("Some claims were removed because the cited evidence did not support them",),
        )
        verifier_events = [
            event
            for event in state.artifacts[AGENT_TRACE_ARTIFACT].events
            if event.tool == "verifier"
        ]
        self.assertEqual(len(verifier_events), 1)
        self.assertEqual(verifier_events[0].status, "salvaged")
        self.assertEqual(verifier_events[0].rejected_claim_count, 1)
        self.assertEqual(verifier_events[0].rejection_reasons, ("unsupported_claim",))

    async def test_all_claims_unsupported_refuses_the_submission(self) -> None:
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
                            "text": "Revenue doubled year over year.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
            verdicts=[
                json.dumps(
                    {
                        "premise": "supported",
                        "claims": [{"index": 0, "support": "contradicted"}],
                    }
                )
            ],
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.answering.rendered.citations, ())

    async def test_malformed_verdict_retries_once_then_fails_closed(self) -> None:
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
                            "text": "Revenue was 10 in 2025.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
            verdicts=["not a verdict", '{"premise":"maybe","claims":[]}'],
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.answering.rendered.citations, ())
        verifier_requests = [request for request in model.requests if not request.tools]
        self.assertEqual(len(verifier_requests), 2)
        self.assertIn("not a valid verification verdict",
                      verifier_requests[1].messages[-1].content)
        verifier_events = [
            event
            for event in state.artifacts[AGENT_TRACE_ARTIFACT].events
            if event.tool == "verifier"
        ]
        self.assertEqual(verifier_events[0].status, "refused")
        self.assertEqual(
            verifier_events[0].rejection_reasons, ("unverifiable_submission",)
        )
        verifier_calls = [
            call
            for call in state.answering.model_calls
            if call.operation == "agent_verifier"
        ]
        self.assertEqual(len(verifier_calls), 2)

    async def test_malformed_verdict_recovers_on_retry(self) -> None:
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
                            "text": "Revenue was 10 in 2025.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                            "calculation_refs": [],
                        }
                    ],
                    "unanswered": [],
                },
            ),
            verdicts=[
                "not a verdict",
                json.dumps(
                    {
                        "premise": "none",
                        "claims": [{"index": 0, "support": "supported"}],
                    }
                ),
            ],
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(len(state.answering.validated.claims), 1)

    async def test_token_budget_forces_a_submit_only_wrap_up_round(self) -> None:
        context = replace(
            _context(),
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
                "budget": ChatAgentBudget(max_total_tokens=1000).as_dict(),
            },
        )
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["revenue"]}),
            ChatToolCall("search-2", "search_knowledge_base", {"queries": ["change"]}),
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
            usage={"total_tokens": 600},
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        wrap_up_request = model.requests[2]
        self.assertEqual(
            tuple(tool.name for tool in wrap_up_request.tools),
            ("submit_answer",),
        )
        self.assertIn(
            "retrieval budget",
            wrap_up_request.messages[-1].content,
        )
        submit_events = [
            event
            for event in state.artifacts[AGENT_TRACE_ARTIFACT].events
            if event.tool == "submit_answer"
        ]
        self.assertTrue(submit_events[-1].budget_wrap_up)
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].total_tokens,
            600 * 4,
        )

    async def test_retrieval_call_budget_forces_immediate_wrap_up(self) -> None:
        context = replace(
            _context(),
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
                "budget": ChatAgentBudget(max_retrieval_calls=1).as_dict(),
            },
        )
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
        self.assertEqual(
            tuple(tool.name for tool in model.requests[1].tools),
            ("submit_answer",),
        )
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls,
            1,
        )

    async def test_evidence_budget_drops_extra_items_with_a_notice(self) -> None:
        context = replace(
            _context(),
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
                "budget": ChatAgentBudget(max_evidence_items=1).as_dict(),
            },
        )
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

        state = await _agent(model, _Retriever(_pack(context, count=3))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        self.assertEqual(trace.evidence_ref_count, 1)
        search_payload = _tool_payload(model.requests[1], "search-1")
        self.assertEqual(search_payload["notice"], "evidence_limit_reached")
        self.assertEqual(search_payload["accepted_new_evidence_count"], 1)
        self.assertEqual(
            tuple(tool.name for tool in model.requests[1].tools),
            ("calculate", "submit_answer"),
        )

    async def test_multi_query_search_never_exceeds_remaining_retrieval_budget(
        self,
    ) -> None:
        context = replace(
            _context(),
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
                "budget": ChatAgentBudget(max_retrieval_calls=2).as_dict(),
            },
        )
        retriever = _Retriever(_pack(context))
        model = _Model(
            ChatToolCall(
                "search-1",
                "search_knowledge_base",
                {"queries": ["revenue", "margin", "guidance"]},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.queries, ["revenue", "margin"])
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls,
            2,
        )
        payload = _tool_payload(model.requests[1], "search-1")
        self.assertEqual(payload["skipped_query_count"], 1)
        self.assertEqual(payload["accepted_new_evidence_count"], 1)
        self.assertEqual(
            tuple(tool.name for tool in model.requests[1].tools),
            ("submit_answer",),
        )

    async def test_two_consecutive_no_new_searches_close_retrieval(self) -> None:
        context = _context()
        retriever = _Retriever(_pack(context))
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["one"]}),
            ChatToolCall("search-2", "search_knowledge_base", {"queries": ["two"]}),
            ChatToolCall("search-3", "search_knowledge_base", {"queries": ["three"]}),
            ChatToolCall("search-4", "search_knowledge_base", {"queries": ["four"]}),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.queries, ["one", "two", "three"])
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls,
            3,
        )
        self.assertEqual(
            _tool_payload(model.requests[2], "search-2")[
                "accepted_new_evidence_count"
            ],
            0,
        )
        self.assertEqual(
            _tool_payload(model.requests[3], "search-3")[
                "accepted_new_evidence_count"
            ],
            0,
        )
        self.assertEqual(
            tuple(tool.name for tool in model.requests[3].tools),
            ("calculate", "submit_answer"),
        )

    async def test_empty_knowledge_base_closes_after_two_no_new_searches(
        self,
    ) -> None:
        context = _context()
        empty_pack = replace(_pack(context), evidence=())
        retriever = _Retriever(empty_pack)
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["one"]}),
            ChatToolCall("search-2", "search_knowledge_base", {"queries": ["two"]}),
            ChatToolCall("search-3", "search_knowledge_base", {"queries": ["three"]}),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.queries, ["one", "two"])
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls,
            2,
        )
        self.assertEqual(
            _tool_payload(model.requests[1], "search-1")[
                "accepted_new_evidence_count"
            ],
            0,
        )
        self.assertEqual(
            _tool_payload(model.requests[2], "search-2")[
                "accepted_new_evidence_count"
            ],
            0,
        )
        self.assertEqual(
            tuple(tool.name for tool in model.requests[2].tools),
            ("calculate", "submit_answer"),
        )
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)

    async def test_simple_and_graph_no_new_searches_share_the_stop_streak(
        self,
    ) -> None:
        context = _adaptive_context()
        simple_pack = _pack(context)
        duplicate_graph_pack = _graph_pack(
            context,
            chunk_id=simple_pack.evidence[0].index_chunk_id,
        )
        no_new_graph = GraphSearchResult(
            "no_evidence",
            duplicate_graph_pack.evidence,
            new_index_chunk_ids=(),
            candidate_count=16,
            path_count=1,
            hydrated_chunk_count=1,
            hop1_count=0,
            hop2_count=1,
            hop3_count=0,
        )
        retriever = _GraphRetriever(simple_pack, [no_new_graph])
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["seed"]}),
            ChatToolCall(
                "search-2",
                "search_knowledge_base",
                {"queries": ["duplicate"]},
            ),
            ChatToolCall(
                "graph-1",
                "search_graph_relations",
                {"query": "same relation", "reason": "direct_relation"},
            ),
            ChatToolCall(
                "search-3",
                "search_knowledge_base",
                {"queries": ["must not execute"]},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.queries, ["seed", "duplicate"])
        self.assertEqual(retriever.graph_queries, ["same relation"])
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls,
            3,
        )
        self.assertEqual(
            _tool_payload(model.requests[2], "search-2")[
                "accepted_new_evidence_count"
            ],
            0,
        )
        graph_payload = _tool_payload(model.requests[3], "graph-1")
        self.assertEqual(graph_payload["new_evidence_count"], 0)
        self.assertEqual(graph_payload["accepted_new_evidence_count"], 0)
        self.assertEqual(
            tuple(tool.name for tool in model.requests[3].tools),
            ("calculate", "submit_answer"),
        )

    async def test_default_evidence_limit_prevents_another_retriever_call(
        self,
    ) -> None:
        context = _context()
        retriever = _Retriever(_pack(context, count=65))
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["fill"]}),
            ChatToolCall(
                "search-2",
                "search_knowledge_base",
                {"queries": ["must not execute"]},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.queries, ["fill"])
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        self.assertEqual(trace.evidence_ref_count, 64)
        self.assertEqual(trace.retrieval_calls, 1)
        payload = _tool_payload(model.requests[1], "search-1")
        self.assertEqual(payload["notice"], "evidence_limit_reached")
        self.assertEqual(payload["accepted_new_evidence_count"], 64)
        self.assertEqual(
            tuple(tool.name for tool in model.requests[1].tools),
            ("calculate", "submit_answer"),
        )

    async def test_new_evidence_resets_the_no_new_search_streak(self) -> None:
        context = _context()
        first = _pack(context, text="first evidence")
        second = _pack(context, text="second evidence")
        retriever = _QueryRetriever(
            {
                "one": first,
                "duplicate-one": first,
                "two": second,
                "duplicate-two-a": second,
                "duplicate-two-b": second,
            }
        )
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["one"]}),
            ChatToolCall(
                "search-2",
                "search_knowledge_base",
                {"queries": ["duplicate-one"]},
            ),
            ChatToolCall("search-3", "search_knowledge_base", {"queries": ["two"]}),
            ChatToolCall(
                "search-4",
                "search_knowledge_base",
                {"queries": ["duplicate-two-a"]},
            ),
            ChatToolCall(
                "search-5",
                "search_knowledge_base",
                {"queries": ["duplicate-two-b"]},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(len(retriever.queries), 5)
        self.assertEqual(
            state.artifacts[AGENT_TRACE_ARTIFACT].retrieval_calls,
            5,
        )
        self.assertEqual(
            _tool_payload(model.requests[3], "search-3")[
                "accepted_new_evidence_count"
            ],
            1,
        )
        self.assertEqual(
            tuple(tool.name for tool in model.requests[5].tools),
            ("calculate", "submit_answer"),
        )

    async def test_search_closed_allows_one_calculation_before_submit(self) -> None:
        context = _context()
        retriever = _Retriever(_pack(context))
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["one"]}),
            ChatToolCall("search-2", "search_knowledge_base", {"queries": ["two"]}),
            ChatToolCall("search-3", "search_knowledge_base", {"queries": ["three"]}),
            ChatToolCall(
                "calc-1",
                "calculate",
                {"expression": "10-5", "evidence_refs": ["ev_1"]},
            ),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "answered",
                    "claims": [
                        {
                            "text": "Revenue increased by 5.",
                            "kind": "fact",
                            "evidence_refs": [],
                            "calculation_refs": ["calc_1"],
                        }
                    ],
                    "unanswered": [],
                },
            ),
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.ANSWERED)
        self.assertEqual(
            tuple(tool.name for tool in model.requests[3].tools),
            ("calculate", "submit_answer"),
        )
        self.assertEqual(
            tuple(tool.name for tool in model.requests[4].tools),
            ("submit_answer",),
        )

    async def test_agent_deadline_does_not_override_deterministic_budgets(self) -> None:
        context = _context()
        model = _Model(
            ChatToolCall("search-1", "search_knowledge_base", {"queries": ["revenue"]}),
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {"outcome": "refused", "claims": [], "unanswered": []},
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(
            context,
            deadline_seconds=0.0,
        )

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(
            tuple(tool.name for tool in model.requests[0].tools),
            ("search_knowledge_base", "calculate", "submit_answer"),
        )

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

    async def test_clarify_submission_completes_with_questions_and_no_citations(self) -> None:
        context = _context()
        model = _Model(
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "clarify",
                    "claims": [],
                    "unanswered": ["Which project do you mean: Apollo or Borealis?"],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.CLARIFY)
        self.assertEqual(state.answering.rendered.citations, ())
        self.assertEqual(
            state.answering.rendered.content,
            "Before I can answer, I need to clarify: "
            "Which project do you mean: Apollo or Borealis?",
        )
        self.assertIsNone(state.answering.rendered.control_reason)
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        self.assertEqual(trace.outcome, "clarify")
        submit_events = [event for event in trace.events if event.tool == "submit_answer"]
        self.assertEqual([event.status for event in submit_events], ["ok"])
        self.assertEqual(len(model.requests), 1)

    async def test_clarify_with_claims_is_rejected_then_valid_clarify_completes(self) -> None:
        context = _context()
        model = _Model(
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "clarify",
                    "claims": [
                        {
                            "text": "Revenue was 10 in 2025.",
                            "kind": "fact",
                            "evidence_refs": ["ev_1"],
                        }
                    ],
                    "unanswered": ["Which project do you mean?"],
                },
            ),
            ChatToolCall(
                "submit-2",
                "submit_answer",
                {
                    "outcome": "clarify",
                    "claims": [],
                    "unanswered": ["Which project do you mean?"],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.CLARIFY)
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        submit_events = [event for event in trace.events if event.tool == "submit_answer"]
        self.assertEqual([event.status for event in submit_events], ["rejected", "ok"])

    async def test_clarify_submission_skips_verification(self) -> None:
        context = replace(_context(), query="这是否是同一个项目？")
        model = _Model(
            ChatToolCall(
                "submit-1",
                "submit_answer",
                {
                    "outcome": "clarify",
                    "claims": [],
                    "unanswered": ["你说的“它”指的是哪个项目？"],
                },
            ),
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.CLARIFY)
        self.assertEqual(
            state.answering.rendered.content,
            "在回答之前，我需要先和你确认：你说的“它”指的是哪个项目？",
        )
        self.assertEqual(len(model.requests), 1)


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
                "version": "native_tool_calling_agent_v3",
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
                "version": "native_tool_calling_agent_v3",
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
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        self.assertTrue(trace.forced_finalize)
        self.assertEqual(trace.stop_reason, "model_round_limit")

    async def test_forced_finalize_still_goes_through_the_verifier(self) -> None:
        context = replace(
            _adaptive_context(),
            query="甲公司是否最终属于某集团？",
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
                "budget": ChatAgentBudget(max_model_rounds=1).as_dict(),
            },
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
            verdicts=[
                json.dumps(
                    {
                        "premise": "unsupported",
                        "claims": [{"index": 0, "support": "unsupported"}],
                    }
                )
            ],
        )

        state = await _agent(model, _Retriever(_pack(context))).run(context)

        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        self.assertEqual(state.answering.rendered.citations, ())
        verifier_events = [
            event
            for event in state.artifacts[AGENT_TRACE_ARTIFACT].events
            if event.tool == "verifier"
        ]
        self.assertEqual(len(verifier_events), 1)
        self.assertEqual(verifier_events[0].status, "refused")

    async def test_forced_refusal_with_a_valid_claim_is_salvaged_as_partial(self) -> None:
        context = _context()
        budget = ChatAgentBudget(max_model_rounds=1)
        context = replace(
            context,
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
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
                        "version": "native_tool_calling_agent_v3",
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
                "version": "native_tool_calling_agent_v3",
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
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )

        state = await _agent(model, retriever).run(context)

        self.assertEqual(retriever.queries, ["one", "two", "three"])
        self.assertEqual(state.answering.rendered.outcome, AnswerOutcome.REFUSED)
        trace = state.artifacts[AGENT_TRACE_ARTIFACT]
        self.assertEqual(trace.retrieval_calls, 3)
        self.assertEqual(trace.retrieval_tool_calls, 1)
        self.assertEqual(trace.simple_tool_calls, 1)
        self.assertEqual(trace.graph_tool_calls, 0)
        self.assertEqual(trace.prompt_tokens, 6)
        self.assertEqual(trace.completion_tokens, 4)
        self.assertEqual(trace.total_tokens, 10)
        self.assertEqual(trace.stop_reason, "submitted")
        self.assertFalse(trace.forced_finalize)

    async def test_forced_finalize_with_invalid_claims_completes_as_refused(self) -> None:
        context = _context()
        budget = ChatAgentBudget(max_model_rounds=2)
        context = replace(
            context,
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
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
                "version": "native_tool_calling_agent_v3",
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
        context = replace(
            _context(),
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
                "budget": ChatAgentBudget(max_evidence_items=128).as_dict(),
            },
        )
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
        context = replace(
            _context(),
            agent_configuration={
                "version": "native_tool_calling_agent_v3",
                "budget": ChatAgentBudget(max_evidence_items=128).as_dict(),
            },
        )
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
