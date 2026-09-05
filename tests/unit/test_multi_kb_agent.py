from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import unittest
from uuid import uuid4

from rag_kb.answering.agent import _tools, _fair_round_groups, _CallOutcome, _query_candidates
from rag_kb.retrieval.eligibility import EvidenceEligibilityPolicy
from rag_kb.answering.scope import BoundedRetriever
from rag_kb.services.chat_visuals import VisualEvidencePreparationStep
from tests.unit.test_native_tool_calling_agent import _adaptive_context, _graph_pack, _graph_result, _native_visual_pack, _AssetMapReader
from rag_kb.answering.scope import target_context
from rag_kb.domain import ChatModelResponse, ChatToolCall, ChatPipelineExecutionError, ChatPipelinePhase, ErrorCode, ServingDocumentList
from rag_kb.domain.chat_scope import ChatKnowledgeBaseSnapshot, normalize_knowledge_base_ids, resolve_scope
from rag_kb.schemas.chat import ChatRunCreate, ChatSessionCreate
from tests.unit.test_native_tool_calling_agent import _context, _pack, _agent


class Model:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        turn = self.turns.pop(0)
        calls = (turn,) if isinstance(turn, ChatToolCall) else turn if isinstance(turn, tuple) else ()
        return ChatModelResponse(content="" if calls else turn, model="fixed-model", finish_reason="tool_calls" if calls else "stop", provider_request_id=None,
            usage={"total_tokens": 100}, tool_calls=calls)


def context_and_packs(count=2):
    base = _context()
    snapshots = tuple(ChatKnowledgeBaseSnapshot(uuid4(), f"Library {i}", uuid4(), base.retrieval_strategy) for i in range(count))
    context = replace(base, knowledge_base_id=None, index_revision_id=None, knowledge_bases=snapshots)
    packs = {item.knowledge_base_id: _pack(target_context(context, item), text=f"Library {i}: the value is {10+i}.") for i, item in enumerate(snapshots)}
    return context, packs


class Retriever:
    def __init__(self, packs, failures=()):
        self.packs = packs
        self.failures = set(failures)
        self.calls = []
        self.active = 0
        self.maximum = 0
        self.delay = 0.001

    async def list_documents(self, context):
        return ServingDocumentList(context.index_revision_id)

    async def keyword_search_capable(self, context): return True
    async def graph_relations_capable(self, context): return False

    async def semantic_search(self, context, query, **kwargs):
        self.calls.append((context.knowledge_base_id, query, dict(context.retrieval_strategy)))
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(self.delay)
            if context.knowledge_base_id in self.failures:
                raise ChatPipelineExecutionError(ErrorCode.CHAT_REVISION_MISMATCH, phase=ChatPipelinePhase.RETRIEVE_EVIDENCE)
            return self.packs[context.knowledge_base_id]
        finally:
            self.active -= 1

    keyword_search = semantic_search


def payload(model, call_id):
    return next(json.loads(message.content) for request in model.requests for message in request.messages if message.role == "tool" and message.tool_call_id == call_id)


class MultiKnowledgeBaseTests(unittest.IsolatedAsyncioTestCase):
    def test_legacy_api_and_explicit_scope_share_normalization(self):
        identifier = uuid4()
        legacy = ChatSessionCreate(knowledge_base_id=identifier)
        explicit = ChatSessionCreate(knowledge_base_ids=(identifier,))
        self.assertEqual(legacy.knowledge_base_ids, explicit.knowledge_base_ids)
        for ids in ((), (identifier, identifier)):
            with self.assertRaises(ValueError): normalize_knowledge_base_ids(ids)
        with self.assertRaises(ValueError): ChatSessionCreate(knowledge_base_id=identifier, knowledge_base_ids=(identifier,))

    def test_tools_require_scope_except_calculate(self):
        for tool in _tools(keyword_ready=True, adaptive=True, graph_ready=True):
            self.assertEqual("knowledge_base_id" in tool.input_schema.get("required", ()), tool.name != "calculate")

    async def test_all_selected_runs_every_query_and_preserves_same_name_sources(self):
        context, packs = context_and_packs()
        retriever = Retriever(packs)
        model = Model(ChatToolCall("both", "semantic_search", {"knowledge_base_id": "all_selected", "queries": ["value", "period"]}), "Values are 10 [ev_1] and 11 [ev_2].")
        state = await _agent(model, retriever).run(context)
        self.assertEqual(len(retriever.calls), 4)
        self.assertEqual({(identifier, query) for identifier, query, _ in retriever.calls}, {(identifier, query) for identifier in packs for query in ("value", "period")})
        self.assertIsNone(state.evidence_pack)
        self.assertEqual(len(state.evidence_packs), 2)
        self.assertEqual({item.evidence.knowledge_base_id for item in state.answering.rendered.citations}, set(packs))
        self.assertEqual({item.evidence.document_display_name for item in state.answering.rendered.citations}, {"Report"})
        self.assertEqual(len(payload(model, "both")["groups"]), 4)

    async def test_missing_invalid_and_outside_scope_never_execute(self):
        for target in (None, "", "Library 0", str(uuid4())):
            with self.subTest(target=target):
                context, packs = context_and_packs()
                arguments = {"queries": ["value"]}
                if target is not None: arguments["knowledge_base_id"] = target
                model = Model(ChatToolCall("bad", "semantic_search", arguments), "Insufficient evidence.")
                retriever = Retriever(packs)
                await _agent(model, retriever).run(context)
                self.assertEqual(retriever.calls, [])
                self.assertEqual(payload(model, "bad")["status"], "error")

    async def test_one_failed_kb_keeps_other_evidence(self):
        context, packs = context_and_packs()
        failed = context.knowledge_bases[0].knowledge_base_id
        retriever = Retriever(packs, failures=(failed,))
        model = Model(ChatToolCall("partial", "semantic_search", {"knowledge_base_id": "all_selected", "queries": ["value"]}), "One value is 11 [ev_1].")
        state = await _agent(model, retriever).run(context)
        self.assertEqual(len(state.answering.rendered.citations), 1)
        self.assertNotEqual(state.answering.rendered.citations[0].evidence.knowledge_base_id, failed)
        self.assertEqual(len(payload(model, "partial")["knowledge_bases"]), 2)

    async def test_first_query_of_b_survives_repeated_empty_a(self):
        context, packs = context_and_packs()
        a, b = context.knowledge_bases
        packs[a.knowledge_base_id] = replace(packs[a.knowledge_base_id], evidence=())
        model = Model(*(ChatToolCall(f"a{i}", "semantic_search", {"knowledge_base_id": str(a.knowledge_base_id), "queries": ["empty"]}) for i in range(2)),
            ChatToolCall("b", "semantic_search", {"knowledge_base_id": str(b.knowledge_base_id), "queries": ["value"]}), "B is 11 [ev_1].")
        retriever = Retriever(packs)
        state = await _agent(model, retriever).run(context)
        self.assertEqual(len(retriever.calls), 3)
        self.assertEqual(state.answering.rendered.citations[0].evidence.knowledge_base_id, b.knowledge_base_id)

    async def test_all_tools_and_queries_share_four_slots(self):
        context, packs = context_and_packs(7)
        retriever = Retriever(packs)
        retriever.delay = 0.01
        calls = tuple(ChatToolCall(str(i), name, {"knowledge_base_id": "all_selected", "queries": ["a", "b", "c"]}) for i, name in enumerate(("semantic_search", "keyword_search")))
        await _agent(Model(calls, "Answer [ev_1]."), retriever).run(context)
        self.assertEqual(len(retriever.calls), 42)
        self.assertEqual(retriever.maximum, 4)

    async def test_cancellation_stops_pending_and_active_targets(self):
        context, packs = context_and_packs(8)
        retriever = Retriever(packs)
        retriever.delay = 10
        model = Model(ChatToolCall("cancel", "semantic_search", {"knowledge_base_id": "all_selected", "queries": ["a", "b"]}))
        task = asyncio.create_task(_agent(model, retriever).run(context))
        for _ in range(100):
            if retriever.active: break
            await asyncio.sleep(0.001)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(retriever.active, 0)
        self.assertLessEqual(len(retriever.calls), 4)

    async def test_read_context_cannot_relabel_a_ref(self):
        context, packs = context_and_packs()
        a, b = context.knowledge_bases
        model = Model(ChatToolCall("a", "semantic_search", {"knowledge_base_id": str(a.knowledge_base_id), "queries": ["value"]}),
            ChatToolCall("wrong", "read_chunk_context", {"knowledge_base_id": str(b.knowledge_base_id), "evidence_refs": ["ev_1"]}), "A is 10 [ev_1].")
        await _agent(model, Retriever(packs)).run(context)
        self.assertIn("anchor_scope_mismatch", json.dumps(payload(model, "wrong")))

    async def test_unavailable_index_remains_in_scope(self):
        context, packs = context_and_packs()
        a, b = context.knowledge_bases
        context = replace(context, knowledge_bases=(replace(a, index_revision_id=None, status="index_unavailable"), b))
        model = Model(ChatToolCall("all", "semantic_search", {"knowledge_base_id": "all_selected", "queries": ["value"]}), "B is 11 [ev_1].")
        await _agent(model, Retriever(packs)).run(context)
        self.assertEqual(payload(model, "all")["knowledge_bases"][0]["status"], "index_unavailable")

    async def test_target_retrieval_configuration_is_not_borrowed(self):
        context, packs = context_and_packs()
        a, b = context.knowledge_bases
        context = replace(context, knowledge_bases=(a, replace(b, retrieval_strategy={**b.retrieval_strategy, "top_k": 7, "rerank_mode": "classic"})))
        retriever = Retriever(packs)
        await _agent(Model(ChatToolCall("all", "semantic_search", {"knowledge_base_id": "all_selected", "queries": ["value"]}), "Answer [ev_1]."), retriever).run(context)
        self.assertEqual([item[2]["top_k"] for item in retriever.calls], [3, 7])


    async def test_malformed_inputs_are_rejected_before_recording_scope_metadata(self):
        for name,arguments in (("semantic_search",{"queries":[123]}),("semantic_search",{"queries":["x"*2049]}),("search_graph_relations",{"query":True,"reason":"direct_relation"}),("list_documents",{"after_document_id":"wrong"})):
            context,packs=context_and_packs()
            model=Model(ChatToolCall('invalid',name,{'knowledge_base_id':'all_selected',**arguments}),'Insufficient evidence.')
            retriever=Retriever(packs)
            state=await _agent(model,retriever).run(context)
            self.assertEqual(retriever.calls,[])
            self.assertEqual(payload(model,'invalid')['status'],'error')
            self.assertEqual(state.answering.rendered.citations,())

    async def test_wrong_returned_pack_identity_is_isolated(self):
        for mismatch in ("knowledge_base_id", "index_revision_id"):
            context, packs = context_and_packs()
            a, b = context.knowledge_bases
            if mismatch == "knowledge_base_id": packs[a.knowledge_base_id] = replace(packs[a.knowledge_base_id], knowledge_base_id=uuid4())
            else: packs[a.knowledge_base_id] = packs[b.knowledge_base_id]
            model = Model(ChatToolCall("all", "semantic_search", {"knowledge_base_id": "all_selected", "queries": ["value"]}), "B [ev_1].")
            state = await _agent(model, Retriever(packs)).run(context)
            self.assertEqual({c.evidence.knowledge_base_id for c in state.answering.rendered.citations}, {b.knowledge_base_id})
            self.assertIn("chat_context_invalid", json.dumps(payload(model, "all")))

    async def test_version_switch_during_retrieval_discards_that_target(self):
        context, packs = context_and_packs()
        a, b = context.knowledge_bases
        class Switching(Retriever):
            async def validate_scope(self, context, *, check_graph=False):
                if context.knowledge_base_id == a.knowledge_base_id and any(k == a.knowledge_base_id for k, _, _ in self.calls):
                    raise ChatPipelineExecutionError(ErrorCode.CHAT_REVISION_MISMATCH, phase=ChatPipelinePhase.RETRIEVE_EVIDENCE)
        model = Model(ChatToolCall("all", "semantic_search", {"knowledge_base_id": "all_selected", "queries": ["value"]}), "B [ev_1].")
        state = await _agent(model, Switching(packs)).run(context)
        self.assertEqual({c.evidence.knowledge_base_id for c in state.answering.rendered.citations}, {b.knowledge_base_id})
        self.assertIn("chat_revision_mismatch", json.dumps(payload(model, "all")))

    async def test_local_graphs_keep_separate_sources_even_with_same_path_id(self):
        context, _ = context_and_packs()
        profile = _adaptive_context().retrieval_strategy
        context = replace(context, retrieval_strategy=profile, knowledge_bases=tuple(replace(s, retrieval_strategy=profile, graph_build_id=uuid4()) for s in context.knowledge_bases))
        packs = {s.knowledge_base_id: _graph_pack(target_context(context, s)) for s in context.knowledge_bases}
        class Graph(Retriever):
            async def graph_relations_capable(self, context):return True
            async def search_graph_relations(self, context, query, **kwargs):
                self.calls.append((context.knowledge_base_id,query,{}))
                return _graph_result(self.packs[context.knowledge_base_id])
        model = Model(ChatToolCall("graphs", "search_graph_relations", {"knowledge_base_id": "all_selected", "query": "dependencies", "reason": "relation_chain"}), "Both sources [ev_1] [ev_2].")
        state = await _agent(model, Graph(packs)).run(context)
        self.assertEqual(len(state.answering.rendered.citations), 2)
        self.assertEqual({group['knowledge_base_id'] for group in payload(model, 'graphs')['groups']}, {str(s.knowledge_base_id) for s in context.knowledge_bases})

    def test_display_budget_round_robins_complete_chunks_and_graph_paths(self):
        context, _ = context_and_packs()
        a, b = context.knowledge_bases
        first = _pack(target_context(context,a),count=3,text='A'*30000)
        second = _pack(target_context(context,b),text='B'*30000)
        call = _CallOutcome(ChatToolCall('all','semantic_search',{}), packs=(first,second))
        groups = _fair_round_groups([(call,(first.evidence,second.evidence))],set(),{})[0][1]
        self.assertEqual([len(group) for group in groups],[2,1])
        self.assertEqual(call.group_metadata[0]['omitted_count'],1)
        self.assertEqual(groups[1][0].text,'B'*30000)
        path = tuple(replace(e,score_kind=__import__('rag_kb.domain',fromlist=['EvidenceScoreKind']).EvidenceScoreKind.GRAPH_PATH,score=1.0,vector_similarity=None,matched_representations=('graph_path','text'),graph_path_id='p',graph_hop_count=2,graph_path_rank=1,graph_anchor_index_chunk_id=first.evidence[0].index_chunk_id) for e in first.evidence)
        call = _CallOutcome(ChatToolCall('all','semantic_search',{}))
        groups = _fair_round_groups([(call,(second.evidence,path))],set(),{})[0][1]
        self.assertEqual([len(group) for group in groups],[1,0])
        self.assertEqual(call.group_metadata[1]['omitted_count'],3)

    async def test_visual_sources_and_image_budget_are_per_source_and_per_run(self):
        context, _ = context_and_packs()
        context = replace(context, model_configuration={"resolved_model":"fixed-model","max_tokens":2048,"vision_enabled":True,"max_visual_images":2,"max_visual_image_bytes":5242880,"max_visual_total_bytes":12582912,"max_visual_pixels":16000000})
        items = [_native_visual_pack(target_context(context,s)) for s in context.knowledge_bases]
        packs = {s.knowledge_base_id:item[0] for s,item in zip(context.knowledge_bases,items)}
        # A wrong-KB image body must not become citeable while another KB's valid image survives.
        wrong = replace(items[0][1], snapshot=replace(items[0][1].snapshot,kb_id=context.knowledge_bases[1].knowledge_base_id))
        model = Model(ChatToolCall('images','semantic_search',{'knowledge_base_id':'all_selected','queries':['chart']}), 'Images [ev_1] [ev_2].')
        state = await _agent(model,Retriever(packs),visual_preparer=VisualEvidencePreparationStep(_AssetMapReader(wrong,items[1][1]))).run(context)
        self.assertEqual(len(state.answering.visual_content),1)
        self.assertEqual({c.evidence.knowledge_base_id for c in state.answering.rendered.citations},{context.knowledge_bases[1].knowledge_base_id})

    async def test_scope_trace_counts_retrieved_admitted_and_displayed(self):
        context,packs=context_and_packs()
        model=Model(ChatToolCall('all','semantic_search',{'knowledge_base_id':'all_selected','queries':['value']}),'Answer [ev_1].')
        state=await _agent(model,Retriever(packs)).run(context)
        records=state.artifacts['chat_agent_trace'].as_dict()['diagnostics']['scope_calls']
        self.assertEqual(len(records),2)
        for record in records:
            group=record['groups'][0]
            self.assertEqual((group['retrieved_count'],group['eligible_count'],group['admitted_count'],group['displayed_count']),(1,1,1,1))


if __name__ == "__main__": unittest.main()
