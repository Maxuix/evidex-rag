from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import unittest
from uuid import uuid4

from rag_kb.adapters.chat_preview.pg_notify import MAX_NOTIFY_PAYLOAD_BYTES, parse_preview_payload, serialize_preview_event
from rag_kb.answering.activity import ChatActivityRecorder
from rag_kb.answering.agent import ChatAgentProgress
from rag_kb.answering.runner import NativeAgentRunner
from rag_kb.domain import ChatExecutionCommand, ChatPipelineExecutionError, ChatPipelinePhase, ChatToolCall, ErrorCode
from rag_kb.domain.chat_activity import CHAT_ACTIVITY_ARTIFACT, MAX_ACTIVITY_BYTES, ActivitySource, ChatActivitySnapshot
from tests.unit.test_native_tool_calling_agent import _Model, _Retriever, _agent, _context, _pack
from tests.unit.test_chat_terminal import _ContextLoader, _Persister, _Reporter


class ActivityTests(unittest.TestCase):
    def test_unicode_preview_is_bounded_and_terminal_keeps_validated_inputs(self):
        events = []
        recorder = ChatActivityRecorder(uuid4(), 1, emit=events.append)
        queries = tuple(('中文查询' * 500) + str(i) for i in range(3))
        key = recorder.begin('tool', 'semantic_search', queries=queries)
        sources = tuple(ActivitySource(str(uuid4()), str(uuid4()), '来源标题' * 120) for _ in range(100))
        recorder.update(key, 'succeeded', sources=sources, returned_count=100)
        wire = serialize_preview_event(events[-1])
        self.assertLessEqual(len(wire.encode()), MAX_NOTIFY_PAYLOAD_BYTES)
        parsed = parse_preview_payload(wire)
        self.assertTrue(parsed.step.details_truncated)
        self.assertEqual(parsed.step.returned_count, 100)
        snapshot = recorder.snapshot('completed')
        self.assertEqual(snapshot.steps[0].queries, queries)
        self.assertEqual(ChatActivitySnapshot.from_dict(json.loads(json.dumps(snapshot.as_dict()))), snapshot)
        bad = json.loads(wire)
        bad['step']['reasoning'] = 'must not be accepted'
        with self.assertRaises(ValueError):
            parse_preview_payload(json.dumps(bad))
        for patch in ({"run_id": None}, {"step": {**json.loads(wire)["step"], "kind": []}}):
            with self.assertRaises(ValueError):
                parse_preview_payload(json.dumps({**json.loads(wire), **patch}))

    def test_size_budget_preserves_steps_before_dropping_details(self):
        recorder = ChatActivityRecorder(uuid4(), 1)
        for i in range(90):
            key = recorder.begin('tool', 'semantic_search', queries=('中' * 2048,) * 3)
            recorder.update(key, 'succeeded', returned_count=i)
        snapshot = recorder.snapshot('completed')
        self.assertEqual(len(snapshot.steps), 90)
        self.assertEqual(snapshot.omitted_step_count, 0)
        self.assertTrue(any(step.details_truncated for step in snapshot.steps))
        self.assertLessEqual(len(json.dumps(snapshot.as_dict(), ensure_ascii=False).encode()), MAX_ACTIVITY_BYTES)

    def test_step_bound_reports_exact_omissions_and_delivery_failure_is_harmless(self):
        def broken(event):
            raise OSError('offline')
        recorder = ChatActivityRecorder(uuid4(), 1, emit=broken)
        for _ in range(1030):
            key = recorder.begin('tool', 'calculate')
            recorder.update(key, 'succeeded', result_value='2')
        snapshot = recorder.snapshot('completed')
        self.assertEqual(snapshot.omitted_step_count, 6)
        self.assertEqual(len(snapshot.steps), 1024)
        self.assertEqual(snapshot.steps[-1].ordinal, 1030)


class AgentActivityTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_unavailability_is_not_reported_as_success(self):
        from rag_kb.domain import GraphSearchResult
        from tests.unit.test_native_tool_calling_agent import _GraphRetriever, _adaptive_context
        context = _adaptive_context()
        calls = (ChatToolCall("graph", "search_graph_relations", {"query": "关系", "reason": "direct_relation"}),)
        state = await _agent(_Model(calls, None), _GraphRetriever(_pack(context), [GraphSearchResult("unavailable", ())])).run(context)
        step = next(step for step in state.artifacts[CHAT_ACTIVITY_ARTIFACT].steps if step.kind == "tool")
        self.assertEqual(step.status, "failed")
        self.assertEqual(step.result_code, "unavailable")

    async def test_parallel_return_is_visible_before_slow_call_and_preserves_order(self):
        context = _context()
        released, fast_returned = asyncio.Event(), asyncio.Event()
        events = []
        class Retriever(_Retriever):
            async def semantic_search(self, context, query, **kwargs):
                await released.wait()
                return self.pack
        def observe(event):
            events.append(event)
            if event.step.name == 'keyword_search' and event.step.status == 'processing':
                fast_returned.set()
        retriever = Retriever(_pack(context), keyword_ready=True)
        model = _Model((ChatToolCall('slow', 'semantic_search', {'queries': ['收入']}), ChatToolCall('fast', 'keyword_search', {'queries': ['营业收入']})), None)
        progress = ChatAgentProgress(activity=ChatActivityRecorder(context.run_id, 1, emit=observe))
        task = asyncio.create_task(_agent(model, retriever).run(context, progress=progress))
        try:
            await asyncio.wait_for(fast_returned.wait(), 2)
            states = {e.step.name: e.step.status for e in events}
            self.assertEqual(states['semantic_search'], 'running')
            self.assertEqual(states['keyword_search'], 'processing')
        finally:
            released.set()
        state = await task
        steps = [step for step in state.artifacts[CHAT_ACTIVITY_ARTIFACT].steps if step.kind == 'tool']
        self.assertEqual([step.name for step in steps], ['semantic_search', 'keyword_search'])
        self.assertEqual([step.round for step in steps], [1, 1])
        self.assertEqual([step.status for step in steps], ['succeeded', 'succeeded'])
        self.assertEqual(sum(step.new_evidence_count for step in steps), 1)
        self.assertEqual(steps[0].sources[0].ref, steps[1].sources[0].ref)

    async def test_seventy_tools_survive_legacy_64_event_limit(self):
        context = _context()
        calls = tuple(ChatToolCall(str(i), 'calculate', {'expression': '120 * 3 + 80'}) for i in range(70))
        state = await _agent(_Model(calls, None), _Retriever(_pack(context))).run(context)
        steps = [step for step in state.artifacts[CHAT_ACTIVITY_ARTIFACT].steps if step.kind == 'tool']
        self.assertEqual(len(steps), 70)
        self.assertEqual(len({step.step_id for step in steps}), 70)
        self.assertTrue(all(step.result_value == '440' for step in steps))

    async def test_invalid_parameters_never_enter_activity_and_failure_is_per_tool(self):
        context = _context()
        class Retriever(_Retriever):
            async def semantic_search(self, *args, **kwargs):
                raise ChatPipelineExecutionError(ErrorCode.CHAT_PROVIDER_UNAVAILABLE, phase=ChatPipelinePhase.GENERATE_OR_REFUSE)
        calls = (
            ChatToolCall('bad', 'semantic_search', {'queries': ['valid'], 'secret': 'PRIVATE_PROVIDER_CONTENT'}),
            ChatToolCall('offline', 'semantic_search', {'queries': ['query']}),
            ChatToolCall('good', 'calculate', {'expression': '1 + 2'}),
        )
        state = await _agent(_Model(calls, None), Retriever(_pack(context))).run(context)
        snapshot = state.artifacts[CHAT_ACTIVITY_ARTIFACT]
        steps = [step for step in snapshot.steps if step.kind == 'tool']
        self.assertEqual([step.status for step in steps], ['rejected', 'failed', 'succeeded'])
        self.assertEqual(steps[0].queries, ())
        self.assertNotIn('PRIVATE_PROVIDER_CONTENT', json.dumps(snapshot.as_dict()))
        self.assertEqual(steps[-1].result_value, '3')

    async def test_worker_cancellation_carries_safe_partial_activity(self):
        context = _context()
        entered = asyncio.Event()
        class Model:
            async def complete(self, request):
                entered.set()
                await asyncio.Event().wait()
        runner = NativeAgentRunner(_ContextLoader(context), _agent(Model(), _Retriever(_pack(context))), _Persister(None), deadline_seconds=10, progress_reporter_factory=lambda *_: _Reporter())
        task = asyncio.create_task(runner.execute(ChatExecutionCommand(context.lease)))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await task
        snapshot = ChatActivitySnapshot.from_dict(raised.exception.agent_trace['activity'])
        self.assertEqual(snapshot.status, 'cancelled')
        self.assertEqual(snapshot.steps[-1].status, 'cancelled')
        self.assertEqual(snapshot.steps[0].status, 'succeeded')
        from datetime import UTC, datetime
        from rag_kb.domain import ChatTerminalWriteStatus
        from rag_kb.services.chat_terminal import ChatFailureSettlementService
        from rag_kb.repositories.sqlalchemy_chat import _serialized_failure
        from tests.unit.test_chat_terminal import _Repository, _Factory
        repository = _Repository(ChatTerminalWriteStatus.APPLIED)
        await ChatFailureSettlementService(_Factory(repository), max_attempts=3,
            base_delay_seconds=1, max_delay_seconds=5, clock=lambda: datetime.now(UTC)
        ).settle(context.lease, raised.exception)
        stored = _serialized_failure(repository.failure)["agent_trace"]["activity"]
        self.assertEqual(ChatActivitySnapshot.from_dict(stored), snapshot)



class ActivityPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_serialization_retains_same_activity_as_live(self):
        from datetime import UTC, datetime
        from rag_kb.services.chat_terminal import ChatResultPersistenceStep
        from rag_kb.domain import ChatTerminalWriteStatus
        from rag_kb.domain import ChatAgentTrace, ChatAgentBudget
        from rag_kb.answering.agent import CHAT_AGENT_TRACE_ARTIFACT
        from tests.unit.test_chat_terminal import _completed_state, _Repository, _Factory
        observed = datetime(2026, 9, 5, tzinfo=UTC)
        state = _completed_state(observed)
        events = []
        recorder = ChatActivityRecorder(state.context.run_id, state.context.lease.attempt, emit=events.append)
        key = recorder.begin("tool", "calculate", expression="1 + 2")
        recorder.update(key, "succeeded", result_value="3")
        snapshot = recorder.snapshot()
        state = replace(state, artifacts={**state.artifacts, CHAT_ACTIVITY_ARTIFACT: snapshot,
            CHAT_AGENT_TRACE_ARTIFACT: ChatAgentTrace(budget=ChatAgentBudget(), events=(), model_rounds=1, retrieval_calls=0, calculation_calls=1, evidence_ref_count=0, outcome="refused")})
        repository = _Repository(ChatTerminalWriteStatus.APPLIED)
        await ChatResultPersistenceStep(_Factory(repository), clock=lambda: observed).run(state)
        stored = json.loads(json.dumps(dict(repository.success.agent_trace)))["activity"]
        self.assertEqual(ChatActivitySnapshot.from_dict(stored), snapshot)
        self.assertEqual(ChatActivitySnapshot.from_dict(stored).steps[0], events[-1].step)
