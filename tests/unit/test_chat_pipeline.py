from __future__ import annotations

import asyncio
import json
import time
import unittest
from datetime import UTC, datetime
from uuid import uuid4

from rag_kb.adapters.model_api.openai_compatible_chat import (
    OpenAICompatibleChatModelAdapter,
    _RetryableProviderError,
)
from rag_kb.domain import (
    ChatExecutionCommand,
    ChatExecutionContext,
    ChatModelExecutionError,
    ChatModelMessage,
    ChatModelRequest,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ChatRunLease,
    ErrorCode,
    EvidencePack,
    RetrievalStrategy,
)
from rag_kb.services.chat_pipeline import ChatEvidenceRetriever, DirectChatPipeline
from rag_kb.workflows import DirectGraphRunner, GraphRunner


def context() -> ChatExecutionContext:
    return ChatExecutionContext(
        run_id=uuid4(),
        workspace_id=uuid4(),
        knowledge_base_id=uuid4(),
        session_id=uuid4(),
        user_message_id=uuid4(),
        assistant_message_id=uuid4(),
        index_revision_id=uuid4(),
        principal_id="principal",
        client_id="client",
        query="What is frozen?",
        effective_policy={"grounding_policy": "evidence_only"},
        retrieval_strategy={
            "strategy": "exact_vector",
            "top_k": 3,
            "rerank": False,
        },
        model_configuration={"requested_model": "fixed"},
        attempt=1,
    )


class _Loader:
    def __init__(self, value: ChatExecutionContext, events: list[str]) -> None:
        self.value = value
        self.events = events

    async def load(self, command: ChatExecutionCommand) -> ChatExecutionContext:
        self.events.append("load_context")
        return self.value


class _Retriever:
    def __init__(self, value: EvidencePack, events: list[str]) -> None:
        self.value = value
        self.events = events

    async def retrieve(self, value: ChatExecutionContext) -> EvidencePack:
        self.events.append("retrieve_evidence")
        return self.value


class _Step:
    def __init__(self, name: str, events: list[str], *, fail: bool = False) -> None:
        self.name = name
        self.events = events
        self.fail = fail

    async def run(self, state: ChatPipelineState) -> ChatPipelineState:
        self.events.append(self.name)
        if self.fail:
            raise RuntimeError("content must not cross the boundary")
        return ChatPipelineState(
            context=state.context,
            evidence_pack=state.evidence_pack,
            artifacts={**state.artifacts, self.name: True},
        )


class DirectPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_enforces_the_six_frozen_phases(self) -> None:
        execution_context = context()
        events: list[str] = []
        pack = EvidencePack(
            knowledge_base_id=execution_context.knowledge_base_id,
            index_revision_id=execution_context.index_revision_id,
            strategy=RetrievalStrategy.EXACT_VECTOR,
        )
        pipeline = DirectChatPipeline(
            _Loader(execution_context, events),  # type: ignore[arg-type]
            _Retriever(pack, events),  # type: ignore[arg-type]
            _Step("assess_evidence", events),
            _Step("generate_or_refuse", events),
            _Step("validate_structure", events),
            _Step("persist_result", events),
            deadline_seconds=1,
        )
        runner = DirectGraphRunner(pipeline)
        self.assertIsInstance(runner, GraphRunner)
        command = ChatExecutionCommand(
            ChatRunLease(
                run_id=execution_context.run_id,
                workspace_id=execution_context.workspace_id,
                claimed_by="worker",
                attempt=1,
                claimed_at=datetime.now(UTC),
            )
        )

        result = await runner.run(command)

        self.assertEqual(
            events,
            [phase.value for phase in ChatPipelinePhase],
        )
        self.assertTrue(result.artifacts["persist_result"])
        with self.assertRaises(TypeError):
            execution_context.effective_policy["x"] = "y"  # type: ignore[index]

    async def test_failure_is_mapped_to_the_current_phase_without_content(self) -> None:
        execution_context = context()
        events: list[str] = []
        pack = EvidencePack(
            knowledge_base_id=execution_context.knowledge_base_id,
            index_revision_id=execution_context.index_revision_id,
            strategy=RetrievalStrategy.EXACT_VECTOR,
        )
        pipeline = DirectChatPipeline(
            _Loader(execution_context, events),  # type: ignore[arg-type]
            _Retriever(pack, events),  # type: ignore[arg-type]
            _Step("assess_evidence", events, fail=True),
            _Step("generate_or_refuse", events),
            _Step("validate_structure", events),
            _Step("persist_result", events),
            deadline_seconds=1,
        )

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await pipeline.execute(
                ChatExecutionCommand(
                    ChatRunLease(
                        execution_context.run_id,
                        execution_context.workspace_id,
                        "worker",
                        1,
                        datetime.now(UTC),
                    )
                )
            )

        self.assertEqual(raised.exception.code, ErrorCode.CHAT_PIPELINE_STEP_FAILED)
        self.assertEqual(raised.exception.phase, ChatPipelinePhase.ASSESS_EVIDENCE)
        self.assertNotIn("content must not", str(raised.exception.diagnostic))

    async def test_retrieval_fails_closed_when_active_revision_moved(self) -> None:
        execution_context = context()

        class Retrieval:
            async def retrieve(self, auth, request):
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=uuid4(),
                    strategy=request.strategy,
                )

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            retriever = ChatEvidenceRetriever(Retrieval())  # type: ignore[arg-type]
            await retriever.retrieve(execution_context)

        self.assertEqual(raised.exception.code, ErrorCode.CHAT_REVISION_MISMATCH)

    async def test_pipeline_deadline_is_stable_and_phase_specific(self) -> None:
        execution_context = context()
        events: list[str] = []

        class SlowLoader(_Loader):
            async def load(
                self, command: ChatExecutionCommand
            ) -> ChatExecutionContext:
                await asyncio.sleep(0.05)
                return await super().load(command)

        pipeline = DirectChatPipeline(
            SlowLoader(execution_context, events),  # type: ignore[arg-type]
            _Retriever(
                EvidencePack(
                    knowledge_base_id=execution_context.knowledge_base_id,
                    index_revision_id=execution_context.index_revision_id,
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                ),
                events,
            ),  # type: ignore[arg-type]
            _Step("assess_evidence", events),
            _Step("generate_or_refuse", events),
            _Step("validate_structure", events),
            _Step("persist_result", events),
            deadline_seconds=0.001,
        )
        lease = ChatRunLease(
            execution_context.run_id,
            execution_context.workspace_id,
            "worker",
            1,
            datetime.now(UTC),
        )

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await pipeline.execute(ChatExecutionCommand(lease))

        self.assertEqual(
            raised.exception.code, ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED
        )
        self.assertEqual(raised.exception.phase, ChatPipelinePhase.LOAD_CONTEXT)


class ChatModelAdapterTests(unittest.IsolatedAsyncioTestCase):
    def adapter(self) -> OpenAICompatibleChatModelAdapter:
        return OpenAICompatibleChatModelAdapter(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="fixed-model",
            timeout_seconds=1,
            max_retries=1,
            max_concurrency=1,
        )

    def test_json_response_is_strictly_decoded(self) -> None:
        response = self.adapter()._decode(
            json.dumps(
                {
                    "model": "fixed-model",
                    "choices": [
                        {
                            "message": {"content": '{"answer":"ok"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            ).encode(),
            provider_request_id="request-1",
        )
        self.assertEqual(response.provider_request_id, "request-1")
        self.assertEqual(response.usage["prompt_tokens"], 3)
        with self.assertRaises(ChatModelExecutionError) as raised:
            self.adapter()._decode(b"{}", provider_request_id=None)
        self.assertEqual(raised.exception.code, ErrorCode.CHAT_RESPONSE_INVALID)

    async def test_retry_is_finite_and_concurrency_is_bounded(self) -> None:
        adapter = self.adapter()
        active = 0
        maximum = 0
        calls = 0

        def request(value):
            nonlocal active, maximum, calls
            calls += 1
            if calls == 1:
                raise _RetryableProviderError
            active += 1
            maximum = max(maximum, active)
            time.sleep(0.02)
            active -= 1
            return adapter._decode(
                b'{"model":"fixed-model","choices":[{"message":{"content":"{}"}}]}',
                provider_request_id=None,
            )

        adapter._request = request  # type: ignore[method-assign]
        model_request = ChatModelRequest((ChatModelMessage("user", "hello"),))
        await asyncio.gather(
            adapter.complete(model_request),
            adapter.complete(model_request),
        )
        self.assertEqual(maximum, 1)
        self.assertEqual(calls, 3)

    async def test_total_timeout_is_content_safe(self) -> None:
        adapter = OpenAICompatibleChatModelAdapter(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="fixed-model",
            timeout_seconds=0.001,
            max_retries=0,
            max_concurrency=1,
        )

        def request(value):
            time.sleep(0.02)
            raise RuntimeError("provider content")

        adapter._request = request  # type: ignore[method-assign]
        with self.assertRaises(ChatModelExecutionError) as raised:
            await adapter.complete(
                ChatModelRequest((ChatModelMessage("user", "hello"),))
            )
        self.assertEqual(raised.exception.code, ErrorCode.CHAT_PROVIDER_UNAVAILABLE)
        self.assertNotIn("provider content", str(raised.exception.diagnostic))


if __name__ == "__main__":
    unittest.main()
