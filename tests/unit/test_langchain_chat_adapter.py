from __future__ import annotations

import asyncio
import hashlib
import json
import unittest
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_kb.adapters.model_api.langchain_chat import LangChainChatModelAdapter
from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelMessage,
    ChatModelRequest,
    ChatModelVisualContent,
    ChatToolCall,
    ChatToolDefinition,
    ErrorCode,
)


class _ToolModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.calls: list[list[object]] = []
        self.bindings: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def bind_tools(self, tools, **kwargs):
        self.bindings.append((tools, kwargs))
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.response


class _BlockingModel(_ToolModel):
    def __init__(self) -> None:
        super().__init__(AIMessage(content="never"))

    async def ainvoke(self, messages):
        await asyncio.sleep(10)


def _tool(name: str = "search_knowledge_base") -> ChatToolDefinition:
    return ChatToolDefinition(
        name=name,
        description="Execute one bounded operation.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def _adapter(model, *, timeout_seconds: float = 1.0):
    return LangChainChatModelAdapter(
        base_url="https://provider.invalid/v1",
        api_key="secret",
        model="configured-model",
        timeout_seconds=timeout_seconds,
        max_retries=0,
        max_concurrency=1,
        chat_model=model,
    )


class LangChainChatAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_tool_call_preserves_usage_and_metadata(self) -> None:
        model = _ToolModel(
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "call-1",
                    "name": "search_knowledge_base",
                    "args": {"query": "bounded query"},
                    "type": "tool_call",
                }],
                response_metadata={
                    "model_name": "resolved-model",
                    "finish_reason": "tool_calls",
                    "headers": {"x-request-id": "request-1"},
                },
                usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )
        )
        response = await _adapter(model).complete(
            ChatModelRequest(
                messages=(ChatModelMessage("user", "search"),),
                tools=(_tool(),),
                tool_choice="required",
            )
        )
        self.assertEqual(response.tool_calls[0].arguments["query"], "bounded query")
        self.assertEqual(response.provider_request_id, "request-1")
        self.assertEqual(response.usage["total_tokens"], 5)
        self.assertEqual(model.bindings[0][1]["parallel_tool_calls"], False)
        json.dumps(model.bindings[0][0])

    async def test_tool_result_continuation_and_forced_submit_preserve_order(self) -> None:
        model = _ToolModel(
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "call-2",
                    "name": "submit_answer",
                    "args": {"outcome": "refused", "claims": [], "unanswered": []},
                    "type": "tool_call",
                }],
                response_metadata={"model_name": "resolved-model"},
            )
        )
        prior = ChatToolCall("call-1", "search_knowledge_base", {"query": "q"})
        await _adapter(model).complete(
            ChatModelRequest(
                messages=(
                    ChatModelMessage("user", "question"),
                    ChatModelMessage("assistant", "", tool_calls=(prior,)),
                    ChatModelMessage("tool", '{"evidence_refs":[]}', tool_call_id="call-1"),
                ),
                tools=(_tool(), _tool("submit_answer")),
                tool_choice="submit_answer",
            )
        )
        self.assertIsInstance(model.calls[0][0], HumanMessage)
        self.assertIsInstance(model.calls[0][1], AIMessage)
        self.assertIsInstance(model.calls[0][2], ToolMessage)
        self.assertEqual(model.bindings[0][1]["tool_choice"]["function"]["name"], "submit_answer")

    async def test_nested_frozen_tool_arguments_are_counted(self) -> None:
        model = _ToolModel(
            AIMessage(content="ok", response_metadata={"model_name": "resolved-model"})
        )
        prior = ChatToolCall(
            "call-nested",
            "submit_answer",
            {
                "outcome": "partial",
                "claims": [{"text": "claim", "citation_ids": ["cite_1"]}],
                "unanswered": ["remaining"],
            },
        )
        await _adapter(model).complete(
            ChatModelRequest(
                messages=(
                    ChatModelMessage("user", "question"),
                    ChatModelMessage("assistant", "", tool_calls=(prior,)),
                ),
            )
        )
        self.assertIsInstance(model.calls[0][1], AIMessage)

    async def test_visual_evidence_is_sent_after_tool_result(self) -> None:
        model = _ToolModel(AIMessage(content="ok", response_metadata={"model_name": "resolved-model"}))
        await _adapter(model).complete(
            ChatModelRequest(
                messages=(
                    ChatModelMessage("user", "inspect"),
                    ChatModelMessage("tool", "{}", tool_call_id="call-1"),
                    ChatModelMessage(
                        "evidence",
                        "EvidenceRef E1",
                        visual_content=(ChatModelVisualContent(
                            citation_ids=("cite_1",),
                            asset_id=uuid4(),
                            media_type="image/png",
                            checksum_sha256=hashlib.sha256(b"png").hexdigest(),
                            content=b"png",
                            width=1,
                            height=1,
                        ),),
                    ),
                ),
            )
        )
        self.assertIsInstance(model.calls[0][1], ToolMessage)
        self.assertIsInstance(model.calls[0][2], HumanMessage)
        self.assertIsInstance(model.calls[0][2].content, list)

    def test_parallel_tool_calls_are_rejected_by_domain_contract(self) -> None:
        with self.assertRaises(ValueError):
            ChatModelRequest(
                messages=(ChatModelMessage("user", "search"),),
                tools=(_tool(),),
                parallel_tool_calls=True,
            )

    async def test_total_timeout_is_content_safe(self) -> None:
        with self.assertRaises(ChatModelExecutionError) as raised:
            await _adapter(_BlockingModel(), timeout_seconds=0.001).complete(
                ChatModelRequest(messages=(ChatModelMessage("user", "hello"),))
            )
        self.assertEqual(raised.exception.code, ErrorCode.CHAT_PROVIDER_UNAVAILABLE)
        self.assertEqual(raised.exception.diagnostic, {"check": "total_timeout"})


if __name__ == "__main__":
    unittest.main()
