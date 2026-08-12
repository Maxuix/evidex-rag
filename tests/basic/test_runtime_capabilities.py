from __future__ import annotations

import json
import unittest
from importlib.metadata import version
from io import BytesIO

import httpx
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RuntimeCapabilityTests(unittest.IsolatedAsyncioTestCase):
    def test_docling_public_local_conversion_capability(self) -> None:
        self.assertEqual(version("docling"), "2.114.0")
        self.assertEqual(version("docling-core"), "2.87.1")
        result = DocumentConverter(allowed_formats=[InputFormat.MD]).convert(
            DocumentStream(
                name="guide.md",
                stream=BytesIO(b"# Overview\n\nLocal parsing evidence.\n\n| K | V |\n| - | - |\n| a | 1 |"),
            ),
            raises_on_error=False,
        )
        self.assertIs(result.status, ConversionStatus.SUCCESS)
        self.assertIn(
            DocItemLabel.TABLE,
            {item.label for item, _level in result.document.iterate_items()},
        )

    async def test_chat_openai_native_tool_call_capability(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["parallel_tool_calls"], False)
            self.assertEqual(payload["tool_choice"], "required")
            return httpx.Response(
                200,
                headers={"x-request-id": "request-probe"},
                json={
                    "id": "completion-probe",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "resolved-probe-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "probe", "arguments": "{}"},
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )

        async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            model = ChatOpenAI(
                model="configured-probe-model",
                api_key="probe-key",
                base_url="https://provider.invalid/v1",
                max_retries=0,
                include_response_headers=True,
                use_responses_api=False,
                http_async_client=async_client,
            ).bind_tools(
                [{"name": "probe", "description": "probe", "parameters": {"type": "object", "properties": {}}}],
                tool_choice="required",
                parallel_tool_calls=False,
            )
            response = await model.ainvoke([("human", "probe")])
            self.assertEqual(response.tool_calls[0]["name"], "probe")
            self.assertEqual(response.usage_metadata["total_tokens"], 5)
        finally:
            await async_client.aclose()

    async def test_openai_embeddings_public_async_and_dimensions_capabilities(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            inputs = payload["input"]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "resolved-embedding-model",
                    "data": [
                        {"object": "embedding", "index": index, "embedding": [1.0] * 1024}
                        for index, _ in enumerate(inputs)
                    ],
                    "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
                },
            )

        async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            embeddings = OpenAIEmbeddings(
                model="configured-embedding-model",
                dimensions=1024,
                api_key="probe-key",
                base_url="https://provider.invalid/v1",
                max_retries=0,
                check_embedding_ctx_length=False,
                http_async_client=async_client,
            )
            self.assertEqual(len(await embeddings.aembed_query("query")), 1024)
        finally:
            await async_client.aclose()


if __name__ == "__main__":
    unittest.main()
