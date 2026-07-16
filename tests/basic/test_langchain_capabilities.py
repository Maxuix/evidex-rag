from __future__ import annotations

import json
import unittest

import httpx
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph


class _StructuredProbe(BaseModel):
    answer: str


class LangChainCapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_openai_public_async_and_metadata_capabilities(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            return httpx.Response(
                200,
                headers={"x-request-id": "request-probe"},
                json={
                    "id": "completion-probe",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "resolved-probe-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )

        async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            model = ChatOpenAI(
                model="configured-probe-model",
                api_key="probe-key",
                base_url="https://provider.invalid/v1",
                timeout=1.0,
                max_retries=0,
                include_response_headers=True,
                use_responses_api=False,
                http_async_client=async_client,
            )
            response = await model.ainvoke([("human", "probe")])

            self.assertEqual(response.content, '{"answer":"ok"}')
            self.assertEqual(response.response_metadata["model_name"], "resolved-probe-model")
            self.assertEqual(response.response_metadata["finish_reason"], "stop")
            self.assertEqual(response.response_metadata["headers"]["x-request-id"], "request-probe")
            self.assertEqual(response.usage_metadata["input_tokens"], 3)
            self.assertEqual(response.usage_metadata["output_tokens"], 2)
            self.assertEqual(response.usage_metadata["total_tokens"], 5)

            structured = model.with_structured_output(
                _StructuredProbe,
                method="json_schema",
                include_raw=True,
            )
            self.assertTrue(callable(structured.ainvoke))
        finally:
            await async_client.aclose()

    async def test_openai_embeddings_public_async_and_dimensions_capabilities(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/embeddings")
            payload = json.loads(request.content)
            self.assertEqual(payload["dimensions"], 1024)
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
                timeout=1.0,
                max_retries=0,
                check_embedding_ctx_length=False,
                http_async_client=async_client,
            )

            documents = await embeddings.aembed_documents(["first", "second"])
            query = await embeddings.aembed_query("query")
            self.assertEqual(len(documents), 2)
            self.assertTrue(all(len(vector) == 1024 for vector in documents))
            self.assertEqual(len(query), 1024)
        finally:
            await async_client.aclose()

    def test_langgraph_state_graph_compiles_without_checkpointing(self) -> None:
        graph = StateGraph(dict)
        graph.add_node("probe", lambda state: {**state, "ok": True})
        graph.add_edge(START, "probe")
        graph.add_edge("probe", END)

        compiled = graph.compile(checkpointer=None)

        self.assertEqual(compiled.invoke({"ok": False}), {"ok": True})


if __name__ == "__main__":
    unittest.main()
