from __future__ import annotations

from types import SimpleNamespace
import unittest

from pydantic import BaseModel, Field

from rag_kb.adapters.graphiti.client import (
    GraphitiSchemaEchoError,
    SchemaEchoRepairingLLMClient,
    as_graphiti_llm_client,
    is_schema_echo_payload,
    required_model_field_names,
)


class _ExtractedEntities(BaseModel):
    extracted_entities: list[str]


class _EdgeDuplicate(BaseModel):
    duplicate_facts: list[int]
    contradicted_facts: list[int] = Field(default_factory=list)


class _OptionalOnly(BaseModel):
    note: str | None = None


class _FakeLLM:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.generate_calls: list[tuple[object, dict[str, object]]] = []
        self.raw_calls: list[tuple[object, object]] = []
        self.tracer = None
        self.token_tracker = "tracker"

    def set_tracer(self, tracer: object) -> None:
        self.tracer = tracer

    async def generate_response(self, messages: object, **kwargs: object) -> object:
        self.generate_calls.append((messages, kwargs))
        last = messages[-1]
        last.content += "\n\nSCHEMA"
        return self.responses.pop(0)

    async def _generate_response(
        self,
        messages: object,
        response_model: object = None,
        **kwargs: object,
    ) -> object:
        self.raw_calls.append((messages, response_model))
        return self.responses.pop(0)


class SchemaEchoHelperTests(unittest.TestCase):
    def test_required_fields_ignore_defaults(self) -> None:
        self.assertEqual(
            required_model_field_names(_ExtractedEntities),
            ("extracted_entities",),
        )
        self.assertEqual(
            required_model_field_names(_EdgeDuplicate),
            ("duplicate_facts",),
        )

    def test_schema_document_is_detected(self) -> None:
        self.assertTrue(
            is_schema_echo_payload(
                {
                    "$defs": {},
                    "properties": {},
                    "required": ["extracted_entities"],
                    "title": "ExtractedEntities",
                    "type": "object",
                },
                _ExtractedEntities,
            )
        )

    def test_valid_payload_is_not_echo(self) -> None:
        self.assertFalse(
            is_schema_echo_payload(
                {"extracted_entities": []},
                _ExtractedEntities,
            )
        )

    def test_missing_model_skips_detection(self) -> None:
        self.assertFalse(is_schema_echo_payload({"type": "object"}, None))
        self.assertFalse(is_schema_echo_payload({"note": None}, _OptionalOnly))


class SchemaEchoRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_first_response_does_not_repair(self) -> None:
        inner = _FakeLLM([{"extracted_entities": ["A"]}])
        client = SchemaEchoRepairingLLMClient(inner)
        messages = [SimpleNamespace(role="user", content="extract")]

        result = await client.generate_response(
            messages,
            response_model=_ExtractedEntities,
            prompt_name="extract_nodes.extract_text",
        )

        self.assertEqual(result, {"extracted_entities": ["A"]})
        self.assertEqual(len(inner.generate_calls), 1)
        self.assertEqual(inner.raw_calls, [])
        self.assertIn("SCHEMA", messages[-1].content)

    async def test_schema_echo_is_repaired_without_reappending_schema(self) -> None:
        inner = _FakeLLM(
            [
                {
                    "$defs": {},
                    "properties": {},
                    "required": ["extracted_entities"],
                    "title": "ExtractedEntities",
                    "type": "object",
                },
                {"extracted_entities": ["B"]},
            ]
        )
        client = SchemaEchoRepairingLLMClient(inner)
        messages = [SimpleNamespace(role="user", content="extract")]

        result = await client.generate_response(
            messages,
            response_model=_ExtractedEntities,
        )

        self.assertEqual(result, {"extracted_entities": ["B"]})
        self.assertEqual(len(inner.generate_calls), 1)
        self.assertEqual(len(inner.raw_calls), 1)
        repaired, model = inner.raw_calls[0]
        self.assertIs(model, _ExtractedEntities)
        self.assertIn("SCHEMA", repaired[-1].content)
        self.assertIn("Return ONLY a JSON object", repaired[-1].content)
        self.assertIn("extracted_entities", repaired[-1].content)
        self.assertNotIn("Return ONLY a JSON object", messages[-1].content)
        self.assertEqual(messages[-1].content.count("SCHEMA"), 1)

    async def test_wrong_json_schema_keys_are_repaired(self) -> None:
        inner = _FakeLLM(
            [
                {"entities": [{"name": "A"}]},
                {"extracted_entities": ["A"]},
            ]
        )
        client = SchemaEchoRepairingLLMClient(inner)

        result = await client.generate_response(
            [SimpleNamespace(role="user", content="extract")],
            response_model=_ExtractedEntities,
        )

        self.assertEqual(result, {"extracted_entities": ["A"]})
        self.assertEqual(len(inner.raw_calls), 1)

    async def test_exhausted_repairs_fail_closed(self) -> None:
        echo = {"$defs": {}, "properties": {}, "type": "object"}
        inner = _FakeLLM([echo, echo, echo])
        client = SchemaEchoRepairingLLMClient(inner, max_attempts=3)

        with self.assertRaises(GraphitiSchemaEchoError):
            await client.generate_response(
                [SimpleNamespace(role="user", content="extract")],
                response_model=_ExtractedEntities,
            )

        self.assertEqual(len(inner.generate_calls), 1)
        self.assertEqual(len(inner.raw_calls), 2)

    async def test_provider_errors_are_not_repaired(self) -> None:
        class _Broken(_FakeLLM):
            async def generate_response(self, messages: object, **kwargs: object) -> object:
                raise RuntimeError("provider unavailable")

        inner = _Broken([])
        client = SchemaEchoRepairingLLMClient(inner)

        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            await client.generate_response(
                [SimpleNamespace(role="user", content="extract")],
                response_model=_ExtractedEntities,
            )
        self.assertEqual(inner.raw_calls, [])

    async def test_set_tracer_and_token_tracker_delegate(self) -> None:
        inner = _FakeLLM([{"extracted_entities": []}])
        client = SchemaEchoRepairingLLMClient(inner)
        client.set_tracer("trace")
        self.assertEqual(inner.tracer, "trace")
        self.assertEqual(client.token_tracker, "tracker")

    async def test_wrapper_preserves_inner_llm_type(self) -> None:
        class _LLMBase:
            pass

        class _TypedLLM(_LLMBase, _FakeLLM):
            pass

        inner = _TypedLLM([{"extracted_entities": ["C"]}])
        client = as_graphiti_llm_client(inner)
        self.assertIsInstance(client, _LLMBase)
        self.assertIsInstance(client, _TypedLLM)
        result = await client.generate_response(
            [SimpleNamespace(role="user", content="extract")],
            response_model=_ExtractedEntities,
        )
        self.assertEqual(result, {"extracted_entities": ["C"]})
        client.set_tracer("trace")
        self.assertEqual(inner.tracer, "trace")
        self.assertEqual(client.token_tracker, "tracker")
