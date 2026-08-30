from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest

from pydantic import BaseModel, Field

from rag_kb.adapters.graphiti.client import (
    GraphitiSchemaEchoError,
    SchemaEchoRepairingLLMClient,
    as_graphiti_llm_client,
    is_invalid_model_payload,
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


class _OrganizationAttributes(BaseModel):
    short_names: list[str] = Field(default_factory=list)


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

    def test_optional_attribute_schema_document_is_detected(self) -> None:
        self.assertTrue(
            is_schema_echo_payload(
                {
                    "description": "An organization",
                    "properties": {
                        "short_names": {
                            "items": {"type": "string"},
                            "title": "Short Names",
                            "type": "array",
                        }
                    },
                    "title": "OrganizationEntity",
                    "type": "object",
                },
                _OrganizationAttributes,
            )
        )

    def test_valid_optional_attribute_payload_is_not_echo(self) -> None:
        self.assertFalse(
            is_schema_echo_payload(
                {"short_names": ["ASF"]},
                _OrganizationAttributes,
            )
        )

    def test_field_level_schema_fragment_is_invalid(self) -> None:
        self.assertTrue(
            is_invalid_model_payload(
                {
                    "short_names": {
                        "items": {"type": "string"},
                        "title": "Short Names",
                        "type": "array",
                    }
                },
                _OrganizationAttributes,
            )
        )

    def test_attribute_payload_with_schema_metadata_is_strictly_invalid(self) -> None:
        payload = {
            "short_names": [],
            "description": None,
            "properties": {"short_names": []},
            "title": "ProjectEntity",
            "type": "object",
        }
        self.assertFalse(
            is_invalid_model_payload(payload, _OrganizationAttributes)
        )
        self.assertTrue(
            is_invalid_model_payload(
                payload,
                _OrganizationAttributes,
                strict_fields=True,
            )
        )

    def test_missing_model_skips_detection(self) -> None:
        self.assertFalse(is_schema_echo_payload({"type": "object"}, None))
        self.assertFalse(is_schema_echo_payload({"note": None}, _OptionalOnly))


class SchemaEchoRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_semaphore_limits_actual_llm_calls(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        class _BlockingLLM(_FakeLLM):
            def __init__(self) -> None:
                super().__init__([])
                self.active = 0
                self.max_active = 0

            async def generate_response(
                self, messages: object, **kwargs: object
            ) -> object:
                del messages, kwargs
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                entered.set()
                try:
                    await release.wait()
                    return {"extracted_entities": []}
                finally:
                    self.active -= 1

        inner = _BlockingLLM()
        client = SchemaEchoRepairingLLMClient(
            inner,
            provider_semaphore=asyncio.Semaphore(1),
        )
        first = asyncio.create_task(
            client.generate_response(
                [SimpleNamespace(role="user", content="first")],
                response_model=_ExtractedEntities,
            )
        )
        second = asyncio.create_task(
            client.generate_response(
                [SimpleNamespace(role="user", content="second")],
                response_model=_ExtractedEntities,
            )
        )
        await entered.wait()
        await asyncio.sleep(0)
        self.assertEqual(inner.max_active, 1)
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual(inner.max_active, 1)

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

    async def test_optional_attribute_schema_echo_is_repaired(self) -> None:
        inner = _FakeLLM(
            [
                {
                    "description": "An organization",
                    "properties": {
                        "short_names": {
                            "items": {"type": "string"},
                            "title": "Short Names",
                            "type": "array",
                        }
                    },
                    "title": "OrganizationEntity",
                    "type": "object",
                },
                {"short_names": ["ASF"]},
            ]
        )
        client = SchemaEchoRepairingLLMClient(inner)

        result = await client.generate_response(
            [SimpleNamespace(role="user", content="extract attributes")],
            response_model=_OrganizationAttributes,
            attribute_extraction=True,
        )

        self.assertEqual(result, {"short_names": ["ASF"]})
        self.assertEqual(len(inner.raw_calls), 1)
        self.assertIn("short_names", inner.raw_calls[0][0][-1].content)

    async def test_field_level_schema_fragment_is_repaired(self) -> None:
        inner = _FakeLLM(
            [
                {
                    "short_names": {
                        "items": {"type": "string"},
                        "title": "Short Names",
                        "type": "array",
                    }
                },
                {"short_names": ["PSF"]},
            ]
        )
        client = SchemaEchoRepairingLLMClient(inner)

        result = await client.generate_response(
            [SimpleNamespace(role="user", content="extract attributes")],
            response_model=_OrganizationAttributes,
            attribute_extraction=True,
        )

        self.assertEqual(result, {"short_names": ["PSF"]})
        self.assertEqual(len(inner.raw_calls), 1)

    async def test_attribute_schema_metadata_contamination_is_repaired(self) -> None:
        inner = _FakeLLM(
            [
                {
                    "short_names": [],
                    "description": None,
                    "properties": {"short_names": []},
                    "title": "OrganizationEntity",
                    "type": "object",
                },
                {"short_names": ["ASF"]},
            ]
        )
        client = SchemaEchoRepairingLLMClient(inner)

        result = await client.generate_response(
            [SimpleNamespace(role="user", content="extract attributes")],
            response_model=_OrganizationAttributes,
            attribute_extraction=True,
        )

        self.assertEqual(result, {"short_names": ["ASF"]})
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

    async def test_default_budget_repairs_four_consecutive_echoes(self) -> None:
        echo = {
            "properties": {"short_names": {"type": "array"}},
            "title": "OrganizationEntity",
            "type": "object",
        }
        inner = _FakeLLM([echo, echo, echo, echo, {"short_names": ["ASF"]}])
        client = SchemaEchoRepairingLLMClient(inner)

        result = await client.generate_response(
            [SimpleNamespace(role="user", content="extract attributes")],
            response_model=_OrganizationAttributes,
            attribute_extraction=True,
        )

        self.assertEqual(result, {"short_names": ["ASF"]})
        self.assertEqual(len(inner.generate_calls), 1)
        self.assertEqual(len(inner.raw_calls), 4)

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
