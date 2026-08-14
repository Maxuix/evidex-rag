from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from unittest.mock import patch
from uuid import UUID

from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    ChatModelExecutionError,
    ChatModelResponse,
    ErrorCode,
    GraphChunkSource,
    GraphChunkResultStatus,
    GraphConfigSnapshot,
    GraphConfigStatus,
    GraphWorkItem,
    GraphWorkKind,
)
from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from apps.api.routers.graph import _response as graph_config_response
from rag_kb.graph.service import (
    GraphConfigurationService,
    GraphConfigView,
    GraphExtractionWorker,
)


WORKSPACE = UUID("01900000-0000-7000-8000-000000000901")
KB_ID = UUID("01900000-0000-7000-8000-000000000902")
BUILD_ID = UUID("01900000-0000-7000-8000-000000000903")
PROFILE_ID = UUID("01900000-0000-7000-8000-000000000904")
CHUNK_ID = UUID("01900000-0000-7000-8000-000000000905")
REVISION_ID = UUID("01900000-0000-7000-8000-000000000906")
TARGET_ID = UUID("01900000-0000-7000-8000-000000000907")
DOCUMENT_ID = UUID("01900000-0000-7000-8000-000000000908")
DOCUMENT_VERSION_ID = UUID("01900000-0000-7000-8000-000000000909")


class GraphBackfillWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_requests_use_eight_k_output_budget(self) -> None:
        repository = _GraphRepository()
        repository.work = GraphWorkItem(
            GraphWorkKind.CHUNK,
            _config(),
            _chunk("No graph facts."),
        )
        model = _FakeChat('{"entities":[],"relations":[]}')

        await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        self.assertEqual(model.requests[0].max_output_tokens, 8192)

    async def test_api_derives_skip_limit_and_rebuild_generation(self) -> None:
        current = graph_config_response(
            GraphConfigView(replace(_config(), eligible_chunk_count=1873))
        )
        stale = graph_config_response(
            GraphConfigView(
                replace(_config(), extractor_version="entity_graph_v1")
            )
        )

        self.assertEqual(current.allowed_skipped_count, 93)
        self.assertFalse(current.requires_rebuild)
        self.assertTrue(stale.requires_rebuild)

    async def test_retry_explicitly_upgrades_to_current_extractor_version(self) -> None:
        repository = _GraphRepository()
        service = GraphConfigurationService(
            _factory(repository), SingleWorkspaceAccessPolicy(WORKSPACE)
        )

        await service.retry(AuthContext("principal", "client", WORKSPACE), KB_ID)

        self.assertEqual(repository.retry_versions, [GRAPH_EXTRACTOR_VERSION])

    async def test_preflight_requires_a_grounded_relation(self) -> None:
        repository = _GraphRepository()
        repository.work = GraphWorkItem(GraphWorkKind.PREFLIGHT, _config())
        model = _FakeChat(
            '{"entities": [{"id":"only","type":"concept","surface":"Atlas",'
            '"disambiguator":null,"disambiguator_support":null}],'
            '"relations": []}'
        )

        await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        self.assertEqual(repository.failed_codes, ["preflight_protocol_invalid"])
        self.assertEqual(repository.preflight_saves, [])

    async def test_protocol_failure_is_repaired_once_then_saved(self) -> None:
        repository = _GraphRepository()
        chunk = _chunk("Atlas released Orion.")
        repository.work = GraphWorkItem(GraphWorkKind.CHUNK, _config(), chunk)
        model = _FakeChat(
            "not-json",
            '{"entities": ['
            '{"id":"atlas","type":"organization","surface":"Atlas",'
            '"disambiguator":null,"disambiguator_support":null},'
            '{"id":"orion","type":"product","surface":"Orion",'
            '"disambiguator":null,"disambiguator_support":null}],'
            '"relations":[{"subject":"atlas","predicate":"released",'
            '"object":"orion","support":"Atlas released Orion"}]}'
        )

        await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        self.assertEqual(len(model.requests), 2)
        self.assertEqual(repository.saved_statuses, [GraphChunkResultStatus.EXTRACTED])
        self.assertEqual(repository.failed_codes, [])
        self.assertEqual(model.requests[1].response_format, {"type": "json_object"})

    async def test_second_protocol_failure_is_not_reported_as_empty(self) -> None:
        repository = _GraphRepository()
        repository.work = GraphWorkItem(GraphWorkKind.CHUNK, _config(), _chunk("A fact."))
        model = _FakeChat("not-json", "still-not-json")

        await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        self.assertEqual(repository.saved_statuses, [GraphChunkResultStatus.SKIPPED_PROTOCOL])
        self.assertEqual(repository.saved_errors, ["json_invalid"])

    async def test_repair_empty_remains_a_protocol_failure(self) -> None:
        repository = _GraphRepository()
        repository.work = GraphWorkItem(GraphWorkKind.CHUNK, _config(), _chunk("A fact."))
        model = _FakeChat("not-json", '{"entities":[],"relations":[]}')

        await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        self.assertEqual(repository.saved_statuses, [GraphChunkResultStatus.SKIPPED_PROTOCOL])
        self.assertEqual(repository.saved_errors, ["repair_empty_after_protocol_error"])

    async def test_output_truncation_is_resource_skip_without_repair(self) -> None:
        repository = _GraphRepository()
        repository.work = GraphWorkItem(GraphWorkKind.CHUNK, _config(), _chunk("A fact."))
        model = _FakeChat(_response("partial", finish_reason="length"))

        await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        self.assertEqual(len(model.requests), 1)
        self.assertEqual(repository.saved_statuses, [GraphChunkResultStatus.SKIPPED_RESOURCE])
        self.assertEqual(repository.saved_errors, ["output_truncated"])

    async def test_adapter_truncation_marker_is_resource_skip_without_repair(self) -> None:
        repository = _GraphRepository()
        repository.work = GraphWorkItem(GraphWorkKind.CHUNK, _config(), _chunk("A fact."))
        model = _FakeChat(_response('{"_response_truncated":true}'))

        await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        self.assertEqual(len(model.requests), 1)
        self.assertEqual(repository.saved_errors, ["output_truncated"])

    async def test_repair_uses_error_rule_without_replaying_provider_response(self) -> None:
        repository = _GraphRepository()
        repository.work = GraphWorkItem(GraphWorkKind.CHUNK, _config(), _chunk("A fact."))
        model = _FakeChat("provider-secret-not-json", "still-not-json")

        await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        repair_prompt = model.requests[1].messages[1].content
        self.assertIn("json_invalid", repair_prompt)
        self.assertIn("json.loads", repair_prompt)
        self.assertNotIn("provider-secret-not-json", repair_prompt)
        self.assertIsNone(model.requests[0].thinking_enabled)

    async def test_each_primary_protocol_error_gets_its_fixed_repair_rule(self) -> None:
        cases = (
            ("not-json", "json_invalid", "json.loads"),
            ('{"entities":{},"relations":[]}', "schema_type", "specified keys"),
            (
                '{"entities":[{"id":"a","type":"organization","surface":"Absent",'
                '"disambiguator":null,"disambiguator_support":null}],"relations":[]}',
                "entity_surface_not_locatable",
                "Copy every surface",
            ),
            (
                '{"entities":[{"id":"a","type":"organization","surface":"Acme",'
                '"disambiguator":null,"disambiguator_support":null}],'
                '"relations":[{"subject":"a","predicate":"owns","object":"missing",'
                '"support":"Acme owns Beta"}]}',
                "relation_endpoint_unknown",
                "entity id defined",
            ),
            (
                '{"entities":[{"id":"a","type":"organization","surface":"Acme",'
                '"disambiguator":null,"disambiguator_support":null},'
                '{"id":"b","type":"organization","surface":"Beta",'
                '"disambiguator":null,"disambiguator_support":null}],'
                '"relations":[{"subject":"a","predicate":"owns","object":"b",'
                '"support":"Acme owns it"}]}',
                "relation_support_missing_object",
                "both referenced entity surfaces",
            ),
            (
                '{"entities":[{"id":"a","type":"organization","surface":"Acme",'
                '"disambiguator":null,"disambiguator_support":null}],'
                '"relations":[{"subject":"a","predicate":"is","object":"a",'
                '"support":"Acme"}]}',
                "self_relation",
                "same entity",
            ),
        )
        for initial, code, rule_fragment in cases:
            with self.subTest(code=code):
                repository = _GraphRepository()
                repository.work = GraphWorkItem(
                    GraphWorkKind.CHUNK, _config(), _chunk("Acme owns it. Beta exists.")
                )
                model = _FakeChat(initial, "still-not-json")
                await GraphExtractionWorker(_factory(repository), model).process_next_work_item()
                repair_prompt = model.requests[1].messages[1].content
                self.assertIn(code, repair_prompt)
                self.assertIn(rule_fragment, repair_prompt)
                self.assertIn("Preserve every other item", repair_prompt)
                self.assertIn('"disambiguator":null', repair_prompt)

    async def test_initial_prompt_contains_complete_contract_and_soft_limits(self) -> None:
        repository = _GraphRepository()
        repository.work = GraphWorkItem(
            GraphWorkKind.CHUNK, _config(), _chunk("No graph facts.")
        )
        model = _FakeChat('{"entities":[],"relations":[]}')

        await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        prompt = model.requests[0].messages[0].content
        for fragment in (
            "letters, numbers, underscore, or hyphen",
            "exactly reference ids",
            "Never return a self relation",
            "explicitly contain both referenced entity surfaces",
            "one continuous exact substring",
            "pronoun, alias, canonical name, paraphrase",
            "at most 16 entities and 8 relations",
            '"disambiguator":null',
        ):
            self.assertIn(fragment, prompt)

    async def test_protocol_logs_only_content_safe_detail_codes_by_phase(self) -> None:
        repository = _GraphRepository()
        repository.work = GraphWorkItem(
            GraphWorkKind.CHUNK, _config(), _chunk("Acme exists.")
        )
        model = _FakeChat(
            '{"entities":[{"id":"a","type":"organization","surface":"secret-a",'
            '"disambiguator":null,"disambiguator_support":null}],"relations":[]}',
            '{"entities":[{"id":"a","type":"organization","surface":"secret-b",'
            '"disambiguator":null,"disambiguator_support":null}],"relations":[]}',
        )

        with patch("rag_kb.graph.service.log_event") as logged:
            await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        protocol_events = [
            call.kwargs
            for call in logged.call_args_list
            if call.args[1] == "graph_extraction_protocol"
        ]
        self.assertEqual(
            [(item["phase"], item["error_code"]) for item in protocol_events],
            [
                ("initial", "entity_surface_not_locatable"),
                ("repair", "entity_surface_not_locatable"),
            ],
        )
        rendered = repr(logged.call_args_list)
        self.assertNotIn("secret-a", rendered)
        self.assertNotIn("secret-b", rendered)

    async def test_provider_failure_fails_the_build_and_does_not_save_a_chunk(self) -> None:
        repository = _GraphRepository()
        repository.work = GraphWorkItem(GraphWorkKind.CHUNK, _config(), _chunk("A fact."))
        model = _FakeChat(
            ChatModelExecutionError(
                ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                diagnostic={"check": "transport"},
            )
        )

        await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        self.assertEqual(repository.failed_codes, [ErrorCode.GRAPH_PROVIDER_UNAVAILABLE.value])
        self.assertEqual(repository.saved_statuses, [])

    async def test_input_resource_limit_is_a_distinct_skipped_result(self) -> None:
        repository = _GraphRepository()
        repository.work = GraphWorkItem(
            GraphWorkKind.CHUNK,
            _config(),
            _chunk("x" * 32_001),
        )
        model = _FakeChat()

        await GraphExtractionWorker(_factory(repository), model).process_next_work_item()

        self.assertEqual(repository.saved_statuses, [GraphChunkResultStatus.SKIPPED_RESOURCE])
        self.assertEqual(repository.saved_errors, ["input_chars"])
        self.assertEqual(model.requests, [])


class _FakeChat:
    def __init__(self, *responses: object) -> None:
        self._responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, ChatModelResponse):
            return response
        return _response(response)


def _response(content: str, *, finish_reason: str = "stop") -> ChatModelResponse:
    return ChatModelResponse(
        content=content,
        model="fake-chat",
        finish_reason=finish_reason,
        provider_request_id=None,
        usage={},
    )


class _GraphRepository:
    def __init__(self) -> None:
        self.work = None
        self.preflight_saves = []
        self.saved_statuses = []
        self.saved_errors = []
        self.failed_codes = []
        self.retry_versions = []

    async def retry(self, kb_id, *, extractor_version, force_rebuild=False):
        del kb_id, force_rebuild
        self.retry_versions.append(extractor_version)
        return _config()

    async def next_work_item(self):
        work, self.work = self.work, None
        return work

    async def save_preflight_success(self, kb_id, *, build_id, extractor_version):
        self.preflight_saves.append((kb_id, build_id, extractor_version))
        return True

    async def save_chunk_extraction(
        self,
        *,
        kb_id,
        build_id,
        index_chunk_id,
        content_hash,
        extractor_version,
        extraction,
    ):
        del kb_id, build_id, index_chunk_id, content_hash, extractor_version
        self.saved_statuses.append(extraction.result_status)
        self.saved_errors.append(extraction.error_code)
        return True

    async def mark_failed(self, kb_id, *, build_id, error_code):
        del kb_id, build_id
        self.failed_codes.append(error_code)
        return True


class _UnitOfWork:
    def __init__(self, repository: _GraphRepository) -> None:
        self.graph = repository
        self.workspace_id = WORKSPACE

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def commit(self) -> None:
        return None


def _factory(repository: _GraphRepository):
    @asynccontextmanager
    async def _unused_context() -> AsyncIterator[_UnitOfWork]:
        yield _UnitOfWork(repository)

    def factory(*, purpose, mode):
        del purpose, mode
        return _unused_context()

    return factory


def _config() -> GraphConfigSnapshot:
    return GraphConfigSnapshot(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        status=GraphConfigStatus.BUILDING,
        build_id=BUILD_ID,
        chat_profile_revision_id=PROFILE_ID,
        extractor_version=GRAPH_EXTRACTOR_VERSION,
        preflight_extractor_version=None,
        last_error_code=None,
        eligible_chunk_count=1,
    )


def _chunk(content: str) -> GraphChunkSource:
    return GraphChunkSource(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        build_id=BUILD_ID,
        index_chunk_id=CHUNK_ID,
        index_revision_id=REVISION_ID,
        indexed_document_version_id=TARGET_ID,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        ordinal=0,
        modality="text",
        content=content,
        content_hash="a" * 64,
        source_location={"paragraph": 1},
        hierarchy={},
        source_metadata={},
    )


if __name__ == "__main__":
    unittest.main()
