from __future__ import annotations

import asyncio
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from apps.api.dependencies import build_api_dependencies
from apps.worker.dependencies import build_worker_dependencies
from rag_kb.adapters import (
    LangChainChatModelAdapter,
    LangChainEmbeddingModelAdapter,
)
from rag_kb.config import StartupConfigurationError, validate_startup_environment
from rag_kb.config.settings import Settings, load_settings
from rag_kb.db import DatabaseProcess
from rag_kb.domain import WorkLane
from rag_kb.workflows import LangGraphRunner


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def valid_payload(root: Path) -> dict[str, object]:
    return {
        "database": {
            "runtime_dsn": (
                "postgresql+asyncpg://rag_kb_runtime:runtime-secret@localhost/rag_kb"
            ),
            "migration_dsn": (
                "postgresql+asyncpg://rag_kb_migration:migration-secret@localhost/rag_kb"
            ),
        },
        "file_store": {
            "root_path": root,
            "staging_path": root / "staging",
            "final_path": root / "final",
        },
        "model_provider": {
            "chat": {
                "base_url": "https://chat.example.invalid/v1",
                "api_key": "chat-secret",
            },
            "embedding": {
                "base_url": "https://embedding.example.invalid/v1",
                "api_key": "embedding-secret",
            },
        },
    }


def build_settings(root: Path, **overrides: object) -> Settings:
    payload = valid_payload(root)
    payload.update(overrides)
    return Settings(_env_file=None, **payload)  # type: ignore[arg-type]


class SettingsTests(unittest.TestCase):
    def test_valid_settings_use_frozen_p1a_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = build_settings(Path(directory))

        self.assertEqual(settings.app.deployment_profile.value, "development")
        self.assertEqual(str(settings.app.bind_host), "127.0.0.1")
        self.assertEqual(settings.database.configured_pool_capacity, 22)
        self.assertEqual(settings.database.application_connection_budget, 40)
        self.assertEqual(settings.model_provider.embedding.dimension, 1024)
        self.assertEqual(settings.model_provider.embedding.metric, "cosine")
        self.assertTrue(settings.vector_store.exact_search)
        self.assertFalse(settings.vector_store.hnsw_enabled)
        self.assertEqual(settings.retrieval.min_cosine_similarity, 0.35)
        self.assertEqual(settings.retrieval.min_rerank_score, 0.45)
        self.assertTrue(settings.retrieval.rerank_enabled)
        self.assertEqual(settings.identity.provider, "development_fixed")
        self.assertEqual(settings.identity.workspace_id.version, 7)
        self.assertEqual(
            settings.security.allowed_cors_origins,
            ("http://127.0.0.1:3000",),
        )
        self.assertFalse(settings.security.cors_allow_credentials)
        self.assertEqual(settings.file_admission.max_bytes, 10 * 1024 * 1024)
        self.assertEqual(settings.file_admission.max_lines, 200_000)
        self.assertEqual(settings.file_admission.max_archive_entries, 10_000)
        self.assertEqual(
            settings.file_admission.max_expanded_bytes,
            100 * 1024 * 1024,
        )
        self.assertEqual(settings.parser.profile, "unstructured_local_v1")
        self.assertEqual(settings.parser.max_chunks, 20_000)
        self.assertEqual(settings.parser.max_extracted_characters, 5_000_000)
        self.assertEqual(settings.parser.max_metadata_bytes, 65_536)
        self.assertEqual(settings.job_poller.required_worker_connections, 7)
        self.assertEqual(settings.job_poller.indexing_deadline_seconds, 900)
        self.assertEqual(settings.job_poller.chat_deadline_seconds, 120)
        self.assertEqual(settings.job_poller.chat_start_target_seconds, 2.0)
        self.assertEqual(settings.database.required_api_connections, 3)
        self.assertEqual(settings.chat_delivery.poll_interval_seconds, 1.0)
        self.assertEqual(settings.chat_delivery.jitter_ratio, 0.2)
        self.assertEqual(
            settings.chat_delivery.max_connection_duration_seconds,
            600.0,
        )
        self.assertEqual(
            settings.chat_delivery.max_connections_per_principal_run,
            2,
        )
        self.assertEqual(settings.maintenance.batch_size, 100)
        self.assertEqual(settings.maintenance.task_retention_seconds, 604_800)
        self.assertEqual(settings.model_provider.chat.temperature, 0.1)
        self.assertEqual(settings.model_provider.chat.max_tokens, 2048)
        self.assertEqual(settings.session_context.max_turns, 6)
        self.assertEqual(settings.session_context.max_context_tokens, 4000)
        self.assertEqual(settings.session_context.tokenizer, "cl100k_base")
        self.assertEqual(
            settings.session_context.query_schema, "contextual_query_v2"
        )

    def test_session_context_policy_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for override in (
                {"max_turns": 7},
                {"max_context_tokens": 5000},
                {"tokenizer": "provider_tokenizer"},
                {"strategy": "summarized"},
                {"query_schema": "contextual_query_v1"},
            ):
                with self.subTest(override=override), self.assertRaises(
                    ValidationError
                ):
                    build_settings(root, session_context=override)

    def test_maintenance_retention_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValidationError):
                build_settings(
                    Path(directory),
                    maintenance={
                        "retired_data_grace_seconds": 300,
                        "task_retention_seconds": 300,
                    },
                )

    def test_chat_delivery_bounds_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for chat_delivery in (
                {"jitter_ratio": 0.6},
                {"max_connection_duration_seconds": 0},
                {"max_connections_per_principal_run": 0},
            ):
                with self.subTest(chat_delivery=chat_delivery):
                    with self.assertRaises(ValidationError):
                        build_settings(root, chat_delivery=chat_delivery)

    def test_worker_lane_heartbeat_and_deadline_budget_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = valid_payload(root)
            with self.assertRaises(ValidationError):
                Settings(
                    _env_file=None,
                    **{
                        **payload,
                        "job_poller": {"indexing_concurrency": 10},
                    },
                )
            with self.assertRaises(ValidationError):
                Settings(
                    _env_file=None,
                    **{
                        **payload,
                        "job_poller": {"stale_after_seconds": 20},
                    },
                )
            with self.assertRaises(ValidationError):
                Settings(
                    _env_file=None,
                    **{
                        **payload,
                        "job_poller": {
                            "poll_interval_seconds": 2,
                            "chat_start_target_seconds": 1,
                        },
                    },
                )
            with self.assertRaises(ValidationError):
                Settings(
                    _env_file=None,
                    **{
                        **payload,
                        "job_poller": {
                            "retry_base_delay_seconds": 10,
                            "retry_max_delay_seconds": 5,
                        },
                    },
                )
            with self.assertRaises(ValidationError):
                Settings(
                    _env_file=None,
                    **{
                        **payload,
                        "job_poller": {"chat_deadline_seconds": 10},
                    },
                )

    def test_nested_environment_surface_loads_without_global_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "RAG_KB__DATABASE__RUNTIME_DSN": (
                    "postgresql+asyncpg://rag_kb_runtime:runtime-secret@localhost/rag_kb"
                ),
                "RAG_KB__DATABASE__MIGRATION_DSN": (
                    "postgresql+asyncpg://rag_kb_migration:migration-secret@localhost/rag_kb"
                ),
                "RAG_KB__FILE_STORE__ROOT_PATH": str(root),
                "RAG_KB__FILE_STORE__STAGING_PATH": str(root / "staging"),
                "RAG_KB__FILE_STORE__FINAL_PATH": str(root / "final"),
                "RAG_KB__MODEL_PROVIDER__CHAT__BASE_URL": (
                    "https://chat.example.invalid/v1"
                ),
                "RAG_KB__MODEL_PROVIDER__CHAT__API_KEY": "chat-secret",
                "RAG_KB__MODEL_PROVIDER__EMBEDDING__BASE_URL": (
                    "https://embedding.example.invalid/v1"
                ),
                "RAG_KB__MODEL_PROVIDER__EMBEDDING__API_KEY": "embedding-secret",
            }
            with patch.dict(os.environ, environment, clear=True):
                settings = load_settings(env_file=None)

        self.assertEqual(settings.file_store.root_path, root)
        self.assertEqual(settings.model_provider.chat.model, "deepseek-v4-flash")

    def test_removed_ai_implementation_switches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for override in (
                {"model_adapter_backend": "langchain"},
                {"chat_workflow_backend": "langgraph"},
                {"workflow": {"runner": "direct"}},
            ):
                with self.subTest(override=override), self.assertRaises(
                    ValidationError
                ):
                    build_settings(root, **override)

    def test_checked_in_environment_example_parses(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings(env_file=PROJECT_ROOT / ".env.example")

        self.assertEqual(settings.database.runtime_role, "rag_kb_runtime")
        self.assertEqual(settings.model_provider.chat.model, "deepseek-v4-flash")
        assert settings.model_provider.multimodal_embedding is not None
        self.assertEqual(
            settings.model_provider.multimodal_embedding.model,
            "tongyi-embedding-vision-flash-2026-03-06",
        )
        self.assertEqual(settings.model_provider.multimodal_embedding.dimension, 768)

    def test_non_development_and_non_loopback_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for app in (
                {"deployment_profile": "department"},
                {"bind_host": "0.0.0.0"},
            ):
                with self.subTest(app=app), self.assertRaises(ValidationError):
                    build_settings(root, app=app)

    def test_identity_and_cors_settings_fail_closed(self) -> None:
        invalid_overrides = (
            {"identity": {"workspace_id": "550e8400-e29b-41d4-a716-446655440000"}},
            {"identity": {"principal_id": ""}},
            {"identity": {"client_id": "client selected"}},
            {"security": {"allowed_cors_origins": []}},
            {"security": {"allowed_cors_origins": ["*"]}},
            {"security": {"allowed_cors_origins": ["https://example.com"]}},
            {"security": {"allowed_cors_origins": ["http://localhost/"]}},
            {"security": {"allowed_cors_origins": ["http://localhost/a"]}},
            {"security": {"cors_allow_credentials": True}},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for override in invalid_overrides:
                with self.subTest(override=override), self.assertRaises(
                    ValidationError
                ):
                    build_settings(root, **override)

    def test_database_roles_and_pool_budget_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = valid_payload(root)
            database = copy.deepcopy(payload["database"])
            assert isinstance(database, dict)
            database["runtime_role"] = "same_role"
            database["migration_role"] = "same_role"
            with self.assertRaises(ValidationError):
                Settings(_env_file=None, **{**payload, "database": database})

            database = copy.deepcopy(payload["database"])
            assert isinstance(database, dict)
            database["api_pool_size"] = 30
            database["worker_pool_size"] = 30
            with self.assertRaises(ValidationError):
                Settings(_env_file=None, **{**payload, "database": database})

            database = copy.deepcopy(payload["database"])
            assert isinstance(database, dict)
            database["api_pool_size"] = 1
            database["api_max_overflow"] = 0
            with self.assertRaises(ValidationError):
                Settings(_env_file=None, **{**payload, "database": database})

    def test_fixed_embedding_space_rejects_in_place_changes(self) -> None:
        changes = {
            "dimension": 1536,
            "model": "another-model",
            "metric": "l2",
            "configuration_fingerprint": "sha256:changed",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value in changes.items():
                payload = valid_payload(root)
                providers = copy.deepcopy(payload["model_provider"])
                assert isinstance(providers, dict)
                embedding = providers["embedding"]
                assert isinstance(embedding, dict)
                embedding[field] = value
                with self.subTest(field=field), self.assertRaises(ValidationError):
                    Settings(
                        _env_file=None,
                        **{**payload, "model_provider": providers},
                    )

    def test_p1b_capabilities_cannot_be_enabled(self) -> None:
        flags = (
            "second_queue_enabled",
            "retained_event_replay_enabled",
            "multi_runner_recovery_enabled",
            "outbox_delivery_enabled",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for flag in flags:
                with self.subTest(flag=flag), self.assertRaises(ValidationError):
                    build_settings(root, delivery_reliability={flag: True})

    def test_later_retrieval_features_cannot_be_enabled(self) -> None:
        overrides = (
            {"vector_store": {"hnsw_enabled": True}},
            {"retrieval": {"hybrid_enabled": True}},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for override in overrides:
                with self.subTest(override=override), self.assertRaises(
                    ValidationError
                ):
                    build_settings(root, **override)

    def test_reranking_can_be_disabled_as_a_runtime_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = build_settings(
                Path(directory), retrieval={"rerank_enabled": False}
            )
        self.assertFalse(settings.retrieval.rerank_enabled)

    def test_rerank_weights_must_sum_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValidationError):
                build_settings(
                    Path(directory),
                    retrieval={"vector_weight": 0.7, "lexical_weight": 0.35},
                )

    def test_cosine_evidence_threshold_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in (-1.01, 1.01):
                with self.subTest(value=value), self.assertRaises(ValidationError):
                    build_settings(
                        root,
                        retrieval={"min_cosine_similarity": value},
                    )

    def test_file_store_paths_must_share_one_logical_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValidationError):
                build_settings(
                    root,
                    file_store={
                        "root_path": root,
                        "staging_path": root / "staging",
                        "final_path": root.parent / "outside-final",
                    },
                )

    def test_secrets_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = build_settings(Path(directory))

        rendered = f"{settings!r}\n{settings.model_dump_json()}"
        self.assertNotIn("runtime-secret", rendered)
        self.assertNotIn("migration-secret", rendered)
        self.assertNotIn("chat-secret", rendered)
        self.assertNotIn("embedding-secret", rendered)

    def test_invalid_secret_is_redacted_from_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = valid_payload(root)
            database = copy.deepcopy(payload["database"])
            assert isinstance(database, dict)
            database["runtime_dsn"] = (
                "mysql://rag_kb_runtime:must-not-leak@localhost/rag_kb"
            )

            with self.assertRaises(ValidationError) as raised:
                Settings(_env_file=None, **{**payload, "database": database})

        self.assertNotIn("must-not-leak", str(raised.exception))

    def test_empty_provider_key_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = valid_payload(root)
            providers = copy.deepcopy(payload["model_provider"])
            assert isinstance(providers, dict)
            chat = providers["chat"]
            assert isinstance(chat, dict)
            chat["api_key"] = ""

            with self.assertRaises(ValidationError):
                Settings(
                    _env_file=None,
                    **{**payload, "model_provider": providers},
                )

    def test_chat_generation_limits_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for generation_override in (
                {"temperature": -0.1},
                {"temperature": 2.1},
                {"max_tokens": 0},
            ):
                payload = valid_payload(root)
                providers = copy.deepcopy(payload["model_provider"])
                assert isinstance(providers, dict)
                chat = providers["chat"]
                assert isinstance(chat, dict)
                chat.update(generation_override)
                with self.subTest(generation_override=generation_override):
                    with self.assertRaises(ValidationError):
                        Settings(
                            _env_file=None,
                            **{**payload, "model_provider": providers},
                        )


class StartupValidationTests(unittest.TestCase):
    def test_worker_composes_only_the_active_ai_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "staging").mkdir()
            (root / "final").mkdir()
            worker = build_worker_dependencies(build_settings(root))

            self.assertIsInstance(
                worker.chat_model_adapter,
                LangChainChatModelAdapter,
            )
            self.assertIsInstance(
                worker.embedding_provider,
                LangChainEmbeddingModelAdapter,
            )
            self.assertIsInstance(worker.chat_runner, LangGraphRunner)
            self.assertIs(worker.chat_scheduler._runner, worker.chat_runner)
            self.assertIsNone(worker.chat_runner._graph.checkpointer)
            asyncio.run(worker.close())

    def test_missing_storage_fails_without_provisioning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "missing"
            settings = build_settings(root)

            with self.assertRaises(StartupConfigurationError):
                validate_startup_environment(settings)

            self.assertFalse(root.exists())

    def test_api_and_worker_composition_roots_run_same_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "staging").mkdir()
            (root / "final").mkdir()
            settings = build_settings(root)

            api = build_api_dependencies(settings)
            worker = build_worker_dependencies(settings)

            self.assertIs(api.settings, settings)
            self.assertIs(worker.settings, settings)
            self.assertEqual(
                api.startup.storage_device,
                worker.startup.storage_device,
            )
            self.assertEqual(api.startup.configured_pool_capacity, 22)
            self.assertIs(api.database.process, DatabaseProcess.API)
            self.assertIs(worker.database.process, DatabaseProcess.WORKER)
            self.assertTrue(api.database.engine.dialect.is_async)
            self.assertTrue(worker.database.engine.dialect.is_async)
            self.assertNotIn("runtime-secret", repr(api))
            self.assertNotIn("runtime-secret", repr(worker))
            self.assertIsNot(api.unit_of_work(), api.unit_of_work())
            self.assertIsNot(worker.unit_of_work(), worker.unit_of_work())
            self.assertIs(api.chat_terminal_watcher._chat, api.chat_service)
            self.assertEqual(api.chat_sse_connection_limiter._maximum, 2)
            api_context = api.auth_provider.get_context()
            worker_context = worker.auth_provider.get_context()
            self.assertEqual(api_context, worker_context)
            self.assertEqual(api.unit_of_work().workspace_id, api_context.workspace_id)
            self.assertEqual(
                worker.unit_of_work().workspace_id,
                worker_context.workspace_id,
            )
            self.assertEqual(
                worker.evidence_assessor._min_cosine_similarity,
                settings.retrieval.min_cosine_similarity,
            )
            self.assertIs(
                worker.answer_generator._model,
                worker.chat_model_adapter,
            )
            self.assertIs(
                worker.structure_validator._model,
                worker.chat_model_adapter,
            )
            self.assertIs(
                worker.result_persister._unit_of_work,
                worker.unit_of_work,
            )
            self.assertIs(
                worker.failure_settler._unit_of_work,
                worker.unit_of_work,
            )
            self.assertIs(
                worker.chat_runner._evidence_retriever._retrieval,
                worker.retrieval_service,
            )
            self.assertIs(
                worker.chat_scheduler._runner,
                worker.chat_runner,
            )
            self.assertIsInstance(worker.chat_runner, LangGraphRunner)
            self.assertIsInstance(
                worker.chat_model_adapter,
                LangChainChatModelAdapter,
            )
            self.assertIsInstance(
                api.embedding_provider,
                LangChainEmbeddingModelAdapter,
            )
            self.assertIsInstance(
                worker.embedding_provider,
                LangChainEmbeddingModelAdapter,
            )
            self.assertIs(
                worker.worker_scheduler._schedulers[
                    WorkLane.CHAT
                ],
                worker.chat_scheduler,
            )
            self.assertEqual(
                api.access_policy.metadata_filter(api_context).workspace_id,
                api_context.workspace_id,
            )
            asyncio.run(api.close())
            asyncio.run(worker.close())

        self.assertTrue(api.database._closed)
        self.assertTrue(worker.database._closed)

if __name__ == "__main__":
    unittest.main()
