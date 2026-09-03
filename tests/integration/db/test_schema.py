from __future__ import annotations

import asyncio
import importlib
import json
import os
import unittest
from uuid import UUID, uuid4

from alembic.migration import MigrationContext
from alembic.operations import Operations
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine

from tests.integration.db import require_database_test_dsns
from rag_kb.db.readiness import EXPECTED_REVISION, check_database_ready


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")

require_database_test_dsns(
    "RAG_KB_TEST_MIGRATION_DSN",
    "RAG_KB_TEST_RUNTIME_DSN",
    "RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN",
)
class DatabaseSchemaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute("TRUNCATE TABLE workspace CASCADE")
        finally:
            await connection.close()

    async def create_foundation(
        self, connection: asyncpg.Connection, *, suffix: str = "default"
    ) -> tuple[UUID, UUID, UUID]:
        workspace_id = await connection.fetchval(
            "INSERT INTO workspace (name) VALUES ($1) RETURNING id",
            f"workspace-{suffix}",
        )
        embedding_space_id = await connection.fetchval(
            """
            INSERT INTO embedding_space (
                workspace_id, provider_identity, endpoint_identity,
                requested_model, resolved_model, model_version, dimension,
                distance_metric, vector_data_type, normalization,
                configuration_fingerprint, compatibility_fingerprint
            ) VALUES (
                $1, 'alibaba-cloud-model-studio-qwen',
                'alibaba-model-studio-beijing-embedding',
                'qwen3.7-text-embedding', 'qwen3.7-text-embedding',
                'qwen3.7-text-embedding', 1024,
                'cosine', 'float32', 'l2', $2, $3
            ) RETURNING id
            """,
            workspace_id,
            f"configuration-{suffix}",
            f"compatibility-{suffix}",
        )
        kb_id = await connection.fetchval(
            """
            INSERT INTO knowledge_base (workspace_id, name)
            VALUES ($1, $2) RETURNING id
            """,
            workspace_id,
            f"kb-{suffix}",
        )
        return workspace_id, embedding_space_id, kb_id

    async def create_revision(
        self,
        connection: asyncpg.Connection,
        workspace_id: UUID,
        embedding_space_id: UUID,
        kb_id: UUID,
        *,
        status: str = "building",
    ) -> UUID:
        return await connection.fetchval(
            """
            INSERT INTO index_revision (
                workspace_id, kb_id, embedding_space_id, status,
                source_snapshot_seq, parser_config, chunking_config
            ) VALUES ($1, $2, $3, $4, 0, '{}'::jsonb, '{}'::jsonb)
            RETURNING id
            """,
            workspace_id,
            kb_id,
            embedding_space_id,
            status,
        )

    async def create_document_versions(
        self,
        connection: asyncpg.Connection,
        workspace_id: UUID,
        kb_id: UUID,
        *,
        count: int = 2,
    ) -> tuple[UUID, list[UUID]]:
        document_id = await connection.fetchval(
            """
            INSERT INTO document (workspace_id, kb_id, display_name)
            VALUES ($1, $2, 'test document') RETURNING id
            """,
            workspace_id,
            kb_id,
        )
        versions: list[UUID] = []
        for version_number in range(1, count + 1):
            version_id = await connection.fetchval(
                """
                INSERT INTO document_version (
                    workspace_id, kb_id, document_id, version_number,
                    source_status, checksum_sha256, storage_uri,
                    original_filename, media_type, size_bytes
                ) VALUES (
                    $1, $2, $3, $4, 'available', $5, $6,
                    'document.txt', 'text/plain', 10
                ) RETURNING id
                """,
                workspace_id,
                kb_id,
                document_id,
                version_number,
                f"{version_number:064d}",
                f"file:///sources/{version_number}.txt",
            )
            versions.append(version_id)
        await connection.execute(
            "UPDATE document SET current_version_id = $1 WHERE id = $2",
            versions[-1],
            document_id,
        )
        return document_id, versions

    async def test_runtime_role_is_dml_only_and_readiness_is_read_only(self) -> None:
        engine = create_async_engine(RUNTIME_SQLALCHEMY_DSN)
        try:
            await check_database_ready(engine)
        finally:
            await engine.dispose()

        runtime = await asyncpg.connect(RUNTIME_DSN)
        try:
            workspace_id = await runtime.fetchval(
                "INSERT INTO workspace (name) VALUES ('runtime-dml') RETURNING id"
            )
            await runtime.execute("DELETE FROM workspace WHERE id = $1", workspace_id)
            with self.assertRaises(asyncpg.InsufficientPrivilegeError):
                await runtime.execute("CREATE TABLE runtime_must_not_create (id int)")
            with self.assertRaises(asyncpg.InsufficientPrivilegeError):
                await runtime.execute("CREATE TEMP TABLE runtime_temp_forbidden (id int)")
            with self.assertRaises(asyncpg.InsufficientPrivilegeError):
                await runtime.execute(
                    "UPDATE alembic_version SET version_num = 'runtime-mutation'"
                )
            revision = await runtime.fetchval("SELECT version_num FROM alembic_version")
            self.assertEqual(revision, EXPECTED_REVISION)
        finally:
            await runtime.close()

    async def test_attempt_ownership_migration_round_trip_is_idle_and_reversible(
        self,
    ) -> None:
        migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0026_simplify_attempt_ownership"
        )
        engine = create_async_engine(
            MIGRATION_DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
        )
        try:
            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection, migration, "downgrade"
                    )
                )
            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                restored = await connection.fetch(
                    """
                    SELECT table_name FROM information_schema.columns
                     WHERE table_schema = 'public'
                       AND column_name = 'claimed_by'
                       AND table_name IN ('chat_run', 'indexing_job')
                     ORDER BY table_name
                    """
                )
            finally:
                await connection.close()
            self.assertEqual(
                [row["table_name"] for row in restored],
                ["chat_run", "indexing_job"],
            )
            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection, migration, "upgrade"
                    )
                )
            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                remaining = await connection.fetchval(
                    """
                    SELECT count(*) FROM information_schema.columns
                     WHERE table_schema = 'public'
                       AND column_name = 'claimed_by'
                       AND table_name IN ('chat_run', 'indexing_job')
                    """
                )
            finally:
                await connection.close()
            self.assertEqual(remaining, 0)
        finally:
            await engine.dispose()

    async def test_dynamic_identity_removal_round_trip_fails_closed(self) -> None:
        migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0025_remove_dynamic_identity"
        )
        engine = create_async_engine(
            MIGRATION_DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
        )
        workspace_id = UUID(os.environ["RAG_KB__IDENTITY__WORKSPACE_ID"])
        principal = os.environ["RAG_KB__IDENTITY__PRINCIPAL_ID"]
        try:
            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection, migration, "downgrade"
                    )
                )

            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                await connection.execute(
                    "INSERT INTO workspace (id, name) VALUES ($1, 'identity-test')",
                    workspace_id,
                )
                kb_id = await connection.fetchval(
                    """
                    INSERT INTO knowledge_base (workspace_id, name)
                    VALUES ($1, 'identity-test-kb') RETURNING id
                    """,
                    workspace_id,
                )
                session_id = await connection.fetchval(
                    """
                    INSERT INTO chat_session (
                        workspace_id, kb_id, principal_id, title
                    ) VALUES ($1, $2, 'wrong-principal', 'identity-test')
                    RETURNING id
                    """,
                    workspace_id,
                    kb_id,
                )
            finally:
                await connection.close()

            with self.assertRaisesRegex(
                RuntimeError, "dynamic identity removal preflight failed"
            ):
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection, migration, "upgrade"
                        )
                    )

            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                await connection.execute(
                    "UPDATE chat_session SET principal_id = $1 WHERE id = $2",
                    principal,
                    session_id,
                )
            finally:
                await connection.close()

            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection, migration, "upgrade"
                    )
                )

            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                columns = await connection.fetch(
                    """
                    SELECT table_name, column_name
                      FROM information_schema.columns
                     WHERE table_schema = 'public'
                       AND table_name IN (
                           'chat_session', 'chat_run', 'content_mutation'
                       )
                       AND column_name IN ('principal_id', 'client_id')
                    """
                )
                self.assertEqual(columns, [])
                self.assertEqual(
                    await connection.fetchval(
                        "SELECT count(*) FROM chat_session WHERE id = $1",
                        session_id,
                    ),
                    1,
                )
            finally:
                await connection.close()
        finally:
            await self._restore_dynamic_identity_schema(engine)
            await engine.dispose()

    async def test_retired_final_llm_context_migration_round_trip(self) -> None:
        migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0015_remove_retired_chat_state"
        )
        engine = create_async_engine(
            MIGRATION_DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
        )
        try:
            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection,
                        migration,
                        "downgrade",
                    )
                )
            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                self.assertTrue(
                    await connection.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_schema = 'public'
                              AND table_name = 'chat_run'
                              AND column_name = 'final_llm_context'
                        )
                        """
                    )
                )
            finally:
                await connection.close()

            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection,
                        migration,
                        "upgrade",
                    )
                )
        finally:
            await engine.dispose()

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            self.assertFalse(
                await connection.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'chat_run'
                          AND column_name = 'final_llm_context'
                    )
                    """
                )
            )
        finally:
            await connection.close()

    async def test_knowledge_base_default_policy_is_partial_answer(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            _, _, kb_id = await self.create_foundation(
                connection, suffix="partial-policy-default"
            )
            policy = await connection.fetchval(
                """
                SELECT answer_policy_defaults ->> 'insufficiency_policy'
                FROM knowledge_base
                WHERE id = $1
                """,
                kb_id,
            )
            column_default = await connection.fetchval(
                """
                SELECT column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'knowledge_base'
                  AND column_name = 'answer_policy_defaults'
                """
            )
        finally:
            await connection.close()

        self.assertEqual(policy, "partial_answer")
        self.assertIn("partial_answer", column_default)

    async def test_knowledge_base_default_uses_explicit_classic_rerank_mode(
        self,
    ) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            _, _, kb_id = await self.create_foundation(
                connection, suffix="rerank-mode-default"
            )
            defaults = await connection.fetchval(
                "SELECT retrieval_defaults FROM knowledge_base WHERE id = $1",
                kb_id,
            )
            column_default = await connection.fetchval(
                """
                SELECT column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'knowledge_base'
                  AND column_name = 'retrieval_defaults'
                """
            )
        finally:
            await connection.close()

        decoded = json.loads(defaults)
        self.assertEqual(decoded["rerank_mode"], "classic")
        self.assertNotIn("rerank", decoded)
        self.assertIn("rerank_mode", column_default)

    async def test_local_rerank_migration_round_trips_existing_rows(self) -> None:
        assert MIGRATION_DSN is not None
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            _, _, classic_kb_id = await self.create_foundation(
                connection, suffix="rerank-migration-classic"
            )
            _, _, none_kb_id = await self.create_foundation(
                connection, suffix="rerank-migration-none"
            )
            await connection.execute(
                """
                UPDATE knowledge_base
                   SET retrieval_defaults = $2::jsonb
                 WHERE id = $1
                """,
                none_kb_id,
                json.dumps(
                    {
                        "strategy": "exact_vector",
                        "top_k": 10,
                        "rerank_mode": "none",
                    }
                ),
            )
        finally:
            await connection.close()

        migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0008_local_rerank_mode"
        )
        engine = create_async_engine(
            MIGRATION_DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
        )
        try:
            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection,
                        migration,
                        "downgrade",
                    )
                )

            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                legacy_rows = await connection.fetch(
                    """
                    SELECT id, retrieval_defaults
                      FROM knowledge_base
                     WHERE id = ANY($1::uuid[])
                    """,
                    [classic_kb_id, none_kb_id],
                )
                legacy_default = await connection.fetchval(
                    """
                    SELECT column_default
                      FROM information_schema.columns
                     WHERE table_schema = 'public'
                       AND table_name = 'knowledge_base'
                       AND column_name = 'retrieval_defaults'
                    """
                )
            finally:
                await connection.close()

            legacy_by_id = {
                row["id"]: json.loads(row["retrieval_defaults"])
                for row in legacy_rows
            }
            self.assertTrue(legacy_by_id[classic_kb_id]["rerank"])
            self.assertFalse(legacy_by_id[none_kb_id]["rerank"])
            self.assertTrue(
                all("rerank_mode" not in value for value in legacy_by_id.values())
            )
            self.assertIn("rerank", legacy_default)

            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection,
                        migration,
                        "upgrade",
                    )
                )

            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                current_rows = await connection.fetch(
                    """
                    SELECT id, retrieval_defaults
                      FROM knowledge_base
                     WHERE id = ANY($1::uuid[])
                    """,
                    [classic_kb_id, none_kb_id],
                )
                current_default = await connection.fetchval(
                    """
                    SELECT column_default
                      FROM information_schema.columns
                     WHERE table_schema = 'public'
                       AND table_name = 'knowledge_base'
                       AND column_name = 'retrieval_defaults'
                    """
                )
            finally:
                await connection.close()
        finally:
            await engine.dispose()

        current_by_id = {
            row["id"]: json.loads(row["retrieval_defaults"])
            for row in current_rows
        }
        self.assertEqual(current_by_id[classic_kb_id]["rerank_mode"], "classic")
        self.assertEqual(current_by_id[none_kb_id]["rerank_mode"], "none")
        self.assertTrue(all("rerank" not in value for value in current_by_id.values()))
        self.assertIn("rerank_mode", current_default)

    @staticmethod
    def _invoke_migration(sync_connection, migration, direction: str) -> None:
        previous_op = migration.op
        migration.op = Operations(MigrationContext.configure(sync_connection))
        try:
            getattr(migration, direction)()
        finally:
            migration.op = previous_op

    @staticmethod
    def _invoke_migration_operation(
        sync_connection, migration, operation: str
    ) -> None:
        previous_op = migration.op
        migration.op = Operations(MigrationContext.configure(sync_connection))
        try:
            getattr(migration, operation)()
        finally:
            migration.op = previous_op

    async def _prepare_historical_v3_schema(self, engine, migration) -> None:
        """Materialize the pre-0022 v3 boundary for historical migration tests."""

        async with engine.begin() as migration_connection:
            await migration_connection.run_sync(
                lambda sync_connection: self._prepare_historical_v3_schema_sync(
                    sync_connection, migration
                )
            )

    @classmethod
    def _prepare_historical_v3_schema_sync(
        cls, sync_connection, migration
    ) -> None:
        has_principal = sync_connection.exec_driver_sql(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                 WHERE table_schema = 'public'
                   AND table_name = 'chat_session'
                   AND column_name = 'principal_id'
            )
            """
        ).scalar_one()
        if not has_principal:
            identity_migration = importlib.import_module(
                "rag_kb.db.migrations.versions.0025_remove_dynamic_identity"
            )
            cls._invoke_migration(
                sync_connection, identity_migration, "downgrade"
            )
        # 0024 intentionally removes these checks from the current head. The
        # old 0016 round-trip tests need the historical v3 boundary restored
        # temporarily so they can exercise that migration in isolation.
        sync_connection.exec_driver_sql(
            "ALTER TABLE chat_run DROP CONSTRAINT IF EXISTS "
            "ck_chat_run_agent_trace_v3"
        )
        sync_connection.exec_driver_sql(
            "ALTER TABLE chat_run DROP CONSTRAINT IF EXISTS "
            "ck_chat_run_agent_configuration_v3"
        )
        cls._invoke_migration_operation(
            sync_connection, migration, "_create_v3_constraints"
        )

    async def _restore_current_agent_schema(self, engine) -> None:
        """Return a historical migration test database to the current head."""

        for module_name in (
            "rag_kb.db.migrations.versions.0022_agent_resource_budget",
            "rag_kb.db.migrations.versions.0023_agent_trace_diagnostics",
            "rag_kb.db.migrations.versions.0024_remove_agent_deadline_reserve",
        ):
            migration = importlib.import_module(module_name)
            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection, migration, "upgrade"
                    )
                )
        await self._restore_dynamic_identity_schema(engine)

    async def _restore_dynamic_identity_schema(self, engine) -> None:
        """Replay 0025–0028 when a historical test left the shared schema behind."""

        identity_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0025_remove_dynamic_identity"
        )
        ownership_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0026_simplify_attempt_ownership"
        )
        v4_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0027_agent_v4_default"
        )
        v5_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0028_agent_v5_default"
        )
        async with engine.begin() as migration_connection:
            result = await migration_connection.exec_driver_sql(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                     WHERE table_schema = 'public'
                       AND table_name = 'chat_session'
                       AND column_name = 'principal_id'
                )
                """
            )
            if result.scalar_one():
                await migration_connection.exec_driver_sql(
                    "TRUNCATE TABLE workspace CASCADE"
                )
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection, identity_migration, "upgrade"
                    )
                )
        async with engine.begin() as migration_connection:
            result = await migration_connection.exec_driver_sql(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                     WHERE table_schema = 'public'
                       AND table_name = 'chat_run'
                       AND column_name = 'claimed_by'
                )
                """
            )
            if result.scalar_one():
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection, ownership_migration, "upgrade"
                    )
                )
        # 0024 historical upgrades rewrite the column default back to v3.
        async with engine.begin() as migration_connection:
            await migration_connection.run_sync(
                lambda sync_connection: self._invoke_migration(
                    sync_connection, v4_migration, "upgrade"
                )
            )
            await migration_connection.run_sync(
                lambda sync_connection: self._invoke_migration(
                    sync_connection, v5_migration, "upgrade"
                )
            )

    async def test_legacy_chat_workflow_columns_are_removed(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            columns = await connection.fetch(
                """
                SELECT column_name, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'chat_run'
                  AND column_name IN ('workflow_configuration', 'workflow_state')
                ORDER BY column_name
                """
            )
            constraints = await connection.fetch(
                """
                SELECT conname, pg_get_constraintdef(oid) AS definition
                FROM pg_constraint
                WHERE conrelid = 'public.chat_run'::regclass
                  AND (
                    conname LIKE '%ck_chat_run_workflow_configuration_v1'
                    OR conname LIKE '%ck_chat_run_workflow_state_v1'
                  )
                ORDER BY conname
                """
            )
        finally:
            await connection.close()

        self.assertEqual(columns, [])
        self.assertEqual(constraints, [])

    async def test_native_agent_columns_use_current_budget_shape(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            columns = await connection.fetch(
                """
                SELECT column_name, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'chat_run'
                  AND column_name IN ('agent_configuration', 'agent_trace')
                ORDER BY column_name
                """
            )
            constraints = await connection.fetch(
                """
                SELECT conname, pg_get_constraintdef(oid) AS definition
                FROM pg_constraint
                WHERE conrelid = 'public.chat_run'::regclass
                  AND conname LIKE '%ck_chat_run_agent_%_v3'
                ORDER BY conname
                """
            )
        finally:
            await connection.close()

        self.assertEqual(
            [row["column_name"] for row in columns],
            ["agent_configuration", "agent_trace"],
        )
        self.assertEqual(columns[0]["is_nullable"], "NO")
        self.assertEqual(columns[1]["is_nullable"], "YES")
        self.assertIn("native_tool_calling_agent_v5", columns[0]["column_default"])
        self.assertIn("max_total_tokens", columns[0]["column_default"])
        self.assertNotIn("max_model_rounds", columns[0]["column_default"])
        self.assertNotIn("max_graph_calls", columns[0]["column_default"])
        self.assertNotIn("max_evidence_items", columns[0]["column_default"])
        self.assertNotIn("max_retrieval_calls", columns[0]["column_default"])
        self.assertNotIn("soft_deadline_reserve_seconds", columns[0]["column_default"])
        self.assertNotIn("'retrieval_calls'", columns[0]["column_default"])
        self.assertNotIn("'calculation_calls'", columns[0]["column_default"])
        self.assertNotIn("'evidence_refs'", columns[0]["column_default"])
        self.assertEqual(constraints, [])

    async def test_deadline_reserve_migration_preserves_historical_chat_runs(
        self,
    ) -> None:
        """0024 drops enforcement without rewriting historical JSON snapshots."""

        assert MIGRATION_DSN is not None
        first_class_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0016_first_class_graph_tool"
        )
        budget_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0022_agent_resource_budget"
        )
        diagnostics_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0023_agent_trace_diagnostics"
        )
        deadline_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0024_remove_agent_deadline_reserve"
        )
        engine = create_async_engine(
            MIGRATION_DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
        )
        connection = None
        current_head_restored = False
        historical_configuration = {
            "version": "native_tool_calling_agent_v3",
            "budget": {
                "max_model_rounds": 8,
                "max_graph_calls": 2,
                "max_total_tokens": 150000,
                "max_evidence_items": 64,
                "max_retrieval_calls": 16,
                "soft_deadline_reserve_seconds": 60,
            },
        }
        historical_trace = {
            "version": "native_tool_calling_agent_v3",
            "events": [],
            "budget": historical_configuration["budget"],
            "usage": {"model_rounds": 0},
            "diagnostics": {
                "stop_reason": "submitted",
                "forced_finalize": False,
                "consecutive_no_new_evidence": 0,
                "elapsed_ms": 1,
                "deadline_ms": 600000,
                "deadline_remaining_ms": 599999,
                "near_deadline": True,
                "deadline_exceeded": False,
            },
            "outcome": "refused",
        }
        try:
            await self._prepare_historical_v3_schema(
                engine, first_class_migration
            )
            for migration in (budget_migration, diagnostics_migration):
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection, migration=migration: self._invoke_migration(
                            sync_connection, migration, "upgrade"
                        )
                    )

            connection = await asyncpg.connect(MIGRATION_DSN)
            workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                connection, suffix="deadline-reserve-preservation"
            )
            revision_id = await self.create_revision(
                connection,
                workspace_id,
                embedding_space_id,
                kb_id,
            )
            session_id = await connection.fetchval(
                """
                INSERT INTO chat_session (
                    workspace_id, kb_id, principal_id, title
                ) VALUES ($1, $2, 'historical-principal', 'historical-session')
                RETURNING id
                """,
                workspace_id,
                kb_id,
            )
            user_message_id = await connection.fetchval(
                """
                INSERT INTO chat_message (
                    workspace_id, session_id, role, content
                ) VALUES ($1, $2, 'user', 'historical question')
                RETURNING id
                """,
                workspace_id,
                session_id,
            )
            run_id = await connection.fetchval(
                """
                INSERT INTO chat_run (
                    workspace_id, kb_id, session_id, user_message_id,
                    index_revision_id, status, principal_id, client_id,
                    endpoint, idempotency_key, request_hash, requested_policy,
                    effective_policy, retrieval_strategy, model_configuration,
                    agent_configuration, agent_trace, conversation_context,
                    usage, timing, attempt, completed_at
                ) VALUES (
                    $1, $2, $3, $4, $5, 'completed',
                    'historical-principal', 'schema-client',
                    'POST /api/v1/chat/runs', $6, $7,
                    '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb,
                    $8::jsonb, $9::jsonb, '{}'::jsonb,
                    '{}'::jsonb, '{}'::jsonb, 1, now()
                )
                RETURNING id
                """,
                workspace_id,
                kb_id,
                session_id,
                user_message_id,
                revision_id,
                uuid4(),
                "sha256:" + "d" * 64,
                json.dumps(historical_configuration),
                json.dumps(historical_trace),
            )
            await connection.close()
            connection = None

            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection, deadline_migration, "upgrade"
                    )
                )
            current_head_restored = True

            connection = await asyncpg.connect(MIGRATION_DSN)
            row = await connection.fetchrow(
                """
                SELECT agent_configuration, agent_trace
                  FROM chat_run
                 WHERE id = $1
                """,
                run_id,
            )
            self.assertEqual(
                json.loads(row["agent_configuration"]), historical_configuration
            )
            self.assertEqual(json.loads(row["agent_trace"]), historical_trace)
        finally:
            if connection is not None:
                await connection.close()
            if not current_head_restored:
                try:
                    async with engine.begin() as migration_connection:
                        await migration_connection.run_sync(
                            lambda sync_connection: self._invoke_migration(
                                sync_connection, deadline_migration, "upgrade"
                            )
                        )
                except Exception:
                    pass
            await self._restore_dynamic_identity_schema(engine)
            await engine.dispose()

    async def test_native_agent_round_limit_migration_preserves_trace_facts(
        self,
    ) -> None:
        assert MIGRATION_DSN is not None
        connection: asyncpg.Connection | None = None
        original_events = [
            {
                "tool": "search_knowledge_base",
                "status": "ok",
                "tool_call_id": "search-1",
                "refs": ["ev_1", "ev_2"],
                "count": 2,
            },
            {
                "tool": "submit_answer",
                "status": "salvaged",
                "tool_call_id": "submit-1",
                "refs": ["ev_1"],
                "count": 1,
            },
        ]
        original_trace_usage = {
            "model_rounds": 8,
            "retrieval_calls": 5,
            "calculation_calls": 3,
            "evidence_refs": 9,
        }
        original_run_usage = {"totals": {"input_tokens": 11, "output_tokens": 7}}
        original_timing = {"attempts": {"1": {"result": "completed"}}}

        async def create_run(
            *, suffix: str, max_model_rounds: int, trace: dict | None
        ) -> UUID:
            assert connection is not None
            session_id = await connection.fetchval(
                """
                INSERT INTO chat_session (
                    workspace_id, kb_id, principal_id, title
                ) VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                workspace_id,
                kb_id,
                f"principal-{suffix}",
                f"session-{suffix}",
            )
            user_message_id = await connection.fetchval(
                """
                INSERT INTO chat_message (
                    workspace_id, session_id, role, content
                ) VALUES ($1, $2, 'user', $3)
                RETURNING id
                """,
                workspace_id,
                session_id,
                f"question-{suffix}",
            )
            configuration = {
                "version": "native_tool_calling_agent_v2",
                "budget": {"max_model_rounds": max_model_rounds},
            }
            return await connection.fetchval(
                """
                INSERT INTO chat_run (
                    workspace_id, kb_id, session_id, user_message_id,
                    index_revision_id, status, principal_id, client_id,
                    endpoint, idempotency_key, request_hash, requested_policy,
                    effective_policy, retrieval_strategy, model_configuration,
                    agent_configuration, agent_trace, conversation_context,
                    usage, timing, attempt, completed_at
                ) VALUES (
                    $1, $2, $3, $4, $5, 'completed', $6, 'schema-client',
                    'POST /api/v1/chat/runs', $7, $8,
                    '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb,
                    $9::jsonb, $10::jsonb, '{}'::jsonb,
                    $11::jsonb, $12::jsonb, 1, now()
                )
                RETURNING id
                """,
                workspace_id,
                kb_id,
                session_id,
                user_message_id,
                revision_id,
                f"principal-{suffix}",
                uuid4(),
                "sha256:" + suffix[0] * 64,
                json.dumps(configuration),
                json.dumps(trace) if trace is not None else None,
                json.dumps(original_run_usage),
                json.dumps(original_timing),
            )

        first_class_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0016_first_class_graph_tool"
        )
        engine = create_async_engine(
            MIGRATION_DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
        )
        prep_downgraded = False
        try:
            await self._prepare_historical_v3_schema(
                engine, first_class_migration
            )
            # Start this historical round trip from v2 rows.
            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection,
                        first_class_migration,
                        "downgrade",
                    )
                )
            prep_downgraded = True
            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                    connection, suffix="agent-round-migration"
                )
                revision_id = await self.create_revision(
                    connection,
                    workspace_id,
                    embedding_space_id,
                    kb_id,
                )
                traced_run_id = await create_run(
                    suffix="traced",
                    max_model_rounds=8,
                    trace={
                        "version": "native_tool_calling_agent_v2",
                        "events": original_events,
                        "budget": {"max_model_rounds": 8},
                        "usage": original_trace_usage,
                        "outcome": "partial",
                    },
                )
                null_trace_run_id = await create_run(
                    suffix="null-trace",
                    max_model_rounds=6,
                    trace=None,
                )
            finally:
                await connection.close()

            migration = importlib.import_module(
                "rag_kb.db.migrations.versions.0011_native_agent_round_limit"
            )
            downgraded = False
            try:
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection,
                            migration,
                            "downgrade",
                        )
                    )
                downgraded = True

                connection = await asyncpg.connect(MIGRATION_DSN)
                try:
                    downgraded_rows = await connection.fetch(
                        """
                        SELECT id, agent_configuration, agent_trace, usage, timing
                          FROM chat_run
                         WHERE id = ANY($1::uuid[])
                        """,
                        [traced_run_id, null_trace_run_id],
                    )
                finally:
                    await connection.close()

                downgraded_by_id = {row["id"]: row for row in downgraded_rows}
                traced_configuration = json.loads(
                    downgraded_by_id[traced_run_id]["agent_configuration"]
                )
                traced_trace = json.loads(
                    downgraded_by_id[traced_run_id]["agent_trace"]
                )
                self.assertEqual(
                    traced_configuration["version"], "native_tool_calling_agent_v1"
                )
                self.assertEqual(traced_trace["version"], "native_tool_calling_agent_v1")
                self.assertEqual(traced_trace["budget"], traced_configuration["budget"])
                self.assertEqual(traced_trace["events"], original_events)
                self.assertEqual(traced_trace["usage"], original_trace_usage)
                self.assertEqual(traced_trace["outcome"], "partial")
                self.assertIsNone(downgraded_by_id[null_trace_run_id]["agent_trace"])

                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection,
                            migration,
                            "upgrade",
                        )
                    )
                downgraded = False

                connection = await asyncpg.connect(MIGRATION_DSN)
                try:
                    upgraded_rows = await connection.fetch(
                        """
                        SELECT id, agent_configuration, agent_trace, usage, timing
                          FROM chat_run
                         WHERE id = ANY($1::uuid[])
                        """,
                        [traced_run_id, null_trace_run_id],
                    )
                    upgraded_by_id = {row["id"]: row for row in upgraded_rows}
                    traced_configuration = json.loads(
                        upgraded_by_id[traced_run_id]["agent_configuration"]
                    )
                    traced_trace = json.loads(upgraded_by_id[traced_run_id]["agent_trace"])
                    self.assertEqual(
                        traced_configuration,
                        {
                            "version": "native_tool_calling_agent_v2",
                            "budget": {"max_model_rounds": 8},
                        },
                    )
                    self.assertEqual(traced_trace["version"], "native_tool_calling_agent_v2")
                    self.assertEqual(traced_trace["budget"], traced_configuration["budget"])
                    self.assertEqual(traced_trace["events"], original_events)
                    self.assertEqual(traced_trace["usage"], original_trace_usage)
                    self.assertEqual(traced_trace["outcome"], "partial")
                    self.assertEqual(
                        json.loads(upgraded_by_id[traced_run_id]["usage"]),
                        original_run_usage,
                    )
                    self.assertEqual(
                        json.loads(upgraded_by_id[traced_run_id]["timing"]),
                        original_timing,
                    )
                    null_configuration = json.loads(
                        upgraded_by_id[null_trace_run_id]["agent_configuration"]
                    )
                    self.assertEqual(
                        null_configuration["budget"], {"max_model_rounds": 6}
                    )
                    self.assertIsNone(upgraded_by_id[null_trace_run_id]["agent_trace"])

                    invalid_configurations = (
                        {"version": "native_tool_calling_agent_v2"},
                        {
                            "version": "native_tool_calling_agent_v2",
                            "budget": {
                                "model_rounds": 8,
                                "retrieval_calls": 6,
                                "calculation_calls": 4,
                                "evidence_refs": 20,
                            },
                        },
                        {
                            "version": "native_tool_calling_agent_v2",
                            "budget": {"max_model_rounds": 8.5},
                        },
                    )
                    for value in invalid_configurations:
                        with self.subTest(agent_configuration=value), self.assertRaises(
                            asyncpg.CheckViolationError
                        ):
                            await connection.execute(
                                """
                                UPDATE chat_run
                                   SET agent_configuration = $2::jsonb
                                 WHERE id = $1
                                """,
                                traced_run_id,
                                json.dumps(value),
                            )

                    trace_without_budget = {
                        key: value
                        for key, value in traced_trace.items()
                        if key != "budget"
                    }
                    invalid_traces = (
                        trace_without_budget,
                        {
                            **traced_trace,
                            "budget": {
                                "model_rounds": 8,
                                "retrieval_calls": 6,
                                "calculation_calls": 4,
                                "evidence_refs": 20,
                            },
                        },
                        {**traced_trace, "budget": {"max_model_rounds": 7}},
                    )
                    for value in invalid_traces:
                        with self.subTest(agent_trace=value), self.assertRaises(
                            asyncpg.CheckViolationError
                        ):
                            await connection.execute(
                                """
                                UPDATE chat_run
                                   SET agent_trace = $2::jsonb
                                 WHERE id = $1
                                """,
                                traced_run_id,
                                json.dumps(value),
                            )
                finally:
                    await connection.close()
            finally:
                if downgraded:
                    async with engine.begin() as migration_connection:
                        await migration_connection.run_sync(
                            lambda sync_connection: self._invoke_migration(
                                sync_connection,
                                migration,
                                "upgrade",
                            )
                        )
                await engine.dispose()
        finally:
            if prep_downgraded:
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection,
                            first_class_migration,
                            "upgrade",
                        )
                    )
            await self._restore_current_agent_schema(engine)
            await engine.dispose()



    async def test_native_agent_round_limit_downgrade_rejects_forced_round_usage(
        self,
    ) -> None:
        assert MIGRATION_DSN is not None
        first_class_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0016_first_class_graph_tool"
        )
        engine = create_async_engine(
            MIGRATION_DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
        )
        prep_downgraded = False
        try:
            await self._prepare_historical_v3_schema(
                engine, first_class_migration
            )
            # v2 fixtures require the v2 schema first.
            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection,
                        first_class_migration,
                        "downgrade",
                    )
                )
            prep_downgraded = True
            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                    connection, suffix="agent-round-downgrade-guard"
                )
                revision_id = await self.create_revision(
                    connection,
                    workspace_id,
                    embedding_space_id,
                    kb_id,
                )
                session_id = await connection.fetchval(
                    """
                    INSERT INTO chat_session (
                        workspace_id, kb_id, principal_id, title
                    ) VALUES ($1, $2, 'principal-guard', 'session-guard')
                    RETURNING id
                    """,
                    workspace_id,
                    kb_id,
                )
                user_message_id = await connection.fetchval(
                    """
                    INSERT INTO chat_message (
                        workspace_id, session_id, role, content
                    ) VALUES ($1, $2, 'user', 'question-guard')
                    RETURNING id
                    """,
                    workspace_id,
                    session_id,
                )
                await connection.execute(
                    """
                    INSERT INTO chat_run (
                        workspace_id, kb_id, session_id, user_message_id,
                        index_revision_id, status, principal_id, client_id,
                        endpoint, idempotency_key, request_hash, requested_policy,
                        effective_policy, retrieval_strategy, model_configuration,
                        agent_configuration, agent_trace, conversation_context,
                        usage, timing, attempt, completed_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, 'completed', 'principal-guard',
                        'schema-client', 'POST /api/v1/chat/runs', $6, $7,
                        '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb,
                        $8::jsonb, $9::jsonb, '{}'::jsonb,
                        '{}'::jsonb, '{}'::jsonb, 1, now()
                    )
                    """,
                    workspace_id,
                    kb_id,
                    session_id,
                    user_message_id,
                    revision_id,
                    uuid4(),
                    "sha256:" + "f" * 64,
                    json.dumps(
                        {
                            "version": "native_tool_calling_agent_v2",
                            "budget": {"max_model_rounds": 8},
                        }
                    ),
                    json.dumps(
                        {
                            "version": "native_tool_calling_agent_v2",
                            "events": [],
                            "budget": {"max_model_rounds": 8},
                            "usage": {
                                "model_rounds": 9,
                                "retrieval_calls": 0,
                                "calculation_calls": 0,
                                "evidence_refs": 0,
                            },
                            "outcome": "refused",
                        }
                    ),
                )
            finally:
                await connection.close()

            migration = importlib.import_module(
                "rag_kb.db.migrations.versions.0011_native_agent_round_limit"
            )
            try:
                with self.assertRaisesRegex(
                    Exception, "native Agent v2 usage cannot be represented"
                ):
                    async with engine.begin() as migration_connection:
                        await migration_connection.run_sync(
                            lambda sync_connection: self._invoke_migration(
                                sync_connection,
                                migration,
                                "downgrade",
                            )
                        )
            finally:
                await engine.dispose()

            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                self.assertEqual(
                    await connection.fetchval(
                        "SELECT version_num FROM alembic_version"
                    ),
                    EXPECTED_REVISION,
                )
            finally:
                await connection.close()
        finally:
            if prep_downgraded:
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection,
                            first_class_migration,
                            "upgrade",
                        )
                    )
            await self._restore_current_agent_schema(engine)
            await engine.dispose()

    async def test_first_class_graph_relations_migration_round_trip(
        self,
    ) -> None:
        assert MIGRATION_DSN is not None
        original_events = [
            {
                "tool": "search_knowledge_base",
                "status": "ok",
                "tool_call_id": "simple-1",
                "refs": ["ev_1"],
                "count": 1,
                "retrieval_lane": "simple",
                "route_result_code": "not_requested",
            },
            {
                "tool": "graphiti_supplement",
                "status": "ok",
                "tool_call_id": "graph-1",
                "refs": ["ev_2"],
                "count": 1,
                "retrieval_lane": "graphiti_supplement",
                "route_reason_code": "cross_document_relation_gap",
                "route_result_code": "admitted",
                "new_evidence_count": 1,
            },
            {
                "tool": "graphiti_supplement",
                "status": "ok",
                "tool_call_id": "guard_3",
                "refs": ["ev_2"],
                "count": 1,
                "retrieval_lane": "graphiti_supplement",
                "route_reason_code": "relation_chain_gap",
                "route_result_code": "no_new_evidence",
                "new_evidence_count": 0,
            },
            {
                "tool": "submit_answer",
                "status": "ok",
                "tool_call_id": "submit-1",
                "refs": ["ev_1", "ev_2"],
                "count": 2,
            },
        ]
        original_usage = {
            "model_rounds": 7,
            "retrieval_calls": 4,
            "calculation_calls": 2,
            "evidence_refs": 2,
        }
        v2_retrieval = {
            "profile_version": "adaptive_graphiti_v2",
            "strategy": "exact_vector",
            "top_k": 8,
            "rerank_mode": "classic",
            "router": "native_agent_path_guard_v2",
            "augmentation": "graphiti_path_v3",
        }
        legacy_adaptive_route_retrieval = {
            "profile_version": "adaptive_graph_route_v1",
            "strategy": "exact_vector",
            "top_k": 10,
            "rerank_mode": "classic",
            "augmentation": "entity_graph_v1",
            "route_policy": "agent_evidence_aware_v1",
        }
        first_class_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0016_first_class_graph_tool"
        )
        engine = create_async_engine(
            MIGRATION_DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
        )
        await self._prepare_historical_v3_schema(
            engine, first_class_migration
        )
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                connection, suffix="graph-tool-migration"
            )
            revision_id = await self.create_revision(
                connection,
                workspace_id,
                embedding_space_id,
                kb_id,
            )
            session_id = await connection.fetchval(
                """
                INSERT INTO chat_session (
                    workspace_id, kb_id, principal_id, title
                ) VALUES ($1, $2, 'principal-graph-tool', 'session-graph-tool')
                RETURNING id
                """,
                workspace_id,
                kb_id,
            )
            user_message_id = await connection.fetchval(
                """
                INSERT INTO chat_message (
                    workspace_id, session_id, role, content
                ) VALUES ($1, $2, 'user', 'question-graph-tool')
                RETURNING id
                """,
                workspace_id,
                session_id,
            )

            async def insert_run(
                *,
                suffix: str,
                configuration: dict,
                trace: dict | None,
                retrieval: dict,
            ) -> UUID:
                return await connection.fetchval(
                    """
                    INSERT INTO chat_run (
                        workspace_id, kb_id, session_id, user_message_id,
                        index_revision_id, status, principal_id, client_id,
                        endpoint, idempotency_key, request_hash, requested_policy,
                        effective_policy, retrieval_strategy, model_configuration,
                        agent_configuration, agent_trace, conversation_context,
                        usage, timing, attempt, completed_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, 'completed', $6, 'schema-client',
                        'POST /api/v1/chat/runs', $7, $8,
                        '{}'::jsonb, '{}'::jsonb, $9::jsonb, '{}'::jsonb,
                        $10::jsonb, $11::jsonb, '{}'::jsonb,
                        '{}'::jsonb, '{}'::jsonb, 1, now()
                    )
                    RETURNING id
                    """,
                    workspace_id,
                    kb_id,
                    session_id,
                    user_message_id,
                    revision_id,
                    f"principal-{suffix}",
                    uuid4(),
                    "sha256:" + suffix[-1] * 64,
                    json.dumps(retrieval),
                    json.dumps(configuration),
                    json.dumps(trace) if trace is not None else None,
                )

            # Fixture rows must target the v2 schema.
            prep_downgraded = False
            try:
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection,
                            first_class_migration,
                            "downgrade",
                        )
                    )
                prep_downgraded = True

                v2_configuration = {
                    "version": "native_tool_calling_agent_v2",
                    "budget": {"max_model_rounds": 8},
                }
                v2_trace = {
                    "version": "native_tool_calling_agent_v2",
                    "events": original_events,
                    "budget": {"max_model_rounds": 8},
                    "usage": original_usage,
                    "outcome": "answered",
                }
                traced_run_id = await insert_run(
                    suffix="traced",
                    configuration=v2_configuration,
                    trace=v2_trace,
                    retrieval=v2_retrieval,
                )
                null_trace_run_id = await insert_run(
                    suffix="null-trace",
                    configuration=v2_configuration,
                    trace=None,
                    retrieval=v2_retrieval,
                )
                legacy_adaptive_route_run_id = await insert_run(
                    suffix="legacy-route-x",
                    configuration=v2_configuration,
                    trace=None,
                    retrieval=legacy_adaptive_route_retrieval,
                )
                await connection.close()

                # Upgrade to v3 transforms every snapshot deterministically.
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection,
                            first_class_migration,
                            "upgrade",
                        )
                    )
                prep_downgraded = False

                connection = await asyncpg.connect(MIGRATION_DSN)
                try:
                    upgraded = await connection.fetchrow(
                        """
                        SELECT agent_configuration, agent_trace, retrieval_strategy
                          FROM chat_run
                         WHERE id = $1
                        """,
                        traced_run_id,
                    )
                finally:
                    await connection.close()
                configuration = json.loads(upgraded["agent_configuration"])
                trace = json.loads(upgraded["agent_trace"])
                retrieval = json.loads(upgraded["retrieval_strategy"])
                self.assertEqual(
                    configuration,
                    {
                        "version": "native_tool_calling_agent_v3",
                        "budget": {"max_model_rounds": 8, "max_graph_calls": 2},
                    },
                )
                self.assertEqual(
                    retrieval,
                    {
                        "profile_version": "adaptive_graphiti_v3",
                        "strategy": "exact_vector",
                        "top_k": 8,
                        "rerank_mode": "classic",
                        "router": "native_agent_graph_tool_v1",
                        "augmentation": "graphiti_path_v3",
                        "graph_edge_limit": 16,
                        "graph_source_chunk_target": 12,
                        "graph_source_chunk_limit": 16,
                        "graph_call_timeout_seconds": 90,
                    },
                )
                self.assertEqual(trace["version"], "native_tool_calling_agent_v3")
                self.assertEqual(trace["budget"], configuration["budget"])
                self.assertEqual(trace["usage"], original_usage)
                self.assertEqual(trace["outcome"], "answered")
                events = trace["events"]
                self.assertEqual(events[0], original_events[0])
                self.assertEqual(
                    events[1],
                    {
                        "tool": "search_graph_relations",
                        "status": "ok",
                        "tool_call_id": "graph-1",
                        "refs": ["ev_2"],
                        "count": 1,
                        "retrieval_lane": "graph_relations",
                        "route_reason_code": "cross_document_relation",
                        "route_result_code": "admitted",
                        "new_evidence_count": 1,
                        "call_index": 1,
                        "invocation_source": "agent",
                    },
                )
                self.assertEqual(
                    events[2],
                    {
                        "tool": "search_graph_relations",
                        "status": "ok",
                        "tool_call_id": "guard_3",
                        "refs": ["ev_2"],
                        "count": 1,
                        "retrieval_lane": "graph_relations",
                        "route_reason_code": "relation_chain",
                        "route_result_code": "no_evidence",
                        "new_evidence_count": 0,
                        "call_index": 2,
                        "invocation_source": "legacy_guard",
                    },
                )
                self.assertEqual(events[3], original_events[3])
                self.assertNotIn("duration_ms", events[1])
                self.assertNotIn("candidate_count", events[1])
                self.assertNotIn("hop1_count", events[1])

                connection = await asyncpg.connect(MIGRATION_DSN)
                try:
                    legacy_route = await connection.fetchrow(
                        """
                        SELECT agent_configuration, agent_trace, retrieval_strategy
                          FROM chat_run
                         WHERE id = $1
                        """,
                        legacy_adaptive_route_run_id,
                    )
                finally:
                    await connection.close()
                self.assertEqual(
                    json.loads(legacy_route["agent_configuration"]),
                    {
                        "version": "native_tool_calling_agent_v3",
                        "budget": {
                            "max_model_rounds": 8,
                            "max_graph_calls": 2,
                        },
                    },
                )
                self.assertEqual(
                    json.loads(legacy_route["retrieval_strategy"]),
                    {
                        "profile_version": "exact_vector_v2",
                        "strategy": "exact_vector",
                        "top_k": 10,
                        "rerank_mode": "classic",
                    },
                )
                self.assertIsNone(legacy_route["agent_trace"])

                # Downgrade back to v2 restores the historical v2 names.
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection,
                            first_class_migration,
                            "downgrade",
                        )
                    )

                connection = await asyncpg.connect(MIGRATION_DSN)
                try:
                    downgraded = await connection.fetchrow(
                        """
                        SELECT agent_configuration, agent_trace, retrieval_strategy
                          FROM chat_run
                         WHERE id = $1
                        """,
                        traced_run_id,
                    )
                finally:
                    await connection.close()
                configuration = json.loads(downgraded["agent_configuration"])
                trace = json.loads(downgraded["agent_trace"])
                retrieval = json.loads(downgraded["retrieval_strategy"])
                self.assertEqual(
                    configuration,
                    {
                        "version": "native_tool_calling_agent_v2",
                        "budget": {"max_model_rounds": 8},
                    },
                )
                self.assertEqual(
                    retrieval,
                    {
                        "profile_version": "adaptive_graphiti_v2",
                        "strategy": "exact_vector",
                        "top_k": 8,
                        "rerank_mode": "classic",
                        "router": "native_agent_path_guard_v2",
                        "augmentation": "graphiti_path_v3",
                    },
                )
                self.assertEqual(trace["version"], "native_tool_calling_agent_v2")
                self.assertEqual(trace["budget"], configuration["budget"])
                events = trace["events"]
                self.assertEqual(events[0], original_events[0])
                self.assertEqual(events[1]["tool"], "graphiti_supplement")
                self.assertEqual(
                    events[1]["route_reason_code"], "cross_document_relation_gap"
                )
                self.assertEqual(events[1]["route_result_code"], "admitted")
                self.assertEqual(events[2]["tool"], "graphiti_supplement")
                self.assertEqual(events[2]["route_result_code"], "no_new_evidence")
                self.assertNotIn("call_index", events[1])
                self.assertNotIn("invocation_source", events[2])
                self.assertEqual(events[3], original_events[3])
                connection = await asyncpg.connect(MIGRATION_DSN)
                try:
                    self.assertIsNone(
                        await connection.fetchval(
                            "SELECT agent_trace FROM chat_run WHERE id = $1",
                            null_trace_run_id,
                        )
                    )
                finally:
                    await connection.close()

                # A second Graph call cannot be represented by v2.
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection,
                            first_class_migration,
                            "upgrade",
                        )
                    )
                connection = await asyncpg.connect(MIGRATION_DSN)
                try:
                    v3_events = [
                        dict(original_events[1]),
                        dict(original_events[1]),
                    ]
                    v3_events[0].update(
                        tool="search_graph_relations",
                        retrieval_lane="graph_relations",
                        route_reason_code="relation_chain",
                        route_result_code="no_evidence",
                        new_evidence_count=0,
                        call_index=1,
                        invocation_source="agent",
                    )
                    v3_events[1].update(
                        tool="search_graph_relations",
                        retrieval_lane="graph_relations",
                        route_reason_code="relation_chain",
                        route_result_code="no_evidence",
                        new_evidence_count=0,
                        call_index=2,
                        invocation_source="agent",
                    )
                    await connection.execute(
                        """
                        UPDATE chat_run
                           SET agent_configuration = $2::jsonb,
                               agent_trace = $3::jsonb,
                               retrieval_strategy = $4::jsonb
                         WHERE id = $1
                        """,
                        traced_run_id,
                        json.dumps(
                            {
                                "version": "native_tool_calling_agent_v3",
                                "budget": {
                                    "max_model_rounds": 8,
                                    "max_graph_calls": 2,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "version": "native_tool_calling_agent_v3",
                                "events": v3_events,
                                "budget": {
                                    "max_model_rounds": 8,
                                    "max_graph_calls": 2,
                                },
                                "usage": original_usage,
                                "outcome": "answered",
                            }
                        ),
                        json.dumps(
                            {
                                "profile_version": "adaptive_graphiti_v3",
                                "strategy": "exact_vector",
                                "top_k": 8,
                                "rerank_mode": "classic",
                                "router": "native_agent_graph_tool_v1",
                                "augmentation": "graphiti_path_v3",
                                "graph_edge_limit": 16,
                                "graph_source_chunk_target": 12,
                                "graph_source_chunk_limit": 16,
                                "graph_call_timeout_seconds": 90,
                            }
                        ),
                    )
                finally:
                    await connection.close()
                with self.assertRaisesRegex(
                    Exception, "second Graph call"
                ):
                    async with engine.begin() as migration_connection:
                        await migration_connection.run_sync(
                            lambda sync_connection: self._invoke_migration(
                                sync_connection,
                                first_class_migration,
                                "downgrade",
                            )
                        )

            finally:
                if prep_downgraded:
                    async with engine.begin() as migration_connection:
                        await migration_connection.run_sync(
                            lambda sync_connection: self._invoke_migration(
                                sync_connection,
                                first_class_migration,
                                "upgrade",
                            )
                        )
                await self._restore_current_agent_schema(engine)
                await engine.dispose()
        finally:
            try:
                await connection.close()
            except Exception:
                pass

    async def test_first_class_graph_migration_fails_closed_on_unknown_shapes(
        self,
    ) -> None:
        assert MIGRATION_DSN is not None
        first_class_migration = importlib.import_module(
            "rag_kb.db.migrations.versions.0016_first_class_graph_tool"
        )
        engine = create_async_engine(
            MIGRATION_DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
        )
        prep_downgraded = False
        try:
            await self._prepare_historical_v3_schema(
                engine, first_class_migration
            )
            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection,
                        first_class_migration,
                        "downgrade",
                    )
                )
            prep_downgraded = True

            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                    connection, suffix="graph-tool-fail-closed"
                )
                revision_id = await self.create_revision(
                    connection,
                    workspace_id,
                    embedding_space_id,
                    kb_id,
                )
                session_id = await connection.fetchval(
                    """
                    INSERT INTO chat_session (
                        workspace_id, kb_id, principal_id, title
                    ) VALUES ($1, $2, 'principal-fail-closed', 'session-fail-closed')
                    RETURNING id
                    """,
                    workspace_id,
                    kb_id,
                )
                user_message_id = await connection.fetchval(
                    """
                    INSERT INTO chat_message (
                        workspace_id, session_id, role, content
                    ) VALUES ($1, $2, 'user', 'question-fail-closed')
                    RETURNING id
                    """,
                    workspace_id,
                    session_id,
                )

                async def insert_run(*, suffix: str, retrieval: dict, trace: dict) -> UUID:
                    return await connection.fetchval(
                        """
                        INSERT INTO chat_run (
                            workspace_id, kb_id, session_id, user_message_id,
                            index_revision_id, status, principal_id, client_id,
                            endpoint, idempotency_key, request_hash, requested_policy,
                            effective_policy, retrieval_strategy, model_configuration,
                            agent_configuration, agent_trace, conversation_context,
                            usage, timing, attempt, completed_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, 'completed', $6, 'schema-client',
                            'POST /api/v1/chat/runs', $7, $8,
                            '{}'::jsonb, '{}'::jsonb, $9::jsonb, '{}'::jsonb,
                            $10::jsonb, $11::jsonb, '{}'::jsonb,
                            '{}'::jsonb, '{}'::jsonb, 1, now()
                        )
                        RETURNING id
                        """,
                        workspace_id,
                        kb_id,
                        session_id,
                        user_message_id,
                        revision_id,
                        f"principal-{suffix}",
                        uuid4(),
                        "sha256:" + suffix[-4:] * 16,
                        json.dumps(retrieval),
                        json.dumps(
                            {
                                "version": "native_tool_calling_agent_v2",
                                "budget": {"max_model_rounds": 8},
                            }
                        ),
                        json.dumps(trace),
                    )

                v2_retrieval = {
                    "profile_version": "adaptive_graphiti_v2",
                    "strategy": "exact_vector",
                    "top_k": 8,
                    "rerank_mode": "classic",
                    "router": "native_agent_path_guard_v2",
                    "augmentation": "graphiti_path_v3",
                }
                base_trace = {
                    "version": "native_tool_calling_agent_v2",
                    "events": [],
                    "budget": {"max_model_rounds": 8},
                    "usage": {},
                    "outcome": "refused",
                }
                unknown_retrieval_run_id = await insert_run(
                    suffix="unknown-profile",
                    retrieval={
                        "profile_version": "future_retrieval_v9",
                        "extra": True,
                    },
                    trace=dict(base_trace),
                )
                unknown_event_run_id = await insert_run(
                    suffix="unknown-event",
                    retrieval=v2_retrieval,
                    trace={
                        **base_trace,
                        "events": [
                            {
                                "tool": "graphiti_supplement",
                                "status": "ok",
                                "tool_call_id": "graph-1",
                                "count": 1,
                                "retrieval_lane": "graphiti_supplement",
                                "route_reason_code": "relation_chain_gap",
                                "route_result_code": "weird_result_code",
                                "new_evidence_count": 0,
                            }
                        ],
                    },
                )
            finally:
                await connection.close()

            with self.assertRaisesRegex(
                Exception, "unknown retrieval snapshot shape"
            ):
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection,
                            first_class_migration,
                            "upgrade",
                        )
                    )

            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                await connection.execute(
                    "DELETE FROM chat_run WHERE id = $1",
                    unknown_retrieval_run_id,
                )
            finally:
                await connection.close()

            with self.assertRaisesRegex(Exception, "unknown Graphiti event shape"):
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection,
                            first_class_migration,
                            "upgrade",
                        )
                    )

            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                await connection.execute(
                    "DELETE FROM chat_run WHERE id = $1",
                    unknown_event_run_id,
                )
            finally:
                await connection.close()

            # With only valid v2 rows left the upgrade completes.
            async with engine.begin() as migration_connection:
                await migration_connection.run_sync(
                    lambda sync_connection: self._invoke_migration(
                        sync_connection,
                        first_class_migration,
                        "upgrade",
                    )
                )
            prep_downgraded = False
        finally:
            if prep_downgraded:
                async with engine.begin() as migration_connection:
                    await migration_connection.run_sync(
                        lambda sync_connection: self._invoke_migration(
                            sync_connection,
                            first_class_migration,
                            "upgrade",
                        )
                    )
            await self._restore_current_agent_schema(engine)
            await engine.dispose()

    async def test_same_kb_selector_and_deferred_active_rule(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, embedding_space_id, kb_one = await self.create_foundation(
                connection, suffix="selector-one"
            )
            kb_two = await connection.fetchval(
                """
                INSERT INTO knowledge_base (workspace_id, name)
                VALUES ($1, 'kb-selector-two') RETURNING id
                """,
                workspace_id,
            )
            revision_two = await self.create_revision(
                connection,
                workspace_id,
                embedding_space_id,
                kb_two,
                status="active",
            )

            transaction = connection.transaction()
            await transaction.start()
            await connection.execute(
                "UPDATE knowledge_base SET active_index_revision_id = $1 WHERE id = $2",
                revision_two,
                kb_one,
            )
            with self.assertRaises(asyncpg.ForeignKeyViolationError):
                await transaction.commit()

            transaction = connection.transaction()
            await transaction.start()
            await connection.execute(
                "UPDATE knowledge_base SET provisioned_at = now() WHERE id = $1",
                kb_one,
            )
            with self.assertRaises(asyncpg.CheckViolationError):
                await transaction.commit()

            transaction = connection.transaction()
            await transaction.start()
            revision_one = await self.create_revision(
                connection,
                workspace_id,
                embedding_space_id,
                kb_one,
                status="active",
            )
            await connection.execute(
                """
                UPDATE knowledge_base
                SET active_index_revision_id = $1, provisioned_at = now()
                WHERE id = $2
                """,
                revision_one,
                kb_one,
            )
            await transaction.commit()
        finally:
            await connection.close()

    async def test_concurrent_active_revision_constraint(self) -> None:
        setup = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                setup, suffix="active-race"
            )
        finally:
            await setup.close()

        async def insert_active() -> UUID:
            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                async with connection.transaction():
                    return await self.create_revision(
                        connection,
                        workspace_id,
                        embedding_space_id,
                        kb_id,
                        status="active",
                    )
            finally:
                await connection.close()

        results = await asyncio.gather(
            insert_active(), insert_active(), return_exceptions=True
        )
        successes = [result for result in results if isinstance(result, UUID)]
        failures = [result for result in results if isinstance(result, Exception)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], asyncpg.UniqueViolationError)

    async def test_serving_constraints_are_database_enforced(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                connection, suffix="serving"
            )
            revision_id = await self.create_revision(
                connection, workspace_id, embedding_space_id, kb_id
            )
            document_id, versions = await self.create_document_versions(
                connection, workspace_id, kb_id
            )

            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    """
                    INSERT INTO indexed_document_version (
                        workspace_id, kb_id, document_id, document_version_id,
                        index_revision_id, source_change_seq,
                        build_status, serving_status
                    ) VALUES ($1, $2, $3, $4, $5, 1, 'queued', 'serving')
                    """,
                    workspace_id,
                    kb_id,
                    document_id,
                    versions[0],
                    revision_id,
                )

            indexed_version_id = await connection.fetchval(
                """
                INSERT INTO indexed_document_version (
                    workspace_id, kb_id, document_id, document_version_id,
                    index_revision_id, source_change_seq,
                    build_status, serving_status
                ) VALUES ($1, $2, $3, $4, $5, 1, 'ready', 'serving')
                RETURNING id
                """,
                workspace_id,
                kb_id,
                document_id,
                versions[0],
                revision_id,
            )
            with self.assertRaises(asyncpg.UniqueViolationError):
                await connection.execute(
                    """
                    INSERT INTO indexed_document_version (
                        workspace_id, kb_id, document_id, document_version_id,
                        index_revision_id, source_change_seq,
                        build_status, serving_status
                    ) VALUES ($1, $2, $3, $4, $5, 2, 'ready', 'serving')
                    """,
                    workspace_id,
                    kb_id,
                    document_id,
                    versions[1],
                    revision_id,
                )

            chunk_id = await connection.fetchval(
                """
                INSERT INTO index_chunk (
                    workspace_id, kb_id, indexed_document_version_id, ordinal,
                    content, content_hash, token_count, source_location,
                    unit_key, modality
                ) VALUES (
                    $1, $2, $3, 0, 'test', $4, 1, '{}'::jsonb,
                    'schema-test', 'text'
                )
                RETURNING id
                """,
                workspace_id,
                kb_id,
                indexed_version_id,
                "0" * 64,
            )
            embedding = "[" + ",".join(["1", *("0" for _ in range(1023))]) + "]"
            await connection.execute(
                """
                INSERT INTO vector_record (
                    workspace_id, kb_id, index_chunk_id,
                    embedding_space_id, embedding_dimension,
                    representation_kind, embedding
                ) VALUES ($1, $2, $3, $4, 1024, 'text', $5::vector)
                """,
                workspace_id,
                kb_id,
                chunk_id,
                embedding_space_id,
                embedding,
            )
            distance = await connection.fetchval(
                """
                SELECT embedding <=> $1::vector
                FROM vector_record WHERE index_chunk_id = $2
                  AND embedding_dimension = 1024
                """,
                embedding,
                chunk_id,
            )
            self.assertAlmostEqual(distance, 0.0)
        finally:
            await connection.close()

    async def test_legacy_entity_graph_projection_is_removed(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            for table_name in (
                "index_graph_chunk",
                "graph_entity_mention",
                "graph_relation_assertion",
            ):
                self.assertIsNone(
                    await connection.fetchval("SELECT to_regclass($1)", table_name)
                )
        finally:
            await connection.close()

    async def test_concurrent_source_change_allocation_has_no_gaps(self) -> None:
        setup = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, _, kb_id = await self.create_foundation(
                setup, suffix="source-sequence"
            )
            document_id, versions = await self.create_document_versions(
                setup, workspace_id, kb_id
            )
        finally:
            await setup.close()

        async def allocate(version_id: UUID) -> int:
            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                async with connection.transaction():
                    sequence = await connection.fetchval(
                        """
                        UPDATE knowledge_base
                        SET source_change_seq = source_change_seq + 1
                        WHERE id = $1
                        RETURNING source_change_seq
                        """,
                        kb_id,
                    )
                    await connection.execute(
                        """
                        INSERT INTO source_change (
                            workspace_id, kb_id, source_change_seq,
                            document_id, document_version_id, change_kind
                        ) VALUES ($1, $2, $3, $4, $5, 'upsert')
                        """,
                        workspace_id,
                        kb_id,
                        sequence,
                        document_id,
                        version_id,
                    )
                    return sequence
            finally:
                await connection.close()

        allocated = await asyncio.gather(*(allocate(version) for version in versions))
        self.assertEqual(sorted(allocated), [1, 2])

        check = await asyncpg.connect(MIGRATION_DSN)
        try:
            ledger = await check.fetch(
                """
                SELECT source_change_seq FROM source_change
                WHERE kb_id = $1 ORDER BY source_change_seq
                """,
                kb_id,
            )
        finally:
            await check.close()
        self.assertEqual([row["source_change_seq"] for row in ledger], [1, 2])

    async def test_composite_relation_rejects_cross_target_edges_and_duplicates(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                connection, suffix="composite-relation"
            )
            revision_id = await self.create_revision(
                connection, workspace_id, embedding_space_id, kb_id, status="active"
            )
            document_id, versions = await self.create_document_versions(
                connection, workspace_id, kb_id, count=2
            )

            async def create_target(version_id: UUID, sequence: int) -> UUID:
                return await connection.fetchval(
                    """
                    INSERT INTO indexed_document_version (
                        workspace_id, kb_id, document_id, document_version_id,
                        index_revision_id, source_change_seq,
                        build_status, serving_status
                    ) VALUES ($1, $2, $3, $4, $5, $6, 'processing', 'candidate')
                    RETURNING id
                    """,
                    workspace_id,
                    kb_id,
                    document_id,
                    version_id,
                    revision_id,
                    sequence,
                )

            first_target = await create_target(versions[0], 1)
            second_target = await create_target(versions[1], 2)
            asset_id = await connection.fetchval(
                """
                INSERT INTO index_asset (
                    workspace_id, kb_id, document_id, document_version_id,
                    indexed_document_version_id, asset_key, kind, storage_uri,
                    media_type, checksum_sha256, source_location
                ) VALUES (
                    $1, $2, $3, $4, $5, 'asset-1', 'image', 'local://asset-1',
                    'image/png', $6, '{}'::jsonb
                ) RETURNING id
                """,
                workspace_id,
                kb_id,
                document_id,
                versions[0],
                first_target,
                "a" * 64,
            )

            async def create_chunk(target_id: UUID, ordinal: int, unit_key: str) -> UUID:
                return await connection.fetchval(
                    """
                    INSERT INTO index_chunk (
                        workspace_id, kb_id, indexed_document_version_id,
                        ordinal, unit_key, modality, content, content_hash,
                        token_count, source_location
                    ) VALUES ($1, $2, $3, $4, $5, 'text', 'body', $6, 1, '{}'::jsonb)
                    RETURNING id
                    """,
                    workspace_id,
                    kb_id,
                    target_id,
                    ordinal,
                    unit_key,
                    f"{ordinal:064d}",
                )

            text_chunk = await create_chunk(first_target, 0, "text-1")
            visual_unit = await create_chunk(first_target, 1, "visual-1")
            other_chunk = await create_chunk(second_target, 0, "text-2")
            statement = """
                INSERT INTO index_chunk_asset_relation (
                    workspace_id, kb_id, indexed_document_version_id,
                    chunk_id, visual_unit_id, asset_id, relation_type,
                    confidence_micros, ordinal, provenance, evidence_group_key
                ) VALUES ($1, $2, $3, $4, $5, $6, 'caption_of',
                    1000000, 0, 'author_caption_v2', 'figure-1')
            """
            await connection.execute(
                statement,
                workspace_id,
                kb_id,
                first_target,
                text_chunk,
                visual_unit,
                asset_id,
            )
            with self.assertRaises(asyncpg.UniqueViolationError):
                await connection.execute(
                    statement,
                    workspace_id,
                    kb_id,
                    first_target,
                    text_chunk,
                    visual_unit,
                    asset_id,
                )
            with self.assertRaises(asyncpg.ForeignKeyViolationError):
                await connection.execute(
                    statement,
                    workspace_id,
                    kb_id,
                    first_target,
                    other_chunk,
                    visual_unit,
                    asset_id,
                )
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    """
                    UPDATE index_chunk
                    SET embedding_text = 'body', embedding_text_hash = NULL
                    WHERE id = $1
                    """,
                    text_chunk,
                )
        finally:
            await connection.close()
