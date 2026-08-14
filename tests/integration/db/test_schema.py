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

from rag_kb.db.readiness import check_database_ready


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")


@unittest.skipUnless(
    MIGRATION_DSN and RUNTIME_DSN and RUNTIME_SQLALCHEMY_DSN,
    "database integration DSNs are not configured",
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
            self.assertEqual(revision, "0012_entity_graph_rag")
        finally:
            await runtime.close()

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

    async def test_native_agent_columns_use_only_the_round_limit_v2(self) -> None:
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
                  AND conname LIKE '%ck_chat_run_agent_%_v2'
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
        self.assertIn("native_tool_calling_agent_v2", columns[0]["column_default"])
        self.assertIn("max_model_rounds", columns[0]["column_default"])
        self.assertNotIn("retrieval_calls", columns[0]["column_default"])
        self.assertNotIn("calculation_calls", columns[0]["column_default"])
        self.assertNotIn("evidence_refs", columns[0]["column_default"])
        self.assertEqual(len(constraints), 2)
        self.assertEqual(
            [row["conname"] for row in constraints],
            [
                "ck_chat_run_agent_configuration_v2",
                "ck_chat_run_agent_trace_v2",
            ],
        )
        self.assertTrue(all("pg_column_size" in row["definition"] for row in constraints))

    async def test_native_agent_round_limit_migration_preserves_trace_facts(
        self,
    ) -> None:
        assert MIGRATION_DSN is not None
        connection = await asyncpg.connect(MIGRATION_DSN)
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
        engine = create_async_engine(
            MIGRATION_DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
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

    async def test_native_agent_round_limit_downgrade_rejects_forced_round_usage(
        self,
    ) -> None:
        assert MIGRATION_DSN is not None
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
        engine = create_async_engine(
            MIGRATION_DSN.replace("postgresql://", "postgresql+asyncpg://", 1)
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
                await connection.fetchval("SELECT version_num FROM alembic_version"),
                "0012_entity_graph_rag",
            )
        finally:
            await connection.close()

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

    async def test_entity_graph_scope_foreign_keys_and_cascades(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                connection, suffix="entity-graph-schema"
            )
            revision_id = await self.create_revision(
                connection, workspace_id, embedding_space_id, kb_id, status="active"
            )
            document_id, versions = await self.create_document_versions(
                connection, workspace_id, kb_id, count=1
            )
            target_id = await connection.fetchval(
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
            chunk_id = await connection.fetchval(
                """
                INSERT INTO index_chunk (
                    workspace_id, kb_id, indexed_document_version_id,
                    ordinal, unit_key, modality, content, content_hash,
                    token_count, source_location
                ) VALUES (
                    $1, $2, $3, 0, 'entity-graph-schema', 'text',
                    'Atlas connects to Apollo', $4, 4, '{}'::jsonb
                )
                RETURNING id
                """,
                workspace_id,
                kb_id,
                target_id,
                "a" * 64,
            )
            build_id = uuid4()
            await connection.execute(
                """
                INSERT INTO knowledge_base_graph_config (
                    kb_id, workspace_id, status, build_id, extractor_version
                ) VALUES ($1, $2, 'disabled', $3, 'entity_graph_v1')
                """,
                kb_id,
                workspace_id,
                build_id,
            )
            await connection.execute(
                """
                INSERT INTO index_graph_chunk (
                    workspace_id, kb_id, build_id, index_chunk_id,
                    content_hash, extractor_version, result_status,
                    entity_count, relation_count
                ) VALUES ($1, $2, $3, $4, $5, 'entity_graph_v1',
                          'extracted', 2, 1)
                """,
                workspace_id,
                kb_id,
                build_id,
                chunk_id,
                "a" * 64,
            )
            await connection.execute(
                """
                INSERT INTO graph_entity_mention (
                    workspace_id, kb_id, build_id, index_chunk_id,
                    mention_id, ordinal, entity_type, surface,
                    normalized_surface, surface_start, surface_end, entity_key
                ) VALUES
                    ($1, $2, $3, $4, 'm-atlas', 0, 'organization',
                     'Atlas', 'atlas', 0, 5, $5),
                    ($1, $2, $3, $4, 'm-apollo', 1, 'system',
                     'Apollo', 'apollo', 18, 24, $6)
                """,
                workspace_id,
                kb_id,
                build_id,
                chunk_id,
                "b" * 64,
                "c" * 64,
            )
            await connection.execute(
                """
                INSERT INTO graph_relation_assertion (
                    workspace_id, kb_id, build_id, index_chunk_id,
                    relation_id, ordinal, subject_mention_id,
                    object_mention_id, subject_entity_key,
                    object_entity_key, predicate, normalized_predicate,
                    support_start, support_end
                ) VALUES (
                    $1, $2, $3, $4, 'r-atlas-apollo', 0, 'm-atlas',
                    'm-apollo', $5, $6, 'connects to', 'connects_to', 0, 24
                )
                """,
                workspace_id,
                kb_id,
                build_id,
                chunk_id,
                "b" * 64,
                "c" * 64,
            )

            other_kb_id = await connection.fetchval(
                """
                INSERT INTO knowledge_base (workspace_id, name)
                VALUES ($1, 'entity-graph-other-kb')
                RETURNING id
                """,
                workspace_id,
            )
            with self.assertRaises(asyncpg.ForeignKeyViolationError):
                await connection.execute(
                    """
                    INSERT INTO index_graph_chunk (
                        workspace_id, kb_id, build_id, index_chunk_id,
                        content_hash, extractor_version, result_status
                    ) VALUES ($1, $2, $3, $4, $5, 'entity_graph_v1', 'empty')
                    """,
                    workspace_id,
                    other_kb_id,
                    build_id,
                    chunk_id,
                    "a" * 64,
                )

            self.assertEqual(
                await connection.fetchval(
                    "SELECT count(*) FROM graph_entity_mention WHERE index_chunk_id = $1",
                    chunk_id,
                ),
                2,
            )
            self.assertEqual(
                await connection.fetchval(
                    "SELECT count(*) FROM graph_relation_assertion WHERE index_chunk_id = $1",
                    chunk_id,
                ),
                1,
            )

            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    """
                    INSERT INTO graph_relation_assertion (
                        workspace_id, kb_id, build_id, index_chunk_id,
                        relation_id, ordinal, subject_mention_id,
                        object_mention_id, subject_entity_key,
                        object_entity_key, predicate, normalized_predicate,
                        support_start, support_end
                    ) VALUES (
                        $1, $2, $3, $4, 'r-self', 1, 'm-atlas',
                        'm-atlas', $5, $5, 'is', 'is', 0, 5
                    )
                    """,
                    workspace_id,
                    kb_id,
                    build_id,
                    chunk_id,
                    "b" * 64,
                )

            await connection.execute("DELETE FROM index_chunk WHERE id = $1", chunk_id)
            self.assertEqual(
                await connection.fetchval(
                    "SELECT count(*) FROM index_graph_chunk WHERE index_chunk_id = $1",
                    chunk_id,
                ),
                0,
            )
            self.assertEqual(
                await connection.fetchval(
                    "SELECT count(*) FROM graph_entity_mention WHERE index_chunk_id = $1",
                    chunk_id,
                ),
                0,
            )
            self.assertEqual(
                await connection.fetchval(
                    "SELECT count(*) FROM graph_relation_assertion WHERE index_chunk_id = $1",
                    chunk_id,
                ),
                0,
            )
            await connection.execute("DELETE FROM knowledge_base WHERE id = $1", kb_id)
            self.assertEqual(
                await connection.fetchval(
                    "SELECT count(*) FROM knowledge_base_graph_config WHERE kb_id = $1",
                    kb_id,
                ),
                0,
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
