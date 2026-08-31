from __future__ import annotations

import asyncio
import os
import unittest
from uuid import UUID

import asyncpg
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, InvalidRequestError

from tests.integration.db import require_database_test_dsns
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.domain import (
    ModelKind,
    ModelProviderProtocol,
    ModelValidationStatus,
    Workspace,
)
from rag_kb.uow import (
    TransactionMode,
    execute_in_transaction,
)
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")
WORKSPACE_ONE = UUID("01900000-0000-7000-8000-000000000101")
WORKSPACE_TWO = UUID("01900000-0000-7000-8000-000000000102")

require_database_test_dsns(
    "RAG_KB_TEST_MIGRATION_DSN",
    "RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN",
)
class AsyncDataAccessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute("TRUNCATE TABLE workspace CASCADE")
        finally:
            await connection.close()

        self.database = create_database_resources(
            RUNTIME_SQLALCHEMY_DSN,
            pool_size=2,
            max_overflow=0,
            process=DatabaseProcess.WORKER,
        )
        self.factory = SqlAlchemyUnitOfWorkFactory(
            self.database.sessions,
            WORKSPACE_ONE,
        )

    async def asyncTearDown(self) -> None:
        await self.database.close()

    async def test_commit_persists_without_implicit_followup_transaction(self) -> None:
        async with self.factory() as unit_of_work:
            created = await unit_of_work.workspaces.add("committed-workspace")
            await unit_of_work.commit()
            with self.assertRaises(InvalidRequestError):
                await unit_of_work.workspaces.get()

        async with self.factory() as unit_of_work:
            loaded = await unit_of_work.workspaces.get()
            await unit_of_work.rollback()

        self.assertEqual(loaded, created)

    async def test_runtime_connections_apply_bounded_server_settings(self) -> None:
        async with self.database.sessions() as session:
            statement_timeout = (
                await session.execute(text("SHOW statement_timeout"))
            ).scalar_one()
            lock_timeout = (
                await session.execute(text("SHOW lock_timeout"))
            ).scalar_one()
            idle_timeout = (
                await session.execute(
                    text("SHOW idle_in_transaction_session_timeout")
                )
            ).scalar_one()
            application_name = (
                await session.execute(text("SHOW application_name"))
            ).scalar_one()
            await session.rollback()

        self.assertEqual(statement_timeout, "1min")
        self.assertEqual(lock_timeout, "5s")
        self.assertEqual(idle_timeout, "30s")
        self.assertEqual(application_name, "rag-kb-worker")
        self.assertEqual(self.database.engine.pool.checkedout(), 0)

    async def test_uncommitted_and_failed_work_rolls_back(self) -> None:
        async with self.factory() as unit_of_work:
            await unit_of_work.workspaces.add("uncommitted")

        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            async with self.factory() as unit_of_work:
                await unit_of_work.workspaces.add("failed")
                raise RuntimeError("injected failure")

        async with self.factory() as unit_of_work:
            self.assertIsNone(await unit_of_work.workspaces.get())
            await unit_of_work.rollback()

    async def test_unverified_model_revision_persists_null_validation_snapshot(
        self,
    ) -> None:
        async with self.factory() as unit_of_work:
            await unit_of_work.workspaces.add("model-settings-workspace")
            provider = await unit_of_work.model_settings.create_provider(
                name="Embedding provider",
                protocol=ModelProviderProtocol.OPENAI_COMPATIBLE,
                base_url="https://provider.invalid/v1",
                secret_reference="test-secret-reference",
                timeout_seconds=30,
                max_retries=0,
                max_concurrency=1,
                configuration_fingerprint="sha256:provider",
            )
            profile = await unit_of_work.model_settings.create_profile(
                provider=provider,
                name="Embedding model",
                kind=ModelKind.TEXT_EMBEDDING,
                model="embedding-model",
                configuration={
                    "type": "embedding",
                    "dimension": 1024,
                    "max_batch_size": 10,
                },
                configuration_fingerprint="sha256:profile-1",
                capability_fingerprint="sha256:capability-1",
                compatibility_fingerprint=None,
                validation_status=ModelValidationStatus.UNVERIFIED,
            )
            updated = await unit_of_work.model_settings.update_profile(
                profile.profile.id,
                provider=provider,
                name=profile.profile.name,
                enabled=True,
                model=profile.current_revision.model,
                configuration={
                    "type": "embedding",
                    "dimension": "auto",
                    "max_batch_size": 10,
                },
                configuration_fingerprint="sha256:profile-2",
                capability_fingerprint="sha256:capability-2",
                compatibility_fingerprint=None,
                validation_status=ModelValidationStatus.UNVERIFIED,
            )
            assert updated is not None
            await unit_of_work.commit()

        async with self.database.sessions() as session:
            snapshot_is_null = await session.scalar(
                text(
                    "SELECT validation_snapshot IS NULL "
                    "FROM model_profile_revision WHERE id = :revision_id"
                ),
                {"revision_id": updated.current_revision.id},
            )
            await session.rollback()

        self.assertEqual(updated.current_revision.revision, 2)
        self.assertIsNone(updated.current_revision.validation_snapshot)
        self.assertTrue(snapshot_is_null)

    async def test_each_concurrent_command_gets_an_independent_session(self) -> None:
        second_factory = SqlAlchemyUnitOfWorkFactory(
            self.database.sessions,
            WORKSPACE_TWO,
        )

        async def create(factory, name: str) -> Workspace:
            async with factory() as unit_of_work:
                workspace = await unit_of_work.workspaces.add(name)
                await unit_of_work.commit()
                return workspace

        first, second = await asyncio.gather(
            create(self.factory, "first"),
            create(second_factory, "second"),
        )
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(self.database.engine.pool.checkedout(), 0)

    async def test_repository_is_bound_to_one_workspace_scope(self) -> None:
        second_factory = SqlAlchemyUnitOfWorkFactory(
            self.database.sessions,
            WORKSPACE_TWO,
        )
        async with self.factory() as unit_of_work:
            first = await unit_of_work.workspaces.add("first-workspace")
            self.assertEqual(unit_of_work.workspace_id, WORKSPACE_ONE)
            await unit_of_work.commit()
        async with second_factory() as unit_of_work:
            second = await unit_of_work.workspaces.add("second-workspace")
            self.assertEqual(unit_of_work.workspace_id, WORKSPACE_TWO)
            await unit_of_work.commit()

        async with self.factory() as unit_of_work:
            self.assertEqual(await unit_of_work.workspaces.get(), first)
            renamed = await unit_of_work.workspaces.rename("renamed-first")
            await unit_of_work.commit()
        async with second_factory() as unit_of_work:
            loaded_second = await unit_of_work.workspaces.get()
            await unit_of_work.rollback()

        self.assertIsNotNone(renamed)
        assert renamed is not None
        self.assertEqual(renamed.id, WORKSPACE_ONE)
        self.assertEqual(renamed.name, "renamed-first")
        self.assertEqual(loaded_second, second)

    async def test_transaction_helper_releases_connection_before_external_io(
        self,
    ) -> None:
        async def persist(unit_of_work) -> Workspace:
            return await unit_of_work.workspaces.add("before-external-io")

        created = await execute_in_transaction(self.factory, persist)
        self.assertEqual(self.database.engine.pool.checkedout(), 0)

        external_io_observed = False

        async def external_io() -> None:
            nonlocal external_io_observed
            await asyncio.sleep(0)
            external_io_observed = True

        await external_io()
        self.assertTrue(external_io_observed)

        async with self.factory() as unit_of_work:
            self.assertEqual(await unit_of_work.workspaces.get(), created)
            await unit_of_work.rollback()

    async def test_repeatable_read_only_mode_rejects_writes(self) -> None:
        isolation_level = None
        transaction_read_only = None
        with self.assertRaises(DBAPIError):
            async with self.factory(
                mode=TransactionMode.REPEATABLE_READ_ONLY,
            ) as unit_of_work:
                session = unit_of_work._session
                assert session is not None
                isolation_level = await session.scalar(
                    text("SHOW transaction_isolation")
                )
                transaction_read_only = await session.scalar(
                    text("SHOW transaction_read_only")
                )
                await unit_of_work.workspaces.add("read-only-must-fail")

        self.assertEqual(isolation_level, "repeatable read")
        self.assertEqual(transaction_read_only, "on")

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            count = await connection.fetchval(
                "SELECT count(*) FROM workspace WHERE name = 'read-only-must-fail'"
            )
        finally:
            await connection.close()
        self.assertEqual(count, 0)
