from __future__ import annotations

import asyncio
import os
import unittest
from uuid import uuid4

import asyncpg
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.domain import Workspace
from rag_kb.uow import (
    TransactionMode,
    UnitOfWorkConcurrencyError,
    UnitOfWorkPurpose,
    UnitOfWorkStateError,
    execute_in_transaction,
)
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")


@unittest.skipUnless(
    MIGRATION_DSN and RUNTIME_SQLALCHEMY_DSN,
    "database integration DSNs are not configured",
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
        self.factory = SqlAlchemyUnitOfWorkFactory(self.database.sessions)

    async def asyncTearDown(self) -> None:
        await self.database.close()

    async def test_commit_persists_and_finalizes_repository(self) -> None:
        async with self.factory(purpose=UnitOfWorkPurpose.REQUEST) as unit_of_work:
            created = await unit_of_work.workspaces.add("committed-workspace")
            repository = unit_of_work.workspaces
            await unit_of_work.commit()
            with self.assertRaises(UnitOfWorkStateError):
                await repository.get(created.id)

        async with self.factory(purpose=UnitOfWorkPurpose.REQUEST) as unit_of_work:
            loaded = await unit_of_work.workspaces.get(created.id)
            await unit_of_work.rollback()

        self.assertEqual(loaded, created)

    async def test_uncommitted_and_failed_work_rolls_back(self) -> None:
        async with self.factory() as unit_of_work:
            uncommitted = await unit_of_work.workspaces.add("uncommitted")

        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            async with self.factory() as unit_of_work:
                failed = await unit_of_work.workspaces.add("failed")
                raise RuntimeError("injected failure")

        async with self.factory() as unit_of_work:
            self.assertIsNone(await unit_of_work.workspaces.get(uncommitted.id))
            self.assertIsNone(await unit_of_work.workspaces.get(failed.id))
            await unit_of_work.rollback()

    async def test_unit_of_work_cannot_cross_asyncio_task_boundary(self) -> None:
        async with self.factory(
            purpose=UnitOfWorkPurpose.HEARTBEAT
        ) as unit_of_work:
            async def use_from_child_task() -> None:
                await unit_of_work.workspaces.get(uuid4())

            with self.assertRaises(UnitOfWorkConcurrencyError):
                await asyncio.create_task(use_from_child_task())
            await unit_of_work.rollback()

    async def test_each_concurrent_command_gets_an_independent_session(self) -> None:
        async def create(name: str) -> Workspace:
            async with self.factory(
                purpose=UnitOfWorkPurpose.COMMAND
            ) as unit_of_work:
                workspace = await unit_of_work.workspaces.add(name)
                await unit_of_work.commit()
                return workspace

        first, second = await asyncio.gather(create("first"), create("second"))
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(self.database.engine.pool.checkedout(), 0)

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
            self.assertEqual(await unit_of_work.workspaces.get(created.id), created)
            await unit_of_work.rollback()

    async def test_repeatable_read_only_mode_rejects_writes(self) -> None:
        isolation_level = None
        transaction_read_only = None
        with self.assertRaises(DBAPIError):
            async with self.factory(
                purpose=UnitOfWorkPurpose.READ_SNAPSHOT,
                mode=TransactionMode.REPEATABLE_READ_ONLY,
            ) as unit_of_work:
                session = unit_of_work._require_session()
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
