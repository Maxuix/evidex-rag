from __future__ import annotations

import unittest
from uuid import UUID

from rag_kb.db.readiness import (
    EXPECTED_REVISION,
    DatabaseReadinessError,
    check_database_ready,
    ensure_local_workspace,
)


class _Connection:
    def __init__(self, revision: str | None) -> None:
        self.revision = revision

    async def scalar(self, statement):
        del statement
        return self.revision


class _ConnectionContext:
    def __init__(self, revision: str | None) -> None:
        self.connection = _Connection(revision)

    async def __aenter__(self) -> _Connection:
        return self.connection

    async def __aexit__(self, *args) -> None:
        del args


class _Engine:
    def __init__(self, revision: str | None) -> None:
        self.revision = revision

    def connect(self) -> _ConnectionContext:
        return _ConnectionContext(self.revision)


class _WriteConnection:
    def __init__(self) -> None:
        self.parameters = None

    async def execute(self, statement, parameters) -> None:
        self.statement = str(statement)
        self.parameters = parameters


class _WriteContext:
    def __init__(self, connection: _WriteConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _WriteConnection:
        return self.connection

    async def __aexit__(self, *args) -> None:
        del args


class _WriteEngine:
    def __init__(self) -> None:
        self.connection = _WriteConnection()

    def begin(self) -> _WriteContext:
        return _WriteContext(self.connection)


class DatabaseReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_revision_is_ready(self) -> None:
        await check_database_ready(  # type: ignore[arg-type]
            _Engine(EXPECTED_REVISION)
        )

    async def test_missing_or_stale_revision_is_rejected(self) -> None:
        for revision in (None, "0000_old"):
            with self.subTest(revision=revision), self.assertRaises(
                DatabaseReadinessError
            ):
                await check_database_ready(_Engine(revision))  # type: ignore[arg-type]

    async def test_local_workspace_bootstrap_is_idempotent_insert(self) -> None:
        workspace_id = UUID("01900000-0000-7000-8000-000000000001")
        engine = _WriteEngine()

        await ensure_local_workspace(engine, workspace_id)  # type: ignore[arg-type]

        self.assertIn("ON CONFLICT (id) DO NOTHING", engine.connection.statement)
        self.assertEqual(
            engine.connection.parameters,
            {
                "workspace_id": workspace_id,
                "name": f"local-{workspace_id}",
            },
        )


if __name__ == "__main__":
    unittest.main()
