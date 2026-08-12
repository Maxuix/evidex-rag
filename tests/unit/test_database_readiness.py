from __future__ import annotations

import unittest

from rag_kb.db.readiness import DatabaseReadinessError, check_database_ready


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


class DatabaseReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_revision_is_ready(self) -> None:
        await check_database_ready(  # type: ignore[arg-type]
            _Engine("0011_native_agent_round_limit")
        )

    async def test_missing_or_stale_revision_is_rejected(self) -> None:
        for revision in (None, "0000_old"):
            with self.subTest(revision=revision), self.assertRaises(
                DatabaseReadinessError
            ):
                await check_database_ready(_Engine(revision))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
