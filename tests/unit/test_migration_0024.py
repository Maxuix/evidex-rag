from __future__ import annotations

import importlib
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from alembic import command
from alembic.config import Config


MIGRATION_MODULE = (
    "rag_kb.db.migrations.versions.0024_remove_agent_deadline_reserve"
)
MIGRATION_REVISION = "0024_remove_agent_deadline_reserve"
PREVIOUS_REVISION = "0023_agent_trace_diagnostics"


class RemoveAgentDeadlineReserveMigrationTests(unittest.TestCase):
    def test_downgrade_rejects_before_any_migration_operation(self) -> None:
        migration = importlib.import_module(MIGRATION_MODULE)

        with patch.object(migration, "op") as operations:
            with self.assertRaisesRegex(
                RuntimeError,
                r"does not support downgrade.*matching backup",
            ):
                migration.downgrade()

        self.assertEqual(operations.mock_calls, [])

    def test_offline_alembic_downgrade_fails_without_version_sql(self) -> None:
        output = io.StringIO()
        # Exercise the real migration environment without fileConfig disabling
        # application loggers in the shared test process.
        config = Config()
        config.set_main_option(
            "script_location",
            str(Path(__file__).resolve().parents[2] / "src/rag_kb/db/migrations"),
        )

        with (
            patch.dict(
                os.environ,
                {
                    "RAG_KB__DATABASE__MIGRATION_DSN": (
                        "postgresql+asyncpg://offline:offline@127.0.0.1:1/unused"
                    )
                },
            ),
            patch("sys.stdout", output),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"does not support downgrade.*matching backup",
            ):
                command.downgrade(
                    config,
                    f"{MIGRATION_REVISION}:{PREVIOUS_REVISION}",
                    sql=True,
                )

        rendered_sql = output.getvalue()
        self.assertNotIn("UPDATE alembic_version", rendered_sql)
        self.assertNotRegex(
            rendered_sql,
            r"(?im)^\s*(?:ALTER|CREATE|DROP|INSERT|UPDATE|DELETE)\b",
        )
        self.assertNotIn("COMMIT;", rendered_sql)


if __name__ == "__main__":
    unittest.main()
