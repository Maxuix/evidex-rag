from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch


MIGRATION_MODULE = "rag_kb.db.migrations.versions.0027_agent_v4_default"


class AgentV4DefaultMigrationTests(unittest.TestCase):
    def test_downgrade_rejects_before_any_migration_operation(self) -> None:
        migration = importlib.import_module(MIGRATION_MODULE)

        with patch.object(migration, "op") as operations:
            with self.assertRaisesRegex(
                RuntimeError,
                r"does not support downgrade.*matching backup",
            ):
                migration.downgrade()

        self.assertEqual(operations.mock_calls, [])
