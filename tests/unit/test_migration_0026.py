from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch


MIGRATION_MODULE = (
    "rag_kb.db.migrations.versions.0026_simplify_attempt_ownership"
)


class AttemptOwnershipMigrationTests(unittest.TestCase):
    def test_upgrade_rejects_active_work_before_dropping_columns(self) -> None:
        migration = importlib.import_module(MIGRATION_MODULE)
        with patch.object(migration, "op") as operations:
            result = operations.get_bind.return_value.execute.return_value
            result.mappings.return_value.one.return_value = {
                "chat_idle": True,
                "indexing_idle": False,
                "graph_idle": True,
            }

            with self.assertRaisesRegex(
                RuntimeError,
                "requires idle work: indexing",
            ):
                migration.upgrade()

            operations.drop_column.assert_not_called()


if __name__ == "__main__":
    unittest.main()
