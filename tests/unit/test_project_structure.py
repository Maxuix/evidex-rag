from __future__ import annotations

import importlib
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProjectStructureTests(unittest.TestCase):
    def test_required_areas_exist(self) -> None:
        required = [
            "apps/api",
            "apps/worker",
            "apps/web-test",
            "src/rag_kb/schemas",
            "src/rag_kb/domain",
            "src/rag_kb/services",
            "src/rag_kb/workflows",
            "src/rag_kb/repositories",
            "src/rag_kb/uow",
            "src/rag_kb/db",
            "src/rag_kb/adapters",
            "evaluation",
            "tests/unit",
            "tests/contract",
            "tests/integration",
            "tests/e2e",
            "deploy",
        ]

        missing = [path for path in required if not (PROJECT_ROOT / path).is_dir()]

        self.assertEqual(missing, [])

    def test_python_boundaries_are_importable(self) -> None:
        modules = [
            "apps.api",
            "apps.worker",
            "rag_kb.config",
            "rag_kb.schemas",
            "rag_kb.domain",
            "rag_kb.services",
            "rag_kb.workflows",
            "rag_kb.indexing",
            "rag_kb.retrieval",
            "rag_kb.answering",
            "rag_kb.auth",
            "rag_kb.memory",
            "rag_kb.repositories",
            "rag_kb.uow",
            "rag_kb.db",
            "rag_kb.adapters",
            "rag_kb.observability",
        ]

        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
