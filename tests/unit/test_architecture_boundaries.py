from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.check_architecture import BoundaryConfig, check_project


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = BoundaryConfig.load(PROJECT_ROOT / "architecture.toml")


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_repository_satisfies_declared_boundaries(self) -> None:
        self.assertEqual(check_project(PROJECT_ROOT, CONFIG), [])

    def test_forbidden_dependency_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "src/rag_kb/domain"
            adapters = root / "src/rag_kb/adapters"
            api = root / "apps/api"
            worker = root / "apps/worker"
            domain.mkdir(parents=True)
            adapters.mkdir(parents=True)
            api.mkdir(parents=True)
            worker.mkdir(parents=True)
            (domain / "bad.py").write_text(
                "from rag_kb import adapters\n", encoding="utf-8"
            )

            violations = check_project(root, CONFIG)

        self.assertEqual(len(violations), 1)
        self.assertIn(
            "rag_kb.domain must not depend on rag_kb.adapters",
            violations[0].message,
        )

    def test_allowed_dependency_is_not_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = root / "src/rag_kb/services"
            domain = root / "src/rag_kb/domain"
            api = root / "apps/api"
            worker = root / "apps/worker"
            service.mkdir(parents=True)
            domain.mkdir(parents=True)
            api.mkdir(parents=True)
            worker.mkdir(parents=True)
            (service / "example.py").write_text(
                "from rag_kb.domain import documents\n", encoding="utf-8"
            )

            violations = check_project(root, CONFIG)

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
