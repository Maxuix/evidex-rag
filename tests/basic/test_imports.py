from __future__ import annotations

import importlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class BasicProjectTests(unittest.TestCase):
    def test_application_modules_import(self) -> None:
        modules = (
            "apps.api.main",
            "apps.worker.main",
            "rag_kb.config",
            "rag_kb.domain",
            "rag_kb.indexing",
            "rag_kb.retrieval",
            "rag_kb.answering",
            "rag_kb.db",
            "langchain_core",
            "langchain_openai",
        )
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_local_entrypoints_exist(self) -> None:
        self.assertTrue((ROOT / "Makefile").is_file())
        self.assertTrue((ROOT / "tools/reset_local.py").is_file())
        self.assertTrue((ROOT / "tools/smoke_local.py").is_file())
        self.assertTrue((ROOT / "examples/demo-document.md").is_file())


if __name__ == "__main__":
    unittest.main()
