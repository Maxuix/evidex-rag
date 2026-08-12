from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "rag_kb"
APPLICATION_DIRECTORIES = (
    "services",
    "retrieval",
    "answering",
    "indexing",
    "scheduling",
)
FORBIDDEN_APPLICATION_IMPORTS = (
    "rag_kb.adapters",
    "rag_kb.db",
    "sqlalchemy",
    "pgvector",
    "httpx",
    "openai",
    "langchain_openai",
    "docling",
)
COMPOSITION_MODULES = {
    ROOT / "apps" / "model_asset_runtime.py",
    ROOT / "apps" / "api" / "dependencies.py",
    ROOT / "apps" / "worker" / "dependencies.py",
    ROOT / "apps" / "maintenance" / "dependencies.py",
}
PURE_INITIALIZERS = tuple(
    sorted((SOURCE_ROOT / "adapters").rglob("__init__.py"))
) + (
    SOURCE_ROOT / "document_processing" / "__init__.py",
    SOURCE_ROOT / "indexing" / "__init__.py",
    SOURCE_ROOT / "scheduling" / "__init__.py",
    SOURCE_ROOT / "services" / "__init__.py",
)
ISOLATED_IMPORTS = (
    "rag_kb.ports.files",
    "rag_kb.ports.retrieval",
    "rag_kb.services.files",
    "rag_kb.retrieval.service",
    "rag_kb.answering.agent",
    "rag_kb.scheduling.chat",
)
FORBIDDEN_RUNTIME_ROOTS = {
    "sqlalchemy",
    "pgvector",
    "httpx",
    "openai",
    "langchain_openai",
    "PIL",
    "marko",
    "docling",
    "docling_core",
}


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return tuple(names)


def _matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_application_modules_do_not_import_infrastructure(self) -> None:
        violations: list[str] = []
        for directory in APPLICATION_DIRECTORIES:
            for path in sorted((SOURCE_ROOT / directory).rglob("*.py")):
                for name in _imports(path):
                    if any(
                        _matches(name, forbidden)
                        for forbidden in FORBIDDEN_APPLICATION_IMPORTS
                    ):
                        violations.append(
                            f"{path.relative_to(ROOT)} imports {name}"
                        )
                    if _matches(name, "docling_core") and path != (
                        SOURCE_ROOT / "indexing" / "pipeline.py"
                    ):
                        violations.append(
                            f"{path.relative_to(ROOT)} imports {name}"
                        )

        self.assertEqual(violations, [])

    def test_ports_have_only_runtime_stdlib_and_domain_dependencies(self) -> None:
        violations: list[str] = []
        stdlib = set(sys.stdlib_module_names) | {"__future__"}
        for path in sorted((SOURCE_ROOT / "ports").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            type_checking_nodes = {
                id(child)
                for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
                for child in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            }
            for node in ast.walk(tree):
                if id(node) in type_checking_nodes:
                    continue
                names: tuple[str, ...] = ()
                if isinstance(node, ast.Import):
                    names = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = (node.module,)
                for name in names:
                    if (
                        name.split(".", 1)[0] not in stdlib
                        and not _matches(name, "rag_kb.domain")
                    ):
                        violations.append(
                            f"{path.relative_to(ROOT)} imports {name} at runtime"
                        )

        self.assertEqual(violations, [])

    def test_concrete_adapters_are_imported_only_by_composition_roots(self) -> None:
        violations: list[str] = []
        production_paths = (
            tuple(SOURCE_ROOT.rglob("*.py"))
            + tuple((ROOT / "apps").rglob("*.py"))
        )
        for path in sorted(production_paths):
            if path.is_relative_to(SOURCE_ROOT / "adapters"):
                continue
            for name in _imports(path):
                if _matches(name, "rag_kb.adapters") and path not in COMPOSITION_MODULES:
                    violations.append(f"{path.relative_to(ROOT)} imports {name}")

        self.assertEqual(violations, [])

    def test_boundary_initializers_do_not_import_modules(self) -> None:
        violations: list[str] = []
        for path in PURE_INITIALIZERS:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body):
                violations.append(str(path.relative_to(ROOT)))

        self.assertEqual(violations, [])

    def test_representative_leaf_imports_do_not_load_infrastructure(self) -> None:
        probe = (
            "import importlib, json, sys; "
            "importlib.import_module(sys.argv[1]); "
            "print(json.dumps(sorted({name.split('.')[0] for name in sys.modules})))"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))

        for module in ISOLATED_IMPORTS:
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, "-c", probe, module],
                    cwd=ROOT,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                loaded_roots = set(json.loads(completed.stdout))
                self.assertEqual(
                    sorted(loaded_roots & FORBIDDEN_RUNTIME_ROOTS),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
