from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

from rag_kb.repositories import WorkspaceRepository
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASYNC_DATA_ACCESS_FILES = (
    PROJECT_ROOT / "src/rag_kb/db/session.py",
    PROJECT_ROOT / "src/rag_kb/repositories/workspaces.py",
    PROJECT_ROOT / "src/rag_kb/repositories/sqlalchemy.py",
    PROJECT_ROOT / "src/rag_kb/uow/contracts.py",
    PROJECT_ROOT / "src/rag_kb/uow/operations.py",
    PROJECT_ROOT / "src/rag_kb/uow/sqlalchemy.py",
)
FORBIDDEN_SQLALCHEMY_SYMBOLS = {"Session", "create_engine", "sessionmaker"}


class AsyncDataAccessContractTests(unittest.TestCase):
    def test_public_contracts_are_async(self) -> None:
        for contract, methods in (
            (WorkspaceRepository, ("add", "get")),
            (UnitOfWork, ("__aenter__", "__aexit__", "commit", "rollback")),
        ):
            for method in methods:
                with self.subTest(contract=contract.__name__, method=method):
                    self.assertTrue(
                        inspect.iscoroutinefunction(getattr(contract, method))
                    )

        self.assertTrue(callable(UnitOfWorkFactory))

    def test_implementation_has_no_synchronous_sqlalchemy_path(self) -> None:
        imported_symbols: set[str] = set()
        for path in ASYNC_DATA_ACCESS_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("sqlalchemy"):
                        imported_symbols.update(alias.name for alias in node.names)
                elif isinstance(node, ast.Import):
                    imported_symbols.update(alias.name for alias in node.names)

        self.assertEqual(
            imported_symbols & FORBIDDEN_SQLALCHEMY_SYMBOLS,
            set(),
        )


if __name__ == "__main__":
    unittest.main()
