from __future__ import annotations

import os


def require_database_test_dsns(*names: str) -> None:
    """Fail closed when database tests are imported without a test database."""

    missing = tuple(name for name in names if not os.environ.get(name))
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(
            "Database integration tests require disposable PostgreSQL DSNs; "
            f"missing: {missing_names}. Run "
            "`PYTHONPATH=src:. .venv/bin/python tools/run_database_tests.py` "
            "or provide DSNs for an already-running disposable test database."
        )
