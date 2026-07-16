from __future__ import annotations

import os
import unittest
from unittest import mock

from tools.run_compose_smoke import inherited_smoke_environment


class ComposeSmokeTests(unittest.TestCase):
    def test_host_environment_does_not_inherit_project_runtime_inputs(self) -> None:
        contaminated = {
            "PATH": "/usr/bin",
            "RAG_KB_ENV_FILE": ".env.production",
            "RAG_KB__PROVIDER__API_KEY": "must-not-survive",
            "POSTGRES_ADMIN_PASSWORD": "real-admin-password",
            "RAG_KB_MIGRATION_PASSWORD": "real-migration-password",
            "RAG_KB_RUNTIME_PASSWORD": "real-runtime-password",
        }
        with mock.patch.dict(os.environ, contaminated, clear=True):
            environment = inherited_smoke_environment()

        self.assertEqual(environment, {"PATH": "/usr/bin"})


if __name__ == "__main__":
    unittest.main()
