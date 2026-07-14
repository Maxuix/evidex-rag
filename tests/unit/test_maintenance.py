from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag_kb.config import DeploymentProfile
from tools.reset_local import CONFIRMATION, main, reset_command


class LocalResetTests(unittest.TestCase):
    def test_reset_command_is_project_scoped_and_removes_volumes(self) -> None:
        self.assertEqual(
            reset_command(env_file="local.env", project_name="rag-kb-local"),
            [
                "docker",
                "compose",
                "--env-file",
                "local.env",
                "--profile",
                "tools",
                "--project-name",
                "rag-kb-local",
                "down",
                "--volumes",
                "--remove-orphans",
            ],
        )

    def test_wrong_confirmation_stops_before_external_action(self) -> None:
        with (
            patch.object(sys, "argv", ["reset_local.py", "--confirm", "wrong"]),
            patch("tools.reset_local.load_settings") as load,
            patch("tools.reset_local.subprocess.run") as run,
            self.assertRaises(SystemExit),
        ):
            main()
        load.assert_not_called()
        run.assert_not_called()

    def test_exact_confirmation_allows_only_development(self) -> None:
        settings = SimpleNamespace(
            app=SimpleNamespace(deployment_profile=DeploymentProfile.DEVELOPMENT)
        )
        with (
            patch.object(
                sys,
                "argv",
                ["reset_local.py", "--confirm", CONFIRMATION],
            ),
            patch("tools.reset_local.load_settings", return_value=settings),
            patch("tools.reset_local.subprocess.run") as run,
        ):
            self.assertEqual(main(), 0)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[-3:], ["down", "--volumes", "--remove-orphans"])


if __name__ == "__main__":
    unittest.main()
