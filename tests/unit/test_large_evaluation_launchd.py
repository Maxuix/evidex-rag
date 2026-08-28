from __future__ import annotations

from pathlib import Path
import unittest

from tools.install_large_evaluation_launchd import _plist


class LargeEvaluationLaunchdTests(unittest.TestCase):
    def test_path_state_controls_login_persistence(self) -> None:
        marker = Path("/private/tmp/evaluation-enabled")
        keep_alive = {
            "PathState": {str(marker): True},
        }
        plist = _plist(
            label="com.example.evaluation",
            program_arguments=["/usr/bin/true"],
            working_directory=Path("/private/tmp"),
            stdout_path=Path("/private/tmp/evaluation.out"),
            stderr_path=Path("/private/tmp/evaluation.err"),
            keep_alive=keep_alive,
        )

        self.assertFalse(plist["RunAtLoad"])
        self.assertEqual(plist["KeepAlive"], keep_alive)
        self.assertEqual(plist["ProcessType"], "Background")


if __name__ == "__main__":
    unittest.main()
