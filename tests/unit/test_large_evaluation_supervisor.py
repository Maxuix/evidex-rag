from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import stat
import unittest

from tools.evaluation_campaign_state import write_private_json
from tools.supervise_large_evaluation import (
    STAGES,
    _new_state,
    _persist_stage,
    _resume_paused_stage,
    _resume_failed_stage,
)


class LargeEvaluationSupervisorTests(unittest.TestCase):
    def test_resume_rearms_only_the_failed_stage_and_preserves_order(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "state.json"
            binding = {"stage_root": str(root / "stages")}
            state = _new_state("config-sha")
            state["status"] = "failed"
            state["current_stage"] = "graph"
            state["stages"]["indexing"].update(
                {"status": "completed", "exit_code": 0}
            )
            state["stages"]["graph"].update(
                {
                    "status": "failed",
                    "attempt_count": 1,
                    "exit_code": 2,
                    "failure_code": "graph_child_exit_2",
                    "failure_type": "ControlledStageFailure",
                    "observed": {"processed_chunk_count": 304},
                }
            )
            state["last_failure"] = {
                "stage": "graph",
                "failure_code": "graph_child_exit_2",
            }
            write_private_json(state_path, state)
            _persist_stage(binding, state_path, state, "indexing")
            _persist_stage(binding, state_path, state, "graph")

            result = _resume_failed_stage(
                binding,
                state_path,
                "config-sha",
                "graph",
            )

            self.assertEqual(result["status"], "resumed")
            resumed = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(resumed["status"], "running")
            self.assertEqual(resumed["current_stage"], "graph")
            self.assertEqual(resumed["stages"]["indexing"]["status"], "completed")
            self.assertEqual(resumed["stages"]["graph"]["status"], "pending")
            self.assertEqual(resumed["stages"]["graph"]["attempt_count"], 1)
            self.assertNotIn("exit_code", resumed["stages"]["graph"])
            self.assertNotIn("failure_code", resumed["stages"]["graph"])
            self.assertTrue(
                all(
                    resumed["stages"][stage]["status"] == "pending"
                    for stage in STAGES[STAGES.index("graph") + 1 :]
                )
            )
            self.assertNotIn("last_failure", resumed)
            stage_checkpoint = root / "stages" / "graph.json"
            self.assertEqual(stat.S_IMODE(stage_checkpoint.stat().st_mode), 0o600)
            self.assertEqual(
                json.loads(stage_checkpoint.read_text(encoding="utf-8"))["record"][
                    "status"
                ],
                "pending",
            )
            self.assertTrue(
                any(
                    event.get("event") == "stage_resume_requested"
                    for event in resumed["events"]
                )
            )

    def test_resume_rearms_only_the_paused_stage_and_preserves_attempts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "state.json"
            binding = {"stage_root": str(root / "stages")}
            state = _new_state("config-sha")
            state["status"] = "paused"
            state["current_stage"] = "graph"
            state["stages"]["indexing"].update(
                {"status": "completed", "attempt_count": 1, "exit_code": 0}
            )
            state["stages"]["graph"].update(
                {
                    "status": "paused",
                    "attempt_count": 2,
                    "resume_count": 1,
                    "paused_at": "2026-08-27T06:40:00+00:00",
                    "pause_reason": "operator_request",
                    "paused_child_pid": 31314,
                    "paused_exit_file": str(root / "stages" / "graph-attempt-2-exit.json"),
                    "paused_exit_code": 143,
                    "observed": {"processed_chunk_count": 320},
                }
            )
            write_private_json(state_path, state)
            _persist_stage(binding, state_path, state, "indexing")
            _persist_stage(binding, state_path, state, "graph")

            result = _resume_paused_stage(binding, state_path, "config-sha")

            self.assertEqual(result["status"], "resumed")
            resumed = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(resumed["status"], "running")
            self.assertEqual(resumed["current_stage"], "graph")
            self.assertEqual(resumed["stages"]["indexing"]["status"], "completed")
            self.assertEqual(resumed["stages"]["graph"]["status"], "pending")
            self.assertEqual(resumed["stages"]["graph"]["attempt_count"], 2)
            self.assertEqual(resumed["stages"]["graph"]["resume_count"], 2)
            self.assertNotIn("paused_child_pid", resumed["stages"]["graph"])
            self.assertNotIn("paused_exit_code", resumed["stages"]["graph"])
            self.assertTrue(
                all(
                    resumed["stages"][stage]["status"] == "pending"
                    for stage in STAGES[STAGES.index("graph") + 1 :]
                )
            )
            stage_checkpoint = root / "stages" / "graph.json"
            self.assertEqual(
                json.loads(stage_checkpoint.read_text(encoding="utf-8"))["record"][
                    "status"
                ],
                "pending",
            )


if __name__ == "__main__":
    unittest.main()
