from __future__ import annotations

import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import stat
import tarfile
import unittest

import tools.supervise_large_evaluation as supervisor_module

from tools.evaluation_campaign_state import write_private_json
from tools.supervise_large_evaluation import (
    STAGES,
    SupervisorError,
    _new_state,
    _persist_stage,
    _progress_signature,
    _interrupt_stage_for_shutdown,
    _resume_paused_stage,
    _resume_failed_stage,
    _validate_state,
    _validate_transition_evidence,
    _write_analysis_archive,
)


class LargeEvaluationSupervisorTests(unittest.TestCase):
    def test_sigterm_arms_login_resume_while_sigusr1_requests_pause(self) -> None:
        supervisor_module._pause_requested = False
        supervisor_module._shutdown_requested = False
        try:
            supervisor_module._handle_signal(signal.SIGTERM, None)
            self.assertTrue(supervisor_module._shutdown_requested)
            self.assertFalse(supervisor_module._pause_requested)
            if hasattr(signal, "SIGUSR1"):
                supervisor_module._handle_signal(signal.SIGUSR1, None)
                self.assertTrue(supervisor_module._pause_requested)
        finally:
            supervisor_module._pause_requested = False
            supervisor_module._shutdown_requested = False

    def test_host_shutdown_rearms_current_stage_without_losing_progress(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "state.json"
            binding = {"stage_root": str(root / "stages")}
            state = _new_state("config-sha")
            state["current_stage"] = "graph"
            state["stages"]["indexing"].update(
                {"status": "completed", "attempt_count": 1}
            )
            state["stages"]["graph"].update(
                {
                    "status": "running",
                    "attempt_count": 2,
                    "child_pid": 999999,
                    "observed": {"processed_chunk_count": 328},
                }
            )

            _interrupt_stage_for_shutdown(
                binding,
                state_path,
                state,
                "graph",
                exit_value={"exit_code": 143},
            )

            self.assertEqual(state["status"], "running")
            self.assertEqual(state["stages"]["graph"]["status"], "pending")
            self.assertEqual(state["stages"]["graph"]["attempt_count"], 2)
            self.assertEqual(
                state["stages"]["graph"]["observed"]["processed_chunk_count"],
                328,
            )
            self.assertEqual(
                state["stages"]["graph"]["interrupted_exit_code"], 143
            )
            self.assertEqual(
                stat.S_IMODE((root / "stages" / "graph.json").stat().st_mode),
                0o600,
            )

    def test_analysis_archive_is_private_and_self_contained(self) -> None:
        runtime_root = Path(__file__).resolve().parents[2] / ".runtime"
        runtime_root.mkdir(mode=0o700, exist_ok=True)
        with TemporaryDirectory(dir=runtime_root) as temporary:
            root = Path(temporary)
            artifact = root / "checkpoint.json"
            artifact.write_text('{"status":"completed"}\n', encoding="utf-8")
            artifact.chmod(0o600)
            archive = root / "analysis-bundle.tar.gz"

            sha256 = _write_analysis_archive(archive, (artifact,))

            self.assertEqual(len(sha256), 64)
            self.assertEqual(stat.S_IMODE(archive.stat().st_mode), 0o600)
            with tarfile.open(archive, "r:gz") as handle:
                names = handle.getnames()
            self.assertEqual(len(names), 1)
            self.assertTrue(names[0].endswith("checkpoint.json"))

    def test_provider_heartbeat_does_not_hide_a_stalled_attempt(self) -> None:
        first = {
            "status": "retry_wait",
            "completed_providers": [],
            "attempts": {
                "chat": {
                    "status": "retry_wait",
                    "attempt_count": 2,
                    "next_retry_at": "2026-08-27T12:00:00+00:00",
                    "heartbeat_at": "2026-08-27T11:00:00+00:00",
                }
            },
            "heartbeat_at": "2026-08-27T11:00:00+00:00",
        }
        second = json.loads(json.dumps(first))
        second["attempts"]["chat"]["heartbeat_at"] = (
            "2026-08-27T11:01:00+00:00"
        )
        second["heartbeat_at"] = "2026-08-27T11:01:00+00:00"

        self.assertEqual(
            _progress_signature("provider_smoke", first),
            _progress_signature("provider_smoke", second),
        )

    def test_state_validation_rejects_skipped_gate_order(self) -> None:
        state = _new_state("config-sha")
        state["current_stage"] = "graph"
        state["stages"]["graph"]["status"] = "completed"

        with self.assertRaisesRegex(
            SupervisorError, "supervisor_state_stage_order_invalid"
        ):
            _validate_state(state, "config-sha")

    def test_transition_evidence_requires_each_next_stage_to_start(self) -> None:
        state = _new_state("config-sha")
        state["events"] = []
        for index, stage in enumerate(STAGES):
            state["events"].append({"event": "stage_started", "stage": stage})
            if index < len(STAGES) - 1:
                state["stages"][stage]["status"] = "completed"
                state["events"].append(
                    {"event": "stage_completed", "stage": stage}
                )
        state["stages"][STAGES[-1]]["status"] = "running"

        self.assertEqual(
            _validate_transition_evidence(state)["completed_transition_count"],
            len(STAGES) - 1,
        )
        state["events"] = [
            event
            for event in state["events"]
            if not (
                event.get("event") == "stage_started"
                and event.get("stage") == "provider_smoke"
            )
        ]
        with self.assertRaisesRegex(
            SupervisorError,
            "supervisor_(stage_start_evidence|next_stage_start)_missing",
        ):
            _validate_transition_evidence(state)

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
            self.assertIsInstance(resumed.get("resilience_policy_sha256"), str)
            self.assertIsInstance(resumed.get("implementation_sha256"), str)
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
            self.assertIsInstance(resumed.get("resilience_policy_sha256"), str)
            self.assertIsInstance(resumed.get("implementation_sha256"), str)
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
