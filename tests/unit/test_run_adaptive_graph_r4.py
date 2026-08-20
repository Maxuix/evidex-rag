from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from rag_kb.domain import (
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ErrorCode,
    RetrievalExecutionError,
)
from tools.run_adaptive_graph_r4 import (
    R4_CHECKPOINT_SCHEMA_VERSION,
    R4RunnerError,
    _checkpoint_complete_case,
    _checkpoint_error_code,
    _checkpoint_stage,
    _failure_record,
    _controller_mode,
    _load_or_create_checkpoint,
    _new_checkpoint,
    _ordered_reranked_chunk_ids,
    _safe_usage,
    _safe_model_call_usage,
    _validate_checkpoint,
    _write_json_atomic,
    _write_or_verify_json_artifact,
)
from tools import run_adaptive_graph_r4 as r4_runner


def _identity(*, build_id: str = "build-a") -> dict[str, object]:
    return {
        "schema_version": R4_CHECKPOINT_SCHEMA_VERSION,
        "dataset_id": "routing-rag-v2",
        "scope": "expected_route_graph",
        "manifest_sha256": "a" * 64,
        "runtime": {
            "knowledge_base_id": "kb-a",
            "index_revision_id": "revision-a",
            "graph_build_id": build_id,
            "chat_model_profile_revision_id": "profile-a",
            "chat_model": "model-a",
            "chat_model_source": "profile_revision",
            "chat_model_max_output_tokens": 4096,
            "chat_model_max_retries": 0,
            "graphiti_edge_limit": 8,
            "evaluator_edge_limit": 8,
            "replay_mode": "actual-auto",
            "rerank_mode": "classic",
            "forced_controller_mode": "actual_auto",
            "runner_sha256": "b" * 64,
            "evaluator_sha256": "c" * 64,
        },
        "case_ids": ["route-graph-001", "route-graph-002"],
    }


def _record(case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "columns": {},
        "query_source": "agent_replay",
        "query_count": 0,
        "layer_diagnostics": {},
        "agent_replay_status": "not_requested",
        "layer_metrics": {},
        "capture_metrics": {
            "forced_replacements": 0,
            "model_calls": 2,
            "usage": {"total_tokens": 12},
            "route_requested": False,
            "route_reason_code": None,
        },
    }


class R4CheckpointTests(unittest.TestCase):
    def test_override_uses_single_tool_fallback_controller(self) -> None:
        self.assertEqual(
            _controller_mode(replay_mode="forced", model_override=None),
            "specific_tool_choice",
        )
        self.assertEqual(
            _controller_mode(
                replay_mode="forced",
                model_override="deepseek-v4-flash",
            ),
            "single_tool_required_fallback",
        )
        self.assertEqual(
            _controller_mode(
                replay_mode="actual-auto",
                model_override="deepseek-v4-flash",
            ),
            "actual_auto",
        )

    def test_column_layers_is_bound_to_production_candidate_and_pack_paths(self) -> None:
        source = inspect.getsource(r4_runner._column_layers)
        self.assertIn("_search_graphiti_candidates", source)
        self.assertIn("_pack_graphiti_supplement_evidence", source)
        self.assertNotIn("graphiti_runtime.search", source)
        self.assertNotIn("_rerank_graphiti_candidates_with_scores", source)

    def test_reranked_order_follows_production_paths_without_losing_chunks(self) -> None:
        traversal = SimpleNamespace(
            paths=(
                SimpleNamespace(source_chunk_ids=("chunk-3", "chunk-1")),
                SimpleNamespace(source_chunk_ids=("chunk-2",)),
            )
        )
        self.assertEqual(
            _ordered_reranked_chunk_ids(
                traversal,
                ("chunk-1", "chunk-2", "chunk-3", "chunk-4"),
            ),
            ("chunk-3", "chunk-1", "chunk-2", "chunk-4"),
        )

    def test_atomic_checkpoint_is_owner_only_and_rejects_content(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            checkpoint = _new_checkpoint(_identity())
            digest = _write_json_atomic(path, checkpoint)

            self.assertEqual(len(digest), 64)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
            original = path.read_bytes()

            checkpoint["active_case"] = {
                "case_id": "route-graph-001",
                "stage": "capture",
                "error_code": None,
                "query": "must never persist",
            }
            with self.assertRaisesRegex(
                ValueError,
                "r4_checkpoint_contains_forbidden_field",
            ):
                _write_json_atomic(path, checkpoint)
            self.assertEqual(path.read_bytes(), original)

    def test_resume_preserves_completed_prefix_and_rejects_identity_change(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "checkpoint.json"
            identity = _identity()
            checkpoint = _load_or_create_checkpoint(
                path,
                identity=identity,
                final_outputs=(root / "capture.json", root / "diagnostic.json"),
            )
            _checkpoint_stage(
                checkpoint,
                path,
                case_id="route-graph-001",
                stage="capture",
            )
            _checkpoint_complete_case(
                checkpoint,
                path,
                case_id="route-graph-001",
                capture=None,
                record=_record("route-graph-001"),
            )

            resumed = _load_or_create_checkpoint(
                path,
                identity=identity,
                final_outputs=(root / "capture.json", root / "diagnostic.json"),
            )
            self.assertEqual(len(resumed["completed_cases"]), 1)
            self.assertEqual(
                resumed["completed_cases"][0]["case_id"],
                "route-graph-001",
            )
            self.assertNotIn("question", json.dumps(resumed))

            with self.assertRaisesRegex(
                RuntimeError,
                "r4_checkpoint_identity_mismatch",
            ):
                _validate_checkpoint(resumed, identity=_identity(build_id="build-b"))

    def test_checkpoint_refuses_final_output_without_recovery_identity(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture.json"
            capture.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError,
                "r4_checkpoint_missing_for_existing_output",
            ):
                _load_or_create_checkpoint(
                    root / "checkpoint.json",
                    identity=_identity(),
                    final_outputs=(capture, root / "diagnostic.json"),
                )

    def test_final_artifact_is_idempotent_but_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            artifact = {"schema_version": "test", "case_count": 2}
            first = _write_or_verify_json_artifact(path, artifact)
            second = _write_or_verify_json_artifact(path, artifact)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(
                RuntimeError,
                "r4_existing_artifact_mismatch",
            ):
                _write_or_verify_json_artifact(
                    path,
                    {"schema_version": "test", "case_count": 3},
                )

    def test_error_mapping_is_closed_and_content_safe(self) -> None:
        self.assertEqual(
            _checkpoint_error_code(R4RunnerError("r4_graph_runtime_not_ready")),
            "r4_graph_runtime_not_ready",
        )
        self.assertEqual(_checkpoint_error_code(ValueError("question text")), "r4_validation_failure")
        self.assertEqual(_checkpoint_error_code(RuntimeError("provider payload")), "r4_unexpected_failure")
        with self.assertRaises(ValueError):
            R4RunnerError("r4_undeclared_failure")

    def test_active_checkpoint_cannot_claim_final_artifacts(self) -> None:
        checkpoint = _new_checkpoint(_identity())
        checkpoint["final_artifacts"] = {
            "capture_sha256": "a" * 64,
            "diagnostic_sha256": "b" * 64,
        }
        with self.assertRaisesRegex(ValueError, "r4_checkpoint_completion_invalid"):
            _validate_checkpoint(checkpoint, identity=_identity())

    def test_cancelled_error_maps_to_stable_code_without_content(self) -> None:
        record = _failure_record(asyncio.CancelledError(), phase="agent_replay")
        self.assertEqual(record["error_code"], "r4_interrupted")
        self.assertEqual(record["phase"], "agent_replay")
        self.assertEqual(record["diagnostic"], {})

    def test_domain_errors_keep_only_stable_code_phase_check_and_usage(self) -> None:
        pipeline = ChatPipelineExecutionError(
            ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
            phase=ChatPipelinePhase.GENERATE_OR_REFUSE,
            diagnostic={"check": "model_snapshot", "http_status": 503, "url": "secret"},
        )
        record = _failure_record(pipeline, phase="capability")
        self.assertEqual(record["error_code"], "CHAT_PROVIDER_UNAVAILABLE")
        self.assertEqual(record["phase"], "generate_or_refuse")
        self.assertEqual(record["evaluation_phase"], "capability")
        self.assertTrue(record["retryable"])
        self.assertEqual(record["diagnostic"], {"check": "model_snapshot", "http_status": 503})
        pipeline.model_calls = (
            SimpleNamespace(
                operation="provider-secret-operation",
                usage={"total_tokens": 99},
            ),
        )
        self.assertEqual(_safe_model_call_usage(pipeline), [])

        retrieval = RetrievalExecutionError(
            ErrorCode.GRAPH_NOT_READY,
            diagnostic={"check": "graph_edge_search", "http_status": 700, "url": "secret"},
        )
        retrieval_record = _failure_record(retrieval, phase="agent_replay")
        self.assertEqual(retrieval_record["error_code"], "GRAPH_NOT_READY")
        self.assertEqual(retrieval_record["phase"], "agent_replay")
        self.assertEqual(retrieval_record["diagnostic"], {"check": "graph_edge_search"})

    def test_usage_checkpoint_keeps_only_numeric_token_counts(self) -> None:
        self.assertEqual(
            _safe_usage(
                {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                    "provider_detail": "not retained",
                    "cached_tokens": True,
                }
            ),
            {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        )


if __name__ == "__main__":
    unittest.main()
