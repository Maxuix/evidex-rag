from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import evaluation_runtime as module
from tools.evaluate_agent_complex_qa import _parser as complex_parser


_OWNER = "0123456789abcdef0123456789abcdef"
_IDENTITY = {
    "workspace_id": "01900000-0000-7000-8000-000000000001",
    "knowledge_base_id": "01900000-0000-7000-8000-000000000002",
    "index_revision_id": "01900000-0000-7000-8000-000000000003",
    "graph_build_id": "01900000-0000-7000-8000-000000000004",
    "answer_profile_revision_id": "01900000-0000-7000-8000-000000000005",
    "judge_profile_revision_id": "01900000-0000-7000-8000-000000000006",
    "schema_profile_key": "software_knowledge_v1",
    "schema_profile_digest": "6cae93809f060d21f0c85ba04cde955abdc5259fd93e1b1757fb7445d51eaf38",
    "extractor_version": "graphiti_v4",
}


class EvaluationRuntimeTests(unittest.TestCase):
    def _runtime_files(
        self,
        root: Path,
        *,
        changes: dict[str, object] | None = None,
        mode: int = 0o600,
    ) -> Path:
        root.mkdir(parents=True, mode=0o700)
        root.chmod(0o700)
        for name in ("runtime.env", "compose.env"):
            path = root / name
            path.write_text("HOST_TEST_RUNTIME=1\n", encoding="utf-8")
            path.chmod(0o600)
        value: dict[str, object] = {
            "schema_version": 1,
            "compose_project": "python-host-test",
            "owner": _OWNER,
            "build_revision": "a" * 40,
            "env_file": "runtime.env",
            "compose_env_file": "compose.env",
            "api_base_url": "http://127.0.0.1:28000/api/v1",
            "ports": {"api": 28000, "frontend": 23000, "postgres": 25432, "falkordb": 26379},
            "adaptive_graph": _IDENTITY,
        }
        value.update(changes or {})
        manifest = root / "runtime.json"
        manifest.write_text(json.dumps(value), encoding="utf-8")
        manifest.chmod(mode)
        return manifest

    def test_loads_private_host_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "host-test"
            manifest = self._runtime_files(root)
            with patch.object(module, "DEFAULT_RUNTIME_ROOT", root):
                runtime = module.load_evaluation_runtime(manifest, require_adaptive_graph=True)

        self.assertEqual(runtime.owner, _OWNER)
        self.assertEqual(runtime.compose_project, "python-host-test")
        self.assertEqual(str(runtime.adaptive_graph.knowledge_base_id), _IDENTITY["knowledge_base_id"])
        self.assertNotIn("secret", repr(runtime))

    def test_read_only_loader_can_accept_canonical_checkout_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "primary/.runtime/evaluations/graph-schema-profiles-host"
            manifest = self._runtime_files(root)
            with (
                patch.object(module, "DEFAULT_RUNTIME_ROOT", Path(directory) / "linked"),
                patch.object(module, "canonical_evaluation_runtime_manifest", return_value=manifest),
            ):
                with self.assertRaises(module.EvaluationRuntimeError):
                    module.load_evaluation_runtime(manifest)
                runtime = module.load_evaluation_runtime(
                    manifest,
                    require_adaptive_graph=True,
                    allow_canonical_checkout=True,
                )

        self.assertEqual(runtime.owner, _OWNER)

    def test_rejects_non_host_runtime_identity_and_unsafe_configuration(self) -> None:
        changes = (
            {"compose_project": "legacy-compose-project"},
            {"env_file": "../runtime.env"},
            {"api_base_url": "http://example.com:28000/api/v1"},
            {"ports": {"api": 8000, "frontend": 23000, "postgres": 25432, "falkordb": 26379}},
        )
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "host-test"
                manifest = self._runtime_files(root, changes=change)
                with patch.object(module, "DEFAULT_RUNTIME_ROOT", root):
                    with self.assertRaises(module.EvaluationRuntimeError):
                        module.load_evaluation_runtime(manifest)

    def test_rejects_symlink_and_missing_adaptive_graph_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "host-test"
            manifest = self._runtime_files(root)
            link = root / "linked.json"
            link.symlink_to(manifest)
            with patch.object(module, "DEFAULT_RUNTIME_ROOT", root):
                with self.assertRaises(module.EvaluationRuntimeError):
                    module.load_evaluation_runtime(link)
                manifest.write_text(
                    json.dumps({**json.loads(manifest.read_text(encoding="utf-8")), "adaptive_graph": None}),
                    encoding="utf-8",
                )
                manifest.chmod(0o600)
                with self.assertRaises(module.EvaluationRuntimeError):
                    module.load_evaluation_runtime(manifest, require_adaptive_graph=True)

    def test_evaluator_clis_expose_only_runtime_or_offline_entrypoints(self) -> None:
        parser = complex_parser()
        help_text = parser.format_help()
        self.assertIn("--evaluation-runtime", help_text)
        self.assertIn("--dry-run", help_text)
        self.assertNotIn("--api", help_text)
        self.assertNotIn("--env-file", help_text)
