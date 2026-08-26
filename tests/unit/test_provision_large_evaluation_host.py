from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from tools import provision_large_evaluation_host as module


class LargeEvaluationHostProvisioningTests(unittest.TestCase):
    def test_frozen_specs_have_the_expected_closed_document_sets(self) -> None:
        expected = {
            "graph_rag": 16,
            "routing": 578,
            "enterprise": 138,
            "public": 1498,
        }
        for name, count in expected.items():
            with self.subTest(dataset=name):
                paths = module._paths(module.SPECS[name])
                self.assertEqual(len(paths), count)
                self.assertEqual(len({path.name for path in paths}), count)
                self.assertTrue(all(path.suffix in {".md", ".txt"} for path in paths))

    def test_checkpoint_refuses_a_corpus_binding_change(self) -> None:
        spec = module.SPECS["graph_rag"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            binding = {"dataset_id": spec.dataset_id, "corpus_sha256": "a" * 64}
            path.write_text(
                json.dumps(
                    {
                        "schema_version": module.CHECKPOINT_SCHEMA,
                        "binding": binding,
                        "binding_sha256": module.digest(binding),
                        "status": "started",
                        "events": [],
                    }
                ),
                encoding="utf-8",
            )
            path.chmod(0o600)
            with self.assertRaisesRegex(
                module.ProvisioningError,
                "provisioning_checkpoint_binding_invalid",
            ):
                module._load_checkpoint(
                    path,
                    spec=spec,
                    corpus_digest="b" * 64,
                )

    def test_graph_profile_assignment_matches_the_frozen_suite(self) -> None:
        self.assertEqual(
            module.SPECS["graph_rag"].graph_schema_key,
            "enterprise_knowledge_v1",
        )
        self.assertEqual(
            module.SPECS["routing"].graph_schema_key,
            "generic_open_domain_v1",
        )
        self.assertIsNone(module.SPECS["public"].graph_schema_key)

    def test_dataset_run_lock_rejects_a_second_provisioner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_path = Path(directory) / "runtime.json"
            with module._dataset_run_lock(runtime_path, "graph-rag-v1"):
                with self.assertRaisesRegex(
                    module.ProvisioningError,
                    "provisioning_dataset_already_running",
                ):
                    with module._dataset_run_lock(runtime_path, "graph-rag-v1"):
                        pass

    def test_completed_suite_binding_is_atomic_and_carries_graph_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = SimpleNamespace(runtime_root=root)
            checkpoint = {
                "status": "completed",
                "binding": {"corpus_sha256": "a" * 64},
                "knowledge_base_id": "00000000-0000-0000-0000-000000000001",
                "index_revision_id": "00000000-0000-0000-0000-000000000002",
                "graph_build_id": "00000000-0000-0000-0000-000000000003",
            }
            module._record_suite_binding(
                runtime, spec=module.SPECS["graph_rag"], checkpoint=checkpoint
            )
            bindings = json.loads((root / "large-evaluation-bindings.json").read_text())
            entry = bindings["suites"]["graph-rag-v1"]
            self.assertEqual(entry["corpus_sha256"], "a" * 64)
            self.assertEqual(entry["graph_build_id"], checkpoint["graph_build_id"])
            self.assertEqual((root / "large-evaluation-bindings.json").stat().st_mode & 0o777, 0o600)
