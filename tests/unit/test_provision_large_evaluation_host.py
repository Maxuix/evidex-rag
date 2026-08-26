from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

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

