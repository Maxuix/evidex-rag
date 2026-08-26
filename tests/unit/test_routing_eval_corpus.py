from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools import build_routing_rag_corpus as corpus_builder
from tools.evaluate_adaptive_graph_route import (
    DEFAULT_MANIFEST,
    load_manifest,
    manifest_digest,
    validate_evaluation_readiness,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RoutingEvaluationCorpusTests(unittest.TestCase):
    def test_default_entries_use_v2_and_keep_v1_as_history(self) -> None:
        self.assertEqual(
            corpus_builder.DEFAULT_OUTPUT,
            PROJECT_ROOT / "evaluation" / "routing-rag-v2",
        )
        self.assertEqual(
            DEFAULT_MANIFEST,
            PROJECT_ROOT / "evaluation" / "adaptive-graph-route-v2" / "manifest.json",
        )
        self.assertTrue((PROJECT_ROOT / "evaluation" / "routing-rag-v1").is_dir())
        self.assertTrue((PROJECT_ROOT / "evaluation" / "adaptive-graph-route-v1").is_dir())

        manifest = load_manifest()
        self.assertEqual(manifest["dataset_id"], "routing-rag-v2")
        self.assertEqual(manifest["case_count"], 43)
        self.assertEqual(manifest["empirical_need"]["case_count"], 26)
        self.assertIn("routing-rag-v2", manifest["case_file"])

    def test_v2_graph_locators_follow_relation_source_documents(self) -> None:
        manifest = load_manifest()
        cases = [
            json.loads(line)
            for line in Path(manifest["case_file"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        relation_path = (
            PROJECT_ROOT
            / "evaluation"
            / "routing-rag-v2"
            / "gold"
            / "graph-rag-v1"
            / "relations.jsonl"
        )
        relation_documents = {
            str(row["relation_id"]): str(row["document_id"])
            for row in (
                json.loads(line)
                for line in relation_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        filenames = {
            str(row["document_id"]): str(row["filename"])
            for row in json.loads(
                (
                    PROJECT_ROOT
                    / "evaluation"
                    / "routing-rag-v2"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )["documents"]
        }

        locators = [
            locator
            for case in cases
            if case["expected_route"]["route"] == "graph"
            for locator in (
                case["answer_gold_source_locators"] + case["path_context_locators"]
            )
        ]
        self.assertEqual(len(locators), 52)
        self.assertEqual(
            len({str(locator["relation_id"]) for locator in locators}),
            42,
        )
        for locator in locators:
            relation_id = str(locator["relation_id"])
            self.assertEqual(
                locator["document_filename"],
                filenames[relation_documents[relation_id]],
            )

    def test_builder_rejects_writing_legacy_v1_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy routing-rag-v1"):
            corpus_builder.build(corpus_builder.LEGACY_OUTPUT)

    def test_manifest_digest_does_not_depend_on_resolved_worktree_paths(self) -> None:
        manifest = load_manifest()
        relocated = json.loads(json.dumps(manifest))
        relocated["case_file"] = relocated["case_file"].replace(
            str(PROJECT_ROOT), "/another/worktree/RAG"
        )
        relocated["empirical_need"]["fixture_file"] = relocated["empirical_need"][
            "fixture_file"
        ].replace(str(PROJECT_ROOT), "/another/worktree/RAG")
        self.assertEqual(manifest_digest(manifest), manifest_digest(relocated))

    def test_v1_route_manifest_is_explicitly_read_only_history(self) -> None:
        v1_path = PROJECT_ROOT / "evaluation" / "adaptive-graph-route-v1" / "manifest.json"
        self.assertEqual(load_manifest(v1_path)["dataset_id"], "routing-rag-v1")
        with self.assertRaisesRegex(ValueError, "requires routing-rag-v2"):
            validate_evaluation_readiness(v1_path)

    def test_v1_contract_bytes_are_frozen(self) -> None:
        expected = {
            "evaluation/routing-rag-v1/manifest.json": "66ca09d98668f577ae6a89f49d2488009856082d370e405c1fd4c536cb58db8d",
            "evaluation/routing-rag-v1/cases.jsonl": "da66eacaabac78a2b8f1e98f1f250808e9e4626268db18a70994c4d0d2430e2a",
            "evaluation/adaptive-graph-route-v1/manifest.json": "dad7cf2c4a28c3c557137a905540af2e91813f011139e4863051ae0ae7b1b982",
            "evaluation/adaptive-graph-route-v1/empirical-fixture.jsonl": "27c71c814f55bd0861f71caa9023cc723391b62781ec11891ed728187c96f90f",
        }
        import hashlib

        for relative_path, digest in expected.items():
            actual = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, relative_path)


if __name__ == "__main__":
    unittest.main()
