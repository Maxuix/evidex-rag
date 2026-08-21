from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools import build_open_source_rag_corpus as corpus


class OpenSourceRagCorpusTests(unittest.TestCase):
    def test_frozen_corpus_validates_and_keeps_graph_labels_candidate(self) -> None:
        corpus.validate(corpus.DEFAULT_OUTPUT)
        manifest = json.loads(
            (corpus.DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["dataset_id"], "routing-rag-v3-open-source")
        self.assertEqual(manifest["document_count"], 14)
        self.assertEqual(manifest["relation_count"], 37)
        self.assertEqual(manifest["case_count"], 28)
        self.assertEqual(
            manifest["case_counts"],
            {"graph_candidate": 12, "negative_control": 8, "simple_answerable": 8},
        )
        self.assertFalse(manifest["route_label_policy"]["primary_metric_ready"])
        for document in manifest["documents"]:
            content = (
                corpus.DEFAULT_OUTPUT / "documents" / document["filename"]
            ).read_bytes()
            self.assertEqual(
                document["sha256"], hashlib.sha256(content).hexdigest()
            )

    def test_builder_is_deterministic_in_a_fresh_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rag-open-source-corpus-") as directory:
            output = Path(directory) / "corpus"
            corpus.build(output)
            first = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            corpus.build(output, force=True)
            second = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
        self.assertEqual(first, second)

    def test_relations_have_distinct_endpoints_and_no_relation_id_leakage(self) -> None:
        relations = [
            json.loads(line)
            for line in (corpus.DEFAULT_OUTPUT / "relations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertTrue(relations)
        self.assertTrue(
            all(row["subject_entity_id"] != row["object_entity_id"] for row in relations)
        )
        for document in corpus.DOCUMENTS:
            text = (
                corpus.DEFAULT_OUTPUT / "documents" / document["filename"]
            ).read_text(encoding="utf-8")
            self.assertNotRegex(text, r"\bOSR\d{3}\b")

    def test_cases_cover_negative_kinds_and_closed_world_scope(self) -> None:
        cases = [
            json.loads(line)
            for line in (corpus.DEFAULT_OUTPUT / "cases.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        negative_kinds = {
            case["negative_control_kind"]
            for case in cases
            if case["category"] == "negative_control"
        }
        self.assertEqual(
            negative_kinds,
            {"contradicted", "closed_world_absence", "open_world_unanswerable"},
        )
        closed_world = [
            case
            for case in cases
            if case["negative_control_kind"] == "closed_world_absence"
        ]
        self.assertTrue(closed_world)
        self.assertTrue(
            all(case["absence_scope"] and case["absence_evidence_documents"] for case in closed_world)
        )


if __name__ == "__main__":
    unittest.main()
