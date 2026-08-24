from __future__ import annotations

import json
import unittest

from tools import build_musique_mini_corpus as corpus


def _jsonl(name: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (corpus.DEFAULT_OUTPUT / name)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


class MusiqueMiniCorpusTests(unittest.TestCase):
    def test_frozen_candidate_corpus_validates(self) -> None:
        corpus.validate(corpus.DEFAULT_OUTPUT)
        manifest = json.loads(
            (corpus.DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8")
        )

        self.assertEqual(manifest["dataset_id"], corpus.DATASET_ID)
        self.assertEqual(manifest["case_count"], 16)
        self.assertEqual(manifest["document_count"], 80)
        self.assertEqual(
            manifest["case_counts"],
            {
                "graph_needed_candidate": 10,
                "negative_or_refusal": 3,
                "simple_only": 3,
            },
        )
        self.assertEqual(
            manifest["hop_counts"], {"hop1": 3, "hop2": 6, "hop3": 4}
        )
        self.assertEqual(
            manifest["qualification"]["status"],
            "pending_host_simple_graph_qualification",
        )

    def test_graph_candidates_have_complete_distinct_source_paths(self) -> None:
        graph_cases = [
            case
            for case in _jsonl("cases.jsonl")
            if case["route_label"] == "graph_needed_candidate"
        ]

        self.assertEqual(len(graph_cases), 10)
        for case in graph_cases:
            path = case["required_paths"][0]
            self.assertEqual(len(path), case["hop_count"])
            self.assertEqual(len(path), len(set(path)))
            self.assertTrue(
                all(
                    step["support_document_id"] is not None
                    for step in case["decomposition"]
                )
            )

    def test_negative_controls_have_at_least_one_missing_support_step(self) -> None:
        negative_cases = [
            case
            for case in _jsonl("cases.jsonl")
            if case["route_label"] == "negative_or_refusal"
        ]

        self.assertEqual(len(negative_cases), 3)
        self.assertTrue(
            all(
                any(
                    step["support_document_id"] is None
                    for step in case["decomposition"]
                )
                for case in negative_cases
            )
        )
        self.assertTrue(all(case["required_paths"] == [] for case in negative_cases))

    def test_simple_controls_are_direct_upstream_questions(self) -> None:
        simple_cases = [
            case
            for case in _jsonl("cases.jsonl")
            if case["route_label"] == "simple_only"
        ]

        self.assertEqual(len(simple_cases), 3)
        self.assertTrue(all(case["hop_count"] == 1 for case in simple_cases))
        self.assertTrue(all("#" not in case["question"] for case in simple_cases))
        self.assertTrue(
            all(len(case["required_paths"][0]) == 1 for case in simple_cases)
        )


if __name__ == "__main__":
    unittest.main()
