from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools import lexical_comparison


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "evaluation/configs/lexical-comparison-v1.0.json"


class LexicalTokenizationTests(unittest.TestCase):
    def test_tokenizers_preserve_identifiers_and_apply_frozen_han_modes(self) -> None:
        text = "设备XG-420错误 E217"

        self.assertEqual(
            lexical_comparison.tokenize(text, "contiguous_runs"),
            ("设备", "xg-420", "错误", "e217"),
        )
        self.assertEqual(
            lexical_comparison.tokenize(text, "unigrams"),
            ("设", "备", "xg-420", "错", "误", "e217"),
        )
        self.assertEqual(
            lexical_comparison.tokenize(text, "overlapping_unigrams_and_bigrams"),
            ("设", "备", "设备", "xg-420", "错", "误", "错误", "e217"),
        )

    def test_bm25_ties_are_ordered_by_stable_sample_and_ordinal(self) -> None:
        chunks = (
            lexical_comparison.LexicalChunk("SAMPLE-B", 0, "shared"),
            lexical_comparison.LexicalChunk("SAMPLE-A", 1, "shared"),
            lexical_comparison.LexicalChunk("SAMPLE-A", 0, "shared"),
        )
        index = lexical_comparison.BM25Index(
            chunks,
            han_mode="unigrams",
            k1=1.2,
            b=0.75,
        )

        self.assertEqual(
            [(item.sample_id, item.ordinal) for item in index.search("shared", top_k=3)],
            [("SAMPLE-A", 0), ("SAMPLE-A", 1), ("SAMPLE-B", 0)],
        )
        self.assertEqual(index.search("absent", top_k=3), ())


class LexicalComparisonReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = lexical_comparison.build_report(CONFIG, repository_root=ROOT)

    def test_report_reuses_frozen_inputs_and_filters(self) -> None:
        report = self.report

        self.assertEqual(report["inputs"]["golden_case_count"], 18)
        self.assertEqual(report["inputs"]["eligible_chunk_count"], 61)
        self.assertEqual(len(report["inputs"]["eligible_sample_ids"]), 9)
        self.assertFalse(report["configuration"]["serving_enabled"])
        self.assertEqual(
            report["configuration"]["fixture_workspace_id"],
            "workspace-local-01",
        )
        self.assertIn("corpus_profile_sha256", report["inputs"])
        self.assertIn("quality_baseline_sha256", report["inputs"])
        self.assertEqual(
            report["configuration"]["mandatory_filters"],
            json.loads(CONFIG.read_text(encoding="utf-8"))["mandatory_filters"],
        )

    def test_frozen_chargram_passes_confirmation_with_diagnostic_advantage(self) -> None:
        selection = self.report["selection"]

        self.assertEqual(selection["status"], "confirmed")
        self.assertEqual(
            selection["frozen_selected_strategy_id"],
            "unicode-chargram-bm25-v1",
        )
        self.assertEqual(
            selection["diagnostic_winner_strategy_id"],
            "unicode-han-unigram-bm25-v1",
        )
        self.assertTrue(selection["diagnostic_advantage_recorded"])
        self.assertTrue(all(item["passed"] for item in selection["confirmation_criteria"]))

    def test_confirmation_fails_closed_on_a_forbidden_selected_result(self) -> None:
        candidates = copy.deepcopy(self.report["candidates"])
        selected = next(
            candidate
            for candidate in candidates
            if candidate["strategy_id"] == "unicode-chargram-bm25-v1"
        )
        selected["overall"]["forbidden_result_count"] = 1
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        policy = {
            **config["frozen_confirmation"],
            "diagnostic_ranking_precedence": config["diagnostic_ranking_precedence"],
        }

        confirmation = lexical_comparison._confirmation(
            candidates,
            selected_strategy_id="unicode-chargram-bm25-v1",
            policy=policy,
        )

        self.assertEqual(confirmation["status"], "conflict")
        self.assertFalse(confirmation["confirmation_criteria"][0]["passed"])

    def test_expected_metrics_and_segments_are_reproducible(self) -> None:
        candidates = {
            candidate["strategy_id"]: candidate for candidate in self.report["candidates"]
        }
        chargram = candidates["unicode-chargram-bm25-v1"]
        unigram = candidates["unicode-han-unigram-bm25-v1"]

        self.assertEqual(chargram["overall"]["recall_at_k"]["5"], 1.0)
        self.assertEqual(chargram["overall"]["forbidden_result_count"], 0)
        self.assertEqual(
            chargram["segments"]["tag"]["exact_identifier"]["recall_at_k"]["5"],
            1.0,
        )
        self.assertEqual(
            {
                language: metrics["recall_at_k"]["5"]
                for language, metrics in chargram["segments"]["language"].items()
            },
            {"en": 1.0, "mixed": 1.0, "zh": 1.0},
        )
        self.assertEqual(
            chargram["segments"]["case_group"]["error_code"]["recall_at_k"]["5"],
            1.0,
        )
        self.assertEqual(
            chargram["segments"]["case_group"]["article_number"]["case_count"],
            0,
        )
        self.assertGreater(unigram["overall"]["mrr"], chargram["overall"]["mrr"])
        self.assertEqual(
            lexical_comparison.canonical_json(self.report),
            lexical_comparison.canonical_json(
                lexical_comparison.build_report(CONFIG, repository_root=ROOT)
            ),
        )

    def test_report_schema_is_versioned_json(self) -> None:
        schema_path = (
            ROOT / "evaluation/schemas/lexical-comparison-report-v1.0.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0")
        self.assertIn("selection", schema["required"])


if __name__ == "__main__":
    unittest.main()
