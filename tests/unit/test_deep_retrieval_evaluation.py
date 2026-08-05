from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfReader

from tools.evaluate_deep_retrieval_baseline import (
    DEFAULT_MANIFEST,
    dataset_definition,
    generate_corpus,
    load_manifest,
    manifest_hash,
    summarize_results,
    _source_fingerprints,
    _evaluate_chat_case,
    _evaluate_retrieval_case,
    _visual_binding_facts,
)


class DeepRetrievalEvaluationTests(unittest.TestCase):
    def test_committed_report_matches_frozen_manifest(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        report_path = (
            Path(__file__).resolve().parents[2]
            / "docs/implementation-plans/reports"
            / "2026-08-05-deep-retrieval-phase-1-exact-baseline.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "deep_retrieval_baseline_report_v1")
        self.assertEqual(report["dataset"]["manifest_hash"], manifest["manifest_hash"])
        self.assertEqual(report["dataset"]["case_count"], len(manifest["cases"]))
        self.assertEqual(len(report["retrieval_cases"]), len(manifest["cases"]))
        self.assertEqual(len(report["chat_cases"]), len(manifest["cases"]))
        runner_path = Path(__file__).resolve().parents[2] / "tools/evaluate_deep_retrieval_baseline.py"
        self.assertEqual(
            report["execution"]["source_fingerprints"]["runner"],
            "sha256:" + hashlib.sha256(runner_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            report["execution"]["source_fingerprints"]["manifest"],
            "sha256:" + hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            report["execution"]["manifest_path"],
            "tests/fixtures/deep_retrieval_v1.json",
        )

    def test_frozen_manifest_hash_and_categories_are_valid(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        definition = dataset_definition(manifest)

        self.assertEqual(definition.name, "bounded-adaptive-multi-query-rag")
        self.assertEqual(len(definition.cases), 8)
        self.assertEqual("sha256:" + definition.manifest_hash, manifest["manifest_hash"])
        self.assertEqual(
            {case.expected["category"] for case in definition.cases},
            {"single_fact", "multi_hop", "comparison", "exception", "conflict", "no_answer"},
        )

    def test_manifest_hash_changes_with_corpus_or_annotation(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        changed = deepcopy(manifest)
        changed["corpus"][0]["content"] += " changed"

        self.assertNotEqual(manifest_hash(manifest), manifest_hash(changed))

    def test_generated_corpus_is_self_contained_and_visual_pdf_is_valid(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            corpus = generate_corpus(manifest, Path(directory))

            self.assertEqual(len(corpus), 11)
            self.assertTrue(all(path.exists() for path in corpus.values()))
            visual = corpus["quartz-visual"]
            self.assertEqual(len(PdfReader(visual).pages), 1)
            self.assertIn(
                "QUARTZ-VISUAL-TRIANGLE",
                PdfReader(visual).pages[0].extract_text() or "",
            )

    def test_summary_counts_missing_targets_as_zero_and_no_answer_safety(self) -> None:
        retrieval = [
            {
                "case_key": "a",
                "category": "single_fact",
                "required_target_count": 1,
                "target_ranks": [1],
                "goal_recalled_at_10": [True],
                "complete_chain_at_10": True,
                "ndcg_at_10": 1.0,
                "expects_visual": False,
                "visual_group_recalled": False,
                "elapsed_seconds": 0.1,
            },
            {
                "case_key": "b",
                "category": "multi_hop",
                "required_target_count": 2,
                "target_ranks": [2, None],
                "goal_recalled_at_10": [True, False],
                "complete_chain_at_10": False,
                "ndcg_at_10": 0.5,
                "expects_visual": False,
                "visual_group_recalled": False,
                "elapsed_seconds": 0.2,
            },
        ]
        chat = [
            {
                "expected_outcome": "supported",
                "citation_count": 1,
                "allowed_citation_count": 1,
                "citation_complete": True,
                "outcome_matches": True,
                "outcome": "answered",
                "control_reason": None,
                "expected_control_reason": None,
                "provider_call_count": 1,
                "model_names": ["model-v1"],
                "retrieval_profile_hash": "sha256:profile",
                "input_tokens": 10,
                "output_tokens": 4,
                "repair_attempted": False,
                "elapsed_seconds": 0.3,
            },
            {
                "expected_outcome": "refused",
                "citation_count": 0,
                "allowed_citation_count": 0,
                "citation_complete": True,
                "outcome_matches": True,
                "outcome": "refused",
                "control_reason": "insufficient_evidence",
                "expected_control_reason": "insufficient_evidence",
                "control_reason_matches": True,
                "claim_match_count": 0,
                "provider_call_count": 1,
                "model_names": ["model-v1"],
                "retrieval_profile_hash": "sha256:profile",
                "input_tokens": 8,
                "output_tokens": 2,
                "repair_attempted": False,
                "elapsed_seconds": 0.4,
            },
        ]

        metrics = summarize_results(retrieval, chat)

        self.assertEqual(metrics["evidence_recall_at_10"], 0.666667)
        self.assertEqual(metrics["complete_chain_recall_at_10"], 0.5)
        self.assertEqual(metrics["per_goal_coverage_at_10"], 0.666667)
        self.assertEqual(metrics["citation_precision"], 1.0)
        self.assertEqual(metrics["no_answer_safety"], 1.0)
        self.assertEqual(metrics["out_of_allowlist_citation_count"], 0)
        self.assertEqual(metrics["control_reason_accuracy"], 1.0)

    def test_no_answer_safety_rejects_unsafe_refusal(self) -> None:
        retrieval = []
        chat = [
            {
                "expected_outcome": "refused",
                "outcome": "refused",
                "control_reason": "insufficient_evidence",
                "expected_control_reason": "insufficient_evidence",
                "citation_count": 1,
                "claim_match_count": 0,
                "citation_complete": False,
                "outcome_matches": True,
                "provider_call_count": 0,
                "model_names": [],
                "retrieval_profile_hash": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "repair_attempted": False,
                "elapsed_seconds": 0.0,
            }
        ]

        metrics = summarize_results(retrieval, chat)

        self.assertEqual(metrics["no_answer_safety"], 0.0)

    def test_source_fingerprint_uses_custom_manifest_path_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "custom-manifest.json"
            custom.write_text('{"custom": true}\n', encoding="utf-8")
            custom_fingerprint = _source_fingerprints(custom)["manifest"]
            default_fingerprint = _source_fingerprints(DEFAULT_MANIFEST)["manifest"]
            sources = _source_fingerprints(custom)

        self.assertIsNotNone(custom_fingerprint)
        self.assertIsNotNone(default_fingerprint)
        self.assertNotEqual(custom_fingerprint, default_fingerprint)
        self.assertIsNotNone(sources["multimodal_evaluator"])

    def test_mrr_is_per_case_and_ndcg_deduplicates_documents(self) -> None:
        from tools.evaluate_deep_retrieval_baseline import _mrr_at_k, _ndcg_for_evidence

        self.assertEqual(_mrr_at_k([[2, 5], [None, 4]], 10), 0.375)
        duplicate_results = [
            {"document_id": "doc-a"},
            {"document_id": "doc-a"},
            {"document_id": "doc-b"},
        ]
        score = _ndcg_for_evidence(duplicate_results, {"doc-a", "doc-b"})
        self.assertLessEqual(score, 1.0)
        self.assertEqual(score, 0.919721)

    def test_malicious_manifest_annotations_fail_closed(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malicious.json"
            changed = deepcopy(manifest)
            expected = changed["cases"][0]["expected"]
            expected["required_evidence"] = ["beta-service#NOT-BOUND"]
            changed["manifest_hash"] = manifest_hash(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)

            changed = deepcopy(manifest)
            no_answer = next(
                item for item in changed["cases"] if item["category"] == "no_answer"
            )
            del no_answer["expected"]["expected_control_reason"]
            changed["manifest_hash"] = manifest_hash(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)

            changed = deepcopy(manifest)
            supported = next(
                item for item in changed["cases"] if item["expected"]["expected_outcome"] == "supported"
            )
            supported["expected"]["expected_control_reason"] = "insufficient_evidence"
            changed["manifest_hash"] = manifest_hash(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)

            changed = deepcopy(manifest)
            changed["cases"][0]["expected"]["required_evidence"] = [
                "retention-policy#NOT-PRESENT"
            ]
            changed["cases"][0]["expected"]["allowed_citation_keys"] = [
                "retention-policy#NOT-PRESENT"
            ]
            changed["manifest_hash"] = manifest_hash(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_conflict_pair_visual_relation_and_citation_shape_fail_closed(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malicious.json"
            changed = deepcopy(manifest)
            conflict = next(item for item in changed["cases"] if item["category"] == "conflict")
            conflict["expected"]["conflict_pairs"] = [[
                "gateway-source-a#NIMBUS-PORT-7443",
                "retention-policy#ZF-210",
            ]]
            changed["manifest_hash"] = manifest_hash(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)

            changed = deepcopy(manifest)
            visual = next(item for item in changed["cases"] if item["expected"].get("expects_visual"))
            del visual["expected"]["expected_relation"]
            changed["manifest_hash"] = manifest_hash(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_filename_traversal_and_media_suffix_fail_closed(self) -> None:
        manifest = load_manifest(DEFAULT_MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malicious.json"
            for filename, media_type in (
                ("../escape.txt", "text/plain"),
                (".hidden.txt", "text/plain"),
                ("report.pdf", "text/plain"),
            ):
                changed = deepcopy(manifest)
                changed["corpus"][0]["filename"] = filename
                changed["corpus"][0]["media_type"] = media_type
                changed["manifest_hash"] = manifest_hash(changed)
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    load_manifest(path)

            root = Path(directory) / "root"
            root.mkdir()
            with self.assertRaises(ValueError):
                generate_corpus(
                    {
                        "corpus": [
                            {
                                "document_key": "escape",
                                "filename": "../escape.txt",
                                "media_type": "text/plain",
                                "content": "escape",
                            }
                        ]
                    },
                    root,
                )

    def test_runner_matches_required_markers_and_debug_revision(self) -> None:
        response = {
            "evidence": [
                {
                    "document_id": "doc-a",
                    "index_revision_id": "rev-1",
                    "text": "Unrelated text",
                },
                {
                    "document_id": "doc-a",
                    "index_revision_id": "rev-1",
                    "text": "Evidence marker MARK-A",
                },
            ],
            "debug": {
                "resolved_active_revision_id": "rev-1",
                "query_plan": {
                    "strategy": "exact_vector",
                    "revision_selector": "active",
                    "top_k": 10,
                    "rerank": True,
                    "build_status": "ready",
                    "serving_status": "serving",
                },
            },
        }
        case = {
            "case_key": "case",
            "category": "single_fact",
            "question": "question",
            "expected": {
                "goals": [{"goal_key": "g1", "relevant_targets": ["doc"]}],
                "required_evidence": ["doc#MARK-A"],
                "allowed_citation_keys": ["doc#MARK-A"],
                "expected_outcome": "supported",
                "required_claims": ["claim"],
                "expects_conflict": False,
                "expects_visual": False,
            },
        }
        with patch(
            "tools.evaluate_deep_retrieval_baseline._json_request",
            return_value=response,
        ):
            result = _evaluate_retrieval_case("http://loopback/api/v1", "kb", {"doc": "doc-a"}, case)
        self.assertEqual(result["required_evidence_ranks"], {"doc#MARK-A": 2})
        self.assertEqual(result["required_evidence_recalled_at_10"], [True])
        self.assertEqual(result["index_revision_id"], "rev-1")

    def test_chat_metrics_require_marker_citation_claim_and_visual_relation(self) -> None:
        case = {
            "case_key": "visual",
            "category": "single_fact",
            "question": "question",
            "expected": {
                "goals": [{"goal_key": "g1", "relevant_targets": ["doc"]}],
                "required_evidence": ["doc#MARK-A"],
                "allowed_citation_keys": ["doc#MARK-A"],
                "expected_outcome": "supported",
                "required_claims": ["The triangle is amber."],
                "expects_conflict": False,
                "expects_visual": True,
                "expected_relation": "explicit_figure_reference",
            },
        }
        terminal = {
            "index_revision_id": "rev-1",
            "citations": [
                {
                    "document_id": "doc-a",
                    "quoted_text": "Evidence marker MARK-A",
                    "source_location": {},
                    "modality": "text",
                },
                {
                    "document_id": "doc-a",
                    "quoted_text": "[Figure visual evidence]",
                    "source_location": {},
                    "modality": "image",
                    "asset": {
                        "id": "asset-1",
                        "media_type": "image/png",
                        "relation_type": "explicit_figure_reference",
                        "parent_citation_id": "cite_1",
                        "selection_reason": "selected_explicit_reference",
                    },
                },
            ],
            "answer": "The triangle is amber. [2]",
            "retrieval": {
                "profile_version": "exact_vector_v1",
                "strategy": "exact_vector",
                "top_k": 10,
                "rerank": True,
            },
            "timing": {
                "attempts": {
                    "1": {
                        "outcome": "answered",
                        "control_reason": None,
                        "citation_ids": ["cite_1", "cite_2"],
                        "validation": {},
                        "visual_evidence": {
                            "attached_image_count": 1,
                            "decisions": [
                                {
                                    "asset_id": "asset-1",
                                    "reason_code": "selected_explicit_reference",
                                    "parent_text_citation_ids": ["cite_1"],
                                    "relation_type": "explicit_figure_reference",
                                }
                            ],
                        },
                    }
                }
            },
            "usage": {"calls": {}},
        }
        with patch(
            "tools.evaluate_deep_retrieval_baseline._json_request",
            side_effect=[
                {"id": "session"},
                {"run_id": "run"},
                {
                    "available": True,
                    "media": [
                        {
                            "citation_ids": ["cite_2"],
                            "asset": {"id": "asset-1", "media_type": "image/png"},
                        }
                    ],
                },
            ],
        ), patch(
            "tools.evaluate_deep_retrieval_baseline._wait_for_chat_run",
            return_value=(terminal, 0.1),
        ):
            result = _evaluate_chat_case(
                "http://loopback/api/v1",
                "kb",
                {"doc": "doc-a"},
                case,
                expected_revision_id="rev-1",
                timeout_seconds=1,
                poll_seconds=0.01,
            )
        self.assertEqual(result["required_evidence_citation_match_count"], 1)
        self.assertEqual(result["claim_match_count"], 1)
        self.assertEqual(result["claim_citation_match_count"], 1)
        self.assertTrue(result["visual_complete"])

        terminal["answer"] = "The triangle is amber."
        with patch(
            "tools.evaluate_deep_retrieval_baseline._json_request",
            side_effect=[
                {"id": "session"},
                {"run_id": "run"},
                {
                    "available": True,
                    "media": [
                        {
                            "citation_ids": ["cite_1"],
                            "asset": {"media_type": "image/png"},
                        }
                    ],
                },
            ],
        ), patch(
            "tools.evaluate_deep_retrieval_baseline._wait_for_chat_run",
            return_value=(terminal, 0.1),
        ):
            uncited = _evaluate_chat_case(
                "http://loopback/api/v1",
                "kb",
                {"doc": "doc-a"},
                case,
                expected_revision_id="rev-1",
                timeout_seconds=1,
                poll_seconds=0.01,
            )
        self.assertFalse(uncited["citation_complete"])
        self.assertEqual(uncited["claim_citation_match_count"], 0)

    def test_visual_binding_fails_closed_for_unrelated_or_incomplete_chains(self) -> None:
        attempt = {
            "citation_ids": ["cite_1", "cite_2"],
            "visual_evidence": {
                "decisions": [
                    {
                        "asset_id": "asset-1",
                        "reason_code": "selected_explicit_reference",
                        "parent_text_citation_ids": ["cite_1"],
                        "relation_type": "explicit_figure_reference",
                    }
                ]
            },
        }
        citations = [
            {
                "document_id": "doc-a",
                "quoted_text": "Evidence marker MARK-A",
            },
            {
                "document_id": "doc-a",
                "quoted_text": "[Figure visual evidence]",
                "asset": {
                    "id": "asset-1",
                    "media_type": "image/png",
                    "relation_type": "explicit_figure_reference",
                    "parent_citation_id": "cite_1",
                    "selection_reason": "selected_explicit_reference",
                },
            },
        ]
        media = [
            {
                "citation_ids": ["cite_2"],
                "asset": {"id": "asset-1", "media_type": "image/png"},
            }
        ]
        kwargs = {
            "allowed_keys": ["doc#MARK-A"],
            "documents": {"doc": "doc-a"},
            "expected_relation": "explicit_figure_reference",
        }
        visual, bound = _visual_binding_facts(attempt, citations, media, **kwargs)
        self.assertEqual(len(visual), 1)
        self.assertEqual(len(bound), 1)

        cases = {
            "unrelated_asset": lambda: (
                media,
                {**attempt, "visual_evidence": {"decisions": [{**attempt["visual_evidence"]["decisions"][0], "asset_id": "asset-2"}]}},
            ),
            "wrong_relation": lambda: (
                media,
                {**attempt, "visual_evidence": {"decisions": [{**attempt["visual_evidence"]["decisions"][0], "relation_type": "same_page"}]}},
            ),
            "rejected_decision": lambda: (
                media,
                {**attempt, "visual_evidence": {"decisions": [{**attempt["visual_evidence"]["decisions"][0], "reason_code": "rejected_weak_relation"}]}},
            ),
            "media_not_terminal": lambda: (
                [{"citation_ids": ["cite_9"], "asset": {"id": "asset-1", "media_type": "image/png"}}],
                attempt,
            ),
            "parent_not_allowed": lambda: (
                media,
                {**attempt, "visual_evidence": {"decisions": [{**attempt["visual_evidence"]["decisions"][0], "parent_text_citation_ids": ["cite_9"]}]}},
            ),
            "citation_order_mismatch": lambda: (media, {**attempt, "citation_ids": ["cite_1"]}),
        }
        for name, make_case in cases.items():
            case_media, case_attempt = make_case()
            with self.subTest(case=name):
                visual, bound = _visual_binding_facts(
                    case_attempt,
                    citations,
                    case_media,
                    **kwargs,
                )
                self.assertEqual(visual, [])
                self.assertEqual(bound, [])

if __name__ == "__main__":
    unittest.main()
