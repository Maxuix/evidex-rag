from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.evaluate_agent_complex_qa import (
    _extract_decimal_values,
    _validate_options,
    evaluate_cases,
    _validated_api_base,
    load_document_identity_map,
    score_complex_case,
    summarize_results,
)
from tools.build_document_qa_corpus import COMPLEX_CASE_DEFINITIONS


def _agent() -> dict[str, object]:
    return {
        "version": "native_tool_calling_agent_v1",
        "trace": {
            "outcome": "answered",
            "usage": {},
        },
    }


class AgentComplexEvaluationToolTests(unittest.TestCase):
    def test_batch_stops_after_nonretryable_auth_failure_and_marks_remaining_not_run(
        self,
    ) -> None:
        for http_status in (401, 403):
            cases = [
                {"case_id": case_id, "question": case_id, "aspects": []}
                for case_id in ("complex-04", "complex-02", "complex-05")
            ]
            failed = {
                "case_id": "complex-04",
                "status": "failed",
                "error": {"http_status": http_status, "retryable": False},
                "score": {},
            }

            with self.subTest(http_status=http_status), patch(
                "tools.evaluate_agent_complex_qa._evaluate_one",
                return_value=failed,
            ) as evaluate:
                results = evaluate_cases(
                    "http://127.0.0.1:8000/api/v1",
                    "kb-id",
                    cases,
                    strategy="exact_vector",
                    rerank_mode="classic",
                    top_k=10,
                    parallelism=1,
                    timeout_seconds=900,
                    poll_seconds=1,
                    document_identity_map={},
                )

            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(
                [item["status"] for item in results],
                ["failed", "not_run", "not_run"],
            )
            self.assertTrue(
                all(
                    item.get("not_run_reason")
                    == "provider_nonretryable_auth_failure"
                    for item in results[1:]
                )
            )

    def test_revised_complex_aspects_accept_semantic_and_decimal_variants(self) -> None:
        answers = {
            "complex-01": (
                "AMD reported one customer at 16 percent; Boeing reported U.S. "
                "government contracts at 40 percent. The 24 percentage-point "
                "difference is not directly comparable, and Boeing is cyclical."
            ),
            "complex-02": (
                "American Express's effective tax rate was 24.6% in 2021 and "
                "21.6% in 2022, a decrease of 3.0 percentage points. Boeing's "
                "effective tax rate was 14.8% in 2021 and -0.6% in 2022, a "
                "decrease of 15.4 percentage points. Boeing's change in magnitude "
                "was larger."
            ),
            "complex-07": (
                "Other was 2.95 percent of sales. Fixed Price rose from 1,146.2 "
                "to 1,452.4, offsetting Other so total sales were highest."
            ),
            "complex-08": (
                "The residual was 94.2, and only 1 segment exceeded $50 million."
            ),
        }
        cases = {
            str(case["case_id"]): case
            for case in COMPLEX_CASE_DEFINITIONS
            if case["case_id"] in answers
        }

        for case_id, answer in answers.items():
            case = cases[case_id]
            run = {
                "status": "completed",
                "answer": answer,
                "agent": _agent(),
                "citations": [
                    {"document_id": document_id}
                    for document_id in case["required_citation_document_ids"]
                ],
            }
            with self.subTest(case_id=case_id):
                score = score_complex_case(case, run)
                self.assertTrue(score["strict_correct"], score["aspects"])

    def test_evaluation_parallelism_is_fixed_at_one(self) -> None:
        base = {
            "top_k": 10,
            "parallelism": 1,
            "timeout_seconds": 900.0,
            "poll_seconds": 1.0,
            "strategy": "exact_vector",
            "rerank_mode": "classic",
        }
        _validate_options(argparse.Namespace(**base))
        for parallelism in (0, 2, 3):
            with self.subTest(parallelism=parallelism), self.assertRaises(ValueError):
                _validate_options(
                    argparse.Namespace(**{**base, "parallelism": parallelism})
                )

    def test_loopback_api_validation_rejects_credentials_and_query(self) -> None:
        self.assertEqual(
            _validated_api_base("http://127.0.0.1:8000/api/v1/"),
            "http://127.0.0.1:8000/api/v1",
        )
        for value in (
            "https://127.0.0.1:8000/api/v1",
            "http://user:secret@127.0.0.1:8000/api/v1",
            "http://localhost:8000/api/v1?token=secret",
            "http://localhost.evil:8000/api/v1",
            "http://127.0.0.1:8000/v1",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validated_api_base(value)

    def test_decimal_parser_preserves_parentheses_and_commas(self) -> None:
        self.assertIn(-1234.50, _extract_decimal_values("($1,234.50)"))
        self.assertIn(Decimal("2.95"), _extract_decimal_values("2.95%"))

    def test_scoring_requires_completion_aspects_and_exact_document_citations(self) -> None:
        case = {
            "case_id": "complex-test",
            "required_citation_document_ids": ["doc-a", "doc-b"],
            "forbid_unrelated_citations": True,
            "evaluation_group": "evidence_only",
            "aspects": [
                {
                    "aspect_id": "amount",
                    "answer_variants": ["total", "1,234.50"],
                    "expected_decimal": "1234.50",
                    "numeric_tolerance": "0.01",
                }
            ],
        }
        run = {
            "status": "completed",
            "answer": "The total is $1,234.50.",
            "agent": _agent(),
            "citations": [
                {"document_id": "doc-a"},
                {"document_id": "doc-b"},
            ],
        }
        score = score_complex_case(case, run)
        self.assertTrue(score["strict_correct"])
        self.assertEqual(score["required_document_citation_coverage"], 1.0)

        run["citations"].append({"document_id": "unrelated"})
        score = score_complex_case(case, run)
        self.assertFalse(score["strict_correct"])
        self.assertEqual(score["forbidden_citation_document_ids"], ["unrelated"])

    def test_scoring_maps_runtime_citation_filename_to_corpus_document_id(self) -> None:
        case = {
            "case_id": "complex-filename-map",
            "required_citation_document_ids": ["doc-a"],
            "forbid_unrelated_citations": True,
            "aspects": [
                {
                    "aspect_id": "fact",
                    "answer_variants": ["answer"],
                    "expected_decimal": None,
                }
            ],
        }

        score = score_complex_case(
            case,
            {
                "status": "completed",
                "answer": "The answer is grounded.",
                "agent": _agent(),
                "citations": [
                    {
                        "document_id": "runtime-uuid",
                        "document_original_filename": "DOC-A.pdf",
                    }
                ],
            },
            document_identity_map={"doc-a.pdf": "doc-a"},
        )

        self.assertTrue(score["strict_correct"])
        self.assertEqual(score["cited_document_ids"], ["doc-a"])

    def test_identity_map_accepts_logical_document_filename_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "document_id": "doc-a",
                                "path": "documents/pdf/doc-a.pdf",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            identity_map = load_document_identity_map(root)

        self.assertEqual(identity_map["doc-a.pdf"], "doc-a")

    def test_negative_change_accepts_signed_or_qualified_decrease(self) -> None:
        case = {
            "case_id": "complex-negative",
            "required_citation_document_ids": [],
            "forbid_unrelated_citations": True,
            "aspects": [
                {
                    "aspect_id": "change",
                    "answer_variants": ["decreased", "3.0 percentage points"],
                    "expected_decimal": "-3.0",
                    "numeric_tolerance": "0.01",
                }
            ],
        }
        score = score_complex_case(
            case,
            {
                "status": "completed",
                "answer": "The rate decreased by 3.0 percentage points.",
                "agent": _agent(),
                "citations": [],
            },
        )
        self.assertTrue(score["strict_correct"])

    def test_aspect_answer_match_any_accepts_one_synonym(self) -> None:
        score = score_complex_case(
            {
                "case_id": "complex-synonym",
                "required_citation_document_ids": [],
                "forbid_unrelated_citations": True,
                "aspects": [
                    {
                        "aspect_id": "label",
                        "answer_variants": ["entailment", "entailed"],
                        "answer_match": "any",
                    }
                ],
            },
            {
                "status": "completed",
                "answer": "The statement is entailed.",
                "agent": _agent(),
                "citations": [],
            },
        )
        self.assertTrue(score["strict_correct"])

    def test_summary_separates_domain_inference(self) -> None:
        results = [
            {"score": {"strict_correct": True, "at_least_partial": True, "terminal_completed": True, "evaluation_group": "evidence_only", "required_document_citation_coverage": 1.0, "forbidden_citation_document_ids": []}},
            {"score": {"strict_correct": False, "at_least_partial": True, "terminal_completed": True, "evaluation_group": "domain_inference", "required_document_citation_coverage": 1.0, "forbidden_citation_document_ids": []}},
        ]
        summary = summarize_results(results)
        self.assertEqual(summary["strict_correct"], 1)
        self.assertEqual(summary["evidence_only_strict_correct"], 1)
        self.assertEqual(summary["domain_inference_cases"], 1)

    def test_summary_reports_answered_precision_and_runtime_budget_facts(self) -> None:
        result = {
            "score": {
                "strict_correct": False,
                "at_least_partial": True,
                "terminal_completed": True,
                "evaluation_group": "evidence_only",
                "required_document_citation_coverage": 1.0,
                "forbidden_citation_document_ids": [],
                "agent_outcome": "answered",
                "answered_precision_ok": True,
            },
            "status": "completed",
            "elapsed_seconds": 4.0,
            "usage": {"totals": {"total_tokens": 12}},
            "timing": {"attempts": {"1": {"diagnostic": {}}}},
        }

        summary = summarize_results([result])

        self.assertEqual(summary["answered_cases"], 1)
        self.assertEqual(summary["answered_precision_cases"], 1)
        self.assertEqual(summary["total_tokens"], 12)
        self.assertEqual(summary["median_elapsed_seconds"], 4.0)
        self.assertEqual(summary["max_elapsed_seconds"], 4.0)
        self.assertEqual(summary["chat_run_retry_cases"], 0)
        self.assertEqual(summary["total_timeout_cases"], 0)
        self.assertEqual(summary["requested_cases"], 1)
        self.assertEqual(summary["attempted_cases"], 1)
        self.assertEqual(summary["not_run_cases"], 0)


if __name__ == "__main__":
    unittest.main()
