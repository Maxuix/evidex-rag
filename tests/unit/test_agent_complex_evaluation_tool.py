from __future__ import annotations

import unittest
from decimal import Decimal

from tools.evaluate_agent_complex_qa import (
    _extract_decimal_values,
    _validated_api_base,
    score_complex_case,
    summarize_results,
)


class AgentComplexEvaluationToolTests(unittest.TestCase):
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
            "workflow": {"requested_mode": "agent", "resolved_mode": "agent"},
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
                "workflow": {"requested_mode": "agent", "resolved_mode": "agent"},
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
                "workflow": {"requested_mode": "agent", "resolved_mode": "agent"},
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
                        "answer_variants": ["not_mentioned", "not mentioned"],
                        "answer_match": "any",
                    }
                ],
            },
            {
                "status": "completed",
                "answer": "The result is not mentioned.",
                "workflow": {"requested_mode": "agent", "resolved_mode": "agent"},
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

    def test_scoring_rejects_not_mentioned_without_complete_scan(self) -> None:
        case = {
            "case_id": "complex-absence",
            "required_citation_document_ids": ["doc-a"],
            "forbid_unrelated_citations": True,
            "aspects": [
                {
                    "aspect_id": "absence",
                    "answer_variants": ["not mentioned"],
                    "answer_match": "any",
                    "requires_complete_scan": True,
                }
            ],
        }
        run = {
            "status": "completed",
            "answer": "The statement is not mentioned.",
            "workflow": {
                "requested_mode": "agent",
                "resolved_mode": "agent",
                "research_result": {
                    "status": "sufficient",
                    "complete_scan_document_count": 0,
                },
            },
            "citations": [{"document_id": "doc-a"}],
        }

        score = score_complex_case(case, run)

        self.assertTrue(score["not_mentioned_without_complete_scan"])
        self.assertFalse(score["strict_correct"])

    def test_summary_reports_sufficient_precision_and_runtime_budget_facts(self) -> None:
        result = {
            "score": {
                "strict_correct": False,
                "at_least_partial": True,
                "terminal_completed": True,
                "evaluation_group": "evidence_only",
                "required_document_citation_coverage": 1.0,
                "forbidden_citation_document_ids": [],
                "research_status": "sufficient",
                "sufficient_precision_ok": True,
                "not_mentioned_without_complete_scan": False,
            },
            "status": "completed",
            "elapsed_seconds": 4.0,
            "usage": {"totals": {"total_tokens": 12}},
            "timing": {"attempts": {"1": {"diagnostic": {}}}},
        }

        summary = summarize_results([result])

        self.assertEqual(summary["sufficient_cases"], 1)
        self.assertEqual(summary["sufficient_precision_cases"], 1)
        self.assertEqual(summary["total_tokens"], 12)
        self.assertEqual(summary["median_elapsed_seconds"], 4.0)
        self.assertEqual(summary["max_elapsed_seconds"], 4.0)
        self.assertEqual(summary["chat_run_retry_cases"], 0)
        self.assertEqual(summary["total_timeout_cases"], 0)


if __name__ == "__main__":
    unittest.main()
