from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from tools import retrieval_evaluation


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "evaluation/configs/retrieval-evaluation-v1.0.json"
STARTED_AT = datetime(2026, 7, 15, 4, 0, tzinfo=UTC)


class RetrievalEvaluationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = retrieval_evaluation.load_inputs(CONFIG, ROOT)
        cls.executions = _executions(cls.inputs["cases"])
        cls.report = retrieval_evaluation.build_report(
            cls.inputs,
            cls.executions,
            run_id=UUID("01900000-0000-7000-8000-000000005001"),
            workspace_id=UUID("01900000-0000-7000-8000-000000005002"),
            knowledge_base_id=UUID("01900000-0000-7000-8000-000000005003"),
            revision_id=UUID("01900000-0000-7000-8000-000000005004"),
            started_at=STARTED_AT,
            finished_at=STARTED_AT + timedelta(seconds=1),
            commit="b4c063e000000000000000000000000000000000",
            database_metadata={
                "postgresql_version": "18",
                "pgvector_version": "0.8.2",
                "chunk_count": 61,
                "vector_count": 61,
                "eligible_chunk_count_after_mandatory_filters": 61,
                "concurrency": 1,
                "warm_up_queries": 0,
                "provider_network_included_in_query_embedding_ms": True,
            },
        )

    def test_report_compares_the_same_cases_and_passes_exact_gate(self) -> None:
        report = self.report

        self.assertEqual(report["failures"]["attempted_cases"], 18)
        self.assertEqual(report["failures"]["failure_rate"], 0.0)
        self.assertEqual(report["gate_decisions"]["exact_vector"]["status"], "passed")
        self.assertFalse(report["gate_decisions"]["hnsw"]["enabled"])
        self.assertEqual(
            {strategy["strategy_id"] for strategy in report["strategies"]},
            {"exact-vector-cosine-v1", "unicode-chargram-bm25-v1"},
        )
        self.assertEqual(
            report["strategies"][0]["overall"]["recall_at_k"]["5"],
            1.0,
        )
        self.assertEqual(
            report["inputs"]["embedding_model"]["resolved_model"],
            "text-embedding-v4",
        )

    def test_required_segments_and_latency_are_recorded(self) -> None:
        segments = self.report["segments"]

        self.assertTrue(
            any(
                item["dimension"] == "language" and item["value"] == "mixed"
                for item in segments
            )
        )
        self.assertTrue(
            any(
                item["dimension"] == "tag" and item["value"] == "exact_identifier"
                for item in segments
            )
        )
        self.assertTrue(
            any(item["dimension"] == "expected_empty_reason" for item in segments)
        )
        self.assertEqual(self.report["latency"]["query_embedding_ms"]["count"], 18)
        self.assertEqual(self.report["latency"]["retrieval_database_ms"]["p95"], 2.0)
        self.assertEqual(
            self.report["answer_metrics"]["label_source"],
            "not_evaluated_stage04_retrieval_only",
        )

    def test_report_validation_rejects_failed_gate_and_hnsw_enablement(self) -> None:
        retrieval_evaluation.validate_report(self.report, self.inputs)
        changed = {**self.report, "gate_decisions": {**self.report["gate_decisions"]}}
        changed["gate_decisions"]["exact_vector"] = {
            **changed["gate_decisions"]["exact_vector"],
            "status": "failed",
        }

        with self.assertRaisesRegex(ValueError, "regression gate"):
            retrieval_evaluation.validate_report(changed, self.inputs)


def _executions(cases) -> tuple[retrieval_evaluation.CaseExecution, ...]:
    executions = []
    for case in cases:
        relevant = tuple(case["retrieval"]["expected_relevant_sample_ids"])
        chunks = tuple(
            {
                "rank": rank,
                "sample_id": sample_id,
                "index_chunk_id": f"01900000-0000-7000-8000-{rank:012d}",
                "chunk_ordinal": 0,
                "score": 0.9,
            }
            for rank, sample_id in enumerate(relevant, start=1)
        )
        executions.append(
            retrieval_evaluation.CaseExecution(
                case_id=case["case_id"],
                sample_ids=relevant,
                ranked_chunks=chunks,
                query_embedding_ms=10.0,
                retrieval_database_ms=2.0,
                query_plan={
                    "strategy": "exact_vector",
                    "top_k": 10,
                    "revision_selector": "active",
                    "current_document_version_only": True,
                    "build_status": "ready",
                    "serving_status": "serving",
                    "distance_metric": "cosine",
                    "candidate_count": None,
                    "ef_search": None,
                    "iterative_scan": "disabled",
                    "rerank": False,
                },
                result_count=len(relevant),
            )
        )
    return tuple(executions)


if __name__ == "__main__":
    unittest.main()
