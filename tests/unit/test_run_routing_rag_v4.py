from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools import run_routing_rag_v4 as v4
from tools.run_routing_rag_v4 import (
    CONFIRM,
    LOCKED_SCHEMA,
    OBSERVATION_SCHEMA,
    gold_path_by_hop,
    expected_route_labels,
)


def _case(
    *,
    case_id: str,
    semantic_intent: str = "simple",
    answerable: bool = True,
    negative_control_kind: str | None = None,
    gold_paths: tuple[tuple[str, ...], ...] = (),
    answer_gold: tuple[str, ...] = (),
) -> dict:
    return {
        "case_id": case_id,
        "semantic_intent": semantic_intent,
        "answerable": answerable,
        "negative_control_kind": negative_control_kind,
        "valid_paths": [list(path) for path in gold_paths],
        "answer_gold_relation_ids": list(answer_gold),
        "question": f"question-{case_id}",
        "forbidden_claims": [],
    }


class V4LabelsTests(unittest.TestCase):
    def test_labels_are_stable_and_cover_all_three_strata(self) -> None:
        cases = [
            _case(
                case_id="graph-1",
                semantic_intent="graph",
                gold_paths=(("A", "B", "C"),),
                answer_gold=("A",),
            ),
            _case(case_id="graph-2", semantic_intent="graph", gold_paths=(("A",),), answer_gold=("A",)),
            _case(case_id="simple-1"),
            _case(case_id="negative-1", answerable=False),
            _case(case_id="negative-2", negative_control_kind="open_world_unanswerable"),
        ]
        labels = expected_route_labels(cases)
        self.assertEqual(labels["graph-1"], "graph_needed")
        self.assertEqual(labels["graph-2"], "simple_only")
        self.assertEqual(labels["simple-1"], "simple_only")
        self.assertEqual(labels["negative-1"], "negative_or_refusal")
        self.assertEqual(labels["negative-2"], "negative_or_refusal")

    def test_gold_hop_stratification(self) -> None:
        cases = [
            _case(
                case_id="g1",
                semantic_intent="graph",
                gold_paths=(("A", "B"), ("C", "D", "E")),
            ),
            _case(case_id="s1"),
        ]
        by_hop = gold_path_by_hop(cases)
        self.assertEqual(by_hop["hop2"], ["g1"])
        self.assertEqual(by_hop["hop3"], ["g1"])
        self.assertEqual(by_hop["hop1"], [])

    def test_corpus_dry_run_shape(self) -> None:
        corpus = v4._load_corpus()
        labels = expected_route_labels(corpus["cases"])
        by_hop = gold_path_by_hop(corpus["cases"])
        self.assertEqual(corpus["dataset_id"], "routing-rag-v3-open-source")
        self.assertEqual(len(corpus["cases"]), 28)
        self.assertEqual(
            sum(1 for value in labels.values() if value == "graph_needed"), 12
        )
        self.assertEqual(
            sum(1 for value in labels.values() if value == "simple_only"), 8
        )
        self.assertEqual(
            sum(1 for value in labels.values() if value == "negative_or_refusal"), 8
        )
        self.assertEqual(len(by_hop["hop2"]), 8)
        self.assertEqual(len(by_hop["hop3"]), 4)


class V4ReportTests(unittest.TestCase):
    def _checkpoint(self, records: list[dict]) -> Path:
        payload = {
            "schema_version": OBSERVATION_SCHEMA,
            "dataset_id": "routing-rag-v3-open-source",
            "corpus_sha256": "a" * 64,
            "runtime": {"knowledge_base_id": "kb-a"},
            "labels": {},
            "case_observations": records,
        }
        directory = tempfile.mkdtemp()
        path = Path(directory) / "checkpoint.json"
        v4._write_checkpoint(path, payload)
        return path

    def _auto(self, case_id: str, repeat: int, *, admitted: bool, first: int, second: int) -> dict:
        return {
            "case_id": case_id,
            "lane": "auto",
            "repeat": repeat,
            "graph_route_attempted": True,
            "graph_route_admitted": admitted,
            "graph_new_evidence_count": first,
            "graph_call_count": 2,
            "increments": {"first": first, "second": second},
            "graph_call_duration_ms": [100, 200],
            "timeout_count": 0,
            "chatrun_duration_ms": 30_000,
            "usage": {"model_rounds": 5, "retrieval_calls": 3, "calculation_calls": 0, "evidence_refs": 4, "total_tokens": 1000},
            "estimated_cost_usd": 0.001,
            "forbidden_claim_hit": False,
        }

    def test_report_aggregates_route_recall_over_repeats(self) -> None:
        cases = [
            _case(case_id="graph-1", semantic_intent="graph", gold_paths=(("A", "B"),), answer_gold=("A",)),
            _case(case_id="simple-1"),
            _case(case_id="negative-1", answerable=False),
        ]
        labels = expected_route_labels(cases)
        records = [
            self._auto("graph-1", 1, admitted=True, first=1, second=1),
            self._auto("graph-1", 2, admitted=True, first=1, second=0),
            self._auto("graph-1", 3, admitted=True, first=2, second=0),
            {
                "case_id": "graph-1",
                "lane": "simple",
                "repeat": 1,
                "graph_route_attempted": False,
                "graph_route_admitted": False,
                "graph_new_evidence_count": 0,
                "graph_call_count": 0,
                "increments": {"first": 0, "second": 0},
                "graph_call_duration_ms": [],
                "timeout_count": 0,
                "chatrun_duration_ms": 5_000,
                "usage": {"total_tokens": 500},
                "estimated_cost_usd": 0.0,
                "forbidden_claim_hit": False,
            },
        ]
        checkpoint = self._checkpoint(records)
        arguments = type("A", (), {
            "checkpoint": checkpoint,
            "locked_output": None,
        })()
        corpus = {
            "dataset_id": "routing-rag-v3-open-source",
            "manifest_sha256": "a" * 64,
            "cases": cases,
        }
        report = v4._run_report(arguments, corpus, labels)  # type: ignore[arg-type]
        self.assertEqual(report["schema_version"], LOCKED_SCHEMA)
        self.assertEqual(report["graph_needed_case_count"], 1)
        self.assertEqual(report["repeat_count"], 3)
        self.assertEqual(report["route"]["recall"], 1.0)
        self.assertEqual(report["route"]["precision"], 1.0)
        self.assertTrue(report["acceptance"]["route_recall"])
        self.assertEqual(report["graph_call_increments"]["first_call_new_evidence_mean"], 4 / 3)
        self.assertEqual(report["graph_call_increments"]["second_call_new_evidence_mean"], 1 / 3)
        self.assertEqual(report["graph_call_timeouts"], 0)
        self.assertEqual(report["graph_call_duration_p95_ms"], 200)
        self.assertTrue(report["acceptance"]["forbidden_claim_clear"])

    def test_report_counts_a_missed_graph_case_as_false_negative(self) -> None:
        cases = [
            _case(case_id="graph-1", semantic_intent="graph", gold_paths=(("A", "B"),), answer_gold=("A",)),
        ]
        labels = expected_route_labels(cases)
        records = [
            self._auto("graph-1", 1, admitted=False, first=0, second=0),
            self._auto("graph-1", 2, admitted=False, first=0, second=0),
        ]
        checkpoint = self._checkpoint(records)
        arguments = type("A", (), {"checkpoint": checkpoint, "locked_output": None})()
        report = v4._run_report(arguments, {"dataset_id": "d", "manifest_sha256": "a" * 64, "cases": cases}, labels)  # type: ignore[arg-type]
        self.assertEqual(report["route"]["false_negative"], 1)
        self.assertEqual(report["route"]["recall"], 0.0)
        self.assertFalse(report["acceptance"]["route_recall"])

    def test_checkpoint_identity_is_fail_closed(self) -> None:
        payload = {
            "schema_version": OBSERVATION_SCHEMA,
            "dataset_id": "routing-rag-v3-open-source",
            "corpus_sha256": "a" * 64,
            "runtime": {"knowledge_base_id": "kb-a"},
            "case_observations": [],
        }
        directory = tempfile.mkdtemp()
        path = Path(directory) / "checkpoint.json"
        v4._write_checkpoint(path, payload)
        self.assertEqual(
            v4._load_checkpoint(path, {"dataset_id": "routing-rag-v3-open-source"}),
            payload,
        )
        with self.assertRaises(v4.V4RunnerError):
            v4._load_checkpoint(path, {"dataset_id": "other"})
        wrong = dict(payload)
        wrong["schema_version"] = "future_schema"
        other = Path(directory) / "other.json"
        v4._write_checkpoint(other, wrong)
        with self.assertRaises(v4.V4RunnerError):
            v4._load_checkpoint(other, {})

    def test_confirm_flag_contract(self) -> None:
        self.assertEqual(CONFIRM, "RUN_ROUTING_RAG_V4_EXTERNAL_CALLS")
        self.assertTrue(len(CONFIRM) > 16)


if __name__ == "__main__":
    unittest.main()