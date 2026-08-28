#!/usr/bin/env python3
"""Validate frozen corpora and write the binding plan for a long RAG evaluation.

This is an offline preparation step.  It makes no provider call, starts no
host process, and does not ingest any document.  Its private output is the
immutable input to later provision and execution phases.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from tools import build_enterprise_profile_qualification as enterprise
from tools import build_musique_route_candidates as routing
from tools import build_public_rag_benchmark_suite as public
from tools.evaluation_campaign_state import digest, write_private_json


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "evaluation/public-rag-benchmark-suite-v1"
ROUTING_ROOT = ROOT / "evaluation/routing-rag-musique-expanded-v1"
ENTERPRISE_ROOT = ROOT / "evaluation/enterprise-profile-qualification-v1"
GRAPH_RAG_MANIFEST = ROOT / "evaluation/graph-rag-v1/manifest.json"
SCHEMA_VERSION = "large_rag_evaluation_plan_v1"

# This is a measurement, not a prediction: the host-only feasibility probe used
# 11,242 tokens for one deliberately broad Agent question.  It gives a safe
# provisional reservation until a representative per-stratum calibration run.
FEASIBILITY_OBSERVATION_TOKENS = 11_242


class LargeEvaluationPreparationError(RuntimeError):
    """Raised when the frozen data cannot support a meaningful campaign."""


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and isinstance((value := json.loads(line)), dict)
    ]


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_identity(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str):
        raise LargeEvaluationPreparationError("dataset_identity_invalid")
    return {"dataset_id": dataset_id, "manifest_sha256": _file_digest(path)}


def build_plan(*, routing_root: Path = ROUTING_ROOT) -> dict[str, Any]:
    """Return an offline-validated campaign plan for the three target gaps."""

    public_result = public.validate(PUBLIC_ROOT)
    routing_result = routing.validate(routing_root)
    enterprise_result = enterprise.validate(ENTERPRISE_ROOT)
    public_cases = _rows(PUBLIC_ROOT / "cases.jsonl")
    routing_cases = _rows(routing_root / "cases.jsonl")
    graph_contract = json.loads(GRAPH_RAG_MANIFEST.read_text(encoding="utf-8"))
    if graph_contract.get("current_graph_contract", {}).get("expected_online_max_hops") != 3:
        raise LargeEvaluationPreparationError("graph_contract_not_three_hop")
    graph_answer_cases = graph_contract.get("supported_case_count")
    if not isinstance(graph_answer_cases, int) or graph_answer_cases <= 0:
        raise LargeEvaluationPreparationError("graph_answer_denominator_invalid")

    action_counts = Counter(str(case.get("expected_action")) for case in public_cases)
    route_counts = Counter(str(case.get("route_label")) for case in routing_cases)
    graph_candidate_count = route_counts["graph_needed_candidate"]
    routing_manifest = json.loads(
        (routing_root / "manifest.json").read_text(encoding="utf-8")
    )
    qualification = routing_manifest.get("qualification", {})
    qualification_minimum = qualification.get("minimum_qualified_graph_needed_count")
    if not isinstance(qualification_minimum, int) or qualification_minimum < 30:
        raise LargeEvaluationPreparationError("routing_qualification_denominator_invalid")
    if graph_candidate_count < qualification_minimum:
        raise LargeEvaluationPreparationError("routing_candidate_pool_too_small")
    candidate_hops = Counter(
        str(case.get("hop_count"))
        for case in routing_cases
        if case.get("route_label") == "graph_needed_candidate"
    )

    routing_simple_observations = len(routing_cases)
    routing_auto_minimum = qualification_minimum * 3
    routing_auto_maximum = graph_candidate_count * 3
    answer_observation_minimum = (
        len(public_cases)
        + routing_simple_observations
        + routing_auto_minimum
        + enterprise_result["relation_types"]
        + graph_answer_cases
    )
    answer_observation_maximum = (
        len(public_cases)
        + routing_simple_observations
        + routing_auto_maximum
        + enterprise_result["relation_types"]
        + graph_answer_cases
    )
    enterprise_document_count = len(list((ENTERPRISE_ROOT / "documents").glob("*.md")))
    if enterprise_document_count != (
        enterprise_result["positive_relations"] + enterprise_result["controls"]
    ):
        raise LargeEvaluationPreparationError("enterprise_document_denominator_invalid")

    binding = {
        "corpora": [
            _manifest_identity(PUBLIC_ROOT / "manifest.json"),
            _manifest_identity(routing_root / "manifest.json"),
            _manifest_identity(ENTERPRISE_ROOT / "manifest.json"),
            _manifest_identity(GRAPH_RAG_MANIFEST),
        ],
        "provider_contract": {
            "provider": "OpenCode Go",
            "chat_model": "mimo-v2.5",
            "text_embedding_model": "qwen3.7-text-embedding",
            "multimodal_embedding_model": "tongyi-embedding-vision-flash-2026-03-06",
        },
        "runtime_contract": "isolated_host_native_only",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_binding": binding,
        "plan_binding_sha256": digest(binding),
        "created_at": datetime.now(UTC).isoformat(),
        "offline_validation": {
            "public_answer_refusal": public_result,
            "routing_candidates": routing_result,
            "enterprise_profile": enterprise_result,
        },
        "coverage": {
            "answer_refusal_expected_actions": dict(sorted(action_counts.items())),
            "answer_refusal_negative_denominators": {
                "refuse_insufficient_evidence": action_counts["refuse_insufficient_evidence"],
                "decline_or_correct_false_premise": action_counts[
                    "decline_or_correct_false_premise"
                ],
                "refuse_closed_world_absent": action_counts["refuse_closed_world_absent"],
                "surface_evidence_conflict": action_counts["surface_evidence_conflict"],
                "request_clarification": action_counts["request_clarification"],
            },
            "routing": {
                "dataset_id": routing_manifest.get("dataset_id"),
                "candidate_count": graph_candidate_count,
                "candidate_hops": dict(sorted(candidate_hops.items())),
                "simple_control_count": route_counts["simple_only"],
                "negative_or_refusal_count": route_counts["negative_or_refusal"],
                "minimum_qualified_graph_needed_count": qualification_minimum,
                "dynamic_qualification": {
                    "simple_top_k": qualification.get("simple_top_k"),
                    "graph_edge_limit": qualification.get("graph_edge_limit"),
                    "source_chunk_target": qualification.get("source_chunk_target"),
                    "requires_simple_complete_path_absent": True,
                    "requires_graph_complete_path_present": True,
                    "requires_graph_new_source_chunk": True,
                },
            },
            "enterprise": {
                "relation_types": enterprise_result["relation_types"],
                "positive_relations": enterprise_result["positive_relations"],
                "positive_examples_per_relation": 3,
                "negative_controls": enterprise_result["controls"],
                "direct_answer_cases": enterprise_result["relation_types"],
                "graph_final_answer_cases": graph_answer_cases,
                "online_max_hops": 3,
            },
        },
        "run_groups": [
            {
                "id": "public_answer_refusal",
                "observation_count": len(public_cases),
                "required_checkpoint_fields": [
                    "case_id",
                    "expected_action",
                    "actual_outcome",
                    "forbidden_claim_hit",
                    "retrieval_trace",
                    "model_call_usage",
                    "total_tokens",
                    "duration_ms",
                ],
            },
            {
                "id": "routing_qualification",
                "candidate_count": graph_candidate_count,
                "required_result_fields": [
                    "case_id",
                    "simple_complete_path_present",
                    "graph_complete_path_present",
                    "graph_new_source_chunk_count",
                    "qualified_graph_needed",
                ],
                "acceptance": {
                    "minimum_qualified_graph_needed_count": qualification_minimum,
                    "otherwise": "collect_or_build_another_corpus_variant",
                },
            },
            {
                "id": "routing_agent_answers",
                "simple_observation_count": routing_simple_observations,
                "auto_observation_count_range": [routing_auto_minimum, routing_auto_maximum],
                "required_checkpoint_fields": [
                    "case_id",
                    "lane",
                    "repeat",
                    "actual_outcome",
                    "graph_route_attempted",
                    "graph_route_admitted",
                    "graph_new_evidence_count",
                    "model_call_usage",
                    "total_tokens",
                    "duration_ms",
                ],
            },
            {
                "id": "enterprise_extraction_and_answering",
                "extraction_document_count": enterprise_document_count,
                "answer_observation_count": enterprise_result["relation_types"] + graph_answer_cases,
                "required_extraction_metrics": [
                    "micro_precision",
                    "micro_recall",
                    "micro_f1",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "per_relation_support",
                    "entity_type_confusion",
                    "forbidden_edge_hits",
                ],
            },
        ],
        "recovery_contract": {
            "checkpoint_schema": "large_evaluation_campaign_state_v1",
            "atomic_persistence": "fsync_then_replace_after_each_case_transition",
            "completed_case_policy": "skip_only_durably_completed_cases",
            "interrupted_case_policy": "record_and_retry_without_counting_as_complete",
            "locked_output_policy": "write_only_after_all_planned_observations_are_complete",
            "binding_policy": "reject_resume_when_any_corpus_or_provider_binding_changes",
        },
        "token_reservation": {
            "basis": "one real host feasibility observation, intentionally broad generic question",
            "observed_tokens_per_agent_observation": FEASIBILITY_OBSERVATION_TOKENS,
            "agent_observation_range_after_routing_qualification": [
                answer_observation_minimum,
                answer_observation_maximum,
            ],
            "smoke_rate_agent_token_range": [
                answer_observation_minimum * FEASIBILITY_OBSERVATION_TOKENS,
                answer_observation_maximum * FEASIBILITY_OBSERVATION_TOKENS,
            ],
            "enterprise_extraction_documents_not_yet_calibrated": enterprise_document_count,
            "interpretation": "provisional reservation only; calibrate per-stratum token use before setting a hard campaign cap",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional owner-only plan path, normally under .runtime/evaluations/",
    )
    parser.add_argument(
        "--routing-root",
        type=Path,
        default=ROUTING_ROOT,
        help="routing corpus root; defaults to the frozen expanded v1 corpus",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        plan = build_plan(routing_root=arguments.routing_root)
    except (enterprise.CorpusError, public.CorpusError, routing.CorpusError, LargeEvaluationPreparationError) as error:
        print(json.dumps({"status": "blocked", "failure_code": type(error).__name__}, sort_keys=True))
        return 2
    result: dict[str, Any] = {
        "status": "prepared",
        "plan_binding_sha256": plan["plan_binding_sha256"],
        "agent_observation_range": plan["token_reservation"]["agent_observation_range_after_routing_qualification"],
        "smoke_rate_agent_token_range": plan["token_reservation"]["smoke_rate_agent_token_range"],
    }
    if arguments.output is not None:
        result["output"] = str(arguments.output)
        result["output_sha256"] = write_private_json(arguments.output, plan)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
