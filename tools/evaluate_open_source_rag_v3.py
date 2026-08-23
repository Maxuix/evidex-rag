#!/usr/bin/env python3
"""Lock and score source-backed observations for routing-rag-v3.

This module is deliberately offline.  It does not retrieve evidence, call a
model, connect to a database, or manage a runtime.  A separately authorized
host runner may emit the observation artifact described here after using the
frozen semantic-v4 configuration.  This evaluator validates that artifact
against the immutable factual corpus and writes an independent locked result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID, uuid4

from rag_kb.domain import (
    GRAPH_AUGMENTATION_VERSION,
    GRAPH_EXTRACTOR_VERSION,
)
from rag_kb.retrieval.profile import (
    ADAPTIVE_GRAPHITI_PROFILE_VERSION,
    ADAPTIVE_GRAPHITI_ROUTER_VERSION,
    EXACT_PROFILE_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT / "evaluation/routing-rag-v3"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"
CASES_PATH = CORPUS_ROOT / "cases.jsonl"
RELATIONS_PATH = CORPUS_ROOT / "relations.jsonl"
ENTITIES_PATH = CORPUS_ROOT / "entities.jsonl"

DATASET_ID = "routing-rag-v3-open-source"
OBSERVATION_SCHEMA = "open_source_rag_v3_observations_v2"
LOCKED_SCHEMA = "open_source_rag_v3_locked_evaluation_v3"
LAYERS = ("raw", "hydrated", "reranked", "packed")
SIMPLE_TOP_K = 10
GRAPH_EDGE_LIMIT = 8


class V3EvaluationError(ValueError):
    """The v3 observation or frozen corpus violates the evaluation contract."""


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise V3EvaluationError(f"{path.name} contains a non-object row")
        rows.append(value)
    return tuple(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def frozen_configuration() -> dict[str, Any]:
    """Return the evaluator-owned configuration that every v3 run must use."""

    return {
        "parsing_preset": "text_local_v1",
        "chunking_preset": "semantic_balanced_v1",
        "chunking_profile": "semantic_breakpoint_v4",
        "embedding_strategy": "text_only",
        "simple": {
            "profile_version": EXACT_PROFILE_VERSION,
            "strategy": "exact_vector",
            "top_k": SIMPLE_TOP_K,
            "rerank_mode": "classic",
        },
        "graph": {
            "profile_version": ADAPTIVE_GRAPHITI_PROFILE_VERSION,
            "router_version": ADAPTIVE_GRAPHITI_ROUTER_VERSION,
            "augmentation_version": GRAPH_AUGMENTATION_VERSION,
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
            "edge_limit": GRAPH_EDGE_LIMIT,
            "rerank_mode": "classic",
            "layers": list(LAYERS),
        },
    }


def corpus_identity() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("dataset_id") != DATASET_ID:
        raise V3EvaluationError("v3 manifest identity is invalid")
    return {
        "dataset_id": DATASET_ID,
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "cases_sha256": _sha256(CASES_PATH),
        "relations_sha256": _sha256(RELATIONS_PATH),
        "entities_sha256": _sha256(ENTITIES_PATH),
        "document_set_sha256": _document_set_digest(manifest),
    }


def _document_set_digest(manifest: Mapping[str, Any]) -> str:
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise V3EvaluationError("v3 document manifest is invalid")
    digest = hashlib.sha256()
    for row in documents:
        if not isinstance(row, Mapping):
            raise V3EvaluationError("v3 document manifest row is invalid")
        filename = row.get("filename")
        expected = row.get("sha256")
        if not isinstance(filename, str) or not isinstance(expected, str):
            raise V3EvaluationError("v3 document identity is invalid")
        path = CORPUS_ROOT / "documents" / filename
        actual = _sha256(path)
        if actual != expected:
            raise V3EvaluationError("v3 document digest changed")
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(actual))
    return digest.hexdigest()


def _configuration_hash(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(configuration)).hexdigest()


def observation_template() -> dict[str, Any]:
    cases = _jsonl(CASES_PATH)
    configuration = frozen_configuration()
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "corpus": corpus_identity(),
        "configuration": configuration,
        "configuration_sha256": _configuration_hash(configuration),
        "runtime_identity": None,
        "case_observations": [
            {
                "case_id": str(case["case_id"]),
                "simple_relation_ids": None,
                "graph_full_relation_ids_by_layer": {
                    layer: None for layer in LAYERS
                },
                "graph_incremental_packed_relation_ids": None,
                "auto": {
                    "attempted": None,
                    "admitted": None,
                    "new_source_backed_evidence_count": None,
                },
                "actual_outcome": None,
                "forbidden_claim_hit": None,
            }
            for case in cases
        ],
        "graph_extraction": {
            "extracted_relation_count": None,
            "exact_matched_extracted_relation_count": None,
            "exact_gold_relation_ids": None,
            "topology_gold_relation_ids": None,
            "self_loop_count": None,
        },
    }


def _require_uuid_map(value: object) -> dict[str, str]:
    uuid_fields = {
        "workspace_id",
        "knowledge_base_id",
        "index_revision_id",
        "graph_build_id",
        "embedding_profile_revision_id",
        "graph_chat_profile_revision_id",
    }
    digest_fields = {
        "index_configuration_sha256",
        "serving_document_set_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != uuid_fields | digest_fields:
        raise V3EvaluationError("runtime identity is incomplete")
    result: dict[str, str] = {}
    for field in sorted(uuid_fields):
        candidate = value[field]
        if not isinstance(candidate, str):
            raise V3EvaluationError("runtime identity UUID is invalid")
        result[field] = str(UUID(candidate))
    for field in sorted(digest_fields):
        candidate = value[field]
        if (
            not isinstance(candidate, str)
            or len(candidate) != 64
            or any(character not in "0123456789abcdef" for character in candidate)
        ):
            raise V3EvaluationError("runtime identity digest is invalid")
        result[field] = candidate
    return result


def _relation_ids(
    value: object,
    *,
    allowed: frozenset[str],
    field: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise V3EvaluationError(f"{field} must be a relation-id list")
    result = tuple(dict.fromkeys(value))
    if len(result) != len(value) or not set(result) <= allowed:
        raise V3EvaluationError(f"{field} contains duplicate or unknown relations")
    return result


def _bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise V3EvaluationError(f"{field} must be boolean")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V3EvaluationError(f"{field} must be a non-negative integer")
    return value


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6) if denominator else None,
    }


def _path_score(
    paths: Sequence[Sequence[str]],
    observed: Iterable[str],
) -> tuple[int, int, float, bool]:
    observed_set = set(observed)
    candidates = []
    for path in paths:
        unique = tuple(dict.fromkeys(path))
        hits = len(set(unique) & observed_set)
        total = len(unique)
        candidates.append((hits / total, hits, -total, unique))
    if not candidates:
        return 0, 0, 0.0, False
    _, hits, negative_total, chosen = max(candidates)
    total = -negative_total
    return hits, total, hits / total, set(chosen) <= observed_set


def _path_observation(
    paths: Sequence[Sequence[str]],
    observed: Iterable[str],
) -> dict[str, Any]:
    hits, total, value, complete = _path_score(paths, observed)
    return {
        "relation_hits": hits,
        "relation_total": total,
        "path_recall": round(value, 6),
        "complete": complete,
    }


def _aggregate_path(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    relation_hits = sum(int(record[field]["relation_hits"]) for record in records)
    relation_total = sum(int(record[field]["relation_total"]) for record in records)
    complete = sum(bool(record[field]["complete"]) for record in records)
    return {
        "relation_recall": _rate(relation_hits, relation_total),
        "macro_path_recall": {
            "case_count": len(records),
            "value": round(
                sum(float(record[field]["path_recall"]) for record in records)
                / len(records),
                6,
            )
            if records
            else None,
        },
        "complete_path_rate": _rate(complete, len(records)),
    }


def _incremental_observation(
    paths: Sequence[Sequence[str]],
    simple: Iterable[str],
    graph: Iterable[str],
) -> dict[str, Any]:
    simple_set = set(simple)
    graph_set = set(graph)
    valid_path_relations = {
        relation_id for path in paths for relation_id in path
    }
    new_relations = graph_set - simple_set
    new_valid_path_relations = new_relations & valid_path_relations
    return {
        "new_relation_count": len(new_relations),
        "new_valid_path_relation_count": len(new_valid_path_relations),
        "adds_valid_path_relation": bool(new_valid_path_relations),
    }


def _validate_observations(
    value: object,
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    frozenset[str],
]:
    if not isinstance(value, dict) or set(value) != set(observation_template()):
        raise V3EvaluationError("observation artifact schema is invalid")
    if value.get("schema_version") != OBSERVATION_SCHEMA:
        raise V3EvaluationError("observation schema version is invalid")
    identity = corpus_identity()
    if value.get("corpus") != identity:
        raise V3EvaluationError("observation corpus identity changed")
    configuration = frozen_configuration()
    if value.get("configuration") != configuration:
        raise V3EvaluationError("observation configuration is not frozen")
    if value.get("configuration_sha256") != _configuration_hash(configuration):
        raise V3EvaluationError("observation configuration hash is invalid")
    runtime_identity = _require_uuid_map(value.get("runtime_identity"))
    cases = _jsonl(CASES_PATH)
    relations = _jsonl(RELATIONS_PATH)
    allowed = frozenset(str(row["relation_id"]) for row in relations)
    expected_ids = tuple(str(case["case_id"]) for case in cases)
    observations = value.get("case_observations")
    if not isinstance(observations, list) or len(observations) != len(cases):
        raise V3EvaluationError("case observations do not cover the corpus")
    normalized: list[dict[str, Any]] = []
    for expected_id, row in zip(expected_ids, observations):
        if not isinstance(row, Mapping) or set(row) != {
            "case_id",
            "simple_relation_ids",
            "graph_full_relation_ids_by_layer",
            "graph_incremental_packed_relation_ids",
            "auto",
            "actual_outcome",
            "forbidden_claim_hit",
        }:
            raise V3EvaluationError("case observation row is invalid")
        if row.get("case_id") != expected_id:
            raise V3EvaluationError("case observation order or identity changed")
        simple = _relation_ids(
            row.get("simple_relation_ids"), allowed=allowed, field="simple_relation_ids"
        )
        raw_layers = row.get("graph_full_relation_ids_by_layer")
        if not isinstance(raw_layers, Mapping) or set(raw_layers) != set(LAYERS):
            raise V3EvaluationError("Graph observation layers are invalid")
        layers = {
            layer: _relation_ids(
                raw_layers[layer], allowed=allowed, field=f"graph.{layer}"
            )
            for layer in LAYERS
        }
        incremental_packed = _relation_ids(
            row.get("graph_incremental_packed_relation_ids"),
            allowed=allowed,
            field="graph.incremental_packed",
        )
        auto = row.get("auto")
        if not isinstance(auto, Mapping) or set(auto) != {
            "attempted",
            "admitted",
            "new_source_backed_evidence_count",
        }:
            raise V3EvaluationError("Auto observation is invalid")
        attempted = _bool(auto.get("attempted"), field="auto.attempted")
        admitted = _bool(auto.get("admitted"), field="auto.admitted")
        new_count = _non_negative_int(
            auto.get("new_source_backed_evidence_count"),
            field="auto.new_source_backed_evidence_count",
        )
        if admitted and not attempted:
            raise V3EvaluationError("Auto admission requires an attempt")
        outcome = row.get("actual_outcome")
        if outcome not in {"answered", "partial", "refused"}:
            raise V3EvaluationError("answer outcome observation is invalid")
        forbidden = _bool(
            row.get("forbidden_claim_hit"), field="forbidden_claim_hit"
        )
        normalized.append(
            {
                "case_id": expected_id,
                "simple_relation_ids": simple,
                "graph_full_relation_ids_by_layer": layers,
                "graph_incremental_packed_relation_ids": incremental_packed,
                "auto": {
                    "attempted": attempted,
                    "admitted": admitted,
                    "new_source_backed_evidence_count": new_count,
                },
                "actual_outcome": outcome,
                "forbidden_claim_hit": forbidden,
            }
        )
    return runtime_identity, cases, tuple(normalized), allowed


def evaluate(value: object) -> dict[str, Any]:
    runtime_identity, cases, observations, allowed = _validate_observations(value)
    by_case = {row["case_id"]: row for row in observations}
    graph_cases = tuple(case for case in cases if case["semantic_intent"] == "graph")
    records: list[dict[str, Any]] = []
    for case in graph_cases:
        case_id = str(case["case_id"])
        observation = by_case[case_id]
        paths = case["valid_paths"]
        simple_ids = observation["simple_relation_ids"]
        simple_path = _path_observation(paths, simple_ids)
        graph_only_layers = {
            layer: _path_observation(
                paths, observation["graph_full_relation_ids_by_layer"][layer]
            )
            for layer in LAYERS
        }
        augmented_layers = {
            layer: _path_observation(
                paths,
                (*simple_ids, *observation["graph_full_relation_ids_by_layer"][layer]),
            )
            for layer in LAYERS
        }
        incremental_layers = {
            layer: _incremental_observation(
                paths,
                simple_ids,
                (
                    observation["graph_incremental_packed_relation_ids"]
                    if layer == "packed"
                    else observation["graph_full_relation_ids_by_layer"][layer]
                ),
            )
            for layer in LAYERS
        }
        graph_needed = not simple_path["complete"]
        benefit = (
            graph_needed
            and augmented_layers["packed"]["complete"]
            and incremental_layers["packed"]["adds_valid_path_relation"]
        )
        records.append(
            {
                "case_id": case_id,
                "graph_needed": graph_needed,
                "simple": simple_path,
                "graph_only": graph_only_layers,
                "augmented": augmented_layers,
                "incremental": incremental_layers,
                "graph_benefit": benefit,
                "auto": dict(observation["auto"]),
            }
        )

    graph_needed_count = sum(record["graph_needed"] for record in records)
    benefit_count = sum(record["graph_benefit"] for record in records)
    tp = fp = tn = fn = attempts = 0
    for record in records:
        attempted = bool(record["auto"]["attempted"])
        predicted = bool(record["auto"]["admitted"])
        needed = bool(record["graph_needed"])
        attempts += attempted
        tp += predicted and needed
        fp += predicted and not needed
        tn += not predicted and not needed
        fn += not predicted and needed

    extraction = value["graph_extraction"]
    if not isinstance(extraction, Mapping) or set(extraction) != {
        "extracted_relation_count",
        "exact_matched_extracted_relation_count",
        "exact_gold_relation_ids",
        "topology_gold_relation_ids",
        "self_loop_count",
    }:
        raise V3EvaluationError("Graph extraction observation is invalid")
    extracted_count = _non_negative_int(
        extraction.get("extracted_relation_count"), field="extracted_relation_count"
    )
    matched_extracted_count = _non_negative_int(
        extraction.get("exact_matched_extracted_relation_count"),
        field="exact_matched_extracted_relation_count",
    )
    exact_ids = _relation_ids(
        extraction.get("exact_gold_relation_ids"), allowed=allowed, field="exact_gold"
    )
    topology_ids = _relation_ids(
        extraction.get("topology_gold_relation_ids"),
        allowed=allowed,
        field="topology_gold",
    )
    self_loops = _non_negative_int(
        extraction.get("self_loop_count"), field="self_loop_count"
    )
    if (
        not set(exact_ids) <= set(topology_ids)
        or matched_extracted_count > extracted_count
    ):
        raise V3EvaluationError("Graph extraction alignment counts are inconsistent")
    focus_ids = frozenset(
        relation_id
        for case in graph_cases
        for path in case["valid_paths"]
        for relation_id in path
    )

    negative_cases = tuple(case for case in cases if case["category"] == "negative_control")
    outcome_correct = refusal_correct = refusal_total = forbidden_clear = 0
    for case in negative_cases:
        observation = by_case[str(case["case_id"])]
        outcome_correct += observation["actual_outcome"] == case["expected_outcome"]
        if case["expected_outcome"] == "refused":
            refusal_total += 1
            refusal_correct += observation["actual_outcome"] == "refused"
        forbidden_clear += not observation["forbidden_claim_hit"]

    configuration = frozen_configuration()
    observation_sha256 = hashlib.sha256(
        _canonical_bytes(value)  # type: ignore[arg-type]
    ).hexdigest()
    return {
        "schema_version": LOCKED_SCHEMA,
        "status": "computed",
        "corpus": corpus_identity(),
        "configuration": configuration,
        "configuration_sha256": _configuration_hash(configuration),
        "runtime_identity": runtime_identity,
        "observation_sha256": observation_sha256,
        "route_gold_policy": {
            "source": "simple_complete_valid_path_observation",
            "semantic_intent_is_primary_gold": False,
            "graph_full_is_independent_of_simple_evidence": True,
            "incremental_packed_is_reported_separately": True,
            "augmented_is_simple_union_graph": True,
            "graph_needed_rule": (
                "Simple misses every complete valid path and packed Graph adds "
                "at least one missing source-backed valid-path relation that "
                "completes a valid path in the augmented evidence"
            ),
        },
        "case_labels": records,
        "metrics": {
            "simple_path_recall": {
                "scope": "semantic_graph_candidates",
                **_aggregate_path(records, "simple"),
            },
            "graph_only_path_recall": {
                layer: _aggregate_path(
                    [
                        {
                            **record,
                            "selected_graph_layer": record["graph_only"][layer],
                        }
                        for record in records
                    ],
                    "selected_graph_layer",
                )
                for layer in LAYERS
            },
            "augmented_path_recall": {
                layer: _aggregate_path(
                    [
                        {
                            **record,
                            "selected_augmented_layer": record["augmented"][layer],
                        }
                        for record in records
                    ],
                    "selected_augmented_layer",
                )
                for layer in LAYERS
            },
            "graph_incremental": {
                layer: {
                    "new_relation_count": sum(
                        record["incremental"][layer]["new_relation_count"]
                        for record in records
                    ),
                    "new_valid_path_relation_count": sum(
                        record["incremental"][layer][
                            "new_valid_path_relation_count"
                        ]
                        for record in records
                    ),
                    "case_with_new_relation_rate": _rate(
                        sum(
                            record["incremental"][layer]["new_relation_count"] > 0
                            for record in records
                        ),
                        len(records),
                    ),
                    "case_with_new_valid_path_relation_rate": _rate(
                        sum(
                            record["incremental"][layer][
                                "adds_valid_path_relation"
                            ]
                            for record in records
                        ),
                        len(records),
                    ),
                }
                for layer in LAYERS
            },
            "graph_benefit": {
                "graph_needed_case_count": graph_needed_count,
                "benefit_case_count": benefit_count,
                "benefit_capture": _rate(benefit_count, graph_needed_count),
            },
            "auto": {
                "attempt_count": attempts,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": _rate(tp, tp + fp),
                "recall": _rate(tp, tp + fn),
            },
            "relation_extraction": {
                "gold_relation_count": len(allowed),
                "exact_match_recall": _rate(len(exact_ids), len(allowed)),
                "topology_recall": _rate(len(topology_ids), len(allowed)),
                "matched_edge_precision": _rate(
                    matched_extracted_count, extracted_count
                ),
                "focus_relation_count": len(focus_ids),
                "focus_exact_match_recall": _rate(
                    len(set(exact_ids) & focus_ids), len(focus_ids)
                ),
                "self_loop_count": self_loops,
            },
            "negative_controls": {
                "outcome_accuracy": _rate(outcome_correct, len(negative_cases)),
                "refusal_accuracy": _rate(refusal_correct, refusal_total),
                "forbidden_claim_clear_rate": _rate(
                    forbidden_clear, len(negative_cases)
                ),
            },
        },
    }


def _write_atomic(path: Path, value: Mapping[str, Any]) -> str:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def _write_locked(path: Path, value: Mapping[str, Any]) -> str:
    """Create an immutable result, or verify an identical prior result."""

    payload = _canonical_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise V3EvaluationError("locked evaluation output already differs")
        return digest
    return _write_atomic(path, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--write-observation-template", type=Path)
    group.add_argument("--observations", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    template = observation_template()
    if arguments.dry_run:
        print(
            json.dumps(
                {
                    "status": "offline_dry_run_ok",
                    "corpus": template["corpus"],
                    "configuration_sha256": template["configuration_sha256"],
                    "case_count": len(template["case_observations"]),
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.write_observation_template is not None:
        if arguments.output is not None:
            raise SystemExit("--output is only valid with --observations")
        digest = _write_atomic(arguments.write_observation_template, template)
        print(json.dumps({"status": "template_written", "sha256": digest}))
        return 0
    if arguments.output is None:
        raise SystemExit("--output is required with --observations")
    try:
        value = json.loads(arguments.observations.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("observation artifact could not be read") from error
    result = evaluate(value)
    digest = _write_locked(arguments.output, result)
    print(json.dumps({"status": "locked", "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
