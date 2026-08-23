#!/usr/bin/env python3
"""Offline contracts and diagnostics for the adaptive Graph RAG evaluation.

This module deliberately has no HTTP, database, embedding, Graphiti, or Judge
provider execution path.  Provider execution belongs to the separately
authorized R7 harness.  The local functions here are used by R1/R2 fake and
unit tests to freeze the evaluator contract without changing production wire
schemas or ChatRun snapshots.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import unicodedata
from typing import Any
from uuid import UUID

from rag_kb.domain import (
    CHAT_GRAPHITI_ROUTE_REASONS,
    CHAT_GRAPHITI_ROUTE_RESULTS,
    ChatModelMessage,
    ChatModelRequest,
    ChatModelResponse,
    GraphitiEdgeResult,
)
from rag_kb.ports.model_api import ChatModelAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "adaptive-graph-route-v2" / "manifest.json"
ROUTE_IDS = ("vector-only", "hybrid-control", "manual-graph", "auto-route")
LAYERS = ("raw", "hydrated", "reranked", "packed")
CASE_OUTCOMES = frozenset({"answered", "refused"})
EMPIRICAL_NEEDS = (
    "simple_multi_round_solves",
    "simple_retrieves_gold_but_answers_wrong",
    "simple_miss_graph_can_recover",
    "simple_miss_graph_also_miss",
    "must_not_call_graph",
)
NEGATIVE_CONTROL_KINDS = frozenset(
    {"contradicted", "closed_world_absence", "open_world_unanswerable"}
)
ROUTING_JUDGE_SCHEMA_VERSION = "routing_rag_v1_judge_v1"
ROUTING_JUDGE_PROMPT_VERSION = "routing_rag_v1_judge_prompt_v1"
REPLAY_CAPTURE_SCHEMA_VERSION = "adaptive_graph_replay_capture_v2"
REPLAY_CAPTURE_SOURCE = "r2_clean_forced"
REPLAY_QUERY_MAX_CHARS = 2048
EVALUATOR_EDGE_LIMITS = (8, 16, 32, 64)
GRAPH_ROUTE_LABELS = frozenset({"simple", "graph"})
FORCED_CONTROLLER_MODES = frozenset(
    {
        "specific_tool_choice",
        "single_tool_required_fallback",
        "single_tool_auto_fallback",
        "actual_auto",
    }
)
PROVIDER_SAFE_TOOL_ATTEMPTS = 2
_PROVIDER_SAFE_RETRY_PROMPT = {
    "search_knowledge_base": (
        "For this evaluator capture turn, call exactly the supplied "
        "search_knowledge_base tool. Do not submit an answer."
    ),
    "graphiti_supplement": (
        "For this evaluator capture turn, call exactly the supplied "
        "graphiti_supplement tool with one concise query for the remaining "
        "relation gap. Do not submit an answer."
    ),
}
RERANK_MODES = frozenset({"none", "classic", "local_minilm_v1"})
JUDGE_VERDICTS = frozenset({"correct", "partial", "incorrect"})
JUDGE_GROUNDING = frozenset({"supported", "partial", "unsupported"})
JUDGE_STANCES = frozenset({"affirmed", "denied", "abstained", "not_applicable"})
JUDGE_REASON_CODES = frozenset(
    {
        "answer_supported",
        "answer_partially_supported",
        "answer_incorrect",
        "refusal_correct",
        "refusal_incorrect",
        "citation_missing",
        "citation_misaligned",
    }
)
_TERM_PUNCTUATION = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "、": ",",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


def normalize_term(value: str) -> str:
    """Normalize benchmark proxy terms without making them a Judge."""

    if not isinstance(value, str):
        raise TypeError("term must be a string")
    normalized = unicodedata.normalize("NFKC", value).translate(_TERM_PUNCTUATION)
    return re.sub(r"\s+", "", normalized)


def evaluate_graph_extraction(
    observed_edges: Sequence[GraphitiEdgeResult],
    *,
    entity_rows: Sequence[Mapping[str, Any]],
    relation_rows: Sequence[Mapping[str, Any]],
    focus_relation_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Align one extracted graph to relation gold without persisting content.

    Endpoint aliases, relation facts, and entity names are used only in memory.
    The returned diagnostic contains aggregate counts and synthetic relation
    identifiers, so it is safe to add to the redacted evaluator artifact.
    """

    entity_surfaces: dict[str, tuple[str, ...]] = {}
    entity_ids_by_surface: dict[str, set[str]] = {}
    for row in entity_rows:
        entity_id = str(row.get("entity_id", "")).strip()
        canonical = row.get("canonical_name")
        aliases = row.get("aliases", ())
        if (
            not entity_id
            or not isinstance(canonical, str)
            or not canonical.strip()
            or not isinstance(aliases, (list, tuple))
            or any(not isinstance(item, str) or not item.strip() for item in aliases)
        ):
            raise ValueError("graph extraction entity gold is invalid")
        normalized_surfaces = tuple(
            dict.fromkeys(
                surface
                for value in (canonical, *aliases)
                if (surface := _normalize_graph_surface(value))
            )
        )
        if not normalized_surfaces or entity_id in entity_surfaces:
            raise ValueError("graph extraction entity gold is invalid")
        entity_surfaces[entity_id] = normalized_surfaces
        for surface in normalized_surfaces:
            entity_ids_by_surface.setdefault(surface, set()).add(entity_id)

    relations: dict[str, Mapping[str, Any]] = {}
    for row in relation_rows:
        relation_id = str(row.get("relation_id", "")).strip()
        subject_id = str(row.get("subject_entity_id", "")).strip()
        object_id = str(row.get("object_entity_id", "")).strip()
        predicate = row.get("predicate")
        if (
            not relation_id
            or relation_id in relations
            or subject_id not in entity_surfaces
            or object_id not in entity_surfaces
            or not isinstance(predicate, str)
            or not _normalize_graph_surface(predicate)
        ):
            raise ValueError("graph extraction relation gold is invalid")
        relations[relation_id] = row
    if not relations:
        raise ValueError("graph extraction relation gold is empty")

    focus = tuple(dict.fromkeys(str(item) for item in focus_relation_ids))
    if any(item not in relations for item in focus):
        raise ValueError("graph extraction focus relation is unknown")

    endpoint_entity_ids: dict[str, frozenset[str]] = {}
    endpoint_name_by_uuid: dict[str, str] = {}
    ambiguous_endpoint_uuids: set[str] = set()
    unmapped_endpoint_uuids: set[str] = set()
    self_loop_edge_ids: set[str] = set()
    observed: list[
        tuple[GraphitiEdgeResult, frozenset[str], frozenset[str], str, str]
    ] = []
    for edge in observed_edges:
        source_name = _normalize_graph_surface(edge.source_entity_name)
        target_name = _normalize_graph_surface(edge.target_entity_name)
        for entity_uuid, normalized_name in (
            (edge.source_entity_uuid, source_name),
            (edge.target_entity_uuid, target_name),
        ):
            endpoint_name_by_uuid.setdefault(entity_uuid, normalized_name)
            matches = frozenset(entity_ids_by_surface.get(normalized_name, ()))
            endpoint_entity_ids[entity_uuid] = matches
            if len(matches) > 1:
                ambiguous_endpoint_uuids.add(entity_uuid)
            elif not matches:
                unmapped_endpoint_uuids.add(entity_uuid)
        if (
            not edge.source_entity_uuid
            or not edge.target_entity_uuid
            or edge.source_entity_uuid == edge.target_entity_uuid
            or source_name == target_name
        ):
            self_loop_edge_ids.add(edge.edge_uuid)
        observed.append(
            (
                edge,
                endpoint_entity_ids[edge.source_entity_uuid],
                endpoint_entity_ids[edge.target_entity_uuid],
                _normalize_graph_surface(edge.fact),
                _normalize_graph_surface(edge.relation_type),
            )
        )

    def score_scope(relation_ids: Iterable[str]) -> dict[str, Any]:
        ids = tuple(dict.fromkeys(relation_ids))
        directed_hits: set[str] = set()
        undirected_hits: set[str] = set()
        fact_surface_hits: set[str] = set()
        complete_hits: set[str] = set()
        matched_observed_edge_ids: set[str] = set()
        for relation_id in ids:
            row = relations[relation_id]
            subject_id = str(row["subject_entity_id"])
            object_id = str(row["object_entity_id"])
            predicate = _normalize_graph_surface(str(row["predicate"]))
            for edge, source_ids, target_ids, fact, relation_type in observed:
                directed = subject_id in source_ids and object_id in target_ids
                undirected = directed or (
                    subject_id in target_ids and object_id in source_ids
                )
                predicate_hit = relation_type == predicate or (
                    not relation_type and predicate in fact
                )
                surface_hit = bool(
                    any(surface in fact for surface in entity_surfaces[subject_id])
                    and any(surface in fact for surface in entity_surfaces[object_id])
                    and predicate_hit
                )
                if directed:
                    directed_hits.add(relation_id)
                if undirected:
                    undirected_hits.add(relation_id)
                if surface_hit:
                    fact_surface_hits.add(relation_id)
                if directed and predicate_hit:
                    complete_hits.add(relation_id)
                    matched_observed_edge_ids.add(edge.edge_uuid)
        total = len(ids)
        return {
            "gold_relation_count": total,
            "directed_topology": _rate(len(directed_hits), total),
            "undirected_topology": _rate(len(undirected_hits), total),
            "fact_surface": _rate(len(fact_surface_hits), total),
            "complete_relation": _rate(len(complete_hits), total),
            "matched_observed_edge_count": len(matched_observed_edge_ids),
            "directed_relation_ids": sorted(directed_hits),
            "complete_relation_ids": sorted(complete_hits),
            "missing_complete_relation_ids": sorted(set(ids) - complete_hits),
        }

    all_scope = score_scope(relations)
    matched_count = int(all_scope["matched_observed_edge_count"])
    observed_count = len(observed_edges)
    return {
        "schema_version": "graph_extraction_alignment_v2",
        "observed_edge_count": observed_count,
        "self_loop_edge_count": len(self_loop_edge_ids),
        "unique_endpoint_count": len(endpoint_name_by_uuid),
        "unmapped_endpoint_count": len(unmapped_endpoint_uuids),
        "ambiguous_endpoint_count": len(ambiguous_endpoint_uuids),
        "matched_observed_edge_precision": _rate(matched_count, observed_count),
        "all_relations": all_scope,
        "focus_relations": score_scope(focus) if focus else None,
    }


def _normalize_graph_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def validate_evaluator_edge_limit(value: int) -> int:
    """Validate an evaluator-only Graphiti K without changing production defaults."""

    if value not in EVALUATOR_EDGE_LIMITS:
        raise ValueError("evaluator edge limit must be one of 8, 16, 32, or 64")
    return value


def term_proxy(
    answer: str,
    expected_terms: Iterable[str],
) -> dict[str, Any]:
    normalized_answer = normalize_term(answer)
    normalized_terms = tuple(dict.fromkeys(normalize_term(item) for item in expected_terms))
    matched = tuple(item for item in normalized_terms if item and item in normalized_answer)
    return {
        "matched": len(matched),
        "total": len(normalized_terms),
        "all_matched": bool(normalized_terms) and len(matched) == len(normalized_terms),
        "empty_expected_terms": not normalized_terms,
    }


def validate_case_contract(cases: Sequence[Mapping[str, Any]]) -> None:
    """Validate the content contract independently of generated files."""

    seen: set[str] = set()
    negative_kinds: set[str] = set()
    for raw_case in cases:
        case_id = str(raw_case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError(f"duplicate or missing case_id: {case_id}")
        seen.add(case_id)
        outcome = raw_case.get("expected_outcome")
        if outcome not in CASE_OUTCOMES:
            raise ValueError(f"{case_id}: invalid expected_outcome")
        answerable = raw_case.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError(f"{case_id}: answerable must be bool")
        if answerable != (outcome == "answered"):
            raise ValueError(f"{case_id}: answerable/outcome mismatch")
        aspects = raw_case.get("expected_answer_aspects")
        if not isinstance(aspects, list):
            raise ValueError(f"{case_id}: expected_answer_aspects must be a list")
        if outcome == "answered" and not aspects:
            raise ValueError(f"{case_id}: answered case has no aspects")
        for aspect in aspects:
            if not isinstance(aspect, Mapping) or not str(aspect.get("aspect_id", "")):
                raise ValueError(f"{case_id}: invalid answer aspect")
            variants = aspect.get("answer_variants")
            if not isinstance(variants, list) or not variants or any(
                not isinstance(item, str) or not item.strip() for item in variants
            ):
                raise ValueError(f"{case_id}: answer aspect variants are invalid")
        locators = raw_case.get("answer_gold_source_locators")
        if not isinstance(locators, list):
            raise ValueError(f"{case_id}: answer_gold_source_locators must be a list")
        if outcome == "answered" and not locators:
            raise ValueError(f"{case_id}: answered case has no answer gold locator")
        if not isinstance(raw_case.get("path_context_locators"), list):
            raise ValueError(f"{case_id}: path_context_locators must be a list")
        if not isinstance(raw_case.get("forbidden_claims"), list):
            raise ValueError(f"{case_id}: forbidden_claims must be a list")
        kind = raw_case.get("negative_control_kind")
        if raw_case.get("category") == "negative_control":
            if kind not in NEGATIVE_CONTROL_KINDS:
                raise ValueError(f"{case_id}: invalid negative control kind")
            negative_kinds.add(str(kind))
            if kind == "contradicted" and outcome != "answered":
                raise ValueError(f"{case_id}: contradicted control must be answered")
            if kind == "open_world_unanswerable" and outcome != "refused":
                raise ValueError(f"{case_id}: open-world control must be refused")
        elif kind is not None:
            raise ValueError(f"{case_id}: non-negative case has negative kind")
        expected_route = raw_case.get("expected_route")
        if (
            not isinstance(expected_route, Mapping)
            or expected_route.get("route") not in GRAPH_ROUTE_LABELS
        ):
            raise ValueError(f"{case_id}: expected_route label is invalid")
        if expected_route.get("route") == "graph":
            source = raw_case.get("source")
            if not isinstance(source, Mapping):
                raise ValueError(f"{case_id}: graph case source is invalid")
            answer_relation_ids = {
                str(item) for item in source.get("answer_relation_ids", ())
            }
            locator_ids = {
                str(item.get("relation_id"))
                for item in locators
                if isinstance(item, Mapping) and item.get("kind") == "graph_relation"
            }
            if locator_ids != answer_relation_ids:
                raise ValueError(f"{case_id}: answer locator set differs from answer relations")
            context_ids = {
                str(item.get("relation_id"))
                for item in raw_case["path_context_locators"]
                if isinstance(item, Mapping) and item.get("kind") == "graph_relation"
            }
            if locator_ids & context_ids:
                raise ValueError(f"{case_id}: answer/path locators overlap")
            gold_path = source.get("gold_path")
            if (
                not isinstance(gold_path, (list, tuple))
                or not 1 <= len(gold_path) <= 2
                or any(not isinstance(item, str) or not item for item in gold_path)
                or len(set(gold_path)) != len(gold_path)
            ):
                raise ValueError(f"{case_id}: graph gold path is invalid")
            if set(gold_path) != locator_ids | context_ids:
                raise ValueError(
                    f"{case_id}: answer/path locators do not cover the gold path"
                )
    if negative_kinds != set(NEGATIVE_CONTROL_KINDS):
        raise ValueError("negative controls must cover all three semantic kinds")


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_case_contract(cases)
    return cases


def load_empirical_fixture(
    path: Path,
    *,
    source_case_ids: Iterable[str],
    required_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_ids = {str(item) for item in source_case_ids}
    seen: set[str] = set()
    counts = {item: 0 for item in EMPIRICAL_NEEDS}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("empirical fixture row must be an object")
        case_id = str(row.get("case_id", ""))
        source_case_id = str(row.get("source_case_id", ""))
        need = row.get("empirical_need")
        if (
            not case_id
            or case_id in seen
            or source_case_id not in source_ids
            or need not in counts
            or row.get("observation_required") is not True
            or set(row) != {"case_id", "source_case_id", "empirical_need", "observation_required"}
        ):
            raise ValueError("empirical fixture row is invalid")
        seen.add(case_id)
        counts[str(need)] += 1
    for need, minimum in required_counts.items():
        if need not in counts or counts[need] < minimum:
            raise ValueError(f"empirical fixture category is undersized: {need}")
    return [dict(row) for row in rows]


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "adaptive_graph_route_manifest_v2":
        raise ValueError("adaptive route manifest schema mismatch")
    if manifest.get("dataset_id") not in {"routing-rag-v1", "routing-rag-v2"}:
        raise ValueError("adaptive route manifest dataset identity is invalid")
    if tuple(item.get("id") for item in manifest.get("routes", ())) != ROUTE_IDS:
        raise ValueError("adaptive route manifest lanes are not frozen")
    contract = manifest.get("contract")
    primary_metrics = contract.get("primary_metrics") if isinstance(contract, Mapping) else None
    secondary_metrics = contract.get("secondary_metrics") if isinstance(contract, Mapping) else None
    expected_primary_metrics = (
        {
            "route_label": "simple_required_path_completeness",
            "route_recall": "graph_needed_route_recall",
            "route_accuracy": "graph_route_accuracy",
            "graph_recall": "packed_required_path_recall",
            "benefit_capture": "packed_path_completion_over_simple_incomplete",
        }
        if manifest.get("dataset_id") == "routing-rag-v2"
        else {
            "route_label": "expected_route.route",
            "route_recall": "graph_needed_route_recall",
            "route_accuracy": "graph_route_accuracy",
            "graph_recall": "packed_answer_gold_recall",
            "benefit_capture": "packed_new_answer_gold_over_simple_missing",
        }
    )
    if primary_metrics != expected_primary_metrics:
        raise ValueError("adaptive primary metric contract is not frozen")
    if secondary_metrics != {
        "token_ratio": "budget_context_only",
        "p95_latency_ratio": "budget_context_only",
    }:
        raise ValueError("adaptive secondary metric contract is not frozen")
    expected_benefit = (
        "simple_incomplete_required_path_and_agent_replay_completes_packed_path"
        if manifest.get("dataset_id") == "routing-rag-v2"
        else "simple_missing_answer_gold_and_agent_replay_packed_new_answer_gold"
    )
    if (
        contract.get("benefit") != expected_benefit
        or tuple(contract.get("layers", ())) != LAYERS
    ):
        raise ValueError("adaptive evidence layer contract is not frozen")
    if manifest.get("dataset_id") == "routing-rag-v2" and (
        contract.get("route_decision")
        != "admitted_source_backed_new_evidence"
        or contract.get("probe_diagnostic")
        != "graph_attempt_reported_separately_from_route_decision"
    ):
        raise ValueError("adaptive route decision contract is not frozen")
    case_file = (path.parent / str(manifest.get("case_file", ""))).resolve()
    cases = load_cases(case_file)
    if manifest.get("case_count") != len(cases):
        raise ValueError("adaptive route manifest case count mismatch")
    manifest["case_file"] = str(case_file)
    manifest["case_ids"] = [str(item["case_id"]) for item in cases]
    empirical = manifest.get("empirical_need")
    if not isinstance(empirical, Mapping):
        raise ValueError("adaptive empirical need contract is missing")
    if empirical.get("schema_version") != "adaptive_graph_empirical_need_v1":
        raise ValueError("adaptive empirical need schema mismatch")
    fixture_path = (path.parent / str(empirical.get("fixture_file", ""))).resolve()
    required_counts = empirical.get("required_categories")
    if not isinstance(required_counts, Mapping):
        raise ValueError("adaptive empirical need counts are missing")
    fixture = load_empirical_fixture(
        fixture_path,
        source_case_ids=manifest["case_ids"],
        required_counts={str(key): int(value) for key, value in required_counts.items()},
    )
    if empirical.get("case_count") != len(fixture):
        raise ValueError("adaptive empirical fixture count mismatch")
    if tuple(empirical.get("edge_limits", ())) != EVALUATOR_EDGE_LIMITS:
        raise ValueError("adaptive empirical edge limits are not frozen")
    if tuple(empirical.get("query_columns", ())) != (
        "original_question",
        "agent_query",
        "manual_last_hop",
        "candidate_normalized",
    ):
        raise ValueError("adaptive empirical query columns are not frozen")
    manifest["empirical_need"] = {
        **dict(empirical),
        "fixture_file": str(fixture_path),
        "case_ids": [str(item["case_id"]) for item in fixture],
    }
    return manifest


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    normalized = json.loads(json.dumps(manifest, ensure_ascii=False))
    case_file = normalized.get("case_file")
    if isinstance(case_file, str):
        normalized["case_file"] = _portable_evaluation_path(case_file)
    empirical = normalized.get("empirical_need")
    if isinstance(empirical, dict):
        fixture_file = empirical.get("fixture_file")
        if isinstance(fixture_file, str):
            empirical["fixture_file"] = _portable_evaluation_path(fixture_file)
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _portable_evaluation_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    marker = "evaluation/"
    marker_index = normalized.find(marker)
    return normalized[marker_index:] if marker_index >= 0 else normalized


def validate_evaluation_readiness(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the complete offline evaluation boundary without external I/O."""

    loaded = dict(manifest or load_manifest(manifest_path))
    if loaded.get("dataset_id") != "routing-rag-v2":
        raise ValueError("evaluation readiness requires routing-rag-v2")
    case_file = Path(str(loaded["case_file"]))
    if not case_file.is_absolute():
        case_file = (manifest_path.parent / case_file).resolve()
    empirical = loaded.get("empirical_need")
    if not isinstance(empirical, Mapping):
        raise ValueError("evaluation readiness fixture contract is missing")
    fixture_file = Path(str(empirical["fixture_file"]))
    if not fixture_file.is_absolute():
        fixture_file = (manifest_path.parent / fixture_file).resolve()
    cases = load_cases(case_file)
    fixture = load_empirical_fixture(
        fixture_file,
        source_case_ids=(str(item["case_id"]) for item in cases),
        required_counts={
            str(key): int(value)
            for key, value in dict(empirical["required_categories"]).items()
        },
    )
    if len(cases) != 39 or len(fixture) != 22:
        raise ValueError("evaluation readiness case or fixture count changed")
    if [str(item["case_id"]) for item in cases] != list(loaded["case_ids"]):
        raise ValueError("evaluation readiness case order changed")

    corpus_manifest_path = case_file.parent / "manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if (
        corpus_manifest.get("dataset_id") != "routing-rag-v2"
        or corpus_manifest.get("case_count") != len(cases)
        or corpus_manifest.get("route_contract", {}).get("graph")
        != {
            "runtime_mode": "graph",
            "top_k": 10,
            "rerank_mode": "classic",
            "expected_evidence_modality": "text",
        }
    ):
        raise ValueError("evaluation readiness corpus contract changed")

    graph_gold_root = case_file.parent / "gold" / "graph-rag-v1"
    graph_entities_path = graph_gold_root / "entities.jsonl"
    graph_relations_path = graph_gold_root / "relations.jsonl"
    entity_rows = [
        json.loads(line)
        for line in graph_entities_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entity_ids = {
        str(row.get("entity_id", ""))
        for row in entity_rows
        if isinstance(row, Mapping)
    }
    if len(entity_rows) != 240 or len(entity_ids) != 240 or "" in entity_ids:
        raise ValueError("evaluation readiness graph entities are invalid")
    relation_document_ids: dict[str, str] = {}
    relation_rows = []
    for line in graph_relations_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError("evaluation readiness graph relation is invalid")
        relation_rows.append(row)
    for row in relation_rows:
        relation_id = str(row.get("relation_id", ""))
        document_id = str(row.get("document_id", ""))
        if not relation_id or not document_id:
            raise ValueError("evaluation readiness graph relation is invalid")
        if (
            str(row.get("subject_entity_id", "")) not in entity_ids
            or str(row.get("object_entity_id", "")) not in entity_ids
        ):
            raise ValueError("evaluation readiness graph relation endpoint is invalid")
        if relation_id in relation_document_ids:
            raise ValueError("evaluation readiness relation source is ambiguous")
        relation_document_ids[relation_id] = document_id
    if len(relation_rows) != 224:
        raise ValueError("evaluation readiness graph relation count changed")
    graph_locator_candidates = [
        locator
        for case in cases
        if case.get("expected_route", {}).get("route") == "graph"
        for locator in (
            list(case.get("answer_gold_source_locators", ()))
            + list(case.get("path_context_locators", ()))
        )
    ]
    if any(not isinstance(locator, Mapping) for locator in graph_locator_candidates):
        raise ValueError("evaluation readiness graph locator is invalid")
    graph_locators = [
        locator
        for locator in graph_locator_candidates
        if locator.get("kind") == "graph_relation"
    ]
    document_filename_by_id: dict[str, str] = {}
    documents = corpus_manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError("evaluation readiness corpus documents are invalid")
    for item in documents:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("document_id"), str
        ) or not isinstance(item.get("filename"), str):
            raise ValueError("evaluation readiness corpus document is invalid")
        document_filename_by_id[str(item["document_id"])] = str(item["filename"])
    if len(graph_locators) != 40 or len({str(item["relation_id"]) for item in graph_locators}) != 35:
        raise ValueError("evaluation readiness graph locator coverage changed")
    for locator in graph_locators:
        relation_id = str(locator["relation_id"])
        expected_filename = document_filename_by_id.get(relation_document_ids.get(relation_id, ""))
        if expected_filename is None or str(locator.get("document_filename")) != expected_filename:
            raise ValueError("evaluation readiness graph locator source mismatch")

    from tools import run_adaptive_graph_r4 as r4_runner

    rerank_action = next(
        action
        for action in r4_runner._parser()._actions
        if "--rerank-mode" in action.option_strings
    )
    if (
        r4_runner.R4_DIAGNOSTIC_SCHEMA_VERSION != "adaptive_graph_r4_diagnostic_v3"
        or r4_runner.R4_CHECKPOINT_SCHEMA_VERSION != "adaptive_graph_r4_checkpoint_v3"
        or rerank_action.default != "classic"
        or frozenset(rerank_action.choices or ()) != RERANK_MODES
        or not {
            "question",
            "query",
            "text",
            "filename",
            "provider_payload",
        }.issubset(r4_runner._R4_CHECKPOINT_FORBIDDEN_KEYS)
        or "below_threshold_count" in r4_runner._R4_CHECKPOINT_FORBIDDEN_KEYS
    ):
        raise ValueError("evaluation readiness R4 contract changed")

    return {
        "dataset_id": str(loaded["dataset_id"]),
        "case_count": len(cases),
        "fixture_count": len(fixture),
        "graph_locator_count": len(graph_locators),
        "graph_unique_relation_count": len(
            {str(item["relation_id"]) for item in graph_locators}
        ),
        "graph_source_relation_count": len(relation_rows),
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_digest": manifest_digest(loaded),
        "cases_file_sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
        "fixture_file_sha256": hashlib.sha256(fixture_file.read_bytes()).hexdigest(),
        "corpus_manifest_sha256": hashlib.sha256(corpus_manifest_path.read_bytes()).hexdigest(),
        "graph_relations_sha256": hashlib.sha256(graph_relations_path.read_bytes()).hexdigest(),
        "graph_entities_sha256": hashlib.sha256(graph_entities_path.read_bytes()).hexdigest(),
        "rerank_modes": sorted(RERANK_MODES),
        "layers": list(LAYERS),
        "r4_contract": {
            "diagnostic_schema": r4_runner.R4_DIAGNOSTIC_SCHEMA_VERSION,
            "checkpoint_schema": r4_runner.R4_CHECKPOINT_SCHEMA_VERSION,
            "rerank_default": rerank_action.default,
            "rerank_choices": sorted(rerank_action.choices or ()),
            "checkpoint_forbidden_key_count": len(
                r4_runner._R4_CHECKPOINT_FORBIDDEN_KEYS
            ),
        },
    }


def build_routing_judge_packet(
    case: Mapping[str, Any],
    run: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a bounded, routing-specific offline Judge packet."""

    case_id = str(case.get("case_id", ""))
    if not case_id or case.get("expected_outcome") not in CASE_OUTCOMES:
        raise ValueError("Judge case contract is invalid")
    aspects = case.get("expected_answer_aspects")
    if not isinstance(aspects, list):
        raise ValueError("Judge case aspects are invalid")
    citations = run.get("citations", ())
    if not isinstance(citations, Sequence) or isinstance(citations, (str, bytes)):
        citations = ()
    safe_citations: list[dict[str, Any]] = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise ValueError("Judge citation is invalid")
        safe_citations.append(
            {
                "citation_id": str(citation.get("citation_id", "")),
                "index_chunk_id": str(citation.get("index_chunk_id", "")),
                "document_id": str(citation.get("document_id", "")),
                "source_location": citation.get("source_location", {}),
                "modality": citation.get("modality"),
            }
        )
    return {
        "schema_version": ROUTING_JUDGE_SCHEMA_VERSION,
        "prompt_version": ROUTING_JUDGE_PROMPT_VERSION,
        "case_id": case_id,
        "question": str(case["question"]),
        "reference": {
            "expected_outcome": case["expected_outcome"],
            "negative_control_kind": case.get("negative_control_kind"),
            "expected_answer_aspects": [
                {
                    "aspect_id": str(aspect["aspect_id"]),
                    "answer_variants": [str(item) for item in aspect["answer_variants"]],
                }
                for aspect in aspects
            ],
            "answer_gold_source_locators": case["answer_gold_source_locators"],
            "forbidden_claims": [str(item) for item in case["forbidden_claims"]],
        },
        "agent_result": {
            "outcome": str(run.get("outcome", "")),
            "answer": str(run.get("answer", "")),
            "citations": safe_citations,
        },
    }


def routing_judge_cache_key(
    packet: Mapping[str, Any],
    *,
    profile_revision: str = "offline-routing-judge-v1",
) -> str:
    import hashlib

    value = {
        "schema_version": ROUTING_JUDGE_SCHEMA_VERSION,
        "prompt_version": ROUTING_JUDGE_PROMPT_VERSION,
        "profile_revision": profile_revision,
        "packet": packet,
    }
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_routing_judgement(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "prompt_version",
        "answer_correctness",
        "claim_grounding",
        "citation_alignment",
        "outcome_correctness",
        "negative_stance",
        "reason_code",
    }
    if set(value) != expected:
        raise ValueError("routing Judge result fields are invalid")
    if value["schema_version"] != ROUTING_JUDGE_SCHEMA_VERSION:
        raise ValueError("routing Judge schema version is invalid")
    if value["prompt_version"] != ROUTING_JUDGE_PROMPT_VERSION:
        raise ValueError("routing Judge prompt version is invalid")
    if value["answer_correctness"] not in JUDGE_VERDICTS:
        raise ValueError("routing Judge answer correctness is invalid")
    if value["claim_grounding"] not in JUDGE_GROUNDING:
        raise ValueError("routing Judge grounding is invalid")
    if value["citation_alignment"] not in JUDGE_GROUNDING:
        raise ValueError("routing Judge Citation alignment is invalid")
    if value["outcome_correctness"] not in {"correct", "incorrect"}:
        raise ValueError("routing Judge outcome correctness is invalid")
    if value["negative_stance"] not in JUDGE_STANCES:
        raise ValueError("routing Judge negative stance is invalid")
    if value["reason_code"] not in JUDGE_REASON_CODES:
        raise ValueError("routing Judge reason code is invalid")


def _successful_simple_result(messages: Sequence[ChatModelMessage]) -> bool:
    for message in reversed(messages):
        if message.role != "tool":
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError):
            return False
        return isinstance(payload, Mapping) and payload.get("status") == "ok"
    return False


@dataclass
class ForcedGraphitiSupplementChatModelPort:
    """Evaluator-only decorator that controls one bounded supplement turn."""

    delegate: ChatModelAdapter
    controller_mode: str = "single_tool_auto_fallback"
    replacements: int = 0
    model_calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    response_tool_names: list[tuple[str, ...]] = field(default_factory=list)
    response_finish_reasons: list[str | None] = field(default_factory=list)
    response_route_reason_codes: list[str | None] = field(default_factory=list)
    _used: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.controller_mode not in FORCED_CONTROLLER_MODES:
            raise ValueError("Forced controller mode is invalid")

    def reset_case(self) -> None:
        self.replacements = 0
        self.model_calls = 0
        self.usage.clear()
        self.response_tool_names.clear()
        self.response_finish_reasons.clear()
        self.response_route_reason_codes.clear()
        self._used = False

    @staticmethod
    def _single_tool_request(
        request: ChatModelRequest,
        *,
        tool_name: str,
        retry: bool = False,
    ) -> ChatModelRequest:
        tools = tuple(item for item in request.tools if item.name == tool_name)
        if len(tools) != 1:
            raise RuntimeError("Provider-safe tool cardinality is invalid")
        messages = request.messages
        if retry:
            messages = messages + (
                ChatModelMessage("user", _PROVIDER_SAFE_RETRY_PROMPT[tool_name]),
            )
        return replace(
            request,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

    async def _complete_provider_safe(
        self,
        request: ChatModelRequest,
        *,
        expected_tool: str,
    ) -> ChatModelResponse:
        """Use only the provider-neutral ``auto`` contract for R4 control.

        This is intentionally confined to the evaluator decorator.  The
        production Agent still sends its normal multi-tool request and keeps
        its existing native loop contract.
        """
        response: ChatModelResponse | None = None
        for attempt in range(PROVIDER_SAFE_TOOL_ATTEMPTS):
            response = await self.delegate.complete(
                self._single_tool_request(
                    request,
                    tool_name=expected_tool,
                    retry=attempt > 0,
                )
            )
            self._record_response(response)
            if (
                len(response.tool_calls) == 1
                and response.tool_calls[0].name == expected_tool
            ):
                return response
        if response is None:
            raise RuntimeError("Provider-safe tool attempt budget is invalid")
        return response

    def _record_response(self, response: ChatModelResponse) -> None:
        self.model_calls += 1
        self.response_tool_names.append(
            tuple(tool_call.name for tool_call in response.tool_calls)
        )
        self.response_finish_reasons.append(response.finish_reason)
        route_reason = None
        if response.tool_calls:
            candidate = response.tool_calls[0].arguments.get("route_reason_code")
            if isinstance(candidate, str) and candidate in CHAT_GRAPHITI_ROUTE_REASONS:
                route_reason = str(candidate)
        self.response_route_reason_codes.append(route_reason)
        for key, value in response.usage.items():
            self.usage[key] = self.usage.get(key, 0) + value

    async def complete(self, request: ChatModelRequest):
        effective = request
        tool_names = {item.name for item in request.tools}
        provider_safe_initial = (
            self.controller_mode == "single_tool_auto_fallback"
            and not self._used
            and "graphiti_supplement" not in tool_names
            and str(request.tool_choice) == "required"
        )
        provider_safe_supplement = (
            self.controller_mode == "single_tool_auto_fallback"
            and not self._used
            and "graphiti_supplement" in tool_names
            and _successful_simple_result(request.messages)
        )
        if provider_safe_initial:
            return await self._complete_provider_safe(
                request,
                expected_tool="search_knowledge_base",
            )
        if provider_safe_supplement:
            response = await self._complete_provider_safe(
                request,
                expected_tool="graphiti_supplement",
            )
            if (
                len(response.tool_calls) == 1
                and response.tool_calls[0].name == "graphiti_supplement"
            ):
                self._used = True
                self.replacements += 1
            return response
        if (
            self.controller_mode == "single_tool_required_fallback"
            and not self._used
            and "graphiti_supplement" not in tool_names
            and str(request.tool_choice) == "required"
        ):
            # Some thinking-capable providers reject a multi-tool REQUIRED
            # request. Keep the native loop deterministic by exposing only the
            # Simple tool on the first turn; the Agent still validates exactly
            # one tool call.
            simple_tools = tuple(
                item for item in request.tools if item.name == "search_knowledge_base"
            )
            if len(simple_tools) != 1:
                raise RuntimeError("Fallback Simple tool cardinality is invalid")
            effective = replace(request, tools=simple_tools, tool_choice="required")
        elif (
            self.controller_mode != "actual_auto"
            and not self._used
            and "graphiti_supplement" in tool_names
            and _successful_simple_result(request.messages)
        ):
            if self.controller_mode == "single_tool_required_fallback":
                supplement_tools = tuple(
                    item
                    for item in request.tools
                    if item.name == "graphiti_supplement"
                )
                if len(supplement_tools) != 1:
                    raise RuntimeError("Forced supplement tool cardinality is invalid")
                effective = replace(
                    request,
                    tools=supplement_tools,
                    tool_choice="required",
                )
            else:
                effective = replace(request, tool_choice="graphiti_supplement")
            self._used = True
            self.replacements += 1
        response = await self.delegate.complete(effective)
        self._record_response(response)
        return response


class GraphitiSupplementCaptureComplete(RuntimeError):
    """Content-safe evaluator stop after one validated supplement invocation."""


@dataclass(frozen=True, slots=True)
class GraphitiSupplementCapture:
    """One post-validation supplement call captured outside production trace."""

    case_id: str
    query: str = field(repr=False)
    excluded_index_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized_case_id = self.case_id.strip()
        normalized_query = self.query.strip()
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", normalized_case_id)
            is None
            or not normalized_query
            or len(normalized_query) > REPLAY_QUERY_MAX_CHARS
        ):
            raise ValueError("supplement capture is invalid")
        try:
            normalized_ids = tuple(
                str(UUID(str(item))) for item in self.excluded_index_chunk_ids
            )
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("supplement capture exclusions are invalid") from error
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("supplement capture exclusions must be unique")
        object.__setattr__(self, "case_id", normalized_case_id)
        object.__setattr__(self, "query", normalized_query)
        object.__setattr__(self, "excluded_index_chunk_ids", normalized_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "excluded_index_chunk_ids": list(self.excluded_index_chunk_ids),
        }


@dataclass
class CapturingGraphitiSupplementRetriever:
    """Evaluator-only retriever wrapper capturing the accepted Agent query."""

    delegate: Any
    stop_after_capture: bool = False
    _active_case_id: str | None = field(default=None, init=False, repr=False)
    _active_capture: GraphitiSupplementCapture | None = field(
        default=None, init=False, repr=False
    )
    _simple_index_chunk_ids: list[str] = field(
        default_factory=list, init=False, repr=False
    )

    def begin_case(self, case_id: str) -> None:
        if self._active_case_id is not None:
            raise RuntimeError("supplement capture case is already active")
        normalized = case_id.strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", normalized) is None:
            raise ValueError("supplement capture case_id is invalid")
        self._active_case_id = normalized
        self._active_capture = None
        self._simple_index_chunk_ids.clear()

    def finish_case(self) -> GraphitiSupplementCapture:
        capture, _ = self.finish_observed_case()
        if capture is None:
            raise RuntimeError("supplement capture case is incomplete")
        return capture

    def finish_observed_case(
        self,
    ) -> tuple[GraphitiSupplementCapture | None, tuple[str, ...]]:
        if self._active_case_id is None:
            raise RuntimeError("supplement capture case is not active")
        capture = self._active_capture
        simple_ids = tuple(self._simple_index_chunk_ids)
        self._active_case_id = None
        self._active_capture = None
        self._simple_index_chunk_ids.clear()
        return capture, simple_ids

    def abandon_case(self) -> None:
        self._active_case_id = None
        self._active_capture = None
        self._simple_index_chunk_ids.clear()

    async def retrieve_query(self, *args, **kwargs):
        result = await self.delegate.retrieve_query(*args, **kwargs)
        for item in getattr(result, "evidence", ()):
            chunk_id = str(item.index_chunk_id)
            if chunk_id not in self._simple_index_chunk_ids:
                self._simple_index_chunk_ids.append(chunk_id)
        return result

    async def retrieve_graphiti_supplement(
        self,
        context,
        query: str,
        *,
        excluded_index_chunk_ids,
    ):
        if self._active_case_id is None:
            raise RuntimeError("supplement capture has no active case")
        if self._active_capture is not None:
            raise RuntimeError("supplement capture received more than one call")
        capture = GraphitiSupplementCapture(
            case_id=self._active_case_id,
            query=query,
            excluded_index_chunk_ids=tuple(
                str(item) for item in excluded_index_chunk_ids
            ),
        )
        self._active_capture = capture
        if self.stop_after_capture:
            raise GraphitiSupplementCaptureComplete("graphiti_supplement_captured")
        return await self.delegate.retrieve_graphiti_supplement(
            context,
            capture.query,
            excluded_index_chunk_ids=tuple(
                UUID(item) for item in capture.excluded_index_chunk_ids
            ),
        )


def _redacted_capture_dict(value: GraphitiSupplementCapture | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, GraphitiSupplementCapture):
        return value.as_dict()
    if set(value) != {"case_id", "excluded_index_chunk_ids"}:
        raise ValueError("replay capture redaction is invalid")
    case_id = str(value.get("case_id", ""))
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", case_id) is None:
        raise ValueError("replay capture case identity is invalid")
    excluded = value.get("excluded_index_chunk_ids")
    if not isinstance(excluded, list):
        raise ValueError("replay capture exclusions are invalid")
    try:
        normalized = [str(UUID(str(item))) for item in excluded]
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("replay capture exclusions are invalid") from error
    if normalized != excluded or len(normalized) != len(set(normalized)):
        raise ValueError("replay capture exclusions are invalid")
    return {"case_id": case_id, "excluded_index_chunk_ids": normalized}


def build_replay_capture_artifact(
    *,
    dataset_id: str,
    rerank_mode: str,
    manifest_sha256: str,
    knowledge_base_id: str,
    index_revision_id: str,
    graph_build_id: str,
    chat_model_profile_revision_id: str,
    captures: Sequence[GraphitiSupplementCapture | Mapping[str, Any]],
    controller_mode: str = "single_tool_auto_fallback",
    chat_model: str | None = None,
    chat_model_source: str | None = None,
    chat_model_max_output_tokens: int | None = None,
    chat_model_max_retries: int | None = None,
) -> dict[str, Any]:
    """Build a bounded synthetic-corpus replay artifact with no provider secrets."""

    identifiers = {
        "knowledge_base_id": knowledge_base_id,
        "index_revision_id": index_revision_id,
        "graph_build_id": graph_build_id,
        "chat_model_profile_revision_id": chat_model_profile_revision_id,
    }
    try:
        normalized_identifiers = {
            key: str(UUID(str(value))) for key, value in identifiers.items()
        }
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("replay capture runtime identity is invalid") from error
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise ValueError("replay capture manifest digest is invalid")
    if not re.fullmatch(r"routing-rag-v[0-9]+", dataset_id):
        raise ValueError("replay capture dataset identity is invalid")
    if rerank_mode not in RERANK_MODES:
        raise ValueError("replay capture rerank mode is invalid")
    if controller_mode not in FORCED_CONTROLLER_MODES:
        raise ValueError("replay capture controller mode is invalid")
    redacted_captures = tuple(_redacted_capture_dict(item) for item in captures)
    case_ids = [str(item["case_id"]) for item in redacted_captures]
    if (
        len(case_ids) != len(set(case_ids))
        or not captures
        and controller_mode != "actual_auto"
    ):
        raise ValueError("replay captures must contain unique cases")
    capture_source = (
        "r3a_actual_auto" if controller_mode == "actual_auto" else REPLAY_CAPTURE_SOURCE
    )
    if (chat_model is None) != (chat_model_source is None):
        raise ValueError("replay capture chat model identity is incomplete")
    if chat_model is not None:
        if not chat_model.strip() or chat_model_source not in {
            "profile_revision",
            "evaluator_override",
        }:
            raise ValueError("replay capture chat model identity is invalid")
        if chat_model_max_output_tokens is None or chat_model_max_output_tokens < 1:
            raise ValueError("replay capture chat model token limit is invalid")
        if chat_model_max_retries is None or chat_model_max_retries < 0:
            raise ValueError("replay capture chat model retry limit is invalid")
        normalized_identifiers.update(
            {
                "chat_model": chat_model.strip(),
                "chat_model_source": chat_model_source,
                "chat_model_max_output_tokens": chat_model_max_output_tokens,
                "chat_model_max_retries": chat_model_max_retries,
            }
        )
    normalized_identifiers["rerank_mode"] = rerank_mode
    return {
        "schema_version": REPLAY_CAPTURE_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "capture_source": capture_source,
        "controller_mode": controller_mode,
        "manifest_sha256": manifest_sha256,
        "runtime": normalized_identifiers,
        "case_count": len(captures),
        "cases": list(redacted_captures),
    }


def write_replay_capture_artifact(path: Path, artifact: Mapping[str, Any]) -> str:
    """Write one immutable, owner-readable evaluator artifact and return its digest."""

    if artifact.get("schema_version") != REPLAY_CAPTURE_SCHEMA_VERSION:
        raise ValueError("replay capture artifact schema is invalid")
    payload = json.dumps(
        artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
    os.chmod(path, 0o600)
    return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChunkAlignment:
    column: str
    answer_gold_chunk_ids: tuple[str, ...]
    path_context_chunk_ids: tuple[str, ...]
    required_path_chunk_ids: tuple[str, ...]
    simple_chunk_ids: tuple[str, ...]
    layer_chunk_ids: dict[str, tuple[str, ...]]
    first_loss_layer: str | None
    gold_hit_by_layer: dict[str, tuple[str, ...]]
    required_path_hit_by_layer: dict[str, tuple[str, ...]]
    new_answer_gold_chunk_ids: tuple[str, ...]
    new_required_path_chunk_ids: tuple[str, ...]
    simple_path_complete: bool
    packed_path_complete: bool
    redundant_hit: bool
    benefit: bool
    duplicate_count: int
    non_gold_admitted_count: int


def summarize_graph_route_trace(
    trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reduce an Agent trace to content-safe Graph route observations.

    The returned values are route facts only: booleans, counts, and the closed
    route-result enum.  It intentionally does not retain questions, answers,
    evidence bodies, filenames, or provider payloads.
    """

    if trace is None:
        trace = {}
    if not isinstance(trace, Mapping):
        raise ValueError("agent trace must be an object")
    events = trace.get("events", ())
    if not isinstance(events, (list, tuple)):
        raise ValueError("agent trace events must be a list")
    attempted = False
    admitted = False
    new_evidence_count = 0
    result_counts: dict[str, int] = {}
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("retrieval_lane") != "graphiti_supplement":
            continue
        route_result = event.get("route_result_code")
        if route_result not in CHAT_GRAPHITI_ROUTE_RESULTS:
            raise ValueError("Graph route result is outside the closed enum")
        count = event.get("new_evidence_count", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Graph route evidence count is invalid")
        attempted = True
        admitted = admitted or route_result == "admitted"
        new_evidence_count += count
        result_counts[str(route_result)] = result_counts.get(str(route_result), 0) + 1
    return {
        "graph_route_attempted": attempted,
        "graph_route_admitted": admitted,
        "graph_new_evidence_count": new_evidence_count,
        "graph_route_result_counts": dict(sorted(result_counts.items())),
    }


def align_chunk_layers(
    *,
    column: str,
    answer_gold_chunk_ids: Iterable[str],
    simple_chunk_ids: Iterable[str],
    layer_chunk_ids: Mapping[str, Iterable[str]],
    path_context_chunk_ids: Iterable[str] = (),
) -> ChunkAlignment:
    if column not in {"capability", "agent_replay"}:
        raise ValueError("diagnostic column is invalid")
    gold = tuple(dict.fromkeys(str(item) for item in answer_gold_chunk_ids))
    context = tuple(dict.fromkeys(str(item) for item in path_context_chunk_ids))
    required = tuple(dict.fromkeys((*context, *gold)))
    simple = tuple(dict.fromkeys(str(item) for item in simple_chunk_ids))
    layers: dict[str, tuple[str, ...]] = {}
    for layer in LAYERS:
        if layer not in layer_chunk_ids:
            raise ValueError(f"missing diagnostic layer: {layer}")
        layers[layer] = tuple(str(item) for item in layer_chunk_ids[layer])
    gold_hit_by_layer = {
        layer: tuple(item for item in gold if item in set(layers[layer]))
        for layer in LAYERS
    }
    required_path_hit_by_layer = {
        layer: tuple(
            item
            for item in required
            if item in set(simple).union(layers[layer])
        )
        for layer in LAYERS
    }
    first_loss: str | None = None
    for required_id in required:
        if required_id in simple:
            continue
        for layer in LAYERS:
            if required_id not in set(simple).union(layers[layer]):
                first_loss = layer
                break
        if first_loss is not None:
            break
    packed = layers["packed"]
    new_gold = tuple(item for item in gold if item not in simple and item in packed)
    new_required = tuple(
        item for item in required if item not in simple and item in packed
    )
    graph_new = tuple(item for item in packed if item not in simple)
    simple_path_complete = bool(required) and set(required) <= set(simple)
    packed_path_complete = bool(required) and set(required) <= set(simple).union(packed)
    redundant = (
        simple_path_complete
        or bool(graph_new) and not new_required
    )
    return ChunkAlignment(
        column=column,
        answer_gold_chunk_ids=gold,
        path_context_chunk_ids=context,
        required_path_chunk_ids=required,
        simple_chunk_ids=simple,
        layer_chunk_ids=layers,
        first_loss_layer=first_loss,
        gold_hit_by_layer=gold_hit_by_layer,
        required_path_hit_by_layer=required_path_hit_by_layer,
        new_answer_gold_chunk_ids=new_gold,
        new_required_path_chunk_ids=new_required,
        simple_path_complete=simple_path_complete,
        packed_path_complete=packed_path_complete,
        redundant_hit=redundant,
        benefit=not simple_path_complete and packed_path_complete,
        duplicate_count=len(packed) - len(set(packed)),
        non_gold_admitted_count=sum(item not in set(required) for item in packed),
    )


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6) if denominator else None,
    }


def _alignment_value(alignment: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(alignment, ChunkAlignment):
        return getattr(alignment, field_name, default)
    if isinstance(alignment, Mapping):
        return alignment.get(field_name, default)
    return default


def _graph_recall_metrics(
    graph_cases: Sequence[Mapping[str, Any]],
    alignments: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate complete-path recall without retaining gold identifiers."""

    if alignments is None:
        unavailable = {
            "status": "not_computed_missing_layer_evidence",
            "primary_metric": "packed_required_path_recall",
            "by_layer": {},
        }
        return unavailable, {
            "status": "not_computed_missing_layer_evidence",
            "recoverable_case_count": None,
            "benefit_case_count": None,
            "redundant_case_count": None,
        }

    missing_cases = [
        str(case["case_id"])
        for case in graph_cases
        if str(case["case_id"]) not in alignments
    ]
    if missing_cases:
        incomplete = {
            "status": "not_computed_incomplete_layer_evidence",
            "primary_metric": "packed_required_path_recall",
            "observed_case_count": len(graph_cases) - len(missing_cases),
            "required_case_count": len(graph_cases),
            "by_layer": {},
        }
        return incomplete, {
            "status": "not_computed_incomplete_layer_evidence",
            "recoverable_case_count": None,
            "benefit_case_count": None,
            "redundant_case_count": None,
        }

    layer_counts = {
        layer: {
            "answer_gold_hits": 0,
            "answer_gold_total": 0,
            "required_path_hits": 0,
            "required_path_total": 0,
            "case_complete": 0,
            "case_total": len(graph_cases),
        }
        for layer in LAYERS
    }
    recoverable = benefit = redundant = 0
    for case in graph_cases:
        case_id = str(case["case_id"])
        alignment = alignments[case_id]
        gold = tuple(
            dict.fromkeys(
                str(item)
                for item in (_alignment_value(alignment, "answer_gold_chunk_ids", ()) or ())
            )
        )
        required = tuple(
            dict.fromkeys(
                str(item)
                for item in (
                    _alignment_value(alignment, "required_path_chunk_ids", gold)
                    or gold
                )
            )
        )
        simple = set(
            str(item)
            for item in (_alignment_value(alignment, "simple_chunk_ids", ()) or ())
        )
        layers = _alignment_value(alignment, "layer_chunk_ids")
        if layers is None and isinstance(alignment, Mapping):
            layers = alignment.get("layers")
        if not isinstance(layers, Mapping) or set(layers) != set(LAYERS):
            raise ValueError("Graph recall alignment layers are invalid")
        for layer in LAYERS:
            layer_ids = set(str(item) for item in (layers.get(layer, ()) or ()))
            combined_ids = simple.union(layer_ids)
            layer_counts[layer]["answer_gold_hits"] += len(
                set(gold) & combined_ids
            )
            layer_counts[layer]["answer_gold_total"] += len(gold)
            layer_counts[layer]["required_path_hits"] += len(
                set(required) & combined_ids
            )
            layer_counts[layer]["required_path_total"] += len(required)
            layer_counts[layer]["case_complete"] += bool(required) and set(
                required
            ) <= combined_ids
        simple_missing = bool(required) and not set(required) <= simple
        if simple_missing:
            recoverable += 1
        packed_ids = set(str(item) for item in (layers.get("packed", ()) or ()))
        if simple_missing and set(required) <= simple.union(packed_ids):
            benefit += 1
        if bool(_alignment_value(alignment, "redundant_hit", False)):
            redundant += 1

    recall_by_layer = {
        layer: {
            **counts,
            "answer_gold_recall": _rate(
                counts["answer_gold_hits"], counts["answer_gold_total"]
            ),
            "required_path_recall": _rate(
                counts["required_path_hits"], counts["required_path_total"]
            ),
            "case_complete_rate": _rate(
                counts["case_complete"], counts["case_total"]
            ),
        }
        for layer, counts in layer_counts.items()
    }
    graph_recall = {
        "status": "computed",
        "primary_metric": "packed_required_path_recall",
        "by_layer": recall_by_layer,
    }
    benefit_status = "computed" if recoverable else "not_computed_no_recoverable_cases"
    return graph_recall, {
        "status": benefit_status,
        "recoverable_case_count": recoverable,
        "benefit_case_count": benefit,
        "benefit_capture": _rate(benefit, recoverable),
        "redundant_case_count": redundant,
    }


def aggregate_graph_routing_metrics(
    cases: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    *,
    alignments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the primary Graph route and recall metrics.

    ``expected_route.route`` is the frozen synthetic benchmark label.  An
    observation is intentionally limited to route booleans/counts, while the
    optional alignment map is the evaluator-only raw/hydrated/reranked/packed
    evidence summary.  Cost and latency are deliberately outside this function.
    """

    if not cases:
        raise ValueError("Graph routing metrics require cases")
    expected_ids = [str(case.get("case_id", "")) for case in cases]
    if len(expected_ids) != len(set(expected_ids)) or any(not item for item in expected_ids):
        raise ValueError("Graph routing case IDs are invalid")
    if set(observations) != set(expected_ids):
        raise ValueError("Graph routing observations must cover every case exactly once")

    manifest_graph = manifest_simple = 0
    manifest_true_positive = manifest_false_negative = 0
    manifest_true_negative = manifest_false_positive = 0
    evidence_gap_needed = evidence_gap_not_needed = 0
    true_positive = false_negative = true_negative = false_positive = 0
    evidence_gap_attempts = graph_not_needed_attempts = 0
    admitted_graph = effective_graph = 0
    route_result_counts: dict[str, int] = {}
    redundant_attempts = 0
    for case in cases:
        case_id = str(case["case_id"])
        expected_route = case.get("expected_route")
        if not isinstance(expected_route, Mapping):
            raise ValueError(f"{case_id}: expected route is invalid")
        label = expected_route.get("route")
        if label not in GRAPH_ROUTE_LABELS:
            raise ValueError(f"{case_id}: expected route label is invalid")
        observation = observations[case_id]
        attempted = observation.get("graph_route_attempted")
        admitted = observation.get("graph_route_admitted", False)
        new_evidence_count = observation.get("graph_new_evidence_count", 0)
        if not isinstance(attempted, bool) or not isinstance(admitted, bool):
            raise ValueError(f"{case_id}: Graph route observation booleans are invalid")
        if (
            isinstance(new_evidence_count, bool)
            or not isinstance(new_evidence_count, int)
            or new_evidence_count < 0
        ):
            raise ValueError(f"{case_id}: Graph route evidence count is invalid")
        if admitted and not attempted:
            raise ValueError(f"{case_id}: admitted Graph route was not attempted")
        if admitted != (new_evidence_count > 0):
            raise ValueError(f"{case_id}: Graph route admission evidence is inconsistent")
        # Path-aware v2 evaluates the end-to-end route decision: a probe is
        # successful only when source-backed new evidence is actually admitted.
        # The legacy/no-layer contract keeps its historical attempt semantics.
        routed = admitted if alignments is not None else attempted
        raw_counts = observation.get("graph_route_result_counts", {})
        if raw_counts:
            if not isinstance(raw_counts, Mapping):
                raise ValueError(f"{case_id}: Graph route result counts are invalid")
            for result, count in raw_counts.items():
                if result not in CHAT_GRAPHITI_ROUTE_RESULTS:
                    raise ValueError(f"{case_id}: Graph route result is invalid")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"{case_id}: Graph route result count is invalid")
                route_result_counts[str(result)] = route_result_counts.get(str(result), 0) + count
        if label == "graph":
            manifest_graph += 1
            manifest_true_positive += attempted
            manifest_false_negative += not attempted
        else:
            manifest_simple += 1
            manifest_false_positive += attempted
            manifest_true_negative += not attempted

        alignment = alignments.get(case_id) if alignments is not None else None
        if label == "graph" and alignment is not None:
            simple_path_complete = _alignment_value(
                alignment,
                "simple_path_complete",
            )
            if simple_path_complete is None:
                required = set(
                    str(item)
                    for item in (
                        _alignment_value(
                            alignment,
                            "required_path_chunk_ids",
                            _alignment_value(
                                alignment,
                                "answer_gold_chunk_ids",
                                (),
                            ),
                        )
                        or ()
                    )
                )
                simple = set(
                    str(item)
                    for item in (
                        _alignment_value(alignment, "simple_chunk_ids", ()) or ()
                    )
                )
                simple_path_complete = bool(required) and required <= simple
            needs_graph = not bool(simple_path_complete)
        else:
            needs_graph = label == "graph"
        if needs_graph:
            evidence_gap_needed += 1
            evidence_gap_attempts += attempted
            true_positive += routed
            false_negative += not routed
            admitted_graph += admitted
            effective_graph += new_evidence_count > 0
        else:
            evidence_gap_not_needed += 1
            graph_not_needed_attempts += attempted
            false_positive += routed
            true_negative += not routed

    graph_cases = [case for case in cases if case["expected_route"]["route"] == "graph"]
    graph_recall, benefit = _graph_recall_metrics(graph_cases, alignments)
    if alignments is not None and graph_recall["status"] == "computed":
        for case in graph_cases:
            alignment = alignments[str(case["case_id"])]
            if bool(_alignment_value(alignment, "redundant_hit", False)):
                observation = observations[str(case["case_id"])]
                redundant_attempts += bool(observation["graph_route_attempted"])

    route_metrics = {
        "status": "computed",
        "decision_source": (
            "admitted_source_backed_new_evidence"
            if alignments is not None
            else "graph_route_attempted_legacy_fallback"
        ),
        "label_source": (
            "simple_required_path_completeness"
            if alignments is not None
            else "manifest.expected_route.route_fallback"
        ),
        "case_count": len(cases),
        "graph_needed_case_count": evidence_gap_needed,
        "graph_not_needed_case_count": evidence_gap_not_needed,
        "manifest_graph_case_count": manifest_graph,
        "manifest_simple_case_count": manifest_simple,
        "confusion": {
            "true_positive_evidence_gap_and_routed": true_positive,
            "false_negative_evidence_gap_but_not_routed": false_negative,
            "true_negative_complete_path_and_not_routed": true_negative,
            "false_positive_complete_path_but_routed": false_positive,
        },
        "manifest_confusion": {
            "true_positive_graph_label_and_routed": manifest_true_positive,
            "false_negative_graph_label_but_not_routed": manifest_false_negative,
            "true_negative_simple_label_and_not_routed": manifest_true_negative,
            "false_positive_simple_label_but_routed": manifest_false_positive,
        },
        "graph_needed_route_recall": _rate(true_positive, evidence_gap_needed),
        "graph_route_accuracy": _rate(true_positive + true_negative, len(cases)),
        "graph_route_precision": _rate(true_positive, true_positive + false_positive),
        "graph_not_needed_route_rate": _rate(false_positive, evidence_gap_not_needed),
        "graph_needed_probe_rate": _rate(
            evidence_gap_attempts,
            evidence_gap_needed,
        ),
        "graph_not_needed_probe_rate": _rate(
            graph_not_needed_attempts,
            evidence_gap_not_needed,
        ),
        "manifest_graph_route_recall": _rate(
            manifest_true_positive,
            manifest_graph,
        ),
        "manifest_simple_false_positive_rate": _rate(
            manifest_false_positive,
            manifest_simple,
        ),
        "simple_false_positive_rate": _rate(
            manifest_false_positive,
            manifest_simple,
        ),
        "manifest_route_accuracy": _rate(
            manifest_true_positive + manifest_true_negative,
            len(cases),
        ),
        "graph_route_admission_recall": _rate(admitted_graph, evidence_gap_needed),
        "graph_effective_evidence_recall": _rate(effective_graph, evidence_gap_needed),
        "route_result_counts": dict(sorted(route_result_counts.items())),
        "redundant_graph_route_rate": (
            _rate(
                redundant_attempts,
                sum(
                    observations[str(case["case_id"])]["graph_route_attempted"]
                    for case in graph_cases
                ),
            )
            if alignments is not None and graph_recall["status"] == "computed"
            else {
                "numerator": None,
                "denominator": None,
                "value": None,
            }
        ),
    }
    return {
        "primary_metric": "graph_needed_route_recall",
        "route": route_metrics,
        "graph_recall": graph_recall,
        "benefit_capture": benefit,
    }


def diagnostic_record(
    *,
    case_id: str,
    alignments: Mapping[str, ChunkAlignment],
    query_source: str,
    query_count: int,
    layer_diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if set(alignments) != {"capability", "agent_replay"}:
        raise ValueError("diagnostic record requires both columns")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", case_id) is None:
        raise ValueError("diagnostic case_id is invalid")
    if query_source not in {"capability", "agent_replay"} or query_count < 0:
        raise ValueError("diagnostic query metadata is invalid")
    diagnostics = {
        name: _validate_layer_diagnostic(
            layer_diagnostics.get(name, {}) if layer_diagnostics is not None else {}
        )
        for name in ("capability", "agent_replay")
    }
    return {
        "case_id": case_id,
        "columns": {
            name: {
                "first_loss_layer": value.first_loss_layer,
                "answer_gold_chunk_ids": list(value.answer_gold_chunk_ids),
                "path_context_chunk_ids": list(value.path_context_chunk_ids),
                "required_path_chunk_ids": list(value.required_path_chunk_ids),
                "simple_chunk_ids": list(value.simple_chunk_ids),
                "layers": {layer: list(ids) for layer, ids in value.layer_chunk_ids.items()},
                "gold_hit_by_layer": {
                    layer: list(ids) for layer, ids in value.gold_hit_by_layer.items()
                },
                "required_path_hit_by_layer": {
                    layer: list(ids)
                    for layer, ids in value.required_path_hit_by_layer.items()
                },
                "new_answer_gold_chunk_ids": list(value.new_answer_gold_chunk_ids),
                "new_required_path_chunk_ids": list(
                    value.new_required_path_chunk_ids
                ),
                "simple_path_complete": value.simple_path_complete,
                "packed_path_complete": value.packed_path_complete,
                "redundant_hit": value.redundant_hit,
                "benefit": value.benefit,
                "duplicate_count": value.duplicate_count,
                "non_gold_admitted_count": value.non_gold_admitted_count,
            }
            for name, value in alignments.items()
        },
        "query_source": query_source,
        "query_count": query_count,
        "layer_diagnostics": diagnostics,
    }


def _validate_layer_diagnostic(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "requested_k",
        "raw_edge_uuids",
        "raw_episode_count",
        "unique_chunk_count",
        "duplicate_chunk_path_count",
        "gold_rerank_scores",
        "top1_gold_rerank_score",
        "rerank_score_state",
        "rerank_reordered_chunk_count",
        "route_reason_code",
        "route_result_code",
        "salvage_status",
        "final_outcome",
    }
    if set(value) - allowed:
        raise ValueError("layer diagnostic contains unsupported fields")
    requested_k = value.get("requested_k", EVALUATOR_EDGE_LIMITS[0])
    validate_evaluator_edge_limit(requested_k)
    raw_edge_uuids = value.get("raw_edge_uuids", ())
    if not isinstance(raw_edge_uuids, (list, tuple)):
        raise ValueError("raw edge UUIDs must be a list")
    normalized_edge_uuids = tuple(str(item).strip() for item in raw_edge_uuids)
    if any(not item or len(item) > 128 for item in normalized_edge_uuids):
        raise ValueError("raw edge UUID is invalid")
    if len(normalized_edge_uuids) != len(set(normalized_edge_uuids)):
        raise ValueError("raw edge UUIDs must be unique")

    counts: dict[str, int] = {}
    for field_name in (
        "raw_episode_count",
        "unique_chunk_count",
        "duplicate_chunk_path_count",
    ):
        field_value = value.get(field_name, 0)
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
            raise ValueError(f"layer diagnostic count is invalid: {field_name}")
        counts[field_name] = field_value

    raw_scores = value.get("gold_rerank_scores", {})
    if not isinstance(raw_scores, Mapping):
        raise ValueError("gold rerank scores must be an object")
    gold_scores: dict[str, float] = {}
    for chunk_id, score in raw_scores.items():
        normalized_id = str(chunk_id).strip()
        if not normalized_id or len(normalized_id) > 128:
            raise ValueError("gold rerank score chunk id is invalid")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError("gold rerank score is invalid")
        gold_scores[normalized_id] = float(score)
    top_score = value.get("top1_gold_rerank_score")
    if top_score is not None:
        if (
            isinstance(top_score, bool)
            or not isinstance(top_score, (int, float))
            or not math.isfinite(top_score)
            or not 0.0 <= top_score <= 1.0
        ):
            raise ValueError("top gold rerank score is invalid")
        top_score = float(top_score)

    rerank_score_state = value.get("rerank_score_state", "not_applicable")
    if rerank_score_state not in {"scored", "not_applicable"}:
        raise ValueError("rerank score state is invalid")
    reordered_count = value.get("rerank_reordered_chunk_count", 0)
    if (
        isinstance(reordered_count, bool)
        or not isinstance(reordered_count, int)
        or reordered_count < 0
    ):
        raise ValueError("rerank reordered chunk count is invalid")

    route_reason = value.get("route_reason_code")
    if route_reason is not None and route_reason not in CHAT_GRAPHITI_ROUTE_REASONS:
        raise ValueError("layer route reason is invalid")
    route_result = value.get("route_result_code", "not_requested")
    if route_result not in CHAT_GRAPHITI_ROUTE_RESULTS:
        raise ValueError("layer route result is invalid")
    salvage_status = value.get("salvage_status", "not_attempted")
    if salvage_status not in {"not_attempted", "none", "salvaged", "refused"}:
        raise ValueError("layer salvage status is invalid")
    final_outcome = value.get("final_outcome", "not_run")
    if final_outcome not in {"not_run", "answered", "partial", "refused"}:
        raise ValueError("layer final outcome is invalid")
    return {
        "requested_k": requested_k,
        "raw_edge_uuids": list(normalized_edge_uuids),
        **counts,
        "gold_rerank_scores": gold_scores,
        "top1_gold_rerank_score": top_score,
        "rerank_score_state": rerank_score_state,
        "rerank_reordered_chunk_count": reordered_count,
        "route_reason_code": route_reason,
        "route_result_code": route_result,
        "salvage_status": salvage_status,
        "final_outcome": final_outcome,
    }


def aggregate_usage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate safe numeric usage only; unknown pricing remains uncomputed."""

    fields = (
        "total_tokens",
        "model_rounds",
        "retrieval_queries",
        "evidence_refs",
        "rejected_tools",
    )
    totals = {field_name: 0 for field_name in fields}
    elapsed: list[float] = []
    for record in records:
        for field_name in fields:
            value = record.get(field_name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"usage field is invalid: {field_name}")
            totals[field_name] += value
        value = record.get("elapsed_seconds", 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError("elapsed_seconds is invalid")
        elapsed.append(float(value))
    ordered = sorted(elapsed)
    return {
        "case_count": len(records),
        "totals": totals,
        "elapsed_seconds": {
            "p50": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
        },
        "cost": {"status": "not_computed", "reason": "price_table_not_frozen"},
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 6)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
    return round(values[index], 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    try:
        manifest = load_manifest(arguments.manifest)
        readiness = validate_evaluation_readiness(
            arguments.manifest,
            manifest=manifest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if not arguments.dry_run:
        parser.error(
            "provider/database execution is disabled; use --dry-run or the separately authorized R7 harness"
        )
    print(
        json.dumps(
            {
                "status": "dry_run_ok",
                "dataset_id": manifest["dataset_id"],
                "case_count": manifest["case_count"],
                "route_ids": list(ROUTE_IDS),
                "manifest_digest": manifest_digest(manifest),
                "readiness": readiness,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
