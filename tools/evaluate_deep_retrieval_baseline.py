#!/usr/bin/env python3
"""Evaluate the frozen deep-retrieval manifest on the standard exact path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any
from uuid import uuid4

from PIL import Image, ImageDraw

from rag_kb.domain import EvaluationCaseDefinition, EvaluationDatasetDefinition
from tools.evaluate_multimodal_real import (
    _json_request,
    _latest_attempt,
    _required_string,
    _upload,
    _validated_api_base,
    _wait_for_chat_run,
    _wait_for_job,
    _write_pdf,
)


MANIFEST_SCHEMA = "deep_retrieval_eval_manifest_v1"
REQUIRED_CATEGORIES = frozenset(
    {"single_fact", "multi_hop", "comparison", "exception", "conflict", "no_answer"}
)
ALLOWED_OUTCOMES = frozenset({"supported", "partial", "conflict", "refused"})
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "deep_retrieval_v1.json"
)
_CITATION_MARKER = re.compile(r"\[(\d+)]")
_CITATION_ID = re.compile(r"^cite_[1-9][0-9]*$")
_NO_ANSWER_CONTROL_REASON = "insufficient_evidence"
_MEDIA_SUFFIXES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".csv": "text/csv",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--api",
        default="http://127.0.0.1:8000/api/v1",
        help="loopback API base URL ending in /api/v1",
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="measure retrieval only; Chat correctness metrics remain unavailable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional report path; parent directory must already exist",
    )
    arguments = parser.parse_args()
    if arguments.timeout_seconds <= 0 or arguments.poll_seconds <= 0:
        parser.error("timeouts must be positive")
    try:
        api = _validated_api_base(arguments.api)
        manifest = load_manifest(arguments.manifest)
    except ValueError as error:
        parser.error(str(error))

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rag-kb-deep-baseline-") as directory:
        corpus = generate_corpus(manifest, Path(directory))
        report = evaluate_manifest(
            api,
            manifest,
            corpus,
            manifest_path=arguments.manifest,
            timeout_seconds=arguments.timeout_seconds,
            poll_seconds=arguments.poll_seconds,
            evaluate_chat=not arguments.skip_chat,
        )
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if arguments.output is not None:
        output = arguments.output.resolve()
        if not output.parent.is_dir():
            parser.error("--output parent directory must already exist")
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["gate"]["baseline_recorded"] else 1


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("deep retrieval manifest is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("deep retrieval manifest must be an object")
    required = {
        "schema_version",
        "name",
        "version",
        "manifest_hash",
        "metadata",
        "corpus",
        "cases",
    }
    if set(payload) != required or payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("deep retrieval manifest schema is invalid")
    expected_hash = manifest_hash(payload)
    if payload.get("manifest_hash") != expected_hash:
        raise ValueError("deep retrieval manifest hash does not match its content")
    corpus = payload.get("corpus")
    cases = payload.get("cases")
    metadata = payload.get("metadata")
    if not isinstance(corpus, list) or not corpus or not isinstance(cases, list) or not cases:
        raise ValueError("deep retrieval manifest requires corpus and cases")
    if not isinstance(metadata, dict):
        raise ValueError("deep retrieval manifest metadata must be an object")
    document_keys = _validate_corpus(corpus)
    visual_document_keys = {
        str(item["document_key"])
        for item in corpus
        if isinstance(item, dict)
        and item.get("media_type") in {"application/pdf", "image/png", "image/jpeg", "image/webp"}
        and (item.get("generator") == "quartz_visual_v1" or "visual" in str(item.get("document_key", "")))
    }
    corpus_evidence_text = {
        str(item["document_key"]): (
            str(item.get("content", ""))
            if "content" in item
            else "QUARTZ-VISUAL-TRIANGLE"
        )
        for item in corpus
    }
    categories = _validate_cases(
        cases,
        document_keys,
        visual_document_keys=visual_document_keys,
        corpus_evidence_text=corpus_evidence_text,
    )
    if categories != REQUIRED_CATEGORIES:
        raise ValueError("deep retrieval manifest must contain all six categories")
    declared = metadata.get("categories")
    if not isinstance(declared, list) or set(declared) != REQUIRED_CATEGORIES:
        raise ValueError("deep retrieval manifest category metadata is invalid")
    return payload


def manifest_hash(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "manifest_hash"}
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def dataset_definition(payload: dict[str, Any]) -> EvaluationDatasetDefinition:
    cases = tuple(
        EvaluationCaseDefinition(
            case_key=str(case["case_key"]),
            question=str(case["question"]),
            expected={"category": case["category"], **dict(case["expected"])},
            tags=tuple(str(tag) for tag in case["tags"]),
        )
        for case in payload["cases"]
    )
    return EvaluationDatasetDefinition(
        name=str(payload["name"]),
        version=str(payload["version"]),
        manifest_hash=str(payload["manifest_hash"]).removeprefix("sha256:"),
        metadata={
            **dict(payload["metadata"]),
            "schema_version": payload["schema_version"],
            "manifest_hash": payload["manifest_hash"],
        },
        cases=cases,
    )


def generate_corpus(payload: dict[str, Any], root: Path) -> dict[str, Path]:
    root = root.resolve()
    paths: dict[str, Path] = {}
    for item in payload["corpus"]:
        key = str(item["document_key"])
        filename = item.get("filename")
        if not isinstance(filename, str):
            raise ValueError("generated corpus filename must be a string")
        filename_path = _safe_corpus_filename(filename)
        if item.get("media_type") != _MEDIA_SUFFIXES.get(filename_path.suffix.lower()):
            raise ValueError("generated corpus media type does not match filename")
        path = (root / filename_path.name).resolve()
        if path.parent != root or path == root:
            raise ValueError("generated corpus filename escapes its temporary root")
        if "content" in item:
            path.write_text(str(item["content"]), encoding="utf-8")
        elif item.get("generator") == "quartz_visual_v1":
            _write_quartz_visual(path)
        else:  # pragma: no cover - load_manifest rejects this branch
            raise ValueError("unsupported corpus generator")
        paths[key] = path
    return paths


def evaluate_manifest(
    api: str,
    manifest: dict[str, Any],
    corpus: dict[str, Path],
    *,
    manifest_path: Path | None = None,
    timeout_seconds: float,
    poll_seconds: float,
    evaluate_chat: bool,
) -> dict[str, Any]:
    kb = _json_request(
        f"{api}/knowledge-bases",
        method="POST",
        headers={"Idempotency-Key": str(uuid4())},
        payload={
            "name": f"deep-retrieval-baseline-{uuid4().hex[:10]}",
            "parsing": {"preset": "multimodal_local_v2"},
            "chunking": {"preset": "structural_balanced_v2"},
        },
    )
    kb_id = _required_string(kb, "id")
    capabilities = _json_request(f"{api}/retrieval/capabilities")
    exact_capability = _require_exact_vector_capability(capabilities)
    documents: dict[str, str] = {}
    indexing: dict[str, Any] = {}
    revision_ids: set[str] = set()
    for key, path in corpus.items():
        upload = _upload(api, kb_id, key, path)
        document = upload.get("document")
        if not isinstance(document, dict):
            raise RuntimeError("upload response omitted document")
        documents[key] = _required_string(document, "id")
        job, elapsed = _wait_for_job(
            api,
            _required_string(upload, "job_id"),
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        if job.get("status") != "completed":
            raise RuntimeError(f"required corpus item failed indexing: {key}")
        revision_id = job.get("index_revision_id")
        if isinstance(revision_id, str):
            revision_ids.add(revision_id)
        indexing[key] = {
            "status": job.get("status"),
            "build_status": job.get("build_status"),
            "serving_status": job.get("serving_status"),
            "elapsed_seconds": round(elapsed, 6),
            "source_bytes": path.stat().st_size,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    retrieval_results = [
        _evaluate_retrieval_case(api, kb_id, documents, case)
        for case in manifest["cases"]
    ]
    retrieval_revision_ids = {
        str(item["index_revision_id"])
        for item in retrieval_results
        if isinstance(item.get("index_revision_id"), str)
    }
    if len(retrieval_revision_ids) != 1:
        raise RuntimeError("retrieval cases did not use one active index revision")
    observed_revision_id = next(iter(retrieval_revision_ids))
    chat_results = (
        [
            _evaluate_chat_case(
                api,
                kb_id,
                documents,
                case,
                expected_revision_id=observed_revision_id,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
            )
            for case in manifest["cases"]
        ]
        if evaluate_chat
        else []
    )
    metrics = summarize_results(retrieval_results, chat_results)
    model_names = sorted(
        {
            model
            for item in chat_results
            for model in item["model_names"]
        }
    )
    retrieval_profile_hashes = sorted(
        {
            str(item["retrieval_profile_hash"])
            for item in (*retrieval_results, *chat_results)
            if item["retrieval_profile_hash"] is not None
        }
    )
    contract_budget = {
        "wire_version": "deep_retrieval_budget_v1",
        "max_goals": 4,
        "max_query_variants_per_goal": 1,
        "max_adaptive_waves": 0,
        "max_repairs": 1,
        "deadline_seconds": 30.0,
        "max_parallelism": 1,
        "max_retrieval_calls": 1,
        "max_output_tokens": 2048,
    }
    parser_facts = {
        "preset": (kb.get("parsing") or {}).get("preset")
        if isinstance(kb.get("parsing"), dict)
        else None,
        "profile": (kb.get("parsing") or {}).get("profile")
        if isinstance(kb.get("parsing"), dict)
        else None,
    }
    chunking_facts = {
        "preset": (kb.get("chunking") or {}).get("preset")
        if isinstance(kb.get("chunking"), dict)
        else None,
        "profile": (kb.get("chunking") or {}).get("profile")
        if isinstance(kb.get("chunking"), dict)
        else None,
    }
    embedding_space_ids = sorted(
        {
            str(value)
            for value in (kb.get("embedding_space_id"),)
            if isinstance(value, str) and value
        }
        | {
            str(item["embedding_space_id"])
            for item in retrieval_results
            if isinstance(item.get("embedding_space_id"), str)
        }
    )
    retrieval_profile_facts = sorted(
        {
            item["retrieval_profile_hash"]
            for item in retrieval_results
            if isinstance(item.get("retrieval_profile_hash"), str)
        }
        | {
            item["retrieval_profile_hash"]
            for item in chat_results
            if isinstance(item.get("retrieval_profile_hash"), str)
        }
    )
    source_fingerprints = _source_fingerprints(manifest_path)
    code_sources = {
        name: fingerprint
        for name, fingerprint in source_fingerprints.items()
        if name != "manifest" and fingerprint is not None
    }
    fingerprints = {
        "parser": _content_hash(parser_facts) if all(parser_facts.values()) else None,
        "chunking": _content_hash(chunking_facts) if all(chunking_facts.values()) else None,
        "retrieval_profile": _content_hash(retrieval_profile_facts),
        "embedding_space": _content_hash(embedding_space_ids) if embedding_space_ids else None,
        "model": _content_hash(model_names) if model_names else None,
        "code": _content_hash(code_sources) if code_sources else None,
    }
    runtime_fingerprint = _content_hash(
        {
            "manifest_hash": manifest["manifest_hash"],
            "workflow_depth": "standard",
            "retrieval_mode": "vector",
            "retrieval_profile": exact_capability["profile_version"],
            "retrieval_profile_hashes": retrieval_profile_hashes,
            "observed_revision_id": observed_revision_id,
            "parser": parser_facts,
            "chunking": chunking_facts,
            "embedding_space_ids": embedding_space_ids,
            "fingerprints": fingerprints,
            "source_fingerprints": source_fingerprints,
            "model_names": model_names,
            "contract_budget": contract_budget,
        }
    )
    unavailable_facts: list[str] = []
    if fingerprints["parser"] is None:
        unavailable_facts.append("parser preset/profile fingerprint")
    if fingerprints["chunking"] is None:
        unavailable_facts.append("chunking preset/profile fingerprint")
    if not any(isinstance(item.get("rerank"), bool) for item in retrieval_results):
        unavailable_facts.append("retrieval rerank observation")
    if not embedding_space_ids:
        unavailable_facts.append("embedding space identifier")
    if not model_names:
        unavailable_facts.append("provider model name")
    if fingerprints["code"] is None:
        unavailable_facts.append("source code fingerprint")
    return {
        "schema_version": "deep_retrieval_baseline_report_v1",
        "dataset": {
            "name": manifest["name"],
            "version": manifest["version"],
            "manifest_hash": manifest["manifest_hash"],
            "case_count": len(manifest["cases"]),
        },
        "execution": {
            "workflow_depth": "standard",
            "retrieval_mode": "vector",
            "retrieval_profile": exact_capability["profile_version"],
            "top_k": 10,
            "manifest_path": (
                _content_safe_manifest_path(manifest_path)
                if manifest_path is not None
                else None
            ),
            "knowledge_base_id": kb_id,
            "index_revision_ids": sorted(revision_ids),
            "observed_index_revision_id": observed_revision_id,
            "retrieval_capability": exact_capability,
            "parser": parser_facts,
            "chunking": chunking_facts,
            "rerank": {
                "requested": True,
                "observed": sorted(
                    {
                        item["rerank"]
                        for item in retrieval_results
                        if isinstance(item.get("rerank"), bool)
                    }
                    | {
                        item["rerank"]
                        for item in chat_results
                        if isinstance(item.get("rerank"), bool)
                    }
                ),
            },
            "embedding_space_ids": embedding_space_ids,
            "chat_evaluated": evaluate_chat,
            "code_version": _code_version(),
            "model_names": model_names,
            "fingerprints": fingerprints,
            "source_fingerprints": source_fingerprints,
            "model_capability_fingerprint": None,
            "retrieval_profile_hashes": retrieval_profile_hashes,
            "phase_1_contract_budget": contract_budget,
            "runtime_configuration_fingerprint": runtime_fingerprint,
        },
        "corpus": indexing,
        "retrieval_cases": retrieval_results,
        "chat_cases": chat_results,
        "metrics": metrics,
        "gate": {
            "baseline_recorded": (
                len(retrieval_results) == len(manifest["cases"])
                and (not evaluate_chat or len(chat_results) == len(manifest["cases"]))
                and metrics["out_of_allowlist_citation_count"] == 0
            ),
            "phase_2_allowed": None,
            "note": "Phase 1 records the baseline; Phase 2 static benefit is evaluated later.",
        },
        "limitations": {
            "exact_baseline_is_expected_to_miss_some_multi_target_chains": True,
            "provider_response_bodies_persisted": False,
            "questions_persisted_in_report": False,
            "model_capability_fingerprint": (
                "provider capability/configuration fingerprint is not exposed by the terminal API"
            ),
            "embedding_provider_model_fingerprint": (
                "embedding provider/model fingerprint is not exposed by the public API"
            ),
            "runtime_configuration_fingerprint": (
                "derived only from publicly observable response facts and requested bounded inputs"
            ),
            "unavailable_facts": unavailable_facts,
            "partial_outcome": (
                "comparison_partial_001 records observed terminal partial only; the runner does not synthesize partial answers"
            ),
        },
    }


def summarize_results(
    retrieval_results: list[dict[str, Any]],
    chat_results: list[dict[str, Any]],
) -> dict[str, Any]:
    positive = [item for item in retrieval_results if item["required_target_count"] > 0]
    targets = [rank for item in positive for rank in item["target_ranks"]]
    case_target_ranks = [item["target_ranks"] for item in positive]
    required_evidence_recalled = [
        bool(recalled)
        for item in positive
        for recalled in item.get("required_evidence_recalled_at_10", [])
    ]
    goals = [recalled for item in positive for recalled in item["goal_recalled_at_10"]]
    elapsed = [float(item["elapsed_seconds"]) for item in retrieval_results]
    chat_available = bool(chat_results)
    citation_total = sum(int(item.get("citation_count", 0)) for item in chat_results)
    allowed_citations = sum(int(item.get("allowed_citation_count", 0)) for item in chat_results)
    complete = sum(bool(item.get("citation_complete", False)) for item in chat_results)
    outcome = sum(bool(item["outcome_matches"]) for item in chat_results)
    no_answer = [item for item in chat_results if item["expected_outcome"] == "refused"]
    return {
        "case_count": len(retrieval_results),
        "category_counts": {
            category: sum(item["category"] == category for item in retrieval_results)
            for category in sorted(REQUIRED_CATEGORIES)
        },
        "evidence_recall_at_1": _rank_recall(targets, 1),
        "evidence_recall_at_5": _rank_recall(targets, 5),
        "evidence_recall_at_10": _rank_recall(targets, 10),
        "mrr_at_10": _mrr_at_k(case_target_ranks, 10),
        "ndcg_at_10": _mean([float(item["ndcg_at_10"]) for item in positive]),
        "per_goal_coverage_at_10": _mean([1.0 if item else 0.0 for item in goals]),
        "required_evidence_recall_at_10": _ratio(
            sum(required_evidence_recalled), len(required_evidence_recalled)
        ),
        "complete_chain_recall_at_10": _mean(
            [1.0 if item["complete_chain_at_10"] else 0.0 for item in positive]
        ),
        "evidence_group_recall_at_10": _mean(
            [
                1.0 if item["visual_group_recalled"] else 0.0
                for item in positive
                if item["expects_visual"]
            ]
        ),
        "retrieval_call_count": len(retrieval_results),
        "retrieval_p50_seconds": _percentile(elapsed, 0.5),
        "retrieval_p95_seconds": _percentile(elapsed, 0.95),
        "chat_evaluated": chat_available,
        "answer_outcome_accuracy": (_ratio(outcome, len(chat_results)) if chat_available else None),
        "citation_precision": (_ratio(allowed_citations, citation_total) if chat_available else None),
        "citation_completeness": (_ratio(complete, len(chat_results)) if chat_available else None),
        "out_of_allowlist_citation_count": citation_total - allowed_citations,
        "document_only_citation_match_count": sum(
            int(item.get("document_only_allowed_count", 0)) for item in chat_results
        ),
        "required_evidence_citation_recall": _ratio(
            sum(
                int(item.get("required_evidence_citation_match_count", 0))
                for item in chat_results
            ),
            sum(int(item.get("required_evidence_count", 0)) for item in chat_results),
        ),
        "claim_recall": _ratio(
            sum(int(item.get("claim_match_count", 0)) for item in chat_results),
            sum(int(item.get("required_claim_count", 0)) for item in chat_results),
        ),
        "claim_citation_recall": _ratio(
            sum(int(item.get("claim_citation_match_count", 0)) for item in chat_results),
            sum(int(item.get("required_claim_count", 0)) for item in chat_results),
        ),
        "conflict_pair_recall": _ratio(
            sum(int(item.get("conflict_pair_match_count", 0)) for item in chat_results),
            sum(int(item.get("expected_conflict_pair_count", 0)) for item in chat_results),
        ),
        "conflict_case_accuracy": _ratio(
            sum(
                bool(item.get("conflict_complete", False))
                and bool(item.get("outcome_matches", False))
                for item in chat_results
                if item.get("expected_conflict_pair_count", 0)
            ),
            sum(1 for item in chat_results if item.get("expected_conflict_pair_count", 0)),
        ),
        "visual_relation_recall": _ratio(
            sum(bool(item.get("visual_complete", False)) for item in chat_results if item.get("expects_visual")),
            sum(1 for item in chat_results if item.get("expects_visual")),
        ),
        "control_reason_counts": {
            str(reason): sum(item.get("control_reason") == reason for item in chat_results)
            for reason in sorted(
                {
                    item.get("control_reason")
                    for item in chat_results
                    if item.get("control_reason") is not None
                }
            )
        },
        "no_answer_control_reason_present": _ratio(
            sum(bool(item.get("control_reason")) for item in no_answer),
            len(no_answer),
        ),
        "control_reason_accuracy": _ratio(
            sum(
                bool(item.get("control_reason_matches", False))
                for item in chat_results
                if item.get("expected_control_reason") is not None
            ),
            sum(
                item.get("expected_control_reason") is not None
                for item in chat_results
            ),
        ),
        "no_answer_safety": (
            _ratio(
                sum(
                    item.get("outcome") == "refused"
                    and int(item.get("citation_count", 0)) == 0
                    and int(item.get("claim_match_count", 0)) == 0
                    and bool(item.get("citation_complete", False))
                    and item.get("expected_control_reason") == _NO_ANSWER_CONTROL_REASON
                    and item.get("control_reason") == _NO_ANSWER_CONTROL_REASON
                    for item in no_answer
                ),
                len(no_answer),
            )
            if chat_available
            else None
        ),
        "chat_provider_call_count": sum(int(item["provider_call_count"]) for item in chat_results),
        "chat_input_tokens": sum(int(item["input_tokens"]) for item in chat_results),
        "chat_output_tokens": sum(int(item["output_tokens"]) for item in chat_results),
        "chat_repair_count": sum(bool(item["repair_attempted"]) for item in chat_results),
        "chat_p50_seconds": (
            _percentile([float(item["elapsed_seconds"]) for item in chat_results], 0.5)
            if chat_available
            else None
        ),
        "chat_p95_seconds": (
            _percentile([float(item["elapsed_seconds"]) for item in chat_results], 0.95)
            if chat_available
            else None
        ),
    }


def _evaluate_retrieval_case(
    api: str,
    kb_id: str,
    documents: dict[str, str],
    case: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    response = _json_request(
        f"{api}/retrieval/query",
        method="POST",
        payload={
            "knowledge_base_id": kb_id,
            "query": case["question"],
            "top_k": 10,
            "strategy": "exact_vector",
            "rerank": True,
            "include_debug": True,
        },
    )
    elapsed = time.perf_counter() - started
    evidence = response.get("evidence")
    if not isinstance(evidence, list):
        raise RuntimeError("retrieval response omitted evidence")
    debug = response.get("debug")
    if not isinstance(debug, dict):
        raise RuntimeError("retrieval response omitted debug profile")
    query_plan = debug.get("query_plan")
    if not isinstance(query_plan, dict):
        raise RuntimeError("retrieval response omitted debug query plan")
    if query_plan.get("strategy") != "exact_vector":
        raise RuntimeError("retrieval baseline did not use exact_vector strategy")
    if query_plan.get("revision_selector") != "active":
        raise RuntimeError("retrieval baseline did not use active revision selector")
    if query_plan.get("top_k") != 10 or query_plan.get("rerank") is not True:
        raise RuntimeError("retrieval baseline query plan does not match bounded request")
    if query_plan.get("build_status") != "ready" or query_plan.get("serving_status") != "serving":
        raise RuntimeError("retrieval baseline query plan is not ready + serving")
    revision_id = debug.get("resolved_active_revision_id")
    if not isinstance(revision_id, str) or not revision_id:
        raise RuntimeError("retrieval debug omitted resolved active revision")
    if any(
        not isinstance(item, dict) or item.get("index_revision_id") != revision_id
        for item in evidence
    ):
        raise RuntimeError("retrieval evidence and debug revisions do not match")
    expected = case["expected"]
    goals = expected["goals"]
    target_keys = list(
        dict.fromkeys(
            target
            for goal in goals
            for target in goal.get("relevant_targets", [])
        )
    )
    ranks_by_target: dict[str, int | None] = {}
    for target in target_keys:
        document_id = documents[target]
        ranks_by_target[target] = next(
            (
                rank
                for rank, item in enumerate(evidence, start=1)
                if isinstance(item, dict) and item.get("document_id") == document_id
            ),
            None,
        )
    required_keys = list(expected.get("required_evidence", []))
    required_ranks = {
        key: next(
            (
                rank
                for rank, item in enumerate(evidence, start=1)
                if _evidence_matches_annotation(item, key, documents)
            ),
            None,
        )
        for key in required_keys
    }
    required_recalled = [
        rank is not None and rank <= 10 for rank in required_ranks.values()
    ]
    goal_recalled = [
        bool(goal["relevant_targets"])
        and all(
            any(
                _annotation_target(key) == target
                and required_ranks.get(key) is not None
                and (required_ranks[key] or 11) <= 10
                for key in required_keys
            )
            for target in goal["relevant_targets"]
        )
        for goal in goals
    ]
    expected_ids = {documents[target] for target in target_keys}
    expected_relation = expected.get("expected_relation")
    visual_relation_matched = any(
        isinstance(item, dict)
        and item.get("document_id") in expected_ids
        and any(_evidence_matches_annotation(item, key, documents) for key in required_keys)
        and _related_visual_matches(item, expected_relation)
        for item in evidence
    )
    visual_group_recalled = visual_relation_matched if expected["expects_visual"] else False
    profile_hash = _content_hash(query_plan)
    return {
        "case_key": case["case_key"],
        "category": case["category"],
        "expected_outcome": expected["expected_outcome"],
        "required_target_count": len(target_keys),
        "target_ranks": [ranks_by_target[key] for key in target_keys],
        "required_evidence_ranks": required_ranks,
        "required_evidence_recalled_at_10": required_recalled,
        "goal_recalled_at_10": goal_recalled,
        "goal_count": len(goals),
        "goal_missing_count": sum(not recalled for recalled in goal_recalled),
        "complete_chain_at_10": bool(goal_recalled) and all(goal_recalled),
        "ndcg_at_10": _ndcg_for_evidence(evidence, expected_ids),
        "expects_visual": bool(expected["expects_visual"]),
        "visual_group_recalled": visual_group_recalled,
        "visual_relation_matched": visual_relation_matched,
        "expected_relation": expected_relation,
        "result_count": len(evidence),
        "index_revision_id": revision_id,
        "embedding_space_id": (
            str(query_plan.get("embedding_space_id"))
            if query_plan.get("embedding_space_id") is not None
            else None
        ),
        "rerank": query_plan.get("rerank"),
        "retrieval_profile_hash": profile_hash,
        "retrieval_debug": {
            "strategy": query_plan.get("strategy"),
            "revision_selector": query_plan.get("revision_selector"),
            "build_status": query_plan.get("build_status"),
            "serving_status": query_plan.get("serving_status"),
            "top_k": query_plan.get("top_k"),
            "rerank": query_plan.get("rerank"),
        },
        "elapsed_seconds": round(elapsed, 6),
    }


def _evaluate_chat_case(
    api: str,
    kb_id: str,
    documents: dict[str, str],
    case: dict[str, Any],
    *,
    expected_revision_id: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    session = _json_request(
        f"{api}/chat/sessions",
        method="POST",
        payload={"knowledge_base_id": kb_id, "title": f"deep-eval-{case['case_key']}"},
    )
    created = _json_request(
        f"{api}/chat/runs",
        method="POST",
        headers={"Idempotency-Key": str(uuid4())},
        payload={
            "session_id": _required_string(session, "id"),
            "knowledge_base_id": kb_id,
            "message": case["question"],
            "retrieval": {"mode": "vector", "top_k": 10, "rerank": True},
        },
    )
    run_id = _required_string(created, "run_id")
    terminal, elapsed = _wait_for_chat_run(
        api,
        run_id,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    terminal_revision_id = terminal.get("index_revision_id")
    if terminal_revision_id != expected_revision_id:
        raise RuntimeError("Chat terminal revision does not match retrieval baseline revision")
    retrieval_profile = terminal.get("retrieval")
    if not isinstance(retrieval_profile, dict):
        raise RuntimeError("Chat terminal omitted frozen retrieval profile")
    if (
        retrieval_profile.get("profile_version") != "exact_vector_v1"
        or retrieval_profile.get("strategy") != "exact_vector"
        or retrieval_profile.get("top_k") != 10
        or retrieval_profile.get("rerank") is not True
    ):
        raise RuntimeError("Chat terminal retrieval profile is not exact_vector_v1")
    attempt = _latest_attempt(terminal.get("timing"))
    outcome = attempt.get("outcome")
    control_reason = attempt.get("control_reason")
    expected = case["expected"]
    expected_outcome = str(expected["expected_outcome"])
    outcome_matches = {
        "supported": outcome == "answered",
        "partial": outcome == "partial",
        "conflict": False,
        "refused": outcome == "refused",
    }[expected_outcome]
    expected_control_reason = expected.get("expected_control_reason")
    control_reason_matches = (
        control_reason == expected_control_reason
        if expected_control_reason is not None
        else control_reason is None
    )
    citations = terminal.get("citations")
    if not isinstance(citations, list):
        citations = []
    required_keys = list(expected.get("required_evidence", []))
    allowed_keys = list(expected.get("allowed_citation_keys", []))
    allowed_targets = {documents[target] for target, _marker in map(_split_annotation, allowed_keys)}
    citation_document_ids = [item.get("document_id") for item in citations if isinstance(item, dict)]
    citation_matches = [
        _citation_annotation_match(item, allowed_keys, documents) for item in citations
    ]
    allowed_count = sum(match is not None for match in citation_matches)
    document_only_allowed_count = sum(
        item in allowed_targets and match is None
        for item, match in zip(citation_document_ids, citation_matches)
    )
    required_citation_matches = {
        key: any(
            _citation_annotation_match(item, [key], documents) == key
            for item in citations
        )
        for key in required_keys
    }
    evidence_citation_complete = (
        not citations and not expected["required_claims"]
        if expected_outcome == "refused"
        else bool(required_keys)
        and all(required_citation_matches.values())
    )
    usage = terminal.get("usage") if isinstance(terminal.get("usage"), dict) else {}
    calls = usage.get("calls") if isinstance(usage.get("calls"), dict) else {}
    call_rows = [item for item in calls.values() if isinstance(item, dict)]
    call_usage = [
        item["usage"]
        for item in call_rows
        if isinstance(item.get("usage"), dict)
    ]
    answer = terminal.get("answer")
    citation_markers = set(_CITATION_MARKER.findall(answer if isinstance(answer, str) else ""))
    valid_citation_markers = {
        marker
        for marker in citation_markers
        if 1 <= int(marker) <= len(citations)
    }
    claim_matches = [
        _claim_matches_answer(answer, claim) for claim in expected.get("required_claims", [])
    ]
    claim_citation_matches = [
        matched and bool(valid_citation_markers) for matched in claim_matches
    ]
    conflict_pairs = expected.get("conflict_pairs", []) or []
    conflict_pair_matches = [
        all(
            any(
                _citation_annotation_match(item, [key], documents) == key
                for item in citations
            )
            for key in pair
        )
        for pair in conflict_pairs
    ]
    conflict_explicit = (
        outcome == "partial"
        and bool(conflict_pairs)
        and all(conflict_pair_matches)
        and len(valid_citation_markers) >= min(2, len(citations))
    )
    if expected_outcome == "conflict":
        outcome_matches = outcome == "refused" or conflict_explicit
    citation_complete = evidence_citation_complete and (
        all(claim_citation_matches) if claim_citation_matches else True
    )
    expected_relation = expected.get("expected_relation")
    validation = attempt.get("validation") if isinstance(attempt.get("validation"), dict) else {}
    visual_facts = attempt.get("visual_evidence")
    if not isinstance(visual_facts, dict):
        visual_facts = {}
    attached_image_count = visual_facts.get("attached_image_count")
    final_context_media: list[Any] = []
    if expected["expects_visual"]:
        final_context = _json_request(f"{api}/chat/runs/{run_id}/final-context")
        if final_context.get("available") is not True:
            raise RuntimeError("visual Chat case omitted the final model context")
        media = final_context.get("media")
        if not isinstance(media, list):
            raise RuntimeError("visual Chat final context omitted media")
        final_context_media = media
    visual_citations, bound_visual_media = _visual_binding_facts(
        attempt,
        citations,
        final_context_media,
        allowed_keys=allowed_keys,
        documents=documents,
        expected_relation=expected_relation,
    )
    visual_relation_citations = visual_citations
    visual_complete = (
        not expected["expects_visual"]
        or (
            all(required_citation_matches.values())
            and bool(visual_citations or bound_visual_media)
            and isinstance(attached_image_count, int)
            and attached_image_count > 0
            and attached_image_count == len(final_context_media)
        )
    )
    return {
        "case_key": case["case_key"],
        "category": case["category"],
        "expected_outcome": expected_outcome,
        "status": terminal.get("status"),
        "outcome": outcome,
        "control_reason": control_reason,
        "expected_control_reason": expected_control_reason,
        "control_reason_matches": control_reason_matches,
        "outcome_matches": outcome_matches,
        "citation_count": len(citations),
        "allowed_citation_count": allowed_count,
        "document_only_allowed_count": document_only_allowed_count,
        "required_evidence_count": len(required_keys),
        "required_evidence_citation_match_count": sum(required_citation_matches.values()),
        "required_evidence_citation_matches": required_citation_matches,
        "citation_complete": citation_complete,
        "citation_marker_count": len(citation_markers),
        "valid_citation_marker_count": len(valid_citation_markers),
        "required_claim_count": len(claim_matches),
        "claim_match_count": sum(claim_matches),
        "claim_matches": claim_matches,
        "claim_citation_match_count": sum(claim_citation_matches),
        "claim_citation_matches": claim_citation_matches,
        "expected_conflict_pair_count": len(conflict_pairs),
        "conflict_pair_match_count": sum(conflict_pair_matches),
        "conflict_pair_matches": conflict_pair_matches,
        "conflict_explicit": conflict_explicit,
        "conflict_complete": (
            bool(conflict_pairs) and all(conflict_pair_matches) and outcome_matches
        ),
        "expects_visual": bool(expected["expects_visual"]),
        "expected_relation": expected_relation,
        "visual_citation_count": len(visual_citations),
        "visual_relation_citation_count": len(visual_relation_citations),
        "final_context_media_count": len(final_context_media),
        "final_context_bound_visual_count": len(bound_visual_media),
        "attached_image_count": attached_image_count,
        "visual_complete": visual_complete,
        "provider_call_count": len(call_rows),
        "model_names": sorted(
            {
                str(item["model"])
                for item in call_rows
                if isinstance(item.get("model"), str) and item["model"]
            }
        ),
        "index_revision_id": terminal_revision_id,
        "retrieval_profile": retrieval_profile,
        "retrieval_profile_hash": _content_hash(retrieval_profile),
        "rerank": retrieval_profile.get("rerank"),
        "input_tokens": sum(
            int(item.get("prompt_tokens", item.get("input_tokens", 0)) or 0)
            for item in call_usage
        ),
        "output_tokens": sum(
            int(item.get("completion_tokens", item.get("output_tokens", 0)) or 0)
            for item in call_usage
        ),
        "repair_attempted": bool(validation.get("repair_attempted", False)),
        "elapsed_seconds": round(elapsed, 6),
    }


def _split_annotation(value: str) -> tuple[str, str]:
    if value.count("#") != 1:
        raise ValueError("annotation key must contain one # separator")
    return tuple(value.split("#", 1))  # type: ignore[return-value]


def _annotation_target(value: str) -> str:
    return _split_annotation(value)[0]


def _annotation_marker(value: str) -> str:
    return _split_annotation(value)[1]


def _annotation_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_annotation_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_annotation_text(item) for item in value)
    return ""


def _evidence_matches_annotation(
    item: object,
    annotation: str,
    documents: dict[str, str],
) -> bool:
    if not isinstance(item, dict):
        return False
    target, marker = _split_annotation(annotation)
    if item.get("document_id") != documents.get(target):
        return False
    searchable = " ".join(
        _annotation_text(item.get(field))
        for field in (
            "text",
            "quoted_text",
            "source_location",
            "hierarchy",
            "source_metadata",
            "evidence_group_key",
            "related_visuals",
        )
    )
    return marker in searchable


def _citation_annotation_match(
    citation: object,
    annotations: list[str],
    documents: dict[str, str],
) -> str | None:
    if not isinstance(citation, dict):
        return None
    for annotation in annotations:
        if _evidence_matches_annotation(citation, annotation, documents):
            return annotation
    return None


def _terminal_citation_bindings(
    attempt: object,
    citations: object,
) -> dict[str, dict[str, Any]] | None:
    """Align persisted original citation IDs with the public terminal order.

    The public terminal DTO intentionally omits the internal ``cite_N`` value,
    while the persisted attempt keeps it.  The repository preserves the same
    contiguous citation order, so an ambiguity in either side must fail closed
    instead of guessing an asset-to-citation relationship.
    """

    if not isinstance(attempt, dict) or not isinstance(citations, list):
        return None
    citation_ids = attempt.get("citation_ids")
    if not isinstance(citation_ids, list) or len(citation_ids) != len(citations):
        return None
    bindings: dict[str, dict[str, Any]] = {}
    for citation_id, citation in zip(citation_ids, citations, strict=True):
        if (
            not isinstance(citation_id, str)
            or _CITATION_ID.fullmatch(citation_id) is None
            or citation_id in bindings
            or not isinstance(citation, dict)
        ):
            return None
        bindings[citation_id] = citation
    return bindings


def _visual_binding_facts(
    attempt: object,
    citations: object,
    final_context_media: object,
    *,
    allowed_keys: list[str],
    documents: dict[str, str],
    expected_relation: object,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return only citation-bound visual terminal facts.

    A visual hit is valid only when the final model media, terminal citation,
    selected visual decision, parent text citation, target annotation, asset,
    and expected relation all describe the same chain.  This deliberately has
    no permissive fallback for an arbitrary image in the final context.
    """

    bindings = _terminal_citation_bindings(attempt, citations)
    if bindings is None or not isinstance(final_context_media, list):
        return [], []
    visual_facts = attempt.get("visual_evidence") if isinstance(attempt, dict) else None
    decisions = visual_facts.get("decisions") if isinstance(visual_facts, dict) else None
    if not isinstance(decisions, list):
        return [], []

    allowed_parent_ids = {
        citation_id
        for citation_id, citation in bindings.items()
        if _citation_annotation_match(citation, allowed_keys, documents) is not None
    }
    selected_by_asset: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        reason_code = decision.get("reason_code")
        asset_id = decision.get("asset_id")
        parent_ids = decision.get("parent_text_citation_ids")
        if (
            not isinstance(reason_code, str)
            or not reason_code.startswith("selected_")
            or not isinstance(asset_id, str)
            or not asset_id
            or decision.get("relation_type") != expected_relation
            or not isinstance(parent_ids, list)
            or not parent_ids
            or any(not isinstance(item, str) for item in parent_ids)
            or not allowed_parent_ids.intersection(parent_ids)
        ):
            continue
        selected_by_asset[asset_id] = decision

    media_citation_ids: set[str] = set()
    bound_media: list[dict[str, Any]] = []
    for media in final_context_media:
        if not isinstance(media, dict):
            continue
        citation_ids = media.get("citation_ids")
        asset = media.get("asset")
        asset_id = asset.get("id") if isinstance(asset, dict) else None
        media_type = asset.get("media_type") if isinstance(asset, dict) else None
        if (
            not isinstance(citation_ids, list)
            or not citation_ids
            or any(
                not isinstance(citation_id, str) or citation_id not in bindings
                for citation_id in citation_ids
            )
            or not isinstance(asset_id, str)
            or not asset_id
            or not isinstance(media_type, str)
            or not media_type.startswith("image/")
        ):
            continue
        selected = selected_by_asset.get(asset_id)
        if selected is None:
            continue
        matching_terminal_visual = False
        for citation_id in citation_ids:
            terminal = bindings[citation_id]
            terminal_asset = terminal.get("asset")
            if not isinstance(terminal_asset, dict):
                continue
            if (
                str(terminal_asset.get("id")) == asset_id
                and isinstance(terminal_asset.get("media_type"), str)
                and terminal_asset["media_type"].startswith("image/")
                and terminal_asset.get("relation_type") == expected_relation
                and str(terminal_asset.get("selection_reason", "")).startswith("selected_")
            ):
                matching_terminal_visual = True
                break
        if not matching_terminal_visual:
            continue
        media_citation_ids.update(citation_ids)
        bound_media.append(media)

    visual_citations: list[dict[str, Any]] = []
    for citation_id, citation in bindings.items():
        if citation_id not in media_citation_ids:
            continue
        asset = citation.get("asset")
        if not isinstance(asset, dict):
            continue
        asset_id = asset.get("id")
        parent_id = asset.get("parent_citation_id")
        media_type = asset.get("media_type")
        if (
            not isinstance(asset_id, str)
            or asset_id not in selected_by_asset
            or not isinstance(media_type, str)
            or not media_type.startswith("image/")
            or asset.get("relation_type") != expected_relation
            or not str(asset.get("selection_reason", "")).startswith("selected_")
            or not isinstance(parent_id, str)
            or parent_id not in allowed_parent_ids
        ):
            continue
        visual_citations.append(citation)
    return visual_citations, bound_media


def _related_visual_matches(item: dict[str, Any], expected_relation: object) -> bool:
    related = item.get("related_visuals")
    if isinstance(related, list):
        for visual in related:
            if not isinstance(visual, dict):
                continue
            if expected_relation is None or visual.get("relation_type") == expected_relation:
                return True
    return expected_relation is None and item.get("modality") in {"image", "table"}


def _claim_matches_answer(answer: object, claim: str) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return False
    normalized_answer = " ".join(answer.casefold().split())
    normalized_claim = " ".join(claim.casefold().split())
    return bool(normalized_claim) and normalized_claim in normalized_answer


def _require_exact_vector_capability(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("modes"), list):
        raise RuntimeError("retrieval capability response omitted modes")
    for mode in value["modes"]:
        if (
            isinstance(mode, dict)
            and mode.get("mode") == "vector"
            and mode.get("strategy") == "exact_vector"
            and mode.get("profile_version") == "exact_vector_v1"
            and mode.get("enabled") is True
        ):
            return {
                "mode": mode["mode"],
                "strategy": mode["strategy"],
                "profile_version": mode["profile_version"],
                "enabled": mode["enabled"],
            }
    raise RuntimeError("exact_vector_v1 capability is unavailable")


def _validate_corpus(corpus: list[Any]) -> set[str]:
    keys: set[str] = set()
    filenames: set[str] = set()
    for item in corpus:
        if not isinstance(item, dict) or set(item) not in (
            {"document_key", "filename", "media_type", "content"},
            {"document_key", "filename", "media_type", "generator"},
        ):
            raise ValueError("deep retrieval corpus entry is invalid")
        key = item.get("document_key")
        filename = item.get("filename")
        if not isinstance(key, str) or not key or not isinstance(filename, str) or not filename:
            raise ValueError("deep retrieval corpus identity is invalid")
        filename_path = _safe_corpus_filename(filename)
        suffix = filename_path.suffix.lower()
        if item.get("media_type") != _MEDIA_SUFFIXES.get(suffix):
            raise ValueError("deep retrieval corpus media type does not match filename")
        if key in keys or filename in filenames:
            raise ValueError("deep retrieval corpus identities must be unique")
        if "content" in item and (not isinstance(item["content"], str) or not item["content"].strip()):
            raise ValueError("deep retrieval corpus content is invalid")
        if "generator" in item and item["generator"] != "quartz_visual_v1":
            raise ValueError("deep retrieval corpus generator is unsupported")
        keys.add(key)
        filenames.add(filename)
    return keys


def _safe_corpus_filename(filename: str) -> Path:
    """Require a visible, suffix-bearing basename with no traversal syntax."""

    filename_path = Path(filename)
    if (
        "\x00" in filename
        or not filename.strip()
        or filename_path.is_absolute()
        or filename_path.name != filename
        or "/" in filename
        or "\\" in filename
        or filename_path.name.startswith(".")
        or filename_path.name in {".", ".."}
        or any(part in {".", ".."} for part in filename_path.parts)
    ):
        raise ValueError("deep retrieval corpus filename must be a safe basename")
    return filename_path


def _validate_cases(
    cases: list[Any],
    document_keys: set[str],
    *,
    visual_document_keys: set[str] | None = None,
    corpus_evidence_text: dict[str, str] | None = None,
) -> frozenset[str]:
    case_keys: set[str] = set()
    categories: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "case_key", "question", "category", "tags", "expected"
        }:
            raise ValueError("deep retrieval case shape is invalid")
        key = case.get("case_key")
        category = case.get("category")
        tags = case.get("tags")
        if not isinstance(key, str) or not key or key in case_keys:
            raise ValueError("deep retrieval case keys must be non-empty and unique")
        if category not in REQUIRED_CATEGORIES:
            raise ValueError("deep retrieval case category is invalid")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("deep retrieval case question is invalid")
        if (
            not isinstance(tags, list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
            or len(tags) != len(set(tags))
        ):
            raise ValueError("deep retrieval case tags must be unique")
        _validate_expected(
            case["expected"],
            document_keys,
            visual_document_keys=visual_document_keys or set(),
            corpus_evidence_text=corpus_evidence_text or {},
        )
        case_keys.add(key)
        categories.add(category)
    return frozenset(categories)


def _validate_expected(
    expected: Any,
    document_keys: set[str],
    *,
    visual_document_keys: set[str] | None = None,
    corpus_evidence_text: dict[str, str] | None = None,
) -> None:
    required = {
        "goals", "required_evidence", "expected_outcome", "required_claims",
        "allowed_citation_keys", "expects_conflict", "expects_visual"
    }
    optional = {"conflict_pairs", "expected_relation", "expected_control_reason"}
    if not isinstance(expected, dict) or not required <= set(expected) <= required | optional:
        raise ValueError("deep retrieval expected annotation is invalid")
    if expected["expected_outcome"] not in ALLOWED_OUTCOMES:
        raise ValueError("deep retrieval expected outcome is invalid")
    if not isinstance(expected["expects_conflict"], bool) or not isinstance(
        expected["expects_visual"], bool
    ):
        raise ValueError("deep retrieval conflict/visual flags must be booleans")
    goals = expected["goals"]
    if not isinstance(goals, list) or not 1 <= len(goals) <= 4:
        raise ValueError("deep retrieval case must contain one to four goals")
    raw_goal_ids = [goal.get("goal_key") for goal in goals if isinstance(goal, dict)]
    if (
        len(raw_goal_ids) != len(goals)
        or any(not isinstance(goal_id, str) or not goal_id.strip() for goal_id in raw_goal_ids)
        or len(raw_goal_ids) != len(set(raw_goal_ids))
    ):
        raise ValueError("deep retrieval goal IDs must be unique")
    goal_ids = set(raw_goal_ids)
    dependencies: dict[str, list[str]] = {}
    relevant_target_bindings: dict[str, set[str]] = {}
    for goal in goals:
        if not isinstance(goal, dict) or not {"goal_key", "question", "relevant_targets"} <= set(goal) <= {
            "goal_key", "question", "relevant_targets", "depends_on"
        }:
            raise ValueError("deep retrieval goal annotation is invalid")
        if not isinstance(goal["goal_key"], str) or not goal["goal_key"].strip():
            raise ValueError("deep retrieval goal key is invalid")
        if not isinstance(goal["question"], str) or not goal["question"].strip():
            raise ValueError("deep retrieval goal question is invalid")
        targets = goal["relevant_targets"]
        if (
            not isinstance(targets, list)
            or any(not isinstance(target, str) or not target.strip() for target in targets)
            or len(targets) != len(set(targets))
        ):
            raise ValueError("deep retrieval goal targets must be unique")
        if any(not isinstance(target, str) or target not in document_keys for target in targets):
            raise ValueError("deep retrieval goal target is unknown")
        deps = goal.get("depends_on", [])
        if (
            not isinstance(deps, list)
            or any(not isinstance(dep, str) or dep not in goal_ids for dep in deps)
        ):
            raise ValueError("deep retrieval goal dependency is unknown")
        if len(deps) != len(set(deps)) or goal["goal_key"] in deps:
            raise ValueError("deep retrieval goal dependencies must be unique and non-self")
        dependencies[str(goal["goal_key"])] = [str(dep) for dep in deps]
        relevant_target_bindings[str(goal["goal_key"])] = set(targets)
    _require_acyclic(dependencies)
    for field in ("required_evidence", "required_claims", "allowed_citation_keys"):
        values = expected[field]
        if (
            not isinstance(values, list)
            or any(not isinstance(item, str) or not item.strip() for item in values)
            or len(values) != len(set(values))
        ):
            raise ValueError("deep retrieval annotation list is invalid")
    required_evidence = _parse_annotation_keys(expected["required_evidence"])
    allowed_citations = _parse_annotation_keys(expected["allowed_citation_keys"])
    if not set(required_evidence).issubset(allowed_citations):
        raise ValueError("required evidence must be citation-allowlisted")
    required_targets = {
        target for values in relevant_target_bindings.values() for target in values
    }
    evidence_targets = {target for target, _marker in required_evidence}
    if evidence_targets != required_targets:
        raise ValueError("required evidence and goal relevant targets are not bound")
    if any(
        target not in required_targets or target not in document_keys
        for target, _marker in allowed_citations
    ):
        raise ValueError("allowed citation target is not bound to a goal")
    known_text = corpus_evidence_text or {}
    for target, marker in (*required_evidence, *allowed_citations):
        if target in known_text and marker not in known_text[target]:
            raise ValueError("evidence/citation marker is not present in its corpus target")
    if expected["expected_outcome"] != "refused" and required_targets and not expected["required_evidence"]:
        raise ValueError("supported/partial/conflict cases require evidence")
    if expected["expected_outcome"] == "refused" and (
        expected["required_evidence"] or expected["required_claims"] or expected["allowed_citation_keys"]
    ):
        raise ValueError("refused evaluation cases cannot require evidence or claims")
    if bool(expected["expects_conflict"]) != (expected["expected_outcome"] == "conflict"):
        raise ValueError("conflict annotation is inconsistent")
    conflict_pairs = expected.get("conflict_pairs", [])
    if conflict_pairs is None:
        conflict_pairs = []
    if not isinstance(conflict_pairs, list):
        raise ValueError("conflict_pairs must be a list")
    normalized_pairs: set[tuple[str, str]] = set()
    allowed_keys = {f"{target}#{marker}" for target, marker in allowed_citations}
    required_keys = {f"{target}#{marker}" for target, marker in required_evidence}
    for pair in conflict_pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(item, str) for item in pair)
            or pair[0] == pair[1]
            or pair[0] not in allowed_keys
            or pair[1] not in allowed_keys
            or pair[0] not in required_keys
            or pair[1] not in required_keys
        ):
            raise ValueError("conflict pair shape or member is invalid")
        pair_key = tuple(sorted((pair[0], pair[1])))
        if pair_key in normalized_pairs:
            raise ValueError("conflict pairs must be unique")
        if pair[0].split("#", 1)[0] == pair[1].split("#", 1)[0]:
            raise ValueError("conflict pairs must identify distinct targets")
        normalized_pairs.add(pair_key)
    if expected["expects_conflict"] and not normalized_pairs:
        raise ValueError("conflict cases require conflict_pairs")
    if not expected["expects_conflict"] and normalized_pairs:
        raise ValueError("non-conflict cases must not contain conflict_pairs")
    expected_relation = expected.get("expected_relation")
    if expected["expects_visual"]:
        if not isinstance(expected_relation, str) or not expected_relation.strip():
            raise ValueError("visual cases require expected_relation")
        visual_targets = {
            target
            for values in relevant_target_bindings.values()
            for target in values
            if target in (visual_document_keys or set())
        }
        if not visual_targets:
            # The fixture uses a generated visual target.  Do not infer visual
            # semantics from a flag when no visual corpus target is available.
            raise ValueError("visual case has no visual corpus target")
    elif expected_relation is not None:
        raise ValueError("non-visual cases must not contain expected_relation")
    expected_control_reason = expected.get("expected_control_reason")
    if expected_control_reason is not None and (
        not isinstance(expected_control_reason, str)
        or not expected_control_reason.strip()
        or len(expected_control_reason) > 128
    ):
        raise ValueError("expected_control_reason is invalid")
    if expected["expected_outcome"] == "refused":
        if expected_control_reason != _NO_ANSWER_CONTROL_REASON:
            raise ValueError("refused cases require insufficient_evidence control reason")
    elif "expected_control_reason" in expected:
        raise ValueError("only refused cases may declare expected_control_reason")


def _parse_annotation_keys(values: list[str]) -> set[tuple[str, str]]:
    """Parse target-scoped ``document_key#marker`` annotation identities."""

    parsed: set[tuple[str, str]] = set()
    for value in values:
        if value.count("#") != 1:
            raise ValueError("evidence/citation keys must contain one target marker separator")
        target, marker = value.split("#", 1)
        if not target or not marker or target.strip() != target or marker.strip() != marker:
            raise ValueError("evidence/citation key target and marker must be non-empty")
        parsed.add((target, marker))
    return parsed


def _require_acyclic(dependencies: dict[str, list[str]]) -> None:
    visiting: set[str] = set()
    complete: set[str] = set()

    def visit(goal_id: str) -> None:
        if goal_id in visiting:
            raise ValueError("deep retrieval goal dependencies contain a cycle")
        if goal_id in complete:
            return
        visiting.add(goal_id)
        for dependency in dependencies[goal_id]:
            visit(dependency)
        visiting.remove(goal_id)
        complete.add(goal_id)

    for goal_id in dependencies:
        visit(goal_id)


def _write_quartz_visual(path: Path) -> None:
    image = Image.new("RGB", (640, 360), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.polygon(((320, 55), (210, 270), (430, 270)), fill=(224, 154, 37), outline=(92, 65, 15), width=6)
    draw.rectangle((40, 125, 155, 205), fill=(69, 155, 102), outline=(20, 80, 45), width=5)
    draw.rectangle((485, 125, 600, 205), fill=(69, 155, 102), outline=(20, 80, 45), width=5)
    draw.line((155, 165, 250, 165), fill=(55, 71, 79), width=8)
    draw.line((390, 165, 485, 165), fill=(55, 71, 79), width=8)
    _write_pdf(
        path,
        (
            "Quartz deployment diagram - Figure 4",
            "Evidence marker QUARTZ-VISUAL-TRIANGLE",
            "The amber triangle is the routing coordinator.",
        ),
        image,
        image_draws=((86, 280, 440, 248),),
    )


def _rank_recall(ranks: list[int | None], k: int) -> float:
    return _mean([1.0 if rank is not None and rank <= k else 0.0 for rank in ranks])


def _mrr_at_k(
    ranks: list[int | None] | list[list[int | None]],
    k: int,
) -> float:
    """Mean reciprocal rank using the first relevant result per case.

    A case may have multiple required targets.  Flattening those target ranks
    would reward a case twice and is not MRR; only its earliest relevant rank
    contributes.
    """

    if any(isinstance(rank, (list, tuple)) for rank in ranks):
        first_ranks = [
            min(
                (rank for rank in case if rank is not None and rank <= k),
                default=None,
            )
            if isinstance(case, (list, tuple))
            else case
            for case in ranks
        ]
    else:
        first_ranks = list(ranks)  # type: ignore[list-item]
    return _mean(
        [
            1.0 / rank if rank is not None and rank <= k else 0.0
            for rank in first_ranks
        ]
    )


def _ndcg_for_evidence(evidence: list[Any], expected_document_ids: set[str]) -> float:
    if not expected_document_ids:
        return 0.0
    seen_expected: set[str] = set()
    dcg = 0.0
    for index, item in enumerate(evidence[:10]):
        if not isinstance(item, dict):
            continue
        document_id = item.get("document_id")
        if document_id not in expected_document_ids or document_id in seen_expected:
            continue
        seen_expected.add(str(document_id))
        dcg += 1.0 / math.log2(index + 2)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(expected_document_ids), 10)))
    return round(dcg / ideal if ideal else 0.0, 6)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 6)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    return round(
        ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower),
        6,
    )


def _content_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _code_version() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return revision + ("+dirty" if dirty else "")


def _source_fingerprints(manifest_path: Path | None = None) -> dict[str, str | None]:
    """Hash evaluator and actual frozen contract inputs without reading secrets."""

    root = Path(__file__).resolve().parents[1]
    candidates: dict[str, Path] = {
        "runner": Path(__file__).resolve(),
        "multimodal_evaluator": root / "tools/evaluate_multimodal_real.py",
        "domain_contract": root / "src/rag_kb/domain/deep_retrieval.py",
        "wire_contract": root / "src/rag_kb/schemas/deep_retrieval.py",
    }
    if manifest_path is not None:
        candidates["manifest"] = manifest_path.resolve()
    fingerprints: dict[str, str | None] = {}
    for name, path in candidates.items():
        try:
            content_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            fingerprints[name] = "sha256:" + content_sha256
        except (OSError, ValueError):
            fingerprints[name] = None
    return fingerprints


def _content_safe_manifest_path(manifest_path: Path) -> str:
    resolved = manifest_path.resolve()
    repository = Path(__file__).resolve().parents[1]
    try:
        return resolved.relative_to(repository).as_posix()
    except ValueError:
        return resolved.name


if __name__ == "__main__":
    raise SystemExit(main())
