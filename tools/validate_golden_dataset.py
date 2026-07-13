#!/usr/bin/env python3
"""Validate golden JSONL traceability, labels, checksums, and coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_TAGS = {
    "zh",
    "en",
    "mixed",
    "exact_identifier",
    "no_answer",
    "update",
    "delete",
    "revision_filter",
    "status_filter",
    "malicious_document",
    "partial_answer",
}
VALID_OUTCOMES = {"answered", "partial", "refused"}
PROHIBITED_SECRET_KEYS = {"api_key", "authorization", "password", "secret", "token"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--provider-declaration", type=Path)
    parser.add_argument("--evaluation-config", type=Path)
    parser.add_argument("--report-schema", type=Path)
    return parser.parse_args()


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def secret_fields(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in PROHIBITED_SECRET_KEYS:
                findings.append(f"{path}.{key}")
            findings.extend(secret_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(secret_fields(child, f"{path}[{index}]"))
    return findings


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    try:
        corpus = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"corpus manifest failed: {exc}", file=sys.stderr)
        return 1
    corpus_entries = {item["sample_id"]: item for item in corpus.get("documents", [])}
    corpus_ids = set(corpus_entries)
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        args.dataset.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(errors, f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(case, dict):
            fail(errors, f"line {line_number}: case must be an object")
            continue
        cases.append(case)

    ids = [case.get("case_id") for case in cases]
    duplicate_ids = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        fail(errors, f"duplicate case IDs: {duplicate_ids}")

    tag_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    for case in cases:
        case_id = case.get("case_id", "<missing>")
        for field in ("case_id", "question", "language", "tags", "retrieval", "answer"):
            if field not in case:
                fail(errors, f"{case_id}: missing {field}")
        tags = case.get("tags", [])
        if not isinstance(tags, list) or not tags:
            fail(errors, f"{case_id}: tags must be a non-empty list")
            tags = []
        tag_counts.update(tags)
        language_counts.update([case.get("language")])

        retrieval = case.get("retrieval", {})
        expected = set(retrieval.get("expected_relevant_sample_ids", []))
        forbidden = set(retrieval.get("forbidden_sample_ids", []))
        unknown = sorted((expected | forbidden) - corpus_ids)
        if unknown:
            fail(errors, f"{case_id}: unknown sample IDs {unknown}")
        overlap = sorted(expected & forbidden)
        if overlap:
            fail(errors, f"{case_id}: expected/forbidden overlap {overlap}")
        expected_empty = retrieval.get("expected_empty")
        if expected_empty is True and expected:
            fail(errors, f"{case_id}: expected-empty case has relevant samples")
        if expected_empty is False and not expected:
            fail(errors, f"{case_id}: answerable case lacks relevant samples")
        filters = retrieval.get("filters", {})
        for key in ("workspace_id", "current_only", "build_status", "serving_status"):
            if key not in filters:
                fail(errors, f"{case_id}: missing filter {key}")

        evidence_items = case.get("required_evidence", [])
        evidence_ids = {item.get("sample_id") for item in evidence_items}
        if not evidence_ids.issubset(expected):
            fail(errors, f"{case_id}: required evidence is not a relevant sample")
        for item in evidence_items:
            sample_id = item.get("sample_id")
            needles = item.get("must_contain_any", [])
            if not isinstance(needles, list) or not needles:
                fail(errors, f"{case_id}: evidence {sample_id} lacks passage labels")
                continue
            entry = corpus_entries.get(sample_id)
            if entry is None:
                continue
            source_path = args.corpus_manifest.parent / entry["path"]
            try:
                source_text = source_path.read_text(encoding="utf-8")
            except OSError:
                fail(errors, f"{case_id}: evidence source unreadable for {sample_id}")
                continue
            if not any(needle in source_text for needle in needles):
                fail(errors, f"{case_id}: no passage label found in {sample_id}")

        answer = case.get("answer", {})
        outcome = answer.get("expected_outcome")
        outcome_counts.update([outcome])
        if outcome not in VALID_OUTCOMES:
            fail(errors, f"{case_id}: invalid outcome {outcome}")
        facts = answer.get("required_facts", [])
        for fact in facts:
            if not set(fact.get("acceptable_sample_ids", [])).issubset(expected):
                fail(errors, f"{case_id}: fact cites a non-relevant sample")
        if outcome == "refused" and facts:
            fail(errors, f"{case_id}: refused case contains required facts")
        if outcome == "partial" and not answer.get("missing_aspects"):
            fail(errors, f"{case_id}: partial case lacks missing aspects")
        found_secrets = secret_fields(case)
        if found_secrets:
            fail(errors, f"{case_id}: prohibited secret fields {found_secrets}")

    missing_tags = sorted(REQUIRED_TAGS - set(tag_counts))
    if missing_tags:
        fail(errors, f"required tags missing: {missing_tags}")

    dataset_sha = canonical_sha256(args.dataset)
    summary = {
        "schema_version": "1.0",
        "dataset": str(args.dataset),
        "sha256": dataset_sha,
        "case_count": len(cases),
        "language_counts": dict(sorted(language_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "errors": errors,
        "valid": not errors,
    }
    if args.dataset_manifest:
        manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
        if manifest.get("dataset_sha256") != dataset_sha:
            fail(errors, "dataset manifest checksum mismatch")
        if manifest.get("case_count") != len(cases):
            fail(errors, "dataset manifest case count mismatch")
        if manifest.get("corpus_manifest_sha256") != canonical_sha256(
            args.corpus_manifest
        ):
            fail(errors, "corpus manifest checksum mismatch")
        if manifest.get("language_counts") != dict(sorted(language_counts.items())):
            fail(errors, "dataset manifest language counts mismatch")
        if manifest.get("outcome_counts") != dict(sorted(outcome_counts.items())):
            fail(errors, "dataset manifest outcome counts mismatch")
        if not REQUIRED_TAGS.issubset(set(manifest.get("required_coverage", []))):
            fail(errors, "dataset manifest required coverage mismatch")
        if args.provider_declaration:
            provider = json.loads(
                args.provider_declaration.read_text(encoding="utf-8")
            )
            if manifest.get("provider_declaration_sha256") != canonical_sha256(
                args.provider_declaration
            ):
                fail(errors, "provider declaration checksum mismatch")
            fingerprint = provider.get("embedding", {}).get(
                "embedding_space", {}
            ).get("compatibility_fingerprint")
            if manifest.get("embedding_space_fingerprint") != fingerprint:
                fail(errors, "embedding-space fingerprint mismatch")
        if args.evaluation_config:
            json.loads(args.evaluation_config.read_text(encoding="utf-8"))
            if manifest.get("evaluation_config_sha256") != canonical_sha256(
                args.evaluation_config
            ):
                fail(errors, "evaluation config checksum mismatch")
        if args.report_schema:
            json.loads(args.report_schema.read_text(encoding="utf-8"))
            if manifest.get("report_schema_sha256") != canonical_sha256(
                args.report_schema
            ):
                fail(errors, "report schema checksum mismatch")
        summary["errors"] = errors
        summary["valid"] = not errors
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
