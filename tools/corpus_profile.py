#!/usr/bin/env python3
"""Measure a versioned local corpus manifest using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


TOOL_VERSION = "1.0"
TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[._:/+@-][A-Za-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]"
)
IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"(?=[A-Z0-9._/-]*[A-Z])(?=[A-Z0-9._/-]*\d)"
    r"[A-Z][A-Z0-9]*(?:[-._/][A-Z0-9]+)+"
    r"|[A-Z]{1,8}\d{2,}"
    r")(?![A-Za-z0-9_])"
)
HEADING_PATTERN = re.compile(r"^#{1,6}\s+", re.MULTILINE)
ORDERED_LIST_PATTERN = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
UNORDERED_LIST_PATTERN = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
BLOCKQUOTE_PATTERN = re.compile(r"^\s*>\s?", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to the corpus manifest JSON")
    parser.add_argument("--output", type=Path, help="Write canonical JSON to this path")
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values, default=0),
        "p50": round(percentile(values, 0.50), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "max": max(values, default=0),
        "mean": round(sum(values) / len(values), 2) if values else 0.0,
    }


def ratio(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def safe_document_path(manifest_dir: Path, relative_path: str) -> Path:
    candidate = (manifest_dir / relative_path).resolve()
    root = manifest_dir.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"document path escapes manifest directory: {relative_path}")
    return candidate


def analyze_text(text: str, raw: bytes) -> dict[str, Any]:
    tokens = TOKEN_PATTERN.findall(text)
    identifiers = IDENTIFIER_PATTERN.findall(text)
    han_chars = sum("\u3400" <= char <= "\u9fff" for char in text)
    latin_letters = sum(
        ("A" <= char <= "Z") or ("a" <= char <= "z") for char in text
    )
    paragraphs = [part for part in re.split(r"\n\s*\n", text.strip()) if part]
    line_count = 0 if not text else text.count("\n") + (0 if text.endswith("\n") else 1)
    return {
        "bytes": len(raw),
        "characters": len(text),
        "lines": line_count,
        "paragraphs": len(paragraphs),
        "analysis_tokens": len(tokens),
        "han_characters": han_chars,
        "latin_letters": latin_letters,
        "identifier_occurrences": len(identifiers),
        "unique_identifiers": sorted(set(identifiers)),
        "normalization": {
            "has_utf8_bom": raw.startswith(b"\xef\xbb\xbf"),
            "has_crlf": b"\r\n" in raw,
            "is_unicode_nfc": unicodedata.normalize("NFC", text) == text,
        },
        "structure": {
            "headings": len(HEADING_PATTERN.findall(text)),
            "ordered_list_items": len(ORDERED_LIST_PATTERN.findall(text)),
            "unordered_list_items": len(UNORDERED_LIST_PATTERN.findall(text)),
            "table_rows": len(TABLE_ROW_PATTERN.findall(text)),
            "blockquotes": len(BLOCKQUOTE_PATTERN.findall(text)),
            "inline_code_spans": text.count("`") // 2,
            "fenced_code_blocks": text.count("```") // 2,
        },
    }


def measure(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents")
    if manifest.get("schema_version") != "1.0" or not isinstance(documents, list):
        raise ValueError("manifest must use schema_version 1.0 and contain documents[]")
    if not documents:
        raise ValueError("manifest contains no documents")

    manifest_dir = manifest_path.parent
    seen_sample_ids: set[str] = set()
    measured_documents: list[dict[str, Any]] = []
    format_counts: Counter[str] = Counter()
    language_class_counts: Counter[str] = Counter()
    update_frequency_counts: Counter[str] = Counter()
    lifecycle_counts: Counter[str] = Counter()
    build_status_counts: Counter[str] = Counter()
    serving_status_counts: Counter[str] = Counter()
    structure_document_counts: Counter[str] = Counter()
    content_tag_counts: Counter[str] = Counter()

    for entry in documents:
        sample_id = entry["sample_id"]
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)

        path = safe_document_path(manifest_dir, entry["path"])
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(
                f"checksum mismatch for {sample_id}: expected {entry['sha256']}, got {digest}"
            )
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{sample_id} is not valid UTF-8: {exc}") from exc

        metrics = analyze_text(text, raw)
        measured_documents.append(
            {
                "sample_id": sample_id,
                "path": entry["path"],
                "language_class": entry["language_class"],
                "lifecycle_state": entry["lifecycle_state"],
                "build_status": entry["build_status"],
                "serving_status": entry["serving_status"],
                "is_current_version": entry["is_current_version"],
                **metrics,
            }
        )
        format_counts[path.suffix.lower()] += 1
        language_class_counts[entry["language_class"]] += 1
        update_frequency_counts[entry["update_frequency_assumption"]] += 1
        lifecycle_counts[entry["lifecycle_state"]] += 1
        build_status_counts[entry["build_status"]] += 1
        serving_status_counts[entry["serving_status"]] += 1
        for tag in entry["structure_tags"]:
            structure_document_counts[tag] += 1
        for tag in entry["content_tags"]:
            content_tag_counts[tag] += 1

    total = len(measured_documents)
    total_tokens = sum(item["analysis_tokens"] for item in measured_documents)
    total_han = sum(item["han_characters"] for item in measured_documents)
    total_latin = sum(item["latin_letters"] for item in measured_documents)
    language_signal_total = total_han + total_latin
    total_identifier_occurrences = sum(
        item["identifier_occurrences"] for item in measured_documents
    )
    unique_identifiers = sorted(
        {
            identifier
            for item in measured_documents
            for identifier in item["unique_identifiers"]
        }
    )
    serving_eligible = [
        item
        for item in measured_documents
        if item["lifecycle_state"] == "active"
        and item["build_status"] == "ready"
        and item["serving_status"] == "serving"
        and item["is_current_version"]
    ]

    def status_filter_count(key: str, value: Any) -> int:
        return sum(item[key] == value for item in measured_documents)

    serving_language_counts = Counter(item["language_class"] for item in serving_eligible)
    return {
        "schema_version": "1.0",
        "tool": {"name": "tools/corpus_profile.py", "version": TOOL_VERSION},
        "corpus_id": manifest["corpus_id"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "measurement_method": {
            "encoding": "UTF-8 with optional BOM",
            "analysis_token_proxy": "Each CJK unified ideograph is one token; each contiguous Latin/digit identifier-like span is one token. This is not a model tokenizer.",
            "language_signal": "Counts CJK unified ideographs and ASCII Latin letters; ratios exclude punctuation, digits, and whitespace.",
            "identifier_pattern": "Uppercase alphanumeric identifiers containing separators and both letters/digits, plus compact uppercase-prefix numeric codes.",
            "percentiles": "Linear interpolation over per-document measurements.",
        },
        "document_count": total,
        "logical_document_count": len(
            {entry["logical_document_id"] for entry in documents}
        ),
        "format_counts": dict(sorted(format_counts.items())),
        "manifest_language_class": {
            "counts": dict(sorted(language_class_counts.items())),
            "ratios": {
                key: ratio(value, total)
                for key, value in sorted(language_class_counts.items())
            },
        },
        "measured_language_signal": {
            "han_characters": total_han,
            "latin_letters": total_latin,
            "han_ratio": ratio(total_han, language_signal_total),
            "latin_ratio": ratio(total_latin, language_signal_total),
        },
        "length_distributions": {
            metric: distribution([item[metric] for item in measured_documents])
            for metric in ("bytes", "characters", "lines", "paragraphs", "analysis_tokens")
        },
        "exact_identifiers": {
            "occurrences": total_identifier_occurrences,
            "unique_count": len(unique_identifiers),
            "documents_with_identifier": sum(
                item["identifier_occurrences"] > 0 for item in measured_documents
            ),
            "occurrences_per_1000_analysis_tokens": round(
                1000 * total_identifier_occurrences / total_tokens, 2
            )
            if total_tokens
            else 0.0,
            "values": unique_identifiers,
        },
        "structure_document_counts": dict(sorted(structure_document_counts.items())),
        "content_tag_counts": dict(sorted(content_tag_counts.items())),
        "update_frequency_assumption_counts": dict(
            sorted(update_frequency_counts.items())
        ),
        "normalization_observations": {
            "documents_with_bom": sum(
                item["normalization"]["has_utf8_bom"] for item in measured_documents
            ),
            "documents_with_crlf": sum(
                item["normalization"]["has_crlf"] for item in measured_documents
            ),
            "documents_not_nfc": sum(
                not item["normalization"]["is_unicode_nfc"]
                for item in measured_documents
            ),
        },
        "filter_selectivity": {
            "all_samples": {"count": total, "ratio": 1.0},
            "lifecycle_active": {
                "count": status_filter_count("lifecycle_state", "active"),
                "ratio": ratio(status_filter_count("lifecycle_state", "active"), total),
            },
            "current_version": {
                "count": status_filter_count("is_current_version", True),
                "ratio": ratio(status_filter_count("is_current_version", True), total),
            },
            "build_ready": {
                "count": status_filter_count("build_status", "ready"),
                "ratio": ratio(status_filter_count("build_status", "ready"), total),
            },
            "serving_status_serving": {
                "count": status_filter_count("serving_status", "serving"),
                "ratio": ratio(status_filter_count("serving_status", "serving"), total),
            },
            "mandatory_serving_conjunction": {
                "count": len(serving_eligible),
                "ratio": ratio(len(serving_eligible), total),
            },
            "eligible_by_language_class": {
                key: {
                    "count": serving_language_counts.get(key, 0),
                    "ratio_of_all_samples": ratio(serving_language_counts.get(key, 0), total),
                    "ratio_of_eligible": ratio(
                        serving_language_counts.get(key, 0), len(serving_eligible)
                    ),
                }
                for key in sorted(language_class_counts)
            },
        },
        "documents": measured_documents,
        "validation": {
            "manifest_schema_valid": True,
            "sample_ids_unique": True,
            "all_paths_within_manifest_directory": True,
            "all_files_present": True,
            "all_sha256_match": True,
            "all_files_valid_utf8": True,
        },
    }


def main() -> int:
    args = parse_args()
    try:
        result = measure(args.manifest)
    except (KeyError, OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"corpus profile failed: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
