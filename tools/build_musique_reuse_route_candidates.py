#!/usr/bin/env python3
"""Build a source-backed MuSiQue route-case variant over an existing corpus.

The variant deliberately reuses the byte-identical document set from the
completed expanded route corpus.  It adds deterministic answerable MuSiQue
cases whose supporting paragraphs are already present in that corpus, so the
host qualification can test a larger denominator without re-ingesting the
same documents or rotating the completed Graph build.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from tools.build_musique_route_candidates import (
    _canonical_bytes,
    _graph_case,
    _read_source,
    _stable_select,
    _validate_upstream_row,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET_ID = "routing-rag-musique-expanded-v1"
DATASET_ID = "routing-rag-musique-reuse-v2"
SOURCE_ROOT = ROOT / "evaluation/routing-rag-musique-expanded-v1"
DEFAULT_OUTPUT = ROOT / "evaluation/routing-rag-musique-reuse-v2"
GRAPH_HOP_QUOTAS = {2: 29, 3: 23, 4: 11}


class CorpusError(RuntimeError):
    """Raised when the reusable corpus contract is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _rows(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CorpusError(f"JSONL row is not an object: {path}")
        values.append(value)
    return values


def _source_paragraph_hash(paragraph: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "title": str(paragraph.get("title", "")).strip(),
                "paragraph_text": str(paragraph.get("paragraph_text", "")).strip(),
            }
        )
    )


def _select_reusable_rows(
    source_rows: list[dict[str, Any]],
    *,
    source_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], str]]:
    source_hash_map: dict[str, str] = {}
    documents = source_manifest.get("documents", ())
    if not isinstance(documents, list):
        raise CorpusError("source manifest documents are invalid")
    for document in documents:
        if not isinstance(document, Mapping):
            raise CorpusError("source manifest document is invalid")
        source_sha256 = document.get("source_sha256")
        document_id = document.get("document_id")
        if not isinstance(source_sha256, str) or not isinstance(document_id, str):
            raise CorpusError("source manifest document hash is invalid")
        previous = source_hash_map.get(source_sha256)
        if previous is not None and previous != document_id:
            raise CorpusError("source hash maps to multiple documents")
        source_hash_map[source_sha256] = document_id
    selected: list[dict[str, Any]] = []
    mapping: dict[tuple[str, int], str] = {}
    for hops, count in GRAPH_HOP_QUOTAS.items():
        eligible: list[dict[str, Any]] = []
        for row in source_rows:
            if row.get("answerable") is not True:
                continue
            decomposition = row.get("question_decomposition")
            if not isinstance(decomposition, list) or len(decomposition) != hops:
                continue
            paragraphs = {
                paragraph["idx"]: paragraph
                for paragraph in row["paragraphs"]
            }
            support_locations: list[tuple[str, int]] = []
            support_hashes: list[str] = []
            for step in decomposition:
                support_idx = step.get("paragraph_support_idx")
                if not isinstance(support_idx, int):
                    support_locations = []
                    break
                support_locations.append((str(row["id"]), support_idx))
                paragraph = paragraphs.get(support_idx)
                if paragraph is None:
                    support_locations = []
                    break
                support_hashes.append(_source_paragraph_hash(paragraph))
            document_ids = [source_hash_map.get(source_sha256) for source_sha256 in support_hashes]
            if (
                not support_locations
                or any(value is None for value in document_ids)
                or len(set(document_ids)) != len(document_ids)
            ):
                continue
            _validate_upstream_row(row, answerable=True)
            for location, document_id in zip(support_locations, document_ids, strict=True):
                assert document_id is not None
                mapping[location] = document_id
            eligible.append(row)
        selected.extend(_stable_select(eligible, count=count))
    if len({str(row["id"]) for row in selected}) != len(selected):
        raise CorpusError("reusable MuSiQue rows collide")
    return selected, mapping


def _copy_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def build(*, source: Path, output: Path, source_root: Path = SOURCE_ROOT) -> None:
    if output.exists():
        raise CorpusError(f"output already exists: {output}")
    source_manifest = json.loads(
        (source_root / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("dataset_id") != SOURCE_DATASET_ID:
        raise CorpusError("source dataset identity invalid")
    source_documents = _rows(source_root / "documents.jsonl")
    manifest_documents = source_manifest.get("documents")
    if not isinstance(manifest_documents, list) or len(manifest_documents) != len(source_documents):
        raise CorpusError("source document manifest mismatch")
    selected, mapping = _select_reusable_rows(
        _read_source(source), source_manifest=source_manifest
    )
    old_cases = _rows(source_root / "cases.jsonl")
    reused_controls = [
        row
        for row in old_cases
        if row.get("route_label") in {"negative_or_refusal", "simple_only"}
    ]
    if len(reused_controls) != 28:
        raise CorpusError("source control case count invalid")
    graph_cases = [
        _graph_case(index, row, mapping)
        for index, row in enumerate(sorted(selected, key=lambda item: str(item["id"])), start=1)
    ]
    cases = [*graph_cases, *reused_controls]
    output.mkdir(parents=True)
    documents_dir = output / "documents"
    documents_dir.mkdir()
    for document in source_documents:
        filename = document.get("filename")
        if not isinstance(filename, str):
            raise CorpusError("source document filename invalid")
        source_path = source_root / "documents" / filename
        target_path = documents_dir / filename
        if not source_path.is_file() or source_path.is_symlink():
            raise CorpusError(f"source document artifact invalid: {filename}")
        shutil.copy2(source_path, target_path)
    _copy_jsonl(output / "documents.jsonl", source_documents)
    _copy_jsonl(output / "cases.jsonl", cases)
    license_path = source_root / "LICENSE-MUSIQUE.txt"
    (output / "LICENSE-MUSIQUE.txt").write_text(
        license_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "SOURCE_NOTICES.md").write_text(
        """# Source notices

This evaluator-only case variant reuses the byte-identical document
artifacts from `routing-rag-musique-expanded-v1`.  The additional
answerable cases are selected from the same MuSiQue-Full validation
source and reference only support paragraphs already present in that
completed corpus.  See the source corpus notices and included
`LICENSE-MUSIQUE.txt` for attribution and license information.
""",
        encoding="utf-8",
    )
    graph_counts = "、".join(f"{hops}×{count}" for hops, count in GRAPH_HOP_QUOTAS.items())
    (output / "README.md").write_text(
        f"""# MuSiQue reusable Graph-route cases v2

This is a deterministic evaluator-only case expansion over the
completed `routing-rag-musique-expanded-v1` document and Graph build.
It adds {len(graph_cases)} answerable Graph candidates ({graph_counts})
and reuses the original 12 refusal controls plus 16 direct controls.
Every added candidate's support document is already present in the
source corpus byte-for-byte; no document re-ingestion is required for
this variant.  The host binding must therefore point at the completed
expanded-v1 index revision and Graph build, and record the reuse
relationship explicitly.

Qualification remains dynamic and fail-closed: a case counts only when
Simple top-10 lacks a complete path, Graph completes it, and Graph
contributes a new source chunk.  Fewer than 30 qualified cases is not
reported as a route-recall denominator.
""",
        encoding="utf-8",
    )
    source_manifest_sha256 = _sha256_path(source_root / "manifest.json")
    manifest = {
        "schema": "musique_expanded_route_candidates_v1",
        "dataset_id": DATASET_ID,
        "language": "en",
        "license": "CC-BY-4.0",
        "source": {
            "repository": "https://github.com/stonybrooknlp/musique",
            "dataset": "https://huggingface.co/datasets/bdsaglam/musique",
            "file": "musique_full_v1.0_dev.jsonl",
            "source_sha256": source_manifest.get("source", {}).get("source_sha256"),
            "reuse_from": {
                "dataset_id": SOURCE_DATASET_ID,
                "manifest_sha256": source_manifest_sha256,
            },
            "selection": {
                "graph_hop_quotas": GRAPH_HOP_QUOTAS,
                "negative_hop_quotas": {2: 6, 3: 4, 4: 2},
                "simple_control_count": 16,
                "distractors_per_parent": source_manifest.get("source", {})
                .get("selection", {})
                .get("distractors_per_parent"),
                "reused_document_set": True,
            },
        },
        "case_count": len(cases),
        "document_count": len(source_documents),
        "case_counts": dict(Counter(str(case.get("route_label")) for case in cases)),
        "hop_counts": dict(Counter(case.get("hop_count") for case in cases)),
        "documents": manifest_documents,
        "qualification": {
            "status": "pending_host_simple_graph_qualification",
            "candidate_count": len(graph_cases),
            "minimum_qualified_graph_needed_count": 30,
            "simple_top_k": 10,
            "graph_edge_limit": 16,
            "source_chunk_target": 12,
            "source_chunk_limit": 16,
            "required_graph_new_chunk_count": 1,
            "schema_profile_key": "generic_open_domain_v1",
        },
        "artifacts": {},
    }
    for name in (
        "README.md",
        "SOURCE_NOTICES.md",
        "LICENSE-MUSIQUE.txt",
        "cases.jsonl",
        "documents.jsonl",
    ):
        manifest["artifacts"][name] = _sha256_path(output / name)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", choices=("build",))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    build(
        source=arguments.source,
        output=arguments.output,
        source_root=arguments.source_root,
    )
    print(f"built {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
