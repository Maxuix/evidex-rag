#!/usr/bin/env python3
"""Build a larger MuSiQue-Full candidate corpus for Graph-route qualification.

This tool creates *candidates*, not claimed Graph positives.  A host run must
still prove for every included case that Simple top-10 lacks a complete gold
path while Graph supplies a new source chunk that completes one.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation/routing-rag-musique-expanded-v1"
DATASET_ID = "routing-rag-musique-expanded-v1"
SCHEMA = "musique_expanded_route_candidates_v1"
UPSTREAM_REPOSITORY = "https://github.com/stonybrooknlp/musique"
UPSTREAM_DATASET = "https://huggingface.co/datasets/bdsaglam/musique"
UPSTREAM_FILE = "musique_full_v1.0_dev.jsonl"
DISTRACTORS_PER_PARENT = 8
GRAPH_HOP_QUOTAS = {2: 24, 3: 16, 4: 8}
NEGATIVE_HOP_QUOTAS = {2: 6, 3: 4, 4: 2}
SIMPLE_CONTROL_COUNT = 16


class CorpusError(RuntimeError):
    """Raised when a source row or generated corpus violates the contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized[:48] or "untitled"


def _normalize_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def _read_source(path: Path) -> list[dict[str, Any]]:
    rows = [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and isinstance((value := json.loads(line)), dict)
    ]
    if not rows:
        raise CorpusError("MuSiQue source is empty")
    return rows


def _stable_select(
    rows: Sequence[Mapping[str, Any]], *, count: int
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise CorpusError(f"source has {len(rows)} rows, expected at least {count}")
    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (
                _sha256_bytes(str(row.get("id", "")).encode("utf-8")),
                str(row.get("id", "")),
            ),
        )[:count]
    ]


def _validate_upstream_row(row: Mapping[str, Any], *, answerable: bool) -> None:
    required = {
        "id",
        "paragraphs",
        "question",
        "question_decomposition",
        "answer",
        "answer_aliases",
        "answerable",
    }
    if set(row) != required or row.get("answerable") is not answerable:
        raise CorpusError(f"upstream row shape invalid: {row.get('id')}")
    if not isinstance(row["id"], str) or not isinstance(row["question"], str):
        raise CorpusError("upstream ID or question is invalid")
    paragraphs = row["paragraphs"]
    decomposition = row["question_decomposition"]
    if not isinstance(paragraphs, list) or len(paragraphs) != 20:
        raise CorpusError(f"upstream paragraphs invalid: {row['id']}")
    if not isinstance(decomposition, list) or len(decomposition) not in {2, 3, 4}:
        raise CorpusError(f"upstream decomposition invalid: {row['id']}")
    paragraph_ids: set[int] = set()
    support_ids: set[int] = set()
    for paragraph in paragraphs:
        if (
            not isinstance(paragraph, Mapping)
            or set(paragraph) != {"idx", "title", "paragraph_text", "is_supporting"}
            or not isinstance(paragraph.get("idx"), int)
            or not isinstance(paragraph.get("title"), str)
            or not isinstance(paragraph.get("paragraph_text"), str)
            or not isinstance(paragraph.get("is_supporting"), bool)
        ):
            raise CorpusError(f"upstream paragraph invalid: {row['id']}")
        paragraph_ids.add(paragraph["idx"])
        if paragraph["is_supporting"]:
            support_ids.add(paragraph["idx"])
    if len(paragraph_ids) != 20:
        raise CorpusError(f"upstream paragraph IDs invalid: {row['id']}")
    step_supports: list[int | None] = []
    for step in decomposition:
        if (
            not isinstance(step, Mapping)
            or set(step) != {"id", "question", "answer", "paragraph_support_idx"}
            or not isinstance(step.get("id"), int)
            or not isinstance(step.get("question"), str)
            or not isinstance(step.get("answer"), str)
        ):
            raise CorpusError(f"upstream step invalid: {row['id']}")
        support = step["paragraph_support_idx"]
        if support is not None and support not in paragraph_ids:
            raise CorpusError(f"upstream support invalid: {row['id']}")
        if support is not None and support not in support_ids:
            raise CorpusError(f"upstream support flag invalid: {row['id']}")
        step_supports.append(support)
    if answerable and any(item is None for item in step_supports):
        raise CorpusError(f"answerable row lacks a required support: {row['id']}")
    if not answerable and all(item is not None for item in step_supports):
        raise CorpusError(f"negative row is fully supported: {row['id']}")


def _selected_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for hops, count in GRAPH_HOP_QUOTAS.items():
        graph.extend(
            _stable_select(
                [
                    row
                    for row in rows
                    if row.get("answerable") is True
                    and isinstance(row.get("question_decomposition"), list)
                    and len(row["question_decomposition"]) == hops
                ],
                count=count,
            )
        )
    graph_ids = {str(row["id"]) for row in graph}
    for hops, count in NEGATIVE_HOP_QUOTAS.items():
        negative.extend(
            _stable_select(
                [
                    row
                    for row in rows
                    if row.get("answerable") is False
                    and isinstance(row.get("question_decomposition"), list)
                    and len(row["question_decomposition"]) == hops
                    and str(row.get("id")) not in graph_ids
                ],
                count=count,
            )
        )
    for row in graph:
        _validate_upstream_row(row, answerable=True)
    for row in negative:
        _validate_upstream_row(row, answerable=False)
    if len({row["id"] for row in (*graph, *negative)}) != len(graph) + len(negative):
        raise CorpusError("selected upstream IDs collide")
    return graph, negative


def _document_records(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], str]]:
    documents: dict[str, dict[str, Any]] = {}
    mapping: dict[tuple[str, int], str] = {}
    for row in rows:
        support_ids = {
            step["paragraph_support_idx"]
            for step in row["question_decomposition"]
            if step["paragraph_support_idx"] is not None
        }
        distractors = [
            paragraph["idx"]
            for paragraph in sorted(row["paragraphs"], key=lambda item: item["idx"])
            if paragraph["idx"] not in support_ids
        ][:DISTRACTORS_PER_PARENT]
        selected = support_ids | set(distractors)
        for paragraph in row["paragraphs"]:
            if paragraph["idx"] not in selected:
                continue
            source_value = {
                "title": paragraph["title"].strip(),
                "paragraph_text": paragraph["paragraph_text"].strip(),
            }
            source_sha = _sha256_bytes(_canonical_bytes(source_value))
            document_id = f"mus-route-{source_sha[:16]}"
            record = documents.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "title": source_value["title"],
                    "paragraph_text": source_value["paragraph_text"],
                    "source_sha256": source_sha,
                    "upstream_locations": [],
                },
            )
            location = {"upstream_id": row["id"], "paragraph_idx": paragraph["idx"]}
            if location not in record["upstream_locations"]:
                record["upstream_locations"].append(location)
            mapping[(row["id"], paragraph["idx"])] = document_id
    for record in documents.values():
        record["upstream_locations"].sort(
            key=lambda item: (item["upstream_id"], item["paragraph_idx"])
        )
    return documents, mapping


def _decomposition(
    row: Mapping[str, Any], document_mapping: Mapping[tuple[str, int], str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for hop, step in enumerate(row["question_decomposition"], start=1):
        support = step["paragraph_support_idx"]
        result.append(
            {
                "step_id": f"{row['id']}::source-question-{step['id']}",
                "upstream_question_id": step["id"],
                "hop": hop,
                "question": step["question"],
                "answer": step["answer"],
                "support_document_id": (
                    document_mapping[(row["id"], support)] if support is not None else None
                ),
                "upstream_paragraph_idx": support,
            }
        )
    return result


def _graph_case(
    ordinal: int, row: Mapping[str, Any], document_mapping: Mapping[tuple[str, int], str]
) -> dict[str, Any]:
    decomposition = _decomposition(row, document_mapping)
    path = [step["support_document_id"] for step in decomposition]
    if any(item is None for item in path) or len(set(path)) != len(path):
        raise CorpusError(f"graph candidate path invalid: {row['id']}")
    return {
        "schema": "musique_routing_case_v1",
        "case_id": f"graph-{ordinal:03d}",
        "upstream_id": row["id"],
        "question": row["question"],
        "expected_answer": row["answer"],
        "answer_aliases": row["answer_aliases"],
        "expected_outcome": "answered",
        "answerable": True,
        "semantic_intent": "graph",
        "route_label": "graph_needed_candidate",
        "route_label_status": "requires_host_qualification",
        "hop_count": len(decomposition),
        "decomposition": decomposition,
        "required_paths": [path],
        "negative_control_kind": None,
    }


def _negative_case(
    ordinal: int, row: Mapping[str, Any], document_mapping: Mapping[tuple[str, int], str]
) -> dict[str, Any]:
    decomposition = _decomposition(row, document_mapping)
    return {
        "schema": "musique_routing_case_v1",
        "case_id": f"negative-{ordinal:03d}",
        "upstream_id": row["id"],
        "question": row["question"],
        "expected_answer": "insufficient evidence",
        "answer_aliases": [],
        "expected_outcome": "refused",
        "answerable": False,
        "semantic_intent": "simple",
        "route_label": "negative_or_refusal",
        "route_label_status": "frozen_from_upstream_answerability",
        "hop_count": len(decomposition),
        "decomposition": decomposition,
        "required_paths": [],
        "negative_control_kind": "missing_required_support",
    }


def _simple_cases(
    graph_rows: Sequence[Mapping[str, Any]],
    document_mapping: Mapping[tuple[str, int], str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for ordinal, row in enumerate(
        _stable_select(graph_rows, count=SIMPLE_CONTROL_COUNT), start=1
    ):
        first = _decomposition(row, document_mapping)[0]
        if first["support_document_id"] is None or "#" in first["question"]:
            raise CorpusError(f"simple control invalid: {row['id']}")
        cases.append(
            {
                "schema": "musique_routing_case_v1",
                "case_id": f"simple-{ordinal:03d}",
                "upstream_id": row["id"],
                "upstream_step": 0,
                "question": first["question"],
                "expected_answer": first["answer"],
                "answer_aliases": [],
                "expected_outcome": "answered",
                "answerable": True,
                "semantic_intent": "simple",
                "route_label": "simple_only",
                "route_label_status": "frozen_direct_support_control",
                "hop_count": 1,
                "decomposition": [first],
                "required_paths": [[first["support_document_id"]]],
                "negative_control_kind": None,
            }
        )
    return cases


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _source_notices() -> str:
    return """# Source notices

This candidate corpus is adapted from the MuSiQue-Full validation split by
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
MuSiQue is distributed under the Creative Commons Attribution 4.0 International
license; the full license text is included in `LICENSE-MUSIQUE.txt`.

- Official repository: https://github.com/stonybrooknlp/musique
- Dataset mirror used for reproducible retrieval: https://huggingface.co/datasets/bdsaglam/musique
- Official file: `musique_full_v1.0_dev.jsonl`
- Paper: https://doi.org/10.1162/tacl_a_00475

The local package selects validation examples deterministically by the SHA-256
ordering of upstream IDs.  It changes packaging only: selected paragraphs are
stored as Markdown documents and source/document hashes are added.  It does not
claim authorship of the upstream questions, answers, decompositions, or text.
"""


def _readme() -> str:
    return """# MuSiQue expanded Graph-route candidates v1

This is a larger source-backed candidate set for validating Graph auto-routing.
It has 48 answerable multi-hop candidates (24×2-hop, 16×3-hop, 8×4-hop), 12
upstream-unanswerable controls, and 16 direct single-hop controls.  Every
candidate has a complete, distinct source-document path; each source parent
also contributes eight distractors into one shared KB.

`graph_needed_candidate` is deliberately not a route-recall denominator.  A
frozen host qualification must mark a case `graph_needed` only if all are true:

1. Simple exact-vector top-10 lacks every complete `required_paths` path.
2. Graph K=16 plus the frozen packing budget completes a required path.
3. Graph provides at least one new serving source chunk beyond Simple evidence.

The 48-candidate pool makes an Auto route-recall estimate meaningful only after
this dynamic qualification.  If fewer than 30 cases qualify, collect or build
another corpus variant rather than reporting a route-recall percentage with an
insufficient denominator.

Only `documents/` is uploaded to the KB.  `cases.jsonl` and `documents.jsonl`
are evaluator-only gold files and must never be visible to the answering agent.

## Rebuild and validate

```bash
PYTHONPATH=src:. .venv/bin/python tools/build_musique_route_candidates.py build \\
  --source /path/to/musique_full_v1.0_dev.jsonl \\
  --license /path/to/musique/LICENSE
PYTHONPATH=src:. .venv/bin/python tools/build_musique_route_candidates.py validate
```
"""


def build(source: Path, license_path: Path, output: Path) -> None:
    if output.exists():
        raise CorpusError(f"output already exists: {output}")
    source_rows = _read_source(source)
    graph_rows, negative_rows = _selected_rows(source_rows)
    documents, document_mapping = _document_records([*graph_rows, *negative_rows])
    output.mkdir(parents=True)
    directory = output / "documents"
    directory.mkdir()
    document_rows: list[dict[str, Any]] = []
    for ordinal, document in enumerate(sorted(documents.values(), key=lambda item: item["document_id"]), start=1):
        filename = f"{ordinal:04d}-{_slug(document['title'])}-{document['source_sha256'][:8]}.md"
        path = directory / filename
        path.write_text(
            f"# {document['title']}\n\n{document['paragraph_text']}\n", encoding="utf-8"
        )
        document_rows.append(
            {
                "document_id": document["document_id"],
                "filename": filename,
                "title": document["title"],
                "source_sha256": document["source_sha256"],
                "artifact_sha256": _sha256_path(path),
                "upstream_locations": document["upstream_locations"],
            }
        )
    cases = [
        *[
            _graph_case(index, row, document_mapping)
            for index, row in enumerate(sorted(graph_rows, key=lambda item: item["id"]), start=1)
        ],
        *[
            _negative_case(index, row, document_mapping)
            for index, row in enumerate(sorted(negative_rows, key=lambda item: item["id"]), start=1)
        ],
        *_simple_cases(graph_rows, document_mapping),
    ]
    _write_jsonl(output / "cases.jsonl", cases)
    _write_jsonl(output / "documents.jsonl", document_rows)
    (output / "LICENSE-MUSIQUE.txt").write_text(
        _normalize_text(license_path.read_text(encoding="utf-8")) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(_readme(), encoding="utf-8")
    (output / "SOURCE_NOTICES.md").write_text(_source_notices(), encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "dataset_id": DATASET_ID,
        "language": "en",
        "license": "CC-BY-4.0",
        "source": {
            "repository": UPSTREAM_REPOSITORY,
            "dataset": UPSTREAM_DATASET,
            "file": UPSTREAM_FILE,
            "source_sha256": _sha256_path(source),
            "selection": {
                "graph_hop_quotas": GRAPH_HOP_QUOTAS,
                "negative_hop_quotas": NEGATIVE_HOP_QUOTAS,
                "simple_control_count": SIMPLE_CONTROL_COUNT,
                "distractors_per_parent": DISTRACTORS_PER_PARENT,
            },
        },
        "case_count": len(cases),
        "document_count": len(document_rows),
        "case_counts": dict(Counter(case["route_label"] for case in cases)),
        "hop_counts": dict(Counter(case["hop_count"] for case in cases)),
        "documents": document_rows,
        "qualification": {
            "status": "pending_host_simple_graph_qualification",
            "candidate_count": sum(case["route_label"] == "graph_needed_candidate" for case in cases),
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
    manifest["artifacts"] = {
        name: _sha256_path(output / name)
        for name in ("README.md", "SOURCE_NOTICES.md", "LICENSE-MUSIQUE.txt", "cases.jsonl", "documents.jsonl")
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate(output)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_source(path)


def validate(output: Path) -> dict[str, int]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("dataset_id") != DATASET_ID:
        raise CorpusError("manifest identity invalid")
    cases = _jsonl(output / "cases.jsonl")
    documents = _jsonl(output / "documents.jsonl")
    if len(cases) != manifest.get("case_count") or len(documents) != manifest.get("document_count"):
        raise CorpusError("manifest counts invalid")
    document_ids = {row.get("document_id") for row in documents}
    if len(document_ids) != len(documents) or None in document_ids:
        raise CorpusError("document IDs invalid")
    for document in documents:
        path = output / "documents" / str(document.get("filename"))
        if not path.is_file() or _sha256_path(path) != document.get("artifact_sha256"):
            raise CorpusError(f"document artifact invalid: {document.get('document_id')}")
    counts = Counter(case.get("route_label") for case in cases)
    expected_counts = {
        "graph_needed_candidate": sum(GRAPH_HOP_QUOTAS.values()),
        "negative_or_refusal": sum(NEGATIVE_HOP_QUOTAS.values()),
        "simple_only": SIMPLE_CONTROL_COUNT,
    }
    if dict(counts) != expected_counts:
        raise CorpusError(f"route-label counts invalid: {dict(counts)}")
    graph_hops = Counter(
        case.get("hop_count")
        for case in cases
        if case.get("route_label") == "graph_needed_candidate"
    )
    if dict(graph_hops) != GRAPH_HOP_QUOTAS:
        raise CorpusError(f"graph candidate hop counts invalid: {dict(graph_hops)}")
    case_ids: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in case_ids:
            raise CorpusError("case IDs invalid")
        case_ids.add(case_id)
        decomposition = case.get("decomposition")
        if not isinstance(decomposition, list) or len(decomposition) != case.get("hop_count"):
            raise CorpusError(f"decomposition invalid: {case_id}")
        referenced = {
            step.get("support_document_id")
            for step in decomposition
            if step.get("support_document_id") is not None
        }
        if not referenced <= document_ids:
            raise CorpusError(f"unknown source document: {case_id}")
        if case.get("route_label") == "graph_needed_candidate":
            path = case.get("required_paths", [[]])[0]
            if len(path) != case.get("hop_count") or len(path) != len(set(path)):
                raise CorpusError(f"graph path invalid: {case_id}")
        if case.get("route_label") == "negative_or_refusal":
            if case.get("required_paths") != [] or all(
                step.get("support_document_id") is not None for step in decomposition
            ):
                raise CorpusError(f"negative support invalid: {case_id}")
    for name, digest in manifest.get("artifacts", {}).items():
        if _sha256_path(output / name) != digest:
            raise CorpusError(f"artifact digest invalid: {name}")
    return {"case_count": len(cases), "document_count": len(documents)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--source", type=Path, required=True)
    build_parser.add_argument("--license", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "build":
        build(arguments.source, arguments.license, arguments.output)
        print(f"built {arguments.output}")
    else:
        print(json.dumps(validate(arguments.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
