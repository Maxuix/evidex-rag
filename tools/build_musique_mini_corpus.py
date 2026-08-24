#!/usr/bin/env python3
"""Build and validate a small source-backed MuSiQue-Full routing corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "routing-rag-musique-mini"
DATASET_ID = "routing-rag-musique-full-mini-v1"
SCHEMA = "musique_routing_mini_manifest_v1"
UPSTREAM_REPOSITORY = "https://github.com/stonybrooknlp/musique"
UPSTREAM_DATA_FILE = "musique_full_v1.0_dev.jsonl"
UPSTREAM_DRIVE_ID = "15GtqhsI29UGr-mmE4dvu5Z1RgA2FBFE6"
MIRROR_REPOSITORY = "https://huggingface.co/datasets/bdsaglam/musique"
DISTRACTORS_PER_PARENT = 4


GRAPH_CASES = (
    "2hop__510860_22402",
    "2hop__779424_75487",
    "2hop__203985_524737",
    "2hop__215898_67465",
    "2hop__85931_108632",
    "2hop__780238_110949",
    "3hop1__823336_228453_86925",
    "3hop1__857_846_7794",
    "3hop1__354480_834494_34053",
    "3hop1__820301_720914_41132",
)
NEGATIVE_CASES = (
    "2hop__35445_22458",
    "2hop__719184_55227",
    "3hop1__579562_629431_64412",
)
SIMPLE_CONTROLS = (
    ("2hop__510860_22402", 0),
    ("2hop__779424_75487", 0),
    ("3hop1__820301_720914_41132", 0),
)
PARENT_IDS = tuple(dict.fromkeys((*GRAPH_CASES, *NEGATIVE_CASES)))
EXPECTED_ANSWERABLE = {
    **{row_id: True for row_id in GRAPH_CASES},
    **{row_id: False for row_id in NEGATIVE_CASES},
}


class CorpusError(RuntimeError):
    """Raised when an upstream row or generated corpus violates its contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized[:48] or "untitled"


def _jsonl_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise CorpusError(f"upstream row is not an object: {path}")
            yield value


def _source_rows(paths: Sequence[Path]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping) and isinstance(value.get("rows"), list):
            candidates = (
                item.get("row")
                for item in value["rows"]
                if isinstance(item, Mapping)
            )
        elif path.suffix == ".jsonl":
            candidates = _jsonl_rows(path)
        else:
            raise CorpusError(f"unsupported upstream source shape: {path}")
        for row in candidates:
            if not isinstance(row, Mapping):
                continue
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise CorpusError(f"upstream row has no id: {path}")
            if row_id not in EXPECTED_ANSWERABLE:
                continue
            if row.get("answerable") is not EXPECTED_ANSWERABLE[row_id]:
                continue
            previous = rows.get(row_id)
            if previous is not None and previous != row:
                raise CorpusError(f"conflicting upstream row: {row_id}")
            rows[row_id] = row
    missing = sorted(set(PARENT_IDS) - rows.keys())
    if missing:
        raise CorpusError(f"selected upstream rows are missing: {', '.join(missing)}")
    return {row_id: rows[row_id] for row_id in PARENT_IDS}


def _validate_upstream_row(row: Mapping[str, Any], *, expected_answerable: bool) -> None:
    required = {
        "id",
        "paragraphs",
        "question",
        "question_decomposition",
        "answer",
        "answer_aliases",
        "answerable",
    }
    if set(row) != required or row.get("answerable") is not expected_answerable:
        raise CorpusError(f"upstream row shape is invalid: {row.get('id')}")
    paragraphs = row.get("paragraphs")
    decomposition = row.get("question_decomposition")
    if not isinstance(paragraphs, list) or len(paragraphs) != 20:
        raise CorpusError(f"upstream paragraph count is invalid: {row['id']}")
    if not isinstance(decomposition, list) or len(decomposition) not in {2, 3}:
        raise CorpusError(f"upstream hop count is invalid: {row['id']}")
    paragraph_indexes: set[int] = set()
    for paragraph in paragraphs:
        if (
            not isinstance(paragraph, Mapping)
            or set(paragraph) != {"idx", "title", "paragraph_text", "is_supporting"}
            or isinstance(paragraph.get("idx"), bool)
            or not isinstance(paragraph.get("idx"), int)
            or not isinstance(paragraph.get("title"), str)
            or not paragraph["title"].strip()
            or not isinstance(paragraph.get("paragraph_text"), str)
            or not paragraph["paragraph_text"].strip()
            or not isinstance(paragraph.get("is_supporting"), bool)
            or paragraph["idx"] in paragraph_indexes
        ):
            raise CorpusError(f"upstream paragraph is invalid: {row['id']}")
        paragraph_indexes.add(paragraph["idx"])
    supported_steps = 0
    step_ids: set[int] = set()
    for step in decomposition:
        if (
            not isinstance(step, Mapping)
            or set(step) != {"id", "question", "answer", "paragraph_support_idx"}
            or isinstance(step.get("id"), bool)
            or not isinstance(step.get("id"), int)
            or step["id"] in step_ids
            or not isinstance(step.get("question"), str)
            or not step["question"].strip()
            or not isinstance(step.get("answer"), str)
            or not step["answer"].strip()
        ):
            raise CorpusError(f"upstream decomposition is invalid: {row['id']}")
        step_ids.add(step["id"])
        support_idx = step.get("paragraph_support_idx")
        if support_idx is not None:
            if support_idx not in paragraph_indexes:
                raise CorpusError(f"upstream support index is invalid: {row['id']}")
            paragraph = next(item for item in paragraphs if item["idx"] == support_idx)
            if not paragraph["is_supporting"]:
                raise CorpusError(f"upstream support flag is invalid: {row['id']}")
            supported_steps += 1
    if expected_answerable and supported_steps != len(decomposition):
        raise CorpusError(f"answerable row has a missing support step: {row['id']}")
    if not expected_answerable and supported_steps == len(decomposition):
        raise CorpusError(f"unanswerable row has a complete support path: {row['id']}")


def _document_records(
    rows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], str]]:
    by_hash: dict[str, dict[str, Any]] = {}
    mapping: dict[tuple[str, int], str] = {}
    for row_id, row in rows.items():
        supporting_indexes = {
            step["paragraph_support_idx"]
            for step in row["question_decomposition"]
            if step["paragraph_support_idx"] is not None
        }
        distractor_indexes = {
            item["idx"]
            for item in sorted(row["paragraphs"], key=lambda item: item["idx"])
            if item["idx"] not in supporting_indexes
        }
        selected_distractors = set(
            sorted(distractor_indexes)[:DISTRACTORS_PER_PARENT]
        )
        selected_indexes = supporting_indexes | selected_distractors
        for paragraph in row["paragraphs"]:
            if paragraph["idx"] not in selected_indexes:
                continue
            source_value = {
                "title": paragraph["title"].strip(),
                "paragraph_text": paragraph["paragraph_text"].strip(),
            }
            source_hash = _sha256_bytes(_canonical_bytes(source_value))
            record = by_hash.setdefault(
                source_hash,
                {
                    "document_id": f"mus-doc-{source_hash[:16]}",
                    "title": source_value["title"],
                    "paragraph_text": source_value["paragraph_text"],
                    "source_sha256": source_hash,
                    "upstream_locations": [],
                },
            )
            location = {"upstream_id": row_id, "paragraph_idx": paragraph["idx"]}
            if location not in record["upstream_locations"]:
                record["upstream_locations"].append(location)
            mapping[(row_id, paragraph["idx"])] = record["document_id"]
    records = sorted(by_hash.values(), key=lambda item: item["document_id"])
    for record in records:
        record["upstream_locations"].sort(
            key=lambda item: (item["upstream_id"], item["paragraph_idx"])
        )
    return records, mapping


def _decomposition(
    row: Mapping[str, Any], document_mapping: Mapping[tuple[str, int], str]
) -> list[dict[str, Any]]:
    result = []
    for hop, step in enumerate(row["question_decomposition"], start=1):
        support_idx = step["paragraph_support_idx"]
        result.append(
            {
                "step_id": f"{row['id']}::source-question-{step['id']}",
                "upstream_question_id": step["id"],
                "hop": hop,
                "question": step["question"],
                "answer": step["answer"],
                "support_document_id": (
                    document_mapping[(row["id"], support_idx)]
                    if support_idx is not None
                    else None
                ),
                "upstream_paragraph_idx": support_idx,
            }
        )
    return result


def _graph_case(
    ordinal: int,
    row: Mapping[str, Any],
    document_mapping: Mapping[tuple[str, int], str],
) -> dict[str, Any]:
    decomposition = _decomposition(row, document_mapping)
    source_ids = [step["support_document_id"] for step in decomposition]
    if any(value is None for value in source_ids):
        raise CorpusError(f"graph candidate is missing support: {row['id']}")
    return {
        "schema": "musique_routing_case_v1",
        "case_id": f"graph-{ordinal:03d}",
        "upstream_id": row["id"],
        "question": row["question"],
        "expected_answer": row["answer"],
        "answer_aliases": list(row["answer_aliases"]),
        "expected_outcome": "answered",
        "answerable": True,
        "semantic_intent": "graph",
        "route_label": "graph_needed_candidate",
        "route_label_status": "requires_host_qualification",
        "hop_count": len(decomposition),
        "decomposition": decomposition,
        "required_paths": [source_ids],
        "negative_control_kind": None,
    }


def _negative_case(
    ordinal: int,
    row: Mapping[str, Any],
    document_mapping: Mapping[tuple[str, int], str],
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


def _simple_case(
    ordinal: int,
    row: Mapping[str, Any],
    step_index: int,
    document_mapping: Mapping[tuple[str, int], str],
) -> dict[str, Any]:
    step = _decomposition(row, document_mapping)[step_index]
    support_id = step["support_document_id"]
    if support_id is None or "#" in step["question"]:
        raise CorpusError(f"simple control is not standalone: {row['id']}")
    return {
        "schema": "musique_routing_case_v1",
        "case_id": f"simple-{ordinal:03d}",
        "upstream_id": row["id"],
        "upstream_step": step_index,
        "question": step["question"],
        "expected_answer": step["answer"],
        "answer_aliases": [],
        "expected_outcome": "answered",
        "answerable": True,
        "semantic_intent": "simple",
        "route_label": "simple_only",
        "route_label_status": "frozen_direct_support_control",
        "hop_count": 1,
        "decomposition": [step],
        "required_paths": [[support_id]],
        "negative_control_kind": None,
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build(source_paths: Sequence[Path], output: Path, license_path: Path) -> None:
    if output.exists():
        raise CorpusError(f"output already exists: {output}")
    rows = _source_rows(source_paths)
    for row_id in GRAPH_CASES:
        _validate_upstream_row(rows[row_id], expected_answerable=True)
    for row_id in NEGATIVE_CASES:
        _validate_upstream_row(rows[row_id], expected_answerable=False)
    documents, document_mapping = _document_records(rows)
    output.mkdir(parents=True)
    documents_dir = output / "documents"
    documents_dir.mkdir()
    document_manifest: list[dict[str, Any]] = []
    for ordinal, document in enumerate(documents, start=1):
        filename = f"{ordinal:03d}-{_slug(document['title'])}-{document['source_sha256'][:8]}.md"
        content = f"# {document['title']}\n\n{document['paragraph_text']}\n"
        path = documents_dir / filename
        path.write_text(content, encoding="utf-8")
        document_manifest.append(
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
        _graph_case(index, rows[row_id], document_mapping)
        for index, row_id in enumerate(GRAPH_CASES, start=1)
    ]
    cases.extend(
        _negative_case(index, rows[row_id], document_mapping)
        for index, row_id in enumerate(NEGATIVE_CASES, start=1)
    )
    cases.extend(
        _simple_case(index, rows[row_id], step_index, document_mapping)
        for index, (row_id, step_index) in enumerate(SIMPLE_CONTROLS, start=1)
    )
    _write_jsonl(output / "cases.jsonl", cases)
    _write_jsonl(output / "documents.jsonl", document_manifest)
    shutil.copyfile(license_path, output / "LICENSE-MUSIQUE.txt")
    (output / "SOURCE_NOTICES.md").write_text(_source_notices(), encoding="utf-8")
    (output / "README.md").write_text(_readme(), encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "dataset_id": DATASET_ID,
        "language": "en",
        "license": "CC-BY-4.0",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "data_file": UPSTREAM_DATA_FILE,
            "google_drive_file_id": UPSTREAM_DRIVE_ID,
            "selection_mirror": MIRROR_REPOSITORY,
            "split": "validation",
            "selected_upstream_ids": list(PARENT_IDS),
            "distractors_per_parent": DISTRACTORS_PER_PARENT,
        },
        "case_count": len(cases),
        "case_counts": {
            "graph_needed_candidate": len(GRAPH_CASES),
            "negative_or_refusal": len(NEGATIVE_CASES),
            "simple_only": len(SIMPLE_CONTROLS),
        },
        "hop_counts": {
            "hop1": len(SIMPLE_CONTROLS),
            "hop2": sum(len(rows[item]["question_decomposition"]) == 2 for item in GRAPH_CASES),
            "hop3": sum(len(rows[item]["question_decomposition"]) == 3 for item in GRAPH_CASES),
        },
        "document_count": len(document_manifest),
        "documents": document_manifest,
        "qualification": {
            "status": "pending_host_simple_graph_qualification",
            "graph_candidate_k": 16,
            "source_chunk_target": 12,
            "source_chunk_limit": 16,
            "required_graph_new_chunk_count": 1,
        },
        "artifacts": {
            name: _sha256_path(output / name)
            for name in (
                "cases.jsonl",
                "documents.jsonl",
                "README.md",
                "SOURCE_NOTICES.md",
                "LICENSE-MUSIQUE.txt",
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate(output)


def validate(output: Path) -> None:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise CorpusError("manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("dataset_id") != DATASET_ID:
        raise CorpusError("manifest identity is invalid")
    cases = _read_jsonl(output / "cases.jsonl")
    documents = _read_jsonl(output / "documents.jsonl")
    if len(cases) != manifest.get("case_count") or len(documents) != manifest.get(
        "document_count"
    ):
        raise CorpusError("manifest counts are invalid")
    document_ids = {item.get("document_id") for item in documents}
    if None in document_ids or len(document_ids) != len(documents):
        raise CorpusError("document identities are invalid")
    for document in documents:
        path = output / "documents" / document["filename"]
        if not path.is_file() or _sha256_path(path) != document["artifact_sha256"]:
            raise CorpusError(f"document artifact is invalid: {document.get('document_id')}")
    case_ids: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in case_ids:
            raise CorpusError("case identities are invalid")
        case_ids.add(case_id)
        decomposition = case.get("decomposition")
        if not isinstance(decomposition, list) or len(decomposition) != case.get("hop_count"):
            raise CorpusError(f"case decomposition is invalid: {case_id}")
        referenced = {
            step.get("support_document_id")
            for step in decomposition
            if step.get("support_document_id") is not None
        }
        if not referenced <= document_ids:
            raise CorpusError(f"case references an unknown document: {case_id}")
        if case.get("answerable"):
            paths = case.get("required_paths")
            if not isinstance(paths, list) or len(paths) != 1:
                raise CorpusError(f"answerable case path is invalid: {case_id}")
            if any(item not in document_ids for item in paths[0]):
                raise CorpusError(f"answerable path document is invalid: {case_id}")
        elif case.get("required_paths") != [] or all(
            step.get("support_document_id") is not None for step in decomposition
        ):
            raise CorpusError(f"negative case path is invalid: {case_id}")
    for artifact, expected_hash in manifest.get("artifacts", {}).items():
        if _sha256_path(output / artifact) != expected_hash:
            raise CorpusError(f"artifact digest is invalid: {artifact}")


def _source_notices() -> str:
    return """# Source notices

This evaluation subset is adapted from the MuSiQue-Full validation split by
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
MuSiQue is distributed under the Creative Commons Attribution 4.0 International
license. The complete license text is included in `LICENSE-MUSIQUE.txt`.

- Official repository: https://github.com/stonybrooknlp/musique
- Paper: https://doi.org/10.1162/tacl_a_00475
- Official file: `musique_full_v1.0_dev.jsonl`
- Official Google Drive file id: `15GtqhsI29UGr-mmE4dvu5Z1RgA2FBFE6`

This repository changes the packaging only: selected upstream paragraphs are
stored as individual Markdown source documents, direct decomposition questions
are reused as Simple controls, and source/document hashes are added for local
RAG evaluation. It does not claim authorship of the upstream questions,
answers, decompositions, or paragraph text.
"""


def _readme() -> str:
    return """# MuSiQue-Full mini routing corpus

This is a small candidate corpus for host-side Simple versus first-class Graph
Tool qualification. It contains ten answerable two/three-hop questions, three
upstream unanswerable controls, and three direct single-hop controls derived
from the selected questions' first decomposition step.

The Graph cases are deliberately marked `graph_needed_candidate`, not locked
`graph_needed`. A host qualification run must prove that the frozen Simple lane
lacks a complete required source path while Graph K=16 returns that path and at
least one new source chunk. Cases that fail this rule must not enter route
recall denominators.

Only files under `documents/` are ingestion inputs. All selected paragraphs
share one KB so Simple top-10 cannot receive a per-question closed context.
`cases.jsonl` and `documents.jsonl` are evaluator-only gold/provenance files and
must never be indexed or exposed to the answering model.

Rebuild from Hugging Face dataset-server row responses or the official JSONL:

```bash
PYTHONPATH=src:. .venv/bin/python tools/build_musique_mini_corpus.py build \
  --source /path/to/musique_rows.json \
  --license /path/to/official/musique/LICENSE
PYTHONPATH=src:. .venv/bin/python tools/build_musique_mini_corpus.py validate
```

After provisioning this corpus into the isolated host runtime, qualify the ten
Graph candidates with the frozen Simple top-10 and Graph K=16 budgets:

```bash
PYTHONPATH=src:. .venv/bin/python tools/qualify_musique_mini_corpus.py \
  --confirm QUALIFY_ROUTING_RAG_MUSIQUE_MINI
```

The command writes `qualification.json`. A candidate is qualified only when
Simple lacks every complete required path, the incremental Graph evidence
completes one path, and Graph contributes at least one new serving chunk.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source", type=Path, action="append", required=True)
    build_parser.add_argument("--license", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "build":
        build(arguments.source, arguments.output, arguments.license)
        print(f"built {arguments.output}")
    else:
        validate(arguments.output)
        print(f"validated {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
