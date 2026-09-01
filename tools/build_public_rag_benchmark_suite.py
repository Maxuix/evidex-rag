#!/usr/bin/env python3
"""Build and validate a source-backed public RAG quality evaluation suite.

The suite deliberately packages a small, immutable slice of four public
benchmarks instead of mirroring their much larger upstream archives.  Every
case and document keeps its upstream identifier and source digest so a change
in an upstream release cannot silently alter an evaluation denominator.
"""

from __future__ import annotations

import argparse
import bz2
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation/public-rag-benchmark-suite-v1"
DATASET_ID = "public-rag-benchmark-suite-v1"
SCHEMA = "public_rag_benchmark_suite_v1"

CRAG_FALSE_PREMISE_COUNT = 40
MTRAG_QUOTAS = {
    "UNANSWERABLE": 30,
    "UNDERSPECIFIED": 30,
    "PARTIAL": 20,
    "ANSWERABLE": 20,
}
CONFLICT_QUOTAS = {
    "No conflict": 24,
    "Complementary information": 24,
    "Conflicting opinions and research outcomes": 24,
    "Conflict due to outdated information": 24,
    "Conflict due to misinformation": 5,
}
ENTERPRISE_QUOTAS = {"conflicting_info": 20, "info_not_found": 20}
_REMOTE_MTRAG_IMAGE_CORRECTIONS = {
    "ibmcld_16257-2972-4789": 2,
    "ibmcld_16257-1484-3435": 1,
    "ibmcld_05269-23135-24741": 1,
}
_REMOTE_MARKDOWN_IMAGE = re.compile(
    r"!\[[^\]]*\]\((?:https?:)?//[^)]+\)",
    re.IGNORECASE,
)


class CorpusError(RuntimeError):
    """Raised when an upstream source or the derived corpus is invalid."""


@dataclass(frozen=True, slots=True)
class SourceInputs:
    crag: Path
    mtrag: Path
    conflicts: Path
    enterprise_questions: Path
    enterprise_uuid_index: Path
    enterprise_archive: Path
    crag_license: Path
    mtrag_license: Path
    conflicts_license: Path
    enterprise_license: Path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and isinstance((value := json.loads(line)), dict)
    ]


def _read_crag(path: Path) -> list[dict[str, Any]]:
    with bz2.open(path, "rt", encoding="utf-8") as source:
        return [
            value
            for line in source
            if line.strip() and isinstance((value := json.loads(line)), dict)
        ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _stable_select(
    rows: Sequence[Mapping[str, Any]], *, key: str, count: int
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise CorpusError(f"source has {len(rows)} rows, expected at least {count}")
    ordered = sorted(
        rows,
        key=lambda row: (
            _sha256_bytes(str(row.get(key, "")).encode("utf-8")),
            str(row.get(key, "")),
        ),
    )
    return [dict(row) for row in ordered[:count]]


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return result[:44] or "source"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _html_to_text(value: str, *, limit: int = 12_000) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    result = "\n".join(parser.parts)
    return result[:limit].strip()


def _normalize_text(value: str) -> str:
    """Keep source wording while removing non-semantic line-ending noise."""

    return "\n".join(
        re.sub(
            r"^[ \t]*",
            lambda match: match.group(0).expandtabs(4),
            line,
        ).rstrip()
        for line in value.splitlines()
    ).strip()


def _apply_remote_image_correction(upstream_id: str, text: str) -> str:
    expected = _REMOTE_MTRAG_IMAGE_CORRECTIONS.get(upstream_id)
    if expected is None:
        return text
    corrected, count = _REMOTE_MARKDOWN_IMAGE.subn("", text)
    if count != expected:
        raise CorpusError(
            f"remote image correction mismatch: {upstream_id} ({count}/{expected})"
        )
    return corrected


def _document_id(source: str, upstream_id: str, text: str) -> str:
    digest = _sha256_bytes(
        _canonical_bytes({"source": source, "upstream_id": upstream_id, "text": text})
    )
    return f"{source}-{digest[:16]}"


def _add_document(
    documents: dict[str, dict[str, Any]],
    *,
    source: str,
    upstream_id: str,
    title: str,
    text: str,
    source_sha256: str,
    source_path: str | None = None,
    identity_text: str | None = None,
) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        raise CorpusError(f"empty source document: {source}/{upstream_id}")
    identity = _normalize_text(identity_text) if identity_text is not None else cleaned
    document_id = _document_id(source, upstream_id, identity)
    existing = documents.get(document_id)
    payload = {
        "document_id": document_id,
        "source": source,
        "upstream_id": upstream_id,
        "title": title.strip() or upstream_id,
        "text": cleaned,
        "source_sha256": source_sha256,
        "source_path": source_path,
    }
    if existing is not None and existing != payload:
        raise CorpusError(f"document identity collision: {document_id}")
    documents[document_id] = payload
    return document_id


def _mtrag_answerability(row: Mapping[str, Any]) -> str:
    value = row.get("answerability")
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str):
        raise CorpusError(f"invalid MTRAG answerability: {row.get('task_id')}")
    return value[0]


def _mtrag_expected_action(answerability: str) -> str:
    return {
        "UNANSWERABLE": "refuse_insufficient_evidence",
        "UNDERSPECIFIED": "request_clarification",
        "PARTIAL": "answer_with_explicit_qualification",
        "ANSWERABLE": "answer_with_evidence",
    }[answerability]


def _add_mtrag_cases(
    rows: Sequence[Mapping[str, Any]], documents: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for answerability, count in MTRAG_QUOTAS.items():
        source_rows = [
            row
            for row in rows
            if _mtrag_answerability(row) == answerability
            and isinstance(row.get("contexts"), list)
            and (
                answerability == "UNANSWERABLE"
                or bool(row["contexts"])
            )
        ]
        selected.extend(_stable_select(source_rows, key="task_id", count=count))
    cases: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: str(item["task_id"])):
        contexts = row.get("contexts")
        if not isinstance(contexts, list):
            raise CorpusError(f"MTRAG contexts missing: {row['task_id']}")
        document_ids: list[str] = []
        for context in contexts:
            if not isinstance(context, Mapping):
                raise CorpusError(f"MTRAG context invalid: {row['task_id']}")
            upstream_id = context.get("document_id")
            text = context.get("text")
            if not isinstance(upstream_id, str) or not isinstance(text, str):
                raise CorpusError(f"MTRAG context fields invalid: {row['task_id']}")
            document_ids.append(
                _add_document(
                    documents,
                    source="mtrag_un",
                    upstream_id=upstream_id,
                    title=upstream_id,
                    text=_apply_remote_image_correction(upstream_id, text),
                    source_sha256=_sha256_bytes(_canonical_bytes(dict(context))),
                    identity_text=text,
                )
            )
        answerability = _mtrag_answerability(row)
        input_turns = row.get("input")
        question = ""
        if isinstance(input_turns, list):
            for turn in reversed(input_turns):
                if isinstance(turn, Mapping) and turn.get("speaker") == "user":
                    candidate = turn.get("text")
                    if isinstance(candidate, str):
                        question = candidate.strip()
                        break
        if not question:
            raise CorpusError(f"MTRAG question missing: {row['task_id']}")
        cases.append(
            {
                "schema": "public_rag_case_v1",
                "case_id": f"mtrag-{row['task_id']}",
                "source": "MTRAG-UN",
                "source_case_id": row["task_id"],
                "question": question,
                "dimension": "answer_refusal",
                "stratum": answerability.casefold(),
                "expected_action": _mtrag_expected_action(answerability),
                "gold_answer": row.get("targets"),
                "evidence_document_ids": list(dict.fromkeys(document_ids)),
                "source_metadata": {
                    "collection": row.get("Collection"),
                    "question_type": row.get("Question Type"),
                    "multi_turn": row.get("Multi-Turn"),
                },
            }
        )
    return cases


def _add_crag_cases(
    rows: Sequence[Mapping[str, Any]], documents: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    selected = _stable_select(
        [row for row in rows if row.get("question_type") == "false_premise"],
        key="interaction_id",
        count=CRAG_FALSE_PREMISE_COUNT,
    )
    cases: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: str(item["interaction_id"])):
        results = row.get("search_results")
        if not isinstance(results, list) or not results:
            raise CorpusError(f"CRAG search results missing: {row['interaction_id']}")
        document_ids: list[str] = []
        for ordinal, result in enumerate(results, start=1):
            if not isinstance(result, Mapping):
                raise CorpusError(f"CRAG search result invalid: {row['interaction_id']}")
            page = result.get("page_result")
            if not isinstance(page, str):
                page = result.get("page_snippet")
            if not isinstance(page, str) or not page.strip():
                continue
            upstream_id = f"{row['interaction_id']}:{ordinal}"
            document_ids.append(
                _add_document(
                    documents,
                    source="crag",
                    upstream_id=upstream_id,
                    title=str(result.get("page_name") or upstream_id),
                    text=_html_to_text(page),
                    source_sha256=_sha256_bytes(_canonical_bytes(dict(result))),
                    source_path=str(result.get("page_url") or ""),
                )
            )
        if not document_ids:
            raise CorpusError(f"CRAG selected case has no text: {row['interaction_id']}")
        question = row.get("query")
        if not isinstance(question, str) or not question.strip():
            raise CorpusError(f"CRAG question missing: {row['interaction_id']}")
        cases.append(
            {
                "schema": "public_rag_case_v1",
                "case_id": f"crag-{row['interaction_id']}",
                "source": "CRAG",
                "source_case_id": row["interaction_id"],
                "question": question,
                "dimension": "answer_refusal",
                "stratum": "open_world_false_premise",
                "expected_action": "decline_or_correct_false_premise",
                "gold_answer": row.get("answer"),
                "gold_answer_aliases": row.get("alternative_answers"),
                "evidence_document_ids": list(dict.fromkeys(document_ids)),
                "source_metadata": {
                    "domain": row.get("domain"),
                    "static_or_dynamic": row.get("static_or_dynamic"),
                    "split": row.get("split"),
                },
            }
        )
    return cases


def _conflict_expected_action(conflict_type: str) -> str:
    if conflict_type in {
        "Conflicting opinions and research outcomes",
        "Conflict due to outdated information",
        "Conflict due to misinformation",
    }:
        return "surface_evidence_conflict"
    if conflict_type == "Complementary information":
        return "combine_supported_evidence"
    return "answer_without_false_conflict"


def _add_conflict_cases(
    rows: Sequence[Mapping[str, Any]], documents: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for conflict_type, count in CONFLICT_QUOTAS.items():
        source_rows = [row for row in rows if row.get("conflict_type") == conflict_type]
        selected.extend(_stable_select(source_rows, key="question", count=count))
    cases: list[dict[str, Any]] = []
    for row in sorted(
        selected, key=lambda item: (str(item["conflict_type"]), str(item["question"]))
    ):
        question = row.get("question")
        results = row.get("search_results")
        conflict_type = row.get("conflict_type")
        if not isinstance(question, str) or not isinstance(results, list) or not isinstance(conflict_type, str):
            raise CorpusError("CONFLICTS row is invalid")
        source_case_id = _sha256_bytes(_canonical_bytes(row))[:20]
        document_ids: list[str] = []
        for ordinal, result in enumerate(results, start=1):
            if not isinstance(result, Mapping):
                continue
            text = result.get("short_text") or result.get("response_str") or result.get("snippet")
            if not isinstance(text, str) or not text.strip():
                continue
            upstream_id = f"{source_case_id}:{ordinal}"
            document_ids.append(
                _add_document(
                    documents,
                    source="conflicts",
                    upstream_id=upstream_id,
                    title=str(result.get("title") or upstream_id),
                    text=text,
                    source_sha256=_sha256_bytes(_canonical_bytes(dict(result))),
                    source_path=str(result.get("url") or ""),
                )
            )
        if len(document_ids) < 2:
            raise CorpusError(f"CONFLICTS case has insufficient evidence: {source_case_id}")
        cases.append(
            {
                "schema": "public_rag_case_v1",
                "case_id": f"conflicts-{source_case_id}",
                "source": "CONFLICTS",
                "source_case_id": source_case_id,
                "question": question.strip(),
                "dimension": "answer_refusal",
                "stratum": f"evidence_{_slug(conflict_type)}",
                "expected_action": _conflict_expected_action(conflict_type),
                "gold_answer": row.get("correct_answer"),
                "evidence_document_ids": list(dict.fromkeys(document_ids)),
                "source_metadata": {"conflict_type": conflict_type, "origin": row.get("source")},
            }
        )
    return cases


def _enterprise_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _add_enterprise_cases(
    questions: Sequence[Mapping[str, Any]],
    uuid_index: Mapping[str, Any],
    archive_path: Path,
    documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for question_type, count in ENTERPRISE_QUOTAS.items():
        selected.extend(
            _stable_select(
                [row for row in questions if row.get("question_type") == question_type],
                key="question_id",
                count=count,
            )
        )
    document_paths: dict[str, str] = {}
    for row in selected:
        expected = row.get("expected_doc_ids")
        if not isinstance(expected, list):
            raise CorpusError(f"enterprise expected docs invalid: {row.get('question_id')}")
        for upstream_id in expected:
            source_path = uuid_index.get(upstream_id)
            if not isinstance(upstream_id, str) or not isinstance(source_path, str):
                raise CorpusError(f"enterprise source mapping missing: {upstream_id}")
            document_paths[upstream_id] = source_path
    extracted: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        for upstream_id, source_path in document_paths.items():
            candidate_paths = (source_path, f"all_documents/{source_path}")
            member = next((item for item in candidate_paths if item in members), None)
            if member is None:
                # The release archive preserves source directories but prefixes
                # leaf filenames with the UUID and normalizes them to text.
                matches = [
                    item for item in members if f"{upstream_id}__" in Path(item).name
                ]
                same_parent = [
                    item
                    for item in matches
                    if Path(item).parent == Path(source_path).parent
                ]
                member = (
                    same_parent[0]
                    if len(same_parent) == 1
                    else matches[0]
                    if len(matches) == 1
                    else None
                )
            if member is None:
                raise CorpusError(f"enterprise archive member missing: {source_path}")
            raw = archive.read(member)
            extracted[upstream_id] = _add_document(
                documents,
                source="enterprise_rag_bench",
                upstream_id=upstream_id,
                title=source_path,
                text=_enterprise_text(raw),
                source_sha256=_sha256_bytes(raw),
                source_path=source_path,
            )
    cases: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: str(item["question_id"])):
        question_type = str(row["question_type"])
        expected = row["expected_doc_ids"]
        expected_action = (
            "surface_evidence_conflict"
            if question_type == "conflicting_info"
            else "refuse_closed_world_absent"
        )
        cases.append(
            {
                "schema": "public_rag_case_v1",
                "case_id": f"enterprise-{row['question_id']}",
                "source": "EnterpriseRAG-Bench",
                "source_case_id": row["question_id"],
                "question": row["question"],
                "dimension": "answer_refusal",
                "stratum": question_type,
                "expected_action": expected_action,
                "gold_answer": row.get("gold_answer"),
                "gold_answer_facts": row.get("answer_facts"),
                "evidence_document_ids": [extracted[item] for item in expected],
                "source_metadata": {"source_types": row.get("source_types")},
            }
        )
    return cases


def _write_documents(output: Path, documents: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    directory = output / "documents"
    directory.mkdir()
    manifest_rows: list[dict[str, Any]] = []
    for ordinal, (document_id, document) in enumerate(sorted(documents.items()), start=1):
        filename = f"{ordinal:04d}-{document['source']}-{document_id[-8:]}.md"
        content = f"# {document['title']}\n\n{document['text']}\n"
        path = directory / filename
        path.write_text(content, encoding="utf-8")
        manifest_rows.append(
            {
                key: value
                for key, value in document.items()
                if key != "text"
            }
            | {"filename": filename, "artifact_sha256": _sha256_path(path)}
        )
    return manifest_rows


def _source_notices() -> str:
    return """# Source notices

This immutable local evaluation slice is derived from four public benchmarks.
Only selected question records and their required input documents are repackaged;
the original data formats, question wording, evidence text, answers, and source
identifiers remain attributable to their authors.

| Source | Upstream | License | Included role |
| --- | --- | --- | --- |
| CRAG | https://github.com/facebookresearch/CRAG | CC BY-NC 4.0 | Open-world false-premise controls |
| MTRAG-UN | https://github.com/IBM/mt-rag-benchmark | Apache-2.0 | Unanswerable, underspecified, partial, and answerable controls |
| CONFLICTS | https://github.com/google-research-datasets/rag_conflicts | Apache-2.0 | Conflicting, complementary, and non-conflicting evidence |
| EnterpriseRAG-Bench | https://github.com/onyx-dot-app/EnterpriseRAG-Bench | MIT | Enterprise conflict and closed-world-absent controls |

The complete upstream license texts copied into `licenses/` govern their
respective derivative material.  CRAG's non-commercial restriction therefore
applies to every CRAG-derived file in this corpus.
"""


def _readme() -> str:
    return """# Public RAG quality benchmark suite v1

This is a source-backed, frozen RAG evaluation corpus for diagnosing answer /
refusal policy.  It keeps all source documents in a *single shared KB* and
keeps evaluator-only `cases.jsonl` and `documents.jsonl` out of ingestion.

## What it measures

- `refuse_insufficient_evidence`: retrieved passages cannot answer the question.
- `request_clarification`: the user intent is underspecified, so refusal is not
  the correct outcome.
- `answer_with_explicit_qualification`: evidence is partial; a blanket refusal
  and an unqualified answer are both wrong.
- `decline_or_correct_false_premise`: an open-world premise should not be
  invented or accepted without support.
- `surface_evidence_conflict`: contradictions must be disclosed rather than
  silently resolved into an unsupported claim.
- `refuse_closed_world_absent`: no supporting enterprise document exists in the
  frozen corpus.

The suite has no single composite score.  Report a confusion matrix and
precision/recall per expected action, plus forbidden-claim rate for every
negative stratum.  `gold_answer` is source metadata; it must not be exposed to
the evaluated system.

## Rebuild and validate

```bash
PYTHONPATH=src:. .venv/bin/python tools/build_public_rag_benchmark_suite.py build \\
  --crag /path/to/crag_task_1_and_2_dev_v4.jsonl.bz2 \\
  --mtrag /path/to/reference.jsonl \\
  --conflicts /path/to/conflicts.jsonl \\
  --enterprise-questions /path/to/questions.jsonl \\
  --enterprise-uuid-index /path/to/uuid_index.json \\
  --enterprise-archive /path/to/all_documents.zip \\
  --crag-license /path/to/CRAG/LICENSE \\
  --mtrag-license /path/to/MTRAG/LICENSE \\
  --conflicts-license /path/to/CONFLICTS/LICENSE \\
  --enterprise-license /path/to/EnterpriseRAG-Bench/LICENSE
PYTHONPATH=src:. .venv/bin/python tools/build_public_rag_benchmark_suite.py validate
```

`manifest.json` stores the source input checksums, exact selection quotas, and
artifact checksums.  A rebuilt result must validate before it can be used for
threshold or answer-policy tuning.

## Explicit media correction

The frozen source text originally contained four remote Markdown image references in three MTRAG
documents (`1208`, `1357`, and `1368`). They illustrated UI already described completely by the
adjacent text and supplied no evidence span required by the two cases that cite these documents.
The references were removed explicitly on 2026-09-01 so corpus ingestion never depends on network
access. Logical document IDs and upstream source hashes remain stable; artifact hashes bind the
corrected local text. See `docs/reviews/12-0901-markdown-media-review.md` for the case-level evidence.
"""


def build(inputs: SourceInputs, output: Path) -> None:
    if output.exists():
        raise CorpusError(f"output already exists: {output}")
    crag_rows = _read_crag(inputs.crag)
    mtrag_rows = _read_jsonl(inputs.mtrag)
    conflict_rows = _read_jsonl(inputs.conflicts)
    enterprise_rows = _read_jsonl(inputs.enterprise_questions)
    uuid_index = json.loads(inputs.enterprise_uuid_index.read_text(encoding="utf-8"))
    if not isinstance(uuid_index, Mapping):
        raise CorpusError("enterprise UUID index must be an object")

    documents: dict[str, dict[str, Any]] = {}
    cases = [
        *_add_crag_cases(crag_rows, documents),
        *_add_mtrag_cases(mtrag_rows, documents),
        *_add_conflict_cases(conflict_rows, documents),
        *_add_enterprise_cases(enterprise_rows, uuid_index, inputs.enterprise_archive, documents),
    ]
    output.mkdir(parents=True)
    document_rows = _write_documents(output, documents)
    _write_jsonl(output / "cases.jsonl", sorted(cases, key=lambda item: item["case_id"]))
    _write_jsonl(output / "documents.jsonl", document_rows)
    license_dir = output / "licenses"
    license_dir.mkdir()
    for name, path in (
        ("CRAG-CC-BY-NC-4.0.txt", inputs.crag_license),
        ("MTRAG-APACHE-2.0.txt", inputs.mtrag_license),
        ("CONFLICTS-APACHE-2.0.txt", inputs.conflicts_license),
        ("ENTERPRISE-RAG-BENCH-MIT.txt", inputs.enterprise_license),
    ):
        (license_dir / name).write_text(
            _normalize_text(path.read_text(encoding="utf-8")) + "\n",
            encoding="utf-8",
        )
    (output / "SOURCE_NOTICES.md").write_text(_source_notices(), encoding="utf-8")
    (output / "README.md").write_text(_readme(), encoding="utf-8")
    counts = Counter(str(case["stratum"]) for case in cases)
    manifest = {
        "schema": SCHEMA,
        "dataset_id": DATASET_ID,
        "language": "en",
        "case_count": len(cases),
        "document_count": len(document_rows),
        "case_counts": dict(sorted(counts.items())),
        "selection": {
            "crag_false_premise": CRAG_FALSE_PREMISE_COUNT,
            "mtrag_answerability": MTRAG_QUOTAS,
            "conflicts": CONFLICT_QUOTAS,
            "enterprise": ENTERPRISE_QUOTAS,
        },
        "source_inputs": {
            "crag": {"path": inputs.crag.name, "sha256": _sha256_path(inputs.crag)},
            "mtrag": {"path": inputs.mtrag.name, "sha256": _sha256_path(inputs.mtrag)},
            "conflicts": {"path": inputs.conflicts.name, "sha256": _sha256_path(inputs.conflicts)},
            "enterprise_questions": {
                "path": inputs.enterprise_questions.name,
                "sha256": _sha256_path(inputs.enterprise_questions),
            },
            "enterprise_uuid_index": {
                "path": inputs.enterprise_uuid_index.name,
                "sha256": _sha256_path(inputs.enterprise_uuid_index),
            },
            "enterprise_archive": {
                "path": inputs.enterprise_archive.name,
                "sha256": _sha256_path(inputs.enterprise_archive),
            },
        },
        "artifacts": {},
    }
    manifest["artifacts"] = {
        name: _sha256_path(output / name)
        for name in ("README.md", "SOURCE_NOTICES.md", "cases.jsonl", "documents.jsonl")
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate(output)


def validate(output: Path) -> dict[str, int]:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise CorpusError("manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("dataset_id") != DATASET_ID:
        raise CorpusError("manifest identity is invalid")
    cases = _read_jsonl(output / "cases.jsonl")
    documents = _read_jsonl(output / "documents.jsonl")
    if len(cases) != manifest.get("case_count") or len(documents) != manifest.get("document_count"):
        raise CorpusError("manifest counts are invalid")
    document_ids = {row.get("document_id") for row in documents}
    if None in document_ids or len(document_ids) != len(documents):
        raise CorpusError("document identifiers are invalid")
    for row in documents:
        filename = row.get("filename")
        if not isinstance(filename, str) or not (output / "documents" / filename).is_file():
            raise CorpusError(f"document artifact missing: {row.get('document_id')}")
        if _sha256_path(output / "documents" / filename) != row.get("artifact_sha256"):
            raise CorpusError(f"document artifact digest invalid: {row.get('document_id')}")
    case_ids: set[str] = set()
    expected_counts = {
        "open_world_false_premise": CRAG_FALSE_PREMISE_COUNT,
        "unanswerable": MTRAG_QUOTAS["UNANSWERABLE"],
        "underspecified": MTRAG_QUOTAS["UNDERSPECIFIED"],
        "partial": MTRAG_QUOTAS["PARTIAL"],
        "answerable": MTRAG_QUOTAS["ANSWERABLE"],
        **{f"evidence_{_slug(key)}": value for key, value in CONFLICT_QUOTAS.items()},
        **ENTERPRISE_QUOTAS,
    }
    counts: Counter[str] = Counter()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in case_ids:
            raise CorpusError("case identifiers are invalid")
        case_ids.add(case_id)
        stratum = case.get("stratum")
        action = case.get("expected_action")
        if not isinstance(stratum, str) or not isinstance(action, str):
            raise CorpusError(f"case labels invalid: {case_id}")
        counts[stratum] += 1
        evidence = case.get("evidence_document_ids")
        if not isinstance(evidence, list) or any(item not in document_ids for item in evidence):
            raise CorpusError(f"case evidence invalid: {case_id}")
        if stratum in {"info_not_found", "unanswerable"} and evidence:
            raise CorpusError("absence-control case unexpectedly has evidence")
        if stratum not in {"info_not_found", "unanswerable"} and not evidence:
            raise CorpusError(f"case evidence is empty: {case_id}")
    if dict(counts) != expected_counts:
        raise CorpusError(f"stratum counts invalid: {dict(counts)}")
    for artifact, expected in manifest.get("artifacts", {}).items():
        if _sha256_path(output / artifact) != expected:
            raise CorpusError(f"artifact digest invalid: {artifact}")
    return {"case_count": len(cases), "document_count": len(documents)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--crag", type=Path, required=True)
    build_parser.add_argument("--mtrag", type=Path, required=True)
    build_parser.add_argument("--conflicts", type=Path, required=True)
    build_parser.add_argument("--enterprise-questions", type=Path, required=True)
    build_parser.add_argument("--enterprise-uuid-index", type=Path, required=True)
    build_parser.add_argument("--enterprise-archive", type=Path, required=True)
    build_parser.add_argument("--crag-license", type=Path, required=True)
    build_parser.add_argument("--mtrag-license", type=Path, required=True)
    build_parser.add_argument("--conflicts-license", type=Path, required=True)
    build_parser.add_argument("--enterprise-license", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "validate":
        print(json.dumps(validate(arguments.output), sort_keys=True))
        return 0
    inputs = SourceInputs(
        crag=arguments.crag,
        mtrag=arguments.mtrag,
        conflicts=arguments.conflicts,
        enterprise_questions=arguments.enterprise_questions,
        enterprise_uuid_index=arguments.enterprise_uuid_index,
        enterprise_archive=arguments.enterprise_archive,
        crag_license=arguments.crag_license,
        mtrag_license=arguments.mtrag_license,
        conflicts_license=arguments.conflicts_license,
        enterprise_license=arguments.enterprise_license,
    )
    build(inputs, arguments.output)
    print(f"built {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
