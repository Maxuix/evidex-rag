#!/usr/bin/env python3
"""Build and validate the bounded public document-QA evaluation corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable
from urllib.request import Request, urlopen
import zipfile

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "document-qa-v1"
CORPUS_SCHEMA = "document_qa_corpus_v1"
CASE_SCHEMA = "document_qa_case_v1"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    filename: str
    url: str
    sha256: str
    dataset: str
    revision: str | None


FINANCEBENCH_REVISION = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
TATQA_REVISION = "870accc41953dcde885aabeb963d94aabdc0fbc3"
CONTRACTNLI_REVISION = "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9"
CFQA_REVISION = "61c9ec3c4335d0411a1735cd228af8b3ead114fc"


def _github_raw(repository: str, revision: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repository}/{revision}/{path}"


SOURCE_SPECS = (
    SourceSpec(
        key="financebench_cases",
        filename="financebench_open_source.jsonl",
        url=_github_raw(
            "patronus-ai/financebench",
            FINANCEBENCH_REVISION,
            "data/financebench_open_source.jsonl",
        ),
        sha256="a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877",
        dataset="FinanceBench",
        revision=FINANCEBENCH_REVISION,
    ),
    SourceSpec(
        key="financebench_documents",
        filename="financebench_document_information.jsonl",
        url=_github_raw(
            "patronus-ai/financebench",
            FINANCEBENCH_REVISION,
            "data/financebench_document_information.jsonl",
        ),
        sha256="1c69127783879de8cdadb159d2181f39bc3123b8e0ebf74031c3969d69189575",
        dataset="FinanceBench",
        revision=FINANCEBENCH_REVISION,
    ),
    SourceSpec(
        key="financebench_amd_pdf",
        filename="AMD_2022_10K.pdf",
        url=_github_raw(
            "patronus-ai/financebench",
            FINANCEBENCH_REVISION,
            "pdfs/AMD_2022_10K.pdf",
        ),
        sha256="a3bd74088fae0ad4aa03d04998a1f4c64fd0b3c693841601b0ac49b46cf5c1f4",
        dataset="FinanceBench",
        revision=FINANCEBENCH_REVISION,
    ),
    SourceSpec(
        key="financebench_amex_pdf",
        filename="AMERICANEXPRESS_2022_10K.pdf",
        url=_github_raw(
            "patronus-ai/financebench",
            FINANCEBENCH_REVISION,
            "pdfs/AMERICANEXPRESS_2022_10K.pdf",
        ),
        sha256="d3bbc7ab23d6160e07eab50a359e4d68c40efff9fde3baa168e7c40e06f568e6",
        dataset="FinanceBench",
        revision=FINANCEBENCH_REVISION,
    ),
    SourceSpec(
        key="financebench_boeing_pdf",
        filename="BOEING_2022_10K.pdf",
        url=_github_raw(
            "patronus-ai/financebench",
            FINANCEBENCH_REVISION,
            "pdfs/BOEING_2022_10K.pdf",
        ),
        sha256="09285ff7ee737d3302977104aa05cc53c9dfb0161f3461a89b583dc38b9f2ab6",
        dataset="FinanceBench",
        revision=FINANCEBENCH_REVISION,
    ),
    SourceSpec(
        key="tatqa_dev",
        filename="tatqa_dataset_dev.json",
        url=_github_raw(
            "NExTplusplus/TAT-QA",
            TATQA_REVISION,
            "dataset_raw/tatqa_dataset_dev.json",
        ),
        sha256="8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5",
        dataset="TAT-QA",
        revision=TATQA_REVISION,
    ),
    SourceSpec(
        key="contractnli_zip",
        filename="contract-nli.zip",
        url=_github_raw(
            "stanfordnlp/contract-nli",
            CONTRACTNLI_REVISION,
            "resources/contract-nli.zip",
        ),
        sha256="e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b",
        dataset="ContractNLI",
        revision=CONTRACTNLI_REVISION,
    ),
    SourceSpec(
        key="cfqa_cases",
        filename="cfqa_split_by_company_test.json",
        url=_github_raw(
            "ygan/CFQA",
            CFQA_REVISION,
            "dataset/split_by_company/split_by_company_test.json",
        ),
        sha256="e4c08332a1f6aada430ac94fe62106fd557c3d80cd68c495118412bb7704bc7d",
        dataset="CFQA",
        revision=CFQA_REVISION,
    ),
    SourceSpec(
        key="cfqa_fenghuo_pdf",
        filename="fenghuo-electronics-2022-annual-report.pdf",
        url="https://static.cninfo.com.cn/finalpage/2023-04-12/1216382408.PDF",
        sha256="be9bef33ee8d5df5cd74c14bd01c089a5a583cd2b2a436ce1edf2754623cfdec",
        dataset="CFQA / CNINFO",
        revision=None,
    ),
)

SOURCE_BY_KEY = {source.key: source for source in SOURCE_SPECS}

FINANCE_DOCUMENTS = (
    (
        "AMD_2022_10K",
        "financebench-amd-2022-10k",
        "financebench_amd_pdf",
    ),
    (
        "AMERICANEXPRESS_2022_10K",
        "financebench-american-express-2022-10k",
        "financebench_amex_pdf",
    ),
    (
        "BOEING_2022_10K",
        "financebench-boeing-2022-10k",
        "financebench_boeing_pdf",
    ),
)

TATQA_DOCUMENTS = (
    (
        "3ffd9053-a45d-491c-957a-1b2fa0af0570",
        "tatqa-sales-by-contract-type",
        "Sales by Contract Type",
    ),
    (
        "53474060-2736-46cb-bd97-1eb42f0ff3c1",
        "tatqa-net-sales-by-end-market",
        "Net Sales by Segment and Industry End Market",
    ),
    (
        "285a1ced-709e-4f45-a227-b6cd04e725f9",
        "tatqa-other-operating-expenses",
        "Other Operating Expenses",
    ),
    (
        "ba26cd64-e448-4ffb-bfaa-c6ad4760fba7",
        "tatqa-loan-to-value",
        "Loan-to-Value Ratio",
    ),
)

CONTRACT_DOCUMENTS = (
    (
        488,
        "contractnli-sec-text-488",
        ("nda-11", "nda-2", "nda-16"),
    ),
    (
        15,
        "contractnli-pdf-15",
        ("nda-15", "nda-17", "nda-11"),
    ),
    (
        547,
        "contractnli-sec-html-547",
        ("nda-10", "nda-2", "nda-18"),
    ),
    (
        82,
        "contractnli-pdf-82",
        ("nda-19", "nda-20", "nda-8"),
    ),
)

CFQA_CASE_IDS = frozenset({73, 77, 81, 85, 89, 93, 97, 101})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="download sources and build corpus")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument(
        "--cache",
        type=Path,
        help="source cache (default: <output>/.source-cache)",
    )
    build.add_argument(
        "--offline",
        action="store_true",
        help="fail instead of downloading a missing or invalid cached source",
    )
    build.add_argument(
        "--force",
        action="store_true",
        help="replace only the generated documents, cases, and manifest",
    )

    validate = subparsers.add_parser("validate", help="validate built corpus")
    validate.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)

    arguments = parser.parse_args()
    if arguments.command == "build":
        output = arguments.output.resolve()
        cache = (arguments.cache or output / ".source-cache").resolve()
        report = build_corpus(
            output,
            cache=cache,
            offline=arguments.offline,
            force=arguments.force,
        )
    else:
        report = validate_corpus(arguments.root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_corpus(
    output: Path,
    *,
    cache: Path,
    offline: bool,
    force: bool,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    sources = {
        spec.key: _ensure_source(cache, spec, offline=offline)
        for spec in SOURCE_SPECS
    }
    generated_targets = (
        output / "documents",
        output / "cases.jsonl",
        output / "manifest.json",
    )
    existing = [path for path in generated_targets if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise RuntimeError(f"generated corpus already exists: {joined}; use --force")

    with tempfile.TemporaryDirectory(
        prefix="document-qa-v1-",
        dir=output.parent,
    ) as directory:
        staging = Path(directory)
        documents: list[dict[str, object]] = []
        cases: list[dict[str, object]] = []
        _build_financebench(staging, sources, documents, cases)
        _build_cfqa(staging, sources, documents, cases)
        _build_tatqa(staging, sources, documents, cases)
        _build_contractnli(staging, sources, documents, cases)
        cases.sort(key=lambda case: str(case["case_id"]))
        _write_jsonl(staging / "cases.jsonl", cases)
        manifest = _manifest(staging, sources, documents, cases)
        _write_json(staging / "manifest.json", manifest)
        validate_corpus(staging)

        for target in existing:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(staging / "documents"), output / "documents")
        shutil.move(str(staging / "cases.jsonl"), output / "cases.jsonl")
        shutil.move(str(staging / "manifest.json"), output / "manifest.json")

    return validate_corpus(output)


def _ensure_source(cache: Path, spec: SourceSpec, *, offline: bool) -> Path:
    path = cache / spec.filename
    if path.exists() and _sha256(path) == spec.sha256:
        return path
    if offline:
        raise RuntimeError(f"offline source is missing or invalid: {path}")
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = Request(
        spec.url,
        headers={"User-Agent": "rag-kb-document-qa-corpus/1.0"},
    )
    with urlopen(request, timeout=120) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target)
    actual = _sha256(temporary)
    if actual != spec.sha256:
        temporary.unlink()
        raise RuntimeError(
            f"source checksum mismatch for {spec.key}: "
            f"expected {spec.sha256}, got {actual}"
        )
    temporary.replace(path)
    return path


def _build_financebench(
    root: Path,
    sources: dict[str, Path],
    documents: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> None:
    rows = _read_jsonl(sources["financebench_cases"])
    metadata = {
        row["doc_name"]: row
        for row in _read_jsonl(sources["financebench_documents"])
    }
    for source_name, document_id, source_key in FINANCE_DOCUMENTS:
        relative = Path("documents") / "pdf" / f"{document_id}.pdf"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[source_key], target)
        source_spec = SOURCE_BY_KEY[source_key]
        document_metadata = metadata[source_name]
        documents.append(
            _document_record(
                document_id,
                relative,
                target,
                source_dataset="FinanceBench",
                language="en",
                source_url=source_spec.url,
                source_revision=source_spec.revision,
                original_url=document_metadata.get("doc_link"),
                redistribution="local-only; review original publisher terms",
            )
        )
        selected = [row for row in rows if row["doc_name"] == source_name]
        if len(selected) != 7:
            raise RuntimeError(
                f"expected seven FinanceBench cases for {source_name}, "
                f"found {len(selected)}"
            )
        for row in selected:
            evidence_items = []
            for evidence in row["evidence"]:
                evidence_items.append(
                    {
                        "page": int(evidence["evidence_page_num"]) + 1,
                        "source_page_index": int(evidence["evidence_page_num"]),
                        "quote": _clean_text(evidence["evidence_text"]),
                    }
                )
            cases.append(
                {
                    "schema_version": CASE_SCHEMA,
                    "case_id": row["financebench_id"],
                    "document_id": document_id,
                    "document_path": relative.as_posix(),
                    "source_dataset": "FinanceBench",
                    "source_case_id": row["financebench_id"],
                    "language": "en",
                    "question": row["question"],
                    "gold": {
                        "answer": row["answer"],
                        "acceptable_answers": [row["answer"]],
                        "answerable": True,
                        "scale": None,
                        "numeric_tolerance": None,
                    },
                    "question_type": _normal_key(row.get("question_reasoning")),
                    "evidence": {
                        "kind": "pdf_pages",
                        "page_numbering": "pdf_1_based",
                        "items": evidence_items,
                    },
                    "justification": row.get("justification"),
                }
            )


def _build_cfqa(
    root: Path,
    sources: dict[str, Path],
    documents: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> None:
    document_id = "cfqa-fenghuo-electronics-2022-annual-report"
    relative = Path("documents") / "pdf" / f"{document_id}.pdf"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sources["cfqa_fenghuo_pdf"], target)
    source_spec = SOURCE_BY_KEY["cfqa_fenghuo_pdf"]
    documents.append(
        _document_record(
            document_id,
            relative,
            target,
            source_dataset="CFQA",
            language="zh",
            source_url=source_spec.url,
            source_revision=CFQA_REVISION,
            original_url=source_spec.url,
            redistribution="local-only; public filing remains under publisher terms",
        )
    )
    rows = json.loads(sources["cfqa_cases"].read_text(encoding="utf-8"))
    selected = [row for row in rows if int(row["id"]) in CFQA_CASE_IDS]
    if {int(row["id"]) for row in selected} != CFQA_CASE_IDS:
        raise RuntimeError("CFQA source did not contain the pinned case IDs")
    for row in selected:
        if row["公司"] != "烽火电子" or "2022" not in row["问题"]:
            raise RuntimeError(f"unexpected CFQA case content: {row['id']}")
        cases.append(
            {
                "schema_version": CASE_SCHEMA,
                "case_id": f"cfqa-{row['id']}",
                "document_id": document_id,
                "document_path": relative.as_posix(),
                "source_dataset": "CFQA",
                "source_case_id": row["id"],
                "language": "zh",
                "question": row["问题"],
                "gold": {
                    "answer": row["答案"],
                    "acceptable_answers": _cfqa_acceptable_answers(row["答案"]),
                    "answerable": True,
                    "scale": None,
                    "numeric_tolerance": None,
                },
                "question_type": "financial_report_qa",
                "evidence": {
                    "kind": "pdf_page_alternatives",
                    "page_numbering": "pdf_1_based",
                    "alternatives": [
                        {"pages": [int(page) for page in page_group]}
                        for page_group in row["答案出自"]
                    ],
                },
                "justification": None,
            }
        )


def _build_tatqa(
    root: Path,
    sources: dict[str, Path],
    documents: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> None:
    rows = json.loads(sources["tatqa_dev"].read_text(encoding="utf-8"))
    by_uid = {row["table"]["uid"]: row for row in rows}
    for uid, document_id, title in TATQA_DOCUMENTS:
        row = by_uid.get(uid)
        if row is None:
            raise RuntimeError(f"TAT-QA source did not contain table {uid}")
        relative = Path("documents") / "markdown" / f"{document_id}.md"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_tatqa_markdown(title, uid, row), encoding="utf-8")
        documents.append(
            _document_record(
                document_id,
                relative,
                target,
                source_dataset="TAT-QA",
                language="en",
                source_url=SOURCE_BY_KEY["tatqa_dev"].url,
                source_revision=TATQA_REVISION,
                original_url=None,
                redistribution="CC BY 4.0 derivative",
            )
        )
        if len(row["questions"]) != 6:
            raise RuntimeError(f"expected six TAT-QA cases for {uid}")
        for question in row["questions"]:
            sections = [
                f"paragraph-{order}" for order in question["rel_paragraphs"]
            ]
            if question["answer_from"] in {"table", "table-text"}:
                sections.append("table")
            cases.append(
                {
                    "schema_version": CASE_SCHEMA,
                    "case_id": f"tatqa-{question['uid']}",
                    "document_id": document_id,
                    "document_path": relative.as_posix(),
                    "source_dataset": "TAT-QA",
                    "source_case_id": question["uid"],
                    "language": "en",
                    "question": question["question"],
                    "gold": {
                        "answer": question["answer"],
                        "acceptable_answers": _acceptable_answers(
                            question["answer"], question["scale"]
                        ),
                        "answerable": True,
                        "scale": question["scale"] or None,
                        "numeric_tolerance": (
                            {"absolute": 0.01}
                            if question["answer_type"] == "arithmetic"
                            else None
                        ),
                    },
                    "question_type": question["answer_type"],
                    "evidence": {
                        "kind": "markdown_sections",
                        "sections": list(dict.fromkeys(sections)),
                        "answer_from": question["answer_from"],
                    },
                    "justification": question["derivation"] or None,
                    "requires_comparison": bool(question["req_comparison"]),
                }
            )


def _build_contractnli(
    root: Path,
    sources: dict[str, Path],
    documents: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> None:
    with zipfile.ZipFile(sources["contractnli_zip"]) as archive:
        payload = json.loads(archive.read("contract-nli/dev.json"))
    by_id = {int(document["id"]): document for document in payload["documents"]}
    labels = payload["labels"]
    for source_id, document_id, label_ids in CONTRACT_DOCUMENTS:
        document = by_id.get(source_id)
        if document is None:
            raise RuntimeError(f"ContractNLI source did not contain document {source_id}")
        relative = Path("documents") / "text" / f"{document_id}.txt"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document["text"], encoding="utf-8", newline="\n")
        documents.append(
            _document_record(
                document_id,
                relative,
                target,
                source_dataset="ContractNLI",
                language="en",
                source_url=SOURCE_BY_KEY["contractnli_zip"].url,
                source_revision=CONTRACTNLI_REVISION,
                original_url=document.get("url"),
                redistribution="CC BY 4.0 derivative",
                extra={
                    "source_document_id": source_id,
                    "source_document_type": document["document_type"],
                    "source_filename": document["file_name"],
                },
            )
        )
        annotations = document["annotation_sets"][0]["annotations"]
        for label_id in label_ids:
            annotation = annotations[label_id]
            choice = annotation["choice"]
            spans = []
            for span_index in annotation["spans"]:
                start, end = document["spans"][span_index]
                quote = document["text"][start:end]
                spans.append({"start": start, "end": end, "quote": quote})
            canonical = {
                "Entailment": "entailment",
                "Contradiction": "contradiction",
                "NotMentioned": "not_mentioned",
            }[choice]
            aliases = {
                "entailment": ["entailment", "entailed", "supported"],
                "contradiction": ["contradiction", "contradicted"],
                "not_mentioned": ["not_mentioned", "not mentioned", "unknown"],
            }[canonical]
            hypothesis = labels[label_id]["hypothesis"]
            cases.append(
                {
                    "schema_version": CASE_SCHEMA,
                    "case_id": f"contractnli-{source_id}-{label_id}",
                    "document_id": document_id,
                    "document_path": relative.as_posix(),
                    "source_dataset": "ContractNLI",
                    "source_case_id": f"{source_id}:{label_id}",
                    "language": "en",
                    "question": (
                        "According to the agreement, is the following statement "
                        "entailed, contradicted, or not mentioned? "
                        f"{hypothesis}"
                    ),
                    "gold": {
                        "answer": canonical,
                        "acceptable_answers": aliases,
                        "answerable": canonical != "not_mentioned",
                        "scale": None,
                        "numeric_tolerance": None,
                    },
                    "question_type": "document_nli",
                    "evidence": {
                        "kind": "absence" if not spans else "text_spans",
                        "spans": spans,
                    },
                    "justification": labels[label_id]["short_description"],
                }
            )


def _tatqa_markdown(
    title: str,
    uid: str,
    row: dict[str, Any],
) -> str:
    lines = [
        f"# {title}",
        "",
        (
            "_Derived from a TAT-QA financial-report context under CC BY 4.0; "
            f"source table UID `{uid}`._"
        ),
        "",
        "## Narrative",
        "",
    ]
    for paragraph in sorted(row["paragraphs"], key=lambda item: item["order"]):
        order = paragraph["order"]
        lines.extend(
            (
                f'<a id="paragraph-{order}"></a>',
                f"### Paragraph {order}",
                "",
                paragraph["text"].strip(),
                "",
            )
        )
    table = row["table"]["table"]
    width = max(len(table_row) for table_row in table)
    lines.extend(
        (
            '<a id="table"></a>',
            "## Financial table",
            "",
            _markdown_row([f"Column {index + 1}" for index in range(width)]),
            _markdown_row(["---"] * width),
        )
    )
    for table_row in table:
        values = list(table_row) + [""] * (width - len(table_row))
        lines.append(_markdown_row(values))
    lines.append("")
    return "\n".join(lines)


def _markdown_row(values: Iterable[object]) -> str:
    escaped = [
        str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
        for value in values
    ]
    return "| " + " | ".join(escaped) + " |"


def _document_record(
    document_id: str,
    relative: Path,
    target: Path,
    *,
    source_dataset: str,
    language: str,
    source_url: str,
    source_revision: str | None,
    original_url: str | None,
    redistribution: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    suffix = target.suffix.lower().lstrip(".")
    record: dict[str, object] = {
        "document_id": document_id,
        "path": relative.as_posix(),
        "format": suffix,
        "language": language,
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
        "source_dataset": source_dataset,
        "source_url": source_url,
        "source_revision": source_revision,
        "original_url": original_url,
        "redistribution": redistribution,
    }
    if suffix == "pdf":
        record["pages"] = len(PdfReader(target).pages)
    if extra:
        record.update(extra)
    return record


def _manifest(
    root: Path,
    sources: dict[str, Path],
    documents: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> dict[str, object]:
    documents.sort(key=lambda document: str(document["document_id"]))
    cases_sha256 = _sha256(root / "cases.jsonl")
    dataset_material = {
        "cases_sha256": cases_sha256,
        "documents": [
            (document["document_id"], document["sha256"])
            for document in documents
        ],
    }
    dataset_sha256 = hashlib.sha256(
        json.dumps(
            dataset_material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": CORPUS_SCHEMA,
        "dataset_id": "public-enterprise-document-qa-v1",
        "dataset_sha256": dataset_sha256,
        "cases_path": "cases.jsonl",
        "cases_sha256": cases_sha256,
        "document_count": len(documents),
        "case_count": len(cases),
        "format_counts": dict(sorted(Counter(d["format"] for d in documents).items())),
        "language_counts": dict(
            sorted(Counter(case["language"] for case in cases).items())
        ),
        "source_case_counts": dict(
            sorted(Counter(case["source_dataset"] for case in cases).items())
        ),
        "documents": documents,
        "sources": [
            {
                "key": spec.key,
                "dataset": spec.dataset,
                "url": spec.url,
                "revision": spec.revision,
                "sha256": spec.sha256,
                "cached_sha256": _sha256(sources[spec.key]),
            }
            for spec in SOURCE_SPECS
        ],
    }


def validate_corpus(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    cases_path = root / "cases.jsonl"
    if not manifest_path.is_file() or not cases_path.is_file():
        raise RuntimeError(f"corpus is incomplete: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CORPUS_SCHEMA:
        raise RuntimeError("unsupported corpus schema")
    if _sha256(cases_path) != manifest.get("cases_sha256"):
        raise RuntimeError("cases checksum does not match manifest")
    cases = _read_jsonl(cases_path)
    if len(cases) != manifest.get("case_count"):
        raise RuntimeError("case count does not match manifest")

    documents = manifest.get("documents")
    if not isinstance(documents, list) or len(documents) != manifest.get(
        "document_count"
    ):
        raise RuntimeError("document count does not match manifest")
    by_id: dict[str, dict[str, object]] = {}
    for document in documents:
        document_id = _required_string(document, "document_id")
        if document_id in by_id:
            raise RuntimeError(f"duplicate document ID: {document_id}")
        path = _safe_document_path(root, _required_string(document, "path"))
        if not path.is_file():
            raise RuntimeError(f"document is missing: {path}")
        if _sha256(path) != document.get("sha256"):
            raise RuntimeError(f"document checksum mismatch: {path}")
        if path.stat().st_size != document.get("bytes"):
            raise RuntimeError(f"document size mismatch: {path}")
        expected_format = path.suffix.lower().lstrip(".")
        if document.get("format") != expected_format:
            raise RuntimeError(f"document format mismatch: {path}")
        if expected_format == "pdf":
            pages = len(PdfReader(path).pages)
            if pages != document.get("pages"):
                raise RuntimeError(f"PDF page count mismatch: {path}")
        by_id[document_id] = document

    case_ids: set[str] = set()
    referenced_documents: Counter[str] = Counter()
    for case in cases:
        if case.get("schema_version") != CASE_SCHEMA:
            raise RuntimeError("unsupported case schema")
        case_id = _required_string(case, "case_id")
        if case_id in case_ids:
            raise RuntimeError(f"duplicate case ID: {case_id}")
        case_ids.add(case_id)
        document_id = _required_string(case, "document_id")
        document = by_id.get(document_id)
        if document is None:
            raise RuntimeError(f"case references unknown document: {case_id}")
        if case.get("document_path") != document["path"]:
            raise RuntimeError(f"case document path mismatch: {case_id}")
        referenced_documents[document_id] += 1
        gold = case.get("gold")
        if not isinstance(gold, dict) or "answer" not in gold:
            raise RuntimeError(f"case has no gold answer: {case_id}")
        _validate_evidence(root, document, case)

    missing_cases = set(by_id) - set(referenced_documents)
    if missing_cases:
        raise RuntimeError(f"documents have no cases: {sorted(missing_cases)}")
    actual_formats = dict(sorted(Counter(d["format"] for d in documents).items()))
    if actual_formats != manifest.get("format_counts"):
        raise RuntimeError("format counts do not match manifest")
    dataset_material = {
        "cases_sha256": manifest["cases_sha256"],
        "documents": [
            (document["document_id"], document["sha256"])
            for document in documents
        ],
    }
    actual_dataset_hash = hashlib.sha256(
        json.dumps(
            dataset_material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if actual_dataset_hash != manifest.get("dataset_sha256"):
        raise RuntimeError("dataset checksum does not match manifest")
    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_sha256": manifest["dataset_sha256"],
        "documents": len(documents),
        "cases": len(cases),
        "formats": actual_formats,
        "languages": dict(sorted(Counter(c["language"] for c in cases).items())),
        "source_cases": dict(
            sorted(Counter(c["source_dataset"] for c in cases).items())
        ),
        "status": "valid",
    }


def _validate_evidence(
    root: Path,
    document: dict[str, object],
    case: dict[str, object],
) -> None:
    evidence = case.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError(f"case has invalid evidence: {case['case_id']}")
    kind = evidence.get("kind")
    path = _safe_document_path(root, str(document["path"]))
    if kind == "pdf_pages":
        pages = int(document["pages"])
        for item in evidence.get("items", []):
            _validate_page(item.get("page"), pages, case)
    elif kind == "pdf_page_alternatives":
        pages = int(document["pages"])
        alternatives = evidence.get("alternatives", [])
        if not alternatives:
            raise RuntimeError(f"case has no PDF evidence: {case['case_id']}")
        for alternative in alternatives:
            for page in alternative.get("pages", []):
                _validate_page(page, pages, case)
    elif kind == "markdown_sections":
        markdown = path.read_text(encoding="utf-8")
        for section in evidence.get("sections", []):
            if f'id="{section}"' not in markdown:
                raise RuntimeError(
                    f"missing Markdown evidence section {section}: {case['case_id']}"
                )
    elif kind == "text_spans":
        text = path.read_text(encoding="utf-8")
        spans = evidence.get("spans", [])
        if not spans:
            raise RuntimeError(f"case has no text spans: {case['case_id']}")
        for span in spans:
            start = int(span["start"])
            end = int(span["end"])
            if not 0 <= start < end <= len(text):
                raise RuntimeError(f"invalid evidence span: {case['case_id']}")
            if text[start:end] != span["quote"]:
                raise RuntimeError(f"evidence quote mismatch: {case['case_id']}")
    elif kind == "absence":
        if evidence.get("spans"):
            raise RuntimeError(f"absence case contains spans: {case['case_id']}")
        if case["gold"].get("answerable") is not False:
            raise RuntimeError(f"absence case is marked answerable: {case['case_id']}")
    else:
        raise RuntimeError(f"unknown evidence kind {kind}: {case['case_id']}")


def _validate_page(value: object, pages: int, case: dict[str, object]) -> None:
    if not isinstance(value, int) or not 1 <= value <= pages:
        raise RuntimeError(f"invalid PDF evidence page: {case['case_id']}")


def _safe_document_path(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise RuntimeError(f"unsafe document path: {relative}")
    path = (root / raw).resolve()
    documents_root = (root / "documents").resolve()
    if documents_root not in path.parents:
        raise RuntimeError(f"document path escapes corpus: {relative}")
    return path


def _acceptable_answers(answer: object, scale: str) -> list[str]:
    if isinstance(answer, list):
        values = [str(value) for value in answer]
        if len(values) == 1:
            return values
        return ["; ".join(values), ", ".join(values)]
    rendered = str(answer)
    answers = [rendered]
    if scale:
        answers.append(f"{rendered} {scale}")
    return answers


def _cfqa_acceptable_answers(answer: str) -> list[str]:
    values = [answer]
    without_source_marker = re.sub(r"\[\d+\]\s*$", "", answer).rstrip()
    if without_source_marker != answer:
        values.append(without_source_marker)
    without_layout_spaces = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", answer)
    if without_layout_spaces not in values:
        values.append(without_layout_spaces)
    return values


def _normal_key(value: object) -> str:
    if value is None or str(value).strip().lower() == "none":
        return "unspecified"
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value)
    )
    return "_".join(part for part in normalized.split("_") if part)


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _required_string(value: dict[str, object], key: str) -> str:
    candidate = value.get(key)
    if not isinstance(candidate, str) or not candidate:
        raise RuntimeError(f"missing required string: {key}")
    return candidate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for value in values:
            target.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            target.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
