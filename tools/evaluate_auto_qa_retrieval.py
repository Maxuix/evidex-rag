#!/usr/bin/env python3
"""Evaluate frozen Auto-QA off/on semantic and lexical retrieval."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from statistics import median
import time
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID

from apps.api.dependencies import build_api_dependencies
from rag_kb.config import load_settings
from rag_kb.domain import RerankMode, RetrievalStrategy
from rag_kb.retrieval import RetrievalRequest
from tools.build_document_qa_corpus import COMPLEX_CASES_FILENAME, validate_corpus


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "evaluation/document-qa-v1"
DEFAULT_STATE = ROOT / ".runtime/evaluations/auto-qa-ab-20260904/state.json"
DEFAULT_OUTPUT = ROOT / ".runtime/evaluations/auto-qa-ab-20260904/retrieval.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object expected: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL object expected: {path}")
        rows.append(value)
    return rows


def _evaluation_cases(root: Path) -> list[dict[str, Any]]:
    base = _load_jsonl(root / "cases.jsonl")
    by_id = {str(item["case_id"]): item for item in base}
    cases = [
        {
            **item,
            "evaluation_case_id": str(item["case_id"]),
            "group": "direct" if item["gold"]["answerable"] else "unanswerable",
        }
        for item in base
    ]
    for item in _load_jsonl(root / "auto-qa-paraphrases.jsonl"):
        base_case_id = str(item["base_case_id"])
        if base_case_id not in by_id or not by_id[base_case_id]["gold"]["answerable"]:
            raise RuntimeError(f"invalid paraphrase base case: {base_case_id}")
        cases.append(
            {
                **by_id[base_case_id],
                "evaluation_case_id": str(item["case_id"]),
                "base_case_id": base_case_id,
                "question": str(item["question"]),
                "variant": str(item["variant"]),
                "group": "paraphrase",
            }
        )
    for item in _load_jsonl(root / COMPLEX_CASES_FILENAME):
        cases.append(
            {
                **item,
                "evaluation_case_id": str(item["case_id"]),
                "group": "complex",
            }
        )
    ids = [str(item["evaluation_case_id"]) for item in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("evaluation case identifiers are not unique")
    return cases


def _compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _filename(evidence: Any) -> str:
    metadata = evidence.source_metadata or {}
    return str(metadata.get("original_filename") or "")


def _document_filename(case: Mapping[str, Any]) -> str:
    return Path(str(case["document_path"])).name


def _page_hit(source_location: Mapping[str, Any], pages: Iterable[int]) -> bool:
    start = source_location.get("surface_start")
    end = source_location.get("surface_end")
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    return any(start <= int(page) <= end for page in pages)


def _text_overlap(gold: str, actual: str) -> float:
    gold_words = set(re.findall(r"\w+", _compact(gold)))
    actual_words = set(re.findall(r"\w+", _compact(actual)))
    if not gold_words:
        return 0.0
    return len(gold_words & actual_words) / len(gold_words)


def evidence_matches(case: Mapping[str, Any], evidence: Any) -> bool:
    if case.get("group") == "complex":
        return False
    if _filename(evidence) != _document_filename(case):
        return False
    locator = case["evidence"]
    kind = locator["kind"]
    if kind == "absence":
        return False
    if kind == "pdf_page_alternatives":
        return any(
            _page_hit(evidence.source_location, alternative["pages"])
            for alternative in locator["alternatives"]
        )
    if kind == "pdf_pages":
        return _page_hit(
            evidence.source_location,
            (item["page"] for item in locator["items"]),
        )
    if kind == "text_spans":
        return any(
            _compact(span["quote"]) in _compact(evidence.text)
            or _text_overlap(str(span["quote"]), evidence.text) >= 0.6
            for span in locator["spans"]
        )
    if kind == "markdown_sections":
        titles = " ".join(
            str(item.get("text") or "")
            for item in (evidence.hierarchy or {}).get("titles", ())
            if isinstance(item, Mapping)
        )
        haystack = _compact(f"{titles} {evidence.text[:300]}")
        for section in locator["sections"]:
            normalized = _compact(str(section).replace("-", " "))
            if normalized in haystack:
                return True
            if normalized == "table" and (
                evidence.modality == "table" or "|" in evidence.text
            ):
                return True
        return False
    raise RuntimeError(f"unsupported evidence locator: {kind}")


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return round(ordered[index], 3)


def _matched_question(pack: Any, chunk_id: object) -> str | None:
    if pack.debug is None:
        return None
    for item in pack.debug.matched_questions:
        if item.index_chunk_id == chunk_id:
            return item.question
    return None


def _case_result(case: Mapping[str, Any], pack: Any, elapsed_ms: float) -> dict[str, Any]:
    evidence = list(pack.evidence)
    if case["group"] == "complex":
        required = set(case["required_document_ids"])
        manifest_names = {
            "financebench-amd-2022-10k": "AMD_2022_10K.pdf",
            "financebench-american-express-2022-10k": "AMERICANEXPRESS_2022_10K.pdf",
            "financebench-boeing-2022-10k": "BOEING_2022_10K.pdf",
            "cfqa-fenghuo-electronics-2022-annual-report": "fenghuo-electronics-2022-annual-report.pdf",
            "tatqa-loan-to-value": "tatqa-loan-to-value.md",
            "tatqa-net-sales-by-end-market": "tatqa-net-sales-by-end-market.md",
            "tatqa-other-operating-expenses": "tatqa-other-operating-expenses.md",
            "tatqa-sales-by-contract-type": "tatqa-sales-by-contract-type.md",
            "contractnli-pdf-15": "contractnli-pdf-15.txt",
            "contractnli-pdf-82": "contractnli-pdf-82.txt",
            "contractnli-sec-html-547": "contractnli-sec-html-547.txt",
            "contractnli-sec-text-488": "contractnli-sec-text-488.txt",
        }
        required_names = {manifest_names[item] for item in required}
        found_names = {_filename(item) for item in evidence}
        return {
            "case_id": case["evaluation_case_id"],
            "group": case["group"],
            "elapsed_ms": round(elapsed_ms, 3),
            "required_document_count": len(required_names),
            "required_documents_found": sorted(required_names & found_names),
            "required_document_recall_10": round(
                len(required_names & found_names) / len(required_names), 6
            ),
            "top_filenames": [_filename(item) for item in evidence],
        }
    ranks = [item.rank for item in evidence if evidence_matches(case, item)]
    relevant_rank = min(ranks) if ranks else None
    top = evidence[0] if evidence else None
    relevant = next((item for item in evidence if item.rank == relevant_rank), None)
    return {
        "case_id": case["evaluation_case_id"],
        "base_case_id": case.get("base_case_id"),
        "group": case["group"],
        "variant": case.get("variant"),
        "elapsed_ms": round(elapsed_ms, 3),
        "relevant_rank": relevant_rank,
        "hit_1": relevant_rank == 1,
        "recall_5": relevant_rank is not None and relevant_rank <= 5,
        "recall_10": relevant_rank is not None and relevant_rank <= 10,
        "reciprocal_rank_10": round(1.0 / relevant_rank, 6) if relevant_rank else 0.0,
        "target_document_top_1": bool(top and _filename(top) == _document_filename(case)),
        "target_document_recall_10": any(
            _filename(item) == _document_filename(case) for item in evidence
        ),
        "relevant_representations": (
            list(relevant.matched_representations) if relevant is not None else []
        ),
        "matched_question": (
            _matched_question(pack, relevant.index_chunk_id) if relevant is not None else None
        ),
        "top": (
            {
                "filename": _filename(top),
                "index_chunk_id": str(top.index_chunk_id),
                "ordinal": top.ordinal,
                "source_location": top.source_location,
                "matched_representations": list(top.matched_representations),
                "matched_question": _matched_question(pack, top.index_chunk_id),
            }
            if top is not None
            else None
        ),
    }


def summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latency = [float(item["elapsed_ms"]) for item in results]
    summary: dict[str, Any] = {
        "count": len(results),
        "latency_ms": {
            "p50": round(median(latency), 3) if latency else None,
            "p95": _percentile(latency, 0.95),
        },
    }
    ordinary = [item for item in results if item["group"] not in {"complex", "unanswerable"}]
    if ordinary:
        summary["retrieval"] = _retrieval_summary(ordinary)
        summary["retrieval_by_group"] = {
            group: _retrieval_summary(
                [item for item in ordinary if item["group"] == group]
            )
            for group in ("direct", "paraphrase")
            if any(item["group"] == group for item in ordinary)
        }
    unanswerable = [item for item in results if item["group"] == "unanswerable"]
    if unanswerable:
        summary["unanswerable"] = {
            "count": len(unanswerable),
            "target_document_top_1": sum(
                bool(item["target_document_top_1"]) for item in unanswerable
            ),
            "target_document_recall_10": sum(
                bool(item["target_document_recall_10"]) for item in unanswerable
            ),
        }
    complex_results = [item for item in results if item["group"] == "complex"]
    if complex_results:
        summary["complex"] = {
            "count": len(complex_results),
            "mean_required_document_recall_10": round(
                sum(float(item["required_document_recall_10"]) for item in complex_results)
                / len(complex_results),
                6,
            ),
            "complete_required_documents": sum(
                float(item["required_document_recall_10"]) == 1.0
                for item in complex_results
            ),
        }
    return summary


def _retrieval_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(results),
        "hit_1": sum(bool(item["hit_1"]) for item in results),
        "hit_at_1": round(
            sum(bool(item["hit_1"]) for item in results) / len(results), 6
        ),
        "mrr_at_10": round(
            sum(float(item["reciprocal_rank_10"]) for item in results)
            / len(results),
            6,
        ),
        "recall_at_5": round(
            sum(bool(item["recall_5"]) for item in results) / len(results), 6
        ),
        "recall_at_10": round(
            sum(bool(item["recall_10"]) for item in results) / len(results), 6
        ),
        "relevant_representation_counts": dict(
            Counter(
                representation
                for item in results
                for representation in item["relevant_representations"]
            )
        ),
        "matched_question_count": sum(
            bool(item["matched_question"]) for item in results
        ),
    }


async def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    corpus = validate_corpus(arguments.corpus_root)
    state = _load_json(arguments.state)
    if state.get("status") != "indexed":
        raise RuntimeError("Auto-QA A/B indexing checkpoint is not complete")
    cases = _evaluation_cases(arguments.corpus_root)
    settings = load_settings(env_file=arguments.env_file)
    dependencies = build_api_dependencies(settings=settings)
    await dependencies.start()
    try:
        arms: dict[str, Any] = {}
        for arm in ("off", "on"):
            kb_id = UUID(str(state["arms"][arm]["knowledge_base_id"]))
            lanes: dict[str, Any] = {}
            for lane in ("semantic", "keyword"):
                results = []
                for position, case in enumerate(cases, start=1):
                    request = RetrievalRequest(
                        knowledge_base_id=kb_id,
                        query=str(case["question"]),
                        top_k=arguments.top_k,
                        strategy=RetrievalStrategy.EXACT_VECTOR,
                        rerank_mode=RerankMode.CLASSIC,
                        include_debug=True,
                    )
                    started = time.perf_counter()
                    pack = (
                        await dependencies.retrieval_service.retrieve(request)
                        if lane == "semantic"
                        else await dependencies.retrieval_service.retrieve_lexical_only(request)
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    results.append(_case_result(case, pack, elapsed_ms))
                    print(
                        json.dumps(
                            {
                                "event": "retrieval_case",
                                "arm": arm,
                                "lane": lane,
                                "position": position,
                                "count": len(cases),
                                "case_id": case["evaluation_case_id"],
                            },
                            sort_keys=True,
                        )
                    )
                lanes[lane] = {"summary": summarize(results), "cases": results}
            arms[arm] = lanes
    finally:
        await dependencies.close()
    return {
        "schema_version": "auto_qa_retrieval_ab_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_sha256": corpus["dataset_sha256"],
        "state": str(arguments.state),
        "top_k": arguments.top_k,
        "rerank_mode": "classic",
        "case_counts": dict(Counter(str(item["group"]) for item in cases)),
        "arms": arms,
    }


def main() -> int:
    arguments = _parser().parse_args()
    result = asyncio.run(_run(arguments))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(arguments.output)
    print(json.dumps({"status": "complete", "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
