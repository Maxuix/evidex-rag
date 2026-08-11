#!/usr/bin/env python3
"""Run and deterministically score the pinned Agent complex-QA benchmark."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import median
import time
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from tools.build_document_qa_corpus import (
    COMPLEX_CASES_FILENAME,
    validate_corpus,
)


DEFAULT_CORPUS_ROOT = Path(__file__).resolve().parents[1] / "evaluation" / "document-qa-v1"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / ".runtime" / "evaluations"
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
PREFLIGHT_CASE_IDS = ("complex-04", "complex-02", "complex-05")
_NUMBER_RE = re.compile(
    r"(?<![\w])(?P<value>\(?\s*[+-]?\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*\)?)(?P<percent>\s*%)?"
)
_NEGATIVE_WORDS = (
    "decreas",
    "drop",
    "declin",
    "reduc",
    "lower",
    "下降",
    "减少",
    "降低",
    "负",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--kb-id", required=True)
    parser.add_argument(
        "--strategy",
        choices=("exact_vector", "hybrid"),
        default="exact_vector",
    )
    parser.add_argument(
        "--rerank-mode",
        choices=("none", "classic"),
        default="classic",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="run the fixed complex-04, complex-02, complex-05 provider preflight",
    )
    parser.add_argument("--provider-revision-id")
    parser.add_argument("--profile-revision-id")
    parser.add_argument("--provider-timeout-seconds", type=float)
    parser.add_argument("--provider-max-retries", type=int)
    parser.add_argument("--worker-chat-deadline-seconds", type=float)
    parser.add_argument("--code-commit")
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="run only this case; repeat the option for a subset",
    )
    arguments = parser.parse_args()
    try:
        api = _validated_api_base(arguments.api)
        _validate_options(arguments)
        cases = load_complex_cases(arguments.corpus_root)
        document_identity_map = load_document_identity_map(arguments.corpus_root)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))

    requested = tuple(arguments.case_ids or ())
    if arguments.preflight:
        if requested:
            parser.error("--preflight cannot be combined with --case-id")
        requested = PREFLIGHT_CASE_IDS
    if requested:
        unknown = set(requested) - {str(case["case_id"]) for case in cases}
        if unknown:
            parser.error(f"unknown complex case: {sorted(unknown)}")
        by_id = {str(case["case_id"]): case for case in cases}
        cases = [by_id[case_id] for case_id in requested]

    corpus = validate_corpus(arguments.corpus_root)

    started = time.perf_counter()
    results = evaluate_cases(
        api,
        arguments.kb_id,
        cases,
        strategy=arguments.strategy,
        rerank_mode=arguments.rerank_mode,
        top_k=arguments.top_k,
        parallelism=arguments.parallelism,
        timeout_seconds=arguments.timeout_seconds,
        poll_seconds=arguments.poll_seconds,
        document_identity_map=document_identity_map,
    )
    report = {
        "schema_version": "agent_complex_qa_evaluation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "api": api,
            "knowledge_base_id": arguments.kb_id,
            "strategy": arguments.strategy,
            "rerank_mode": arguments.rerank_mode,
            "top_k": arguments.top_k,
            "parallelism": arguments.parallelism,
            "timeout_seconds": arguments.timeout_seconds,
            "poll_seconds": arguments.poll_seconds,
            "case_count": len(cases),
            "document_identity_mapping": "manifest_filename_v1",
            "corpus_sha256": corpus["dataset_sha256"],
            "provider_revision_id": arguments.provider_revision_id,
            "profile_revision_id": arguments.profile_revision_id,
            "provider_timeout_seconds": arguments.provider_timeout_seconds,
            "provider_max_retries": arguments.provider_max_retries,
            "worker_chat_deadline_seconds": arguments.worker_chat_deadline_seconds,
            "code_commit": arguments.code_commit,
            "preflight": arguments.preflight,
        },
        "cases": results,
        "summary": summarize_results(results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output = arguments.output or _default_output(arguments.strategy)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"]["completed"] == report["summary"]["cases"] else 1


def _validate_options(arguments: argparse.Namespace) -> None:
    if not 1 <= arguments.top_k <= 100:
        raise ValueError("--top-k must be between 1 and 100")
    if arguments.parallelism != 1:
        raise ValueError("--parallelism must be 1 for bounded provider evaluation")
    if arguments.timeout_seconds <= 0 or arguments.poll_seconds <= 0:
        raise ValueError("timeouts and polling interval must be positive")
    provider_timeout = getattr(arguments, "provider_timeout_seconds", None)
    provider_retries = getattr(arguments, "provider_max_retries", None)
    worker_deadline = getattr(arguments, "worker_chat_deadline_seconds", None)
    if provider_timeout is not None and provider_timeout <= 0:
        raise ValueError("provider timeout must be positive")
    if provider_retries is not None and provider_retries < 0:
        raise ValueError("provider retries must not be negative")
    if worker_deadline is not None and worker_deadline <= 0:
        raise ValueError("worker chat deadline must be positive")
    if arguments.strategy == "hybrid" and arguments.rerank_mode == "none":
        raise ValueError("hybrid retrieval requires classic reranking")


def load_complex_cases(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    validate_corpus(root)
    path = root / COMPLEX_CASES_FILENAME
    if not path.is_file():
        raise RuntimeError(f"complex case file is missing: {path}")
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != 14:
        raise RuntimeError("complex case file must contain exactly 14 cases")
    return values


def load_document_identity_map(root: Path) -> dict[str, str]:
    """Map generated corpus filenames to stable logical document IDs."""

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise RuntimeError("corpus manifest documents are missing")
    result: dict[str, str] = {}
    for document in documents:
        if not isinstance(document, dict):
            raise RuntimeError("corpus manifest document is invalid")
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise RuntimeError("corpus manifest document ID is invalid")
        path = document.get("path")
        aliases = [document_id, path]
        if isinstance(path, str) and path:
            suffix = Path(path).suffix
            if suffix:
                aliases.append(document_id + suffix)
        source_url = document.get("source_url")
        if isinstance(source_url, str) and source_url:
            aliases.append(urlparse(source_url).path)
        for alias in aliases:
            if isinstance(alias, str) and alias:
                result[_normalize_document_identity(alias)] = document_id
    return result


def evaluate_cases(
    api: str,
    kb_id: str,
    cases: list[dict[str, Any]],
    *,
    strategy: str,
    rerank_mode: str,
    top_k: int,
    parallelism: int,
    timeout_seconds: float,
    poll_seconds: float,
    document_identity_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    def run(case: dict[str, Any]) -> dict[str, Any]:
        return _evaluate_one(
            api,
            kb_id,
            case,
            strategy=strategy,
            rerank_mode=rerank_mode,
            top_k=top_k,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            document_identity_map=document_identity_map,
        )

    if parallelism != 1:
        raise ValueError("evaluation parallelism must be 1")
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        result = run(case)
        results.append(result)
        if _is_nonretryable_provider_403(result):
            results.extend(
                _not_run_result(
                    remaining,
                    reason="provider_nonretryable_403",
                    document_identity_map=document_identity_map,
                )
                for remaining in cases[index + 1 :]
            )
            break
    return results


def _not_run_result(
    case: Mapping[str, Any],
    *,
    reason: str,
    document_identity_map: Mapping[str, str],
) -> dict[str, Any]:
    safe = {
        "status": "not_run",
        "answer": None,
        "citations": [],
        "workflow": None,
        "retrieval": None,
        "error": None,
        "usage": None,
        "timing": None,
    }
    return {
        "case_id": str(case["case_id"]),
        "question": str(case["question"]),
        **safe,
        "research_result": None,
        "search_trace": None,
        "score": score_complex_case(
            dict(case),
            safe,
            document_identity_map=document_identity_map,
        ),
        "elapsed_seconds": 0.0,
        "not_run_reason": reason,
    }


def _is_nonretryable_provider_403(value: Mapping[str, Any]) -> bool:
    error = value.get("error")
    if (
        isinstance(error, Mapping)
        and error.get("http_status") == 403
        and error.get("retryable") is not True
    ):
        return True
    return any(
        isinstance(item.get("diagnostic"), Mapping)
        and item["diagnostic"].get("http_status") == 403
        and item["diagnostic"].get("retryable") is not True
        for item in _attempts(value)
    )


def _evaluate_one(
    api: str,
    kb_id: str,
    case: dict[str, Any],
    *,
    strategy: str,
    rerank_mode: str,
    top_k: int,
    timeout_seconds: float,
    poll_seconds: float,
    document_identity_map: Mapping[str, str],
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    question = str(case["question"])
    started = time.perf_counter()
    try:
        session = _json_request(
            f"{api}/chat/sessions",
            method="POST",
            payload={
                "knowledge_base_id": kb_id,
                "title": f"agent-complex-{case_id}",
            },
        )
        session_id = _required_string(session, "id")
        created = _json_request(
            f"{api}/chat/runs",
            method="POST",
            headers={"Idempotency-Key": str(uuid4())},
            payload={
                "session_id": session_id,
                "knowledge_base_id": kb_id,
                "message": question,
                "workflow": {"mode": "agent"},
                "retrieval": {
                    "mode": "hybrid" if strategy == "hybrid" else "vector",
                    "top_k": top_k,
                    "rerank_mode": rerank_mode,
                },
            },
        )
        run_id = _required_string(created, "run_id")
        terminal = _wait_for_terminal(
            api,
            run_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        safe = _safe_run_snapshot(terminal)
    except _EvaluationRequestError as error:
        safe = {
            "status": "failed",
            "answer": None,
            "citations": [],
            "workflow": None,
            "retrieval": None,
            "error": {"code": error.code, "http_status": error.http_status},
            "usage": None,
            "timing": None,
        }
    scored = score_complex_case(
        case,
        safe,
        document_identity_map=document_identity_map,
    )
    return {
        "case_id": case_id,
        "question": question,
        "answer": safe.get("answer"),
        "citations": safe.get("citations", []),
        "research_result": _research_result(safe.get("workflow")),
        "search_trace": _search_trace(safe.get("workflow")),
        "workflow": safe.get("workflow"),
        "retrieval": safe.get("retrieval"),
        "status": safe.get("status"),
        "error": safe.get("error"),
        "usage": safe.get("usage"),
        "timing": safe.get("timing"),
        "score": scored,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _wait_for_terminal(
    api: str,
    run_id: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    while True:
        run = _json_request(f"{api}/chat/runs/{run_id}")
        if run.get("status") in TERMINAL_STATUSES:
            return run
        if time.perf_counter() - started >= timeout_seconds:
            raise _EvaluationRequestError("evaluation_timeout")
        time.sleep(poll_seconds)


def score_complex_case(
    case: dict[str, Any],
    run: dict[str, Any],
    *,
    document_identity_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    answer = str(run.get("answer") or "")
    workflow = run.get("workflow")
    citations = run.get("citations")
    citations = citations if isinstance(citations, list) else []
    cited_documents = {
        _citation_document_id(item, document_identity_map)
        for item in citations
        if isinstance(item, dict) and item.get("document_id")
    }
    required_documents = {
        str(item) for item in case.get("required_citation_document_ids", ())
    }
    forbidden = sorted(
        cited_documents
        - required_documents
        if case.get("forbid_unrelated_citations")
        else set()
    )
    aspects = []
    for aspect in case.get("aspects", ()):
        text_ok = _match_aspect_text(answer, aspect)
        decimal_ok = _match_aspect_decimal(answer, aspect)
        aspect_ok = text_ok and decimal_ok
        aspects.append(
            {
                "aspect_id": aspect.get("aspect_id"),
                "matched": aspect_ok,
                "text_match": text_ok,
                "decimal_match": decimal_ok,
                "expected_decimal": aspect.get("expected_decimal"),
            }
        )
    workflow_mode_ok = isinstance(workflow, dict) and (
        workflow.get("requested_mode") == "agent"
        and workflow.get("resolved_mode") == "agent"
    )
    status_ok = run.get("status") == "completed"
    research = _research_result(workflow)
    research_status = research.get("status") if isinstance(research, dict) else None
    complete_scan_count = (
        research.get("complete_scan_document_count", 0)
        if isinstance(research, dict)
        else 0
    )
    if (
        isinstance(complete_scan_count, bool)
        or not isinstance(complete_scan_count, int)
        or complete_scan_count < 0
    ):
        complete_scan_count = 0
    requires_complete_scan = any(
        bool(aspect.get("requires_complete_scan"))
        for aspect in case.get("aspects", ())
        if isinstance(aspect, dict)
    )
    not_mentioned_without_complete_scan = bool(
        requires_complete_scan
        and _contains_not_mentioned(answer)
        and complete_scan_count == 0
    )
    coverage = (
        len(required_documents & cited_documents) / len(required_documents)
        if required_documents
        else 1.0
    )
    all_aspects = bool(aspects) and all(item["matched"] for item in aspects)
    sufficient_precision_ok = (
        coverage == 1.0
        and not forbidden
        and not not_mentioned_without_complete_scan
        if research_status == "sufficient"
        else None
    )
    strict = bool(
        status_ok
        and workflow_mode_ok
        and all_aspects
        and coverage == 1.0
        and not forbidden
        and not not_mentioned_without_complete_scan
    )
    return {
        "strict_correct": strict,
        "at_least_partial": bool(status_ok and any(item["matched"] for item in aspects)),
        "terminal_completed": status_ok,
        "workflow_mode_ok": workflow_mode_ok,
        "aspect_accuracy": (
            round(sum(item["matched"] for item in aspects) / len(aspects), 6)
            if aspects
            else 0.0
        ),
        "aspects": aspects,
        "required_document_citation_coverage": round(coverage, 6),
        "cited_document_ids": sorted(cited_documents),
        "forbidden_citation_document_ids": forbidden,
        "research_status": research_status,
        "sufficient_precision_ok": sufficient_precision_ok,
        "not_mentioned_without_complete_scan": not_mentioned_without_complete_scan,
        "evaluation_group": case.get("evaluation_group", "evidence_only"),
    }


def summarize_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(results)
    strict = [item for item in values if item.get("score", {}).get("strict_correct")]
    partial = [item for item in values if item.get("score", {}).get("at_least_partial")]
    completed = [item for item in values if item.get("score", {}).get("terminal_completed")]
    evidence_only = [
        item for item in values
        if item.get("score", {}).get("evaluation_group") == "evidence_only"
    ]
    domain = [
        item for item in values
        if item.get("score", {}).get("evaluation_group") == "domain_inference"
    ]
    elapsed = [
        float(item["elapsed_seconds"])
        for item in values
        if isinstance(item.get("elapsed_seconds"), (int, float))
        and not isinstance(item.get("elapsed_seconds"), bool)
    ]
    return {
        "cases": len(values),
        "requested_cases": len(values),
        "attempted_cases": sum(item.get("status") != "not_run" for item in values),
        "not_run_cases": sum(item.get("status") == "not_run" for item in values),
        "completed": len(completed),
        "at_least_partial": len(partial),
        "strict_correct": len(strict),
        "required_document_citation_coverage_100": sum(
            item.get("score", {}).get("required_document_citation_coverage") == 1.0
            for item in values
        ),
        "forbidden_citation_cases": sum(
            bool(item.get("score", {}).get("forbidden_citation_document_ids"))
            for item in values
        ),
        "evidence_only_strict_correct": sum(
            item.get("score", {}).get("strict_correct") for item in evidence_only
        ),
        "evidence_only_cases": len(evidence_only),
        "domain_inference_strict_correct": sum(
            item.get("score", {}).get("strict_correct") for item in domain
        ),
        "domain_inference_cases": len(domain),
        "sufficient_cases": sum(
            item.get("score", {}).get("research_status") == "sufficient"
            for item in values
        ),
        "sufficient_precision_cases": sum(
            item.get("score", {}).get("sufficient_precision_ok") is True
            for item in values
        ),
        "scope_outside_reference_cases": sum(
            bool(item.get("score", {}).get("forbidden_citation_document_ids"))
            for item in values
        ),
        "not_mentioned_without_complete_scan_cases": sum(
            bool(item.get("score", {}).get("not_mentioned_without_complete_scan"))
            for item in values
        ),
        "chat_run_retry_cases": sum(_has_chat_run_retry(item) for item in values),
        "total_timeout_cases": sum(_has_total_timeout(item) for item in values),
        "total_tokens": sum(_total_tokens(item) for item in values),
        "median_elapsed_seconds": (
            round(float(median(elapsed)), 3) if elapsed else None
        ),
        "max_elapsed_seconds": round(max(elapsed), 3) if elapsed else None,
    }


def _match_aspect_text(answer: str, aspect: dict[str, Any]) -> bool:
    variants = [str(item) for item in aspect.get("answer_variants", ())]
    if not variants:
        return True
    normalized_answer = _normalize(answer)
    normalized_variants = [_normalize(item) for item in variants]
    if aspect.get("answer_match") == "any":
        return any(item in normalized_answer for item in normalized_variants)
    return all(item in normalized_answer for item in normalized_variants)


def _match_aspect_decimal(answer: str, aspect: dict[str, Any]) -> bool:
    expected = aspect.get("expected_decimal")
    if expected is None:
        return True
    try:
        expected_value = Decimal(str(expected))
        tolerance = Decimal(str(aspect.get("numeric_tolerance", "0.01")))
    except (InvalidOperation, ValueError):
        return False
    values = _extract_decimal_values(answer)
    if any(abs(value - expected_value) <= tolerance for value in values):
        return True
    if expected_value < 0 and any(
        abs(abs(value) - abs(expected_value)) <= tolerance
        for value in values
    ):
        return any(word in answer.casefold() or word in answer for word in _NEGATIVE_WORDS)
    return False


def _extract_decimal_values(text: str) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group("value").replace(",", "").replace("$", "")
        raw = re.sub(r"\s+", "", raw)
        negative_parentheses = raw.startswith("(") and raw.endswith(")")
        raw = raw.strip("()")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        if negative_parentheses:
            value = -value
        if match.group("percent"):
            values.append(value)
        values.append(value)
    return tuple(values)


def _research_result(workflow: object) -> dict[str, Any] | None:
    return workflow.get("research_result") if isinstance(workflow, dict) else None


def _search_trace(workflow: object) -> dict[str, Any] | None:
    return workflow.get("search_trace") if isinstance(workflow, dict) else None


def _research_status(workflow: object) -> str | None:
    result = _research_result(workflow)
    return result.get("status") if isinstance(result, dict) else None


def _contains_not_mentioned(value: str) -> bool:
    normalized = value.casefold().replace("-", "_")
    return any(
        marker in normalized
        for marker in ("not_mentioned", "not mentioned", "未提及", "未提到")
    )


def _attempts(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    timing = value.get("timing")
    attempts = timing.get("attempts") if isinstance(timing, Mapping) else None
    if not isinstance(attempts, Mapping):
        return []
    return [item for item in attempts.values() if isinstance(item, Mapping)]


def _has_chat_run_retry(value: Mapping[str, Any]) -> bool:
    return len(_attempts(value)) > 1


def _has_total_timeout(value: Mapping[str, Any]) -> bool:
    return any(
        isinstance(item.get("diagnostic"), Mapping)
        and item["diagnostic"].get("check") == "total_timeout"
        for item in _attempts(value)
    )


def _total_tokens(value: Mapping[str, Any]) -> int:
    usage = value.get("usage")
    totals = usage.get("totals") if isinstance(usage, Mapping) else None
    total = totals.get("total_tokens") if isinstance(totals, Mapping) else 0
    return total if isinstance(total, int) and not isinstance(total, bool) and total >= 0 else 0


def _safe_run_snapshot(run: dict[str, Any]) -> dict[str, Any]:
    citations = run.get("citations")
    safe_citations: list[dict[str, Any]] = []
    if isinstance(citations, list):
        for item in citations:
            if not isinstance(item, dict):
                continue
            safe_citations.append(
                {
                    key: item.get(key)
                    for key in (
                        "ordinal",
                        "index_chunk_id",
                        "document_id",
                        "document_version_id",
                        "document_display_name",
                        "document_original_filename",
                        "quoted_text",
                        "source_location",
                        "score",
                        "modality",
                        "matched_representations",
                    )
                    if key in item
                }
            )
    return {
        "status": run.get("status"),
        "answer": run.get("answer"),
        "citations": safe_citations,
        "workflow": _safe_json_value(run.get("workflow")),
        "retrieval": _safe_json_value(run.get("retrieval")),
        "error": _safe_error(run.get("error")),
        "usage": _safe_json_value(run.get("usage")),
        "timing": _safe_json_value(run.get("timing")),
    }


def _safe_error(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for key in ("code", "retryable", "http_status"):
        if key in value and isinstance(value[key], (str, bool)):
            result[key] = value[key]
        elif key == "http_status" and isinstance(value.get(key), int):
            result[key] = value[key]
    return result or None


def _safe_json_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _safe_json_value(item)
            for key, item in value.items()
            if str(key) not in {"provider_payload", "raw_response", "prompt", "messages"}
        }
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json", **(headers or {})}
    body = None
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            value = json.loads(response.read())
    except HTTPError as error:
        code = "HTTP_ERROR"
        try:
            payload = json.loads(error.read())
            if isinstance(payload, dict) and isinstance(payload.get("code"), str):
                code = payload["code"]
        except (ValueError, TypeError):
            pass
        raise _EvaluationRequestError(code, http_status=error.code) from None
    except URLError:
        raise _EvaluationRequestError("API_UNAVAILABLE") from None
    if not isinstance(value, dict):
        raise _EvaluationRequestError("INVALID_API_RESPONSE")
    return value


class _EvaluationRequestError(RuntimeError):
    def __init__(self, code: str, *, http_status: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.http_status = http_status


def _required_string(value: dict[str, Any], key: str) -> str:
    candidate = value.get(key)
    if not isinstance(candidate, str) or not candidate:
        raise _EvaluationRequestError("INVALID_API_RESPONSE")
    return candidate


def _validated_api_base(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/api/v1"
        or parsed.port is None
    ):
        raise ValueError("--api must be an exact loopback /api/v1 URL")
    return f"http://{parsed.hostname}:{parsed.port}/api/v1"


def _default_output(strategy: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"agent-complex-qa-{timestamp}-{strategy}.json"


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _citation_document_id(
    citation: dict[str, Any],
    document_identity_map: Mapping[str, str] | None,
) -> str:
    raw_id = str(citation.get("document_id"))
    if not document_identity_map:
        return raw_id
    for key in ("document_original_filename", "document_display_name", "document_id"):
        value = citation.get(key)
        if not isinstance(value, str) or not value:
            continue
        mapped = document_identity_map.get(_normalize_document_identity(value))
        if mapped is not None:
            return mapped
    return raw_id


def _normalize_document_identity(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].strip().casefold()


if __name__ == "__main__":
    raise SystemExit(main())
