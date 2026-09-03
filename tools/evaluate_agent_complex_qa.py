#!/usr/bin/env python3
"""Run or semantically rescore the pinned Agent complex-QA benchmark."""

from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import time
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

from tools.build_document_qa_corpus import (
    COMPLEX_CASES_FILENAME,
    validate_corpus,
)
from tools.agent_complex_qa_judge import (
    ComplexQaLlmJudge,
    JUDGE_PROMPT_VERSION,
    JUDGE_SCHEMA_VERSION,
    build_judge_packet,
    judge_packet_sha256,
    load_frozen_judge_runtime,
    semantic_score,
)
from rag_kb.domain import ChatModelExecutionError
from tools.evaluation_runtime import (
    DEFAULT_RUNTIME_MANIFEST,
    EvaluationRuntimeError,
    load_evaluation_runtime,
)


DEFAULT_CORPUS_ROOT = Path(__file__).resolve().parents[1] / "evaluation" / "document-qa-v1"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / ".runtime" / "evaluations"
DEFAULT_JUDGE_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / ".judge-cache"
RESCORE_CORPUS_COMPATIBILITY = {
    (
        "f54689e2b336b90af6fca5a421fe74e44b73ffc56d48118680d1de3360550ae6",
        "625c7fabde8324be13b001936af37434f38bde24239364d44fa3cc5d1a9fec73",
    ): {
        "migration": "complex-reference-corrections-v1",
        "artifact_sha256": (
            "676e1bb103324c8d98dec489a9c01f70407d41606605cfeba221cf1066c07713"
        ),
    },
}
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-runtime",
        type=Path,
        default=DEFAULT_RUNTIME_MANIFEST,
    )
    parser.add_argument("--kb-id")
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
        "--rescore-input",
        type=Path,
        help="rejudge an existing evaluation artifact without calling the answer model",
    )
    parser.add_argument("--judge-profile-revision-id")
    parser.add_argument(
        "--judge-cache-dir",
        type=Path,
        default=DEFAULT_JUDGE_CACHE_ROOT,
        help="content-addressed cache for completed per-case judgements",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate corpus and options without API, database, or Provider access",
    )
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
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    try:
        _validate_options(arguments)
        cases = load_complex_cases(arguments.corpus_root)
        document_identity_map = load_document_identity_map(arguments.corpus_root)
        document_paths = load_document_paths(arguments.corpus_root)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))

    if arguments.dry_run:
        corpus = validate_corpus(arguments.corpus_root)
        print(
            json.dumps(
                {
                    "status": "offline_dry_run_ok",
                    "case_count": len(cases),
                    "corpus_sha256": corpus["dataset_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0

    if arguments.judge_profile_revision_id is None:
        parser.error("--judge-profile-revision-id is required for execution")
    try:
        runtime = load_evaluation_runtime(arguments.evaluation_runtime)
        api = runtime.api_base_url
        judge_profile_revision_id = UUID(arguments.judge_profile_revision_id)
    except (EvaluationRuntimeError, OSError, ValueError):
        parser.error("isolated evaluation runtime is unavailable or profile identity is invalid")

    requested = tuple(arguments.case_ids or ())
    if arguments.preflight:
        if requested:
            parser.error("--preflight cannot be combined with --case-id")
        requested = PREFLIGHT_CASE_IDS
    if requested:
        if arguments.rescore_input is not None:
            parser.error("--rescore-input cannot be combined with --case-id")
        unknown = set(requested) - {str(case["case_id"]) for case in cases}
        if unknown:
            parser.error(f"unknown complex case: {sorted(unknown)}")
        by_id = {str(case["case_id"]): case for case in cases}
        cases = [by_id[case_id] for case_id in requested]

    corpus = validate_corpus(arguments.corpus_root)
    output = arguments.output or (
        _default_rescore_output()
        if arguments.rescore_input is not None
        else _default_output(arguments.strategy)
    )
    if (
        arguments.rescore_input is not None
        and output.resolve() == arguments.rescore_input.resolve()
    ):
        parser.error("--output must not overwrite --rescore-input")

    started = time.perf_counter()
    source_artifact_sha256: str | None = None
    source_corpus_sha256: str | None = None
    source_corpus_compatibility: str | None = None
    if arguments.rescore_input is not None:
        source_report, source_artifact_sha256 = load_evaluation_artifact(
            arguments.rescore_input,
            cases=cases,
            corpus_sha256=str(corpus["dataset_sha256"]),
            compatible_corpus_transitions={
                transition: frozenset({str(details["artifact_sha256"])})
                for transition, details in RESCORE_CORPUS_COMPATIBILITY.items()
            },
        )
        source_config = source_report.get("config")
        assert isinstance(source_config, dict)
        source_corpus_sha256 = str(source_config["corpus_sha256"])
        transition = (
            source_corpus_sha256,
            str(corpus["dataset_sha256"]),
        )
        compatibility = RESCORE_CORPUS_COMPATIBILITY.get(transition)
        source_corpus_compatibility = (
            str(compatibility["migration"])
            if compatibility is not None
            else None
        )
        results = [dict(item) for item in source_report["cases"]]
        base_config = dict(source_report.get("config") or {})
    else:
        assert arguments.kb_id is not None
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
            profile_revision_id=arguments.profile_revision_id,
        )
        base_config = {
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
            "evaluation_owner": runtime.owner,
        }
        answer_checkpoint = _answer_checkpoint_path(output)
        _atomic_write_json(
            answer_checkpoint,
            {
                "schema_version": "native_agent_complex_qa_evaluation_v2",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": base_config,
                "cases": results,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        )
        print(f"answer checkpoint: {answer_checkpoint}", flush=True)
    try:
        results, judge_config = asyncio.run(
            judge_results_with_profile(
                results,
                cases=cases,
                corpus_root=arguments.corpus_root,
                document_identity_map=document_identity_map,
                document_paths=document_paths,
                profile_revision_id=judge_profile_revision_id,
                env_file=runtime.env_file,
                cache_dir=arguments.judge_cache_dir,
            )
        )
    except ChatModelExecutionError as error:
        parser.error(
            "LLM Judge failed: "
            f"{error.code.value} diagnostic="
            f"{json.dumps(error.diagnostic, sort_keys=True)}"
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(f"LLM Judge failed: {error}")
    report = {
        "schema_version": "native_agent_complex_qa_evaluation_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **base_config,
            "case_count": len(cases),
            "corpus_sha256": corpus["dataset_sha256"],
            "rescore_only": arguments.rescore_input is not None,
            "source_artifact_sha256": source_artifact_sha256,
            "source_corpus_sha256": source_corpus_sha256,
            "source_corpus_compatibility": source_corpus_compatibility,
            "evaluation_owner": runtime.owner,
            "judge": judge_config,
        },
        "cases": results,
        "summary": summarize_results(results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _atomic_write_json(output, report)
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
    if (
        (hasattr(arguments, "rescore_input") or hasattr(arguments, "kb_id"))
        and not getattr(arguments, "dry_run", False)
    ):
        rescore_input = getattr(arguments, "rescore_input", None)
        kb_id = getattr(arguments, "kb_id", None)
        if rescore_input is None and not kb_id:
            raise ValueError("--kb-id is required unless --rescore-input is used")
        if rescore_input is not None and getattr(arguments, "preflight", False):
            raise ValueError("--rescore-input cannot be combined with --preflight")


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


def load_document_paths(root: Path) -> dict[str, str]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise RuntimeError("corpus manifest documents are missing")
    result: dict[str, str] = {}
    for document in documents:
        if not isinstance(document, dict):
            raise RuntimeError("corpus manifest document is invalid")
        document_id = document.get("document_id")
        path = document.get("path")
        if (
            not isinstance(document_id, str)
            or not document_id
            or not isinstance(path, str)
            or not path
            or document_id in result
        ):
            raise RuntimeError("corpus manifest document path is invalid")
        result[document_id] = path
    return result


def load_reference_cases(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "cases.jsonl"
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        case_id = value.get("case_id") if isinstance(value, dict) else None
        if not isinstance(case_id, str) or not case_id or case_id in result:
            raise RuntimeError("reference corpus case IDs are invalid")
        result[case_id] = value
    return result


def load_evaluation_artifact(
    path: Path,
    *,
    cases: list[dict[str, Any]],
    corpus_sha256: str,
    compatible_corpus_transitions: Mapping[
        tuple[str, str], frozenset[str]
    ] | None = None,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError(f"evaluation artifact is missing: {path}")
    raw = path.read_bytes()
    try:
        report = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation artifact is not valid JSON") from error
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "native_agent_complex_qa_evaluation_v2"
    ):
        raise ValueError("evaluation artifact must use the v2 pre-Judge schema")
    config = report.get("config")
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    source_corpus_sha256 = (
        config.get("corpus_sha256") if isinstance(config, dict) else None
    )
    compatible_artifacts = (
        compatible_corpus_transitions or {}
    ).get((str(source_corpus_sha256), corpus_sha256), frozenset())
    if (
        not isinstance(source_corpus_sha256, str)
        or (
            source_corpus_sha256 != corpus_sha256
            and artifact_sha256 not in compatible_artifacts
        )
    ):
        raise ValueError("evaluation artifact corpus hash does not match")
    raw_results = report.get("cases")
    if not isinstance(raw_results, list):
        raise ValueError("evaluation artifact cases are missing")
    expected = {str(case["case_id"]): case for case in cases}
    found: dict[str, dict[str, Any]] = {}
    for result in raw_results:
        case_id = result.get("case_id") if isinstance(result, dict) else None
        if not isinstance(case_id, str) or case_id in found:
            raise ValueError("evaluation artifact case IDs are invalid")
        case = expected.get(case_id)
        if case is None or result.get("question") != case.get("question"):
            raise ValueError("evaluation artifact case does not match the corpus")
        score = result.get("score")
        scored_aspects = score.get("aspects") if isinstance(score, dict) else None
        if not isinstance(scored_aspects, list):
            raise ValueError("evaluation artifact scored aspects are invalid")
        expected_aspects = {
            str(item["aspect_id"])
            for item in case.get("aspects", ())
            if isinstance(item, dict)
        }
        found_aspects = {
            str(item["aspect_id"])
            for item in scored_aspects
            if isinstance(item, dict) and isinstance(item.get("aspect_id"), str)
        }
        if found_aspects != expected_aspects or len(scored_aspects) != len(
            expected_aspects
        ):
            raise ValueError("evaluation artifact aspect IDs do not match the corpus")
        found[case_id] = result
    if set(found) != set(expected):
        raise ValueError("evaluation artifact case set does not match the requested corpus")
    report["cases"] = [found[str(case["case_id"])] for case in cases]
    return report, artifact_sha256


async def judge_results_with_profile(
    results: list[dict[str, Any]],
    *,
    cases: list[dict[str, Any]],
    corpus_root: Path,
    document_identity_map: Mapping[str, str],
    document_paths: Mapping[str, str],
    profile_revision_id: UUID,
    env_file: Path,
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = await load_frozen_judge_runtime(
        profile_revision_id,
        env_file=env_file,
    )
    try:
        judge = ComplexQaLlmJudge(
            runtime.model,
            profile_revision_id=runtime.profile_revision_id,
            expected_model=runtime.model_name,
            max_output_tokens=runtime.max_output_tokens,
        )
        judged = await apply_llm_judgements(
            results,
            cases=cases,
            corpus_root=corpus_root,
            document_identity_map=document_identity_map,
            document_paths=document_paths,
            judge=judge,
            cache_dir=cache_dir,
        )
        return judged, runtime.config()
    finally:
        await runtime.close()


async def apply_llm_judgements(
    results: list[dict[str, Any]],
    *,
    cases: list[dict[str, Any]],
    corpus_root: Path,
    document_identity_map: Mapping[str, str],
    document_paths: Mapping[str, str],
    judge: ComplexQaLlmJudge,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    case_by_id = {str(case["case_id"]): case for case in cases}
    reference_cases = load_reference_cases(corpus_root)
    judged: list[dict[str, Any]] = []
    for original in results:
        result = dict(original)
        case_id = result.get("case_id")
        case = case_by_id.get(str(case_id))
        if case is None or result.get("question") != case.get("question"):
            raise ValueError("evaluation result does not match the Judge corpus")
        legacy_score = result.get("score")
        if not isinstance(legacy_score, dict):
            raise ValueError("evaluation result has no legacy score")
        result["legacy_score"] = _safe_json_value(legacy_score)
        if result.get("status") != "completed":
            judgement = {
                "schema_version": JUDGE_SCHEMA_VERSION,
                "status": "not_judged",
                "reason": "run_not_completed",
                "prompt_version": JUDGE_PROMPT_VERSION,
                "input_sha256": None,
                "disputed_aspect_ids": [],
                "aspects": [],
                "calls": {"judge_a": None, "judge_b": None, "judge_c": None},
            }
            semantic = {
                "semantic_strict_correct": False,
                "semantic_at_least_partial": False,
                "semantic_grounding_supported": False,
                "semantic_aspect_accuracy": 0.0,
            }
        else:
            packet = build_judge_packet(
                case,
                result,
                reference_cases=reference_cases,
                citation_document_id=lambda citation: _citation_document_id(
                    dict(citation),
                    document_identity_map,
                ),
                corpus_root=corpus_root,
                document_paths=document_paths,
            )
            input_sha256 = judge_packet_sha256(packet)
            judgement = _load_cached_judgement(
                cache_dir,
                case=case,
                input_sha256=input_sha256,
                profile_revision_id=judge.profile_revision_id,
            )
            if judgement is None:
                judgement = await judge.judge(packet)
                _store_cached_judgement(
                    cache_dir,
                    case_id=str(case_id),
                    judgement=judgement,
                )
                print(f"judge {case_id}: completed", flush=True)
            else:
                print(f"judge {case_id}: cache_hit", flush=True)
            semantic = semantic_score(judgement)
        result["judge"] = judgement
        result["score"] = {
            **semantic,
            "terminal_completed": result.get("status") == "completed",
            "judge_status": judgement["status"],
            "aspects": judgement["aspects"],
            "judge_disagreement_count": len(judgement["disputed_aspect_ids"]),
            "agent_protocol_ok": legacy_score.get("agent_protocol_ok"),
            "required_document_citation_coverage": legacy_score.get(
                "required_document_citation_coverage", 0.0
            ),
            "cited_document_ids": legacy_score.get("cited_document_ids", []),
            "forbidden_citation_document_ids": legacy_score.get(
                "forbidden_citation_document_ids", []
            ),
            "agent_outcome": legacy_score.get("agent_outcome"),
            "answered_precision_ok": legacy_score.get("answered_precision_ok"),
            "evaluation_group": case.get("evaluation_group", "evidence_only"),
        }
        judged.append(result)
    return judged


def _load_cached_judgement(
    cache_dir: Path | None,
    *,
    case: Mapping[str, Any],
    input_sha256: str,
    profile_revision_id: UUID,
) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    case_id = str(case["case_id"])
    path = _judge_cache_path(
        cache_dir,
        case_id,
        input_sha256,
        profile_revision_id=profile_revision_id,
    )
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    judgement = value.get("judgement") if isinstance(value, dict) else None
    expected_aspects = [
        str(item["aspect_id"])
        for item in case.get("aspects", ())
        if isinstance(item, Mapping) and isinstance(item.get("aspect_id"), str)
    ]
    aspects = judgement.get("aspects") if isinstance(judgement, dict) else None
    actual_aspects = [
        str(item.get("aspect_id"))
        for item in aspects or ()
        if isinstance(item, Mapping)
    ]
    if (
        not isinstance(value, dict)
        or value.get("case_id") != case_id
        or not isinstance(judgement, dict)
        or judgement.get("schema_version") != JUDGE_SCHEMA_VERSION
        or judgement.get("status") != "judged"
        or judgement.get("prompt_version") != JUDGE_PROMPT_VERSION
        or judgement.get("profile_revision_id") != str(profile_revision_id)
        or judgement.get("input_sha256") != input_sha256
        or not isinstance(aspects, list)
        or actual_aspects != expected_aspects
    ):
        return None
    try:
        semantic_score(judgement)
    except ValueError:
        return None
    return judgement


def _store_cached_judgement(
    cache_dir: Path | None,
    *,
    case_id: str,
    judgement: Mapping[str, Any],
) -> None:
    if cache_dir is None:
        return
    input_sha256 = judgement.get("input_sha256")
    if not isinstance(input_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", input_sha256) is None:
        raise ValueError("Judge result input hash is invalid")
    try:
        profile_revision_id = UUID(str(judgement.get("profile_revision_id")))
    except ValueError as error:
        raise ValueError("Judge result profile revision is invalid") from error
    path = _judge_cache_path(
        cache_dir,
        case_id,
        input_sha256,
        profile_revision_id=profile_revision_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(
                {"case_id": case_id, "judgement": _safe_json_value(judgement)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _judge_cache_path(
    cache_dir: Path,
    case_id: str,
    input_sha256: str,
    *,
    profile_revision_id: UUID | None = None,
) -> Path:
    if re.fullmatch(r"complex-[0-9]{2}", case_id) is None:
        raise ValueError("Judge cache case ID is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", input_sha256) is None:
        raise ValueError("Judge cache input hash is invalid")
    profile = str(profile_revision_id) if profile_revision_id is not None else "legacy"
    return cache_dir / profile / f"{case_id}-{input_sha256}.json"


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
    profile_revision_id: str | None = None,
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
            profile_revision_id=profile_revision_id,
        )

    if parallelism != 1:
        raise ValueError("evaluation parallelism must be 1")
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        result = run(case)
        results.append(result)
        if _is_nonretryable_provider_auth_failure(result):
            results.extend(
                _not_run_result(
                    remaining,
                    reason="provider_nonretryable_auth_failure",
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
        "agent": None,
        "retrieval": None,
        "error": None,
        "usage": None,
        "timing": None,
    }
    return {
        "case_id": str(case["case_id"]),
        "question": str(case["question"]),
        **safe,
        "score": score_complex_case(
            dict(case),
            safe,
            document_identity_map=document_identity_map,
        ),
        "elapsed_seconds": 0.0,
        "not_run_reason": reason,
    }


def _is_nonretryable_provider_auth_failure(value: Mapping[str, Any]) -> bool:
    error = value.get("error")
    if (
        isinstance(error, Mapping)
        and error.get("http_status") in {401, 403}
        and error.get("retryable") is not True
    ):
        return True
    return any(
        isinstance(item.get("diagnostic"), Mapping)
        and item["diagnostic"].get("http_status") in {401, 403}
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
    profile_revision_id: str | None = None,
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
        run_payload: dict[str, object] = {
            "session_id": session_id,
            "knowledge_base_id": kb_id,
            "message": question,
            "retrieval": {
                "mode": "hybrid" if strategy == "hybrid" else "vector",
                "top_k": top_k,
                "rerank_mode": rerank_mode,
            },
        }
        if profile_revision_id is not None:
            run_payload["model_profile_revision_id"] = profile_revision_id
        created = _json_request(
            f"{api}/chat/runs",
            method="POST",
            headers={"Idempotency-Key": str(uuid4())},
            payload=run_payload,
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
            "agent": None,
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
        "agent": safe.get("agent"),
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
    agent = run.get("agent")
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
    agent_protocol_ok = isinstance(agent, dict) and agent.get("version") in {
        "native_tool_calling_agent_v1",
        "native_tool_calling_agent_v2",
        "native_tool_calling_agent_v3",
        "native_tool_calling_agent_v4",
        "native_tool_calling_agent_v5",
    }
    status_ok = run.get("status") == "completed"
    trace = agent.get("trace") if isinstance(agent, dict) else None
    agent_outcome = trace.get("outcome") if isinstance(trace, dict) else None
    coverage = (
        len(required_documents & cited_documents) / len(required_documents)
        if required_documents
        else 1.0
    )
    all_aspects = bool(aspects) and all(item["matched"] for item in aspects)
    answered_precision_ok = (
        coverage == 1.0
        and not forbidden
        if agent_outcome == "answered"
        else None
    )
    strict = bool(
        status_ok
        and agent_protocol_ok
        and all_aspects
        and coverage == 1.0
        and not forbidden
    )
    return {
        "strict_correct": strict,
        "at_least_partial": bool(status_ok and any(item["matched"] for item in aspects)),
        "terminal_completed": status_ok,
        "agent_protocol_ok": agent_protocol_ok,
        "aspect_accuracy": (
            round(sum(item["matched"] for item in aspects) / len(aspects), 6)
            if aspects
            else 0.0
        ),
        "aspects": aspects,
        "required_document_citation_coverage": round(coverage, 6),
        "cited_document_ids": sorted(cited_documents),
        "forbidden_citation_document_ids": forbidden,
        "agent_outcome": agent_outcome,
        "answered_precision_ok": answered_precision_ok,
        "evaluation_group": case.get("evaluation_group", "evidence_only"),
    }


def summarize_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(results)
    strict = [
        item
        for item in values
        if _primary_score_flag(
            item.get("score"),
            semantic="semantic_strict_correct",
        )
    ]
    partial = [
        item
        for item in values
        if _primary_score_flag(
            item.get("score"),
            semantic="semantic_at_least_partial",
        )
    ]
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
        "semantic_strict_correct": len(strict),
        "semantic_at_least_partial": len(partial),
        "legacy_strict_correct": sum(
            _legacy_score_flag(item, "strict_correct") for item in values
        ),
        "legacy_at_least_partial": sum(
            _legacy_score_flag(item, "at_least_partial") for item in values
        ),
        "semantic_grounding_supported": sum(
            item.get("score", {}).get("semantic_grounding_supported") is True
            for item in values
        ),
        "judge_disagreement_aspects": sum(
            int(item.get("score", {}).get("judge_disagreement_count", 0))
            for item in values
            if isinstance(
                item.get("score", {}).get("judge_disagreement_count", 0), int
            )
            and not isinstance(
                item.get("score", {}).get("judge_disagreement_count", 0), bool
            )
        ),
        "judge_calls": sum(_judge_call_count(item) for item in values),
        "judge_attempts": sum(_judge_attempt_count(item) for item in values),
        "judge_total_tokens": sum(_judge_total_tokens(item) for item in values),
        "required_document_citation_coverage_100": sum(
            item.get("score", {}).get("required_document_citation_coverage") == 1.0
            for item in values
        ),
        "forbidden_citation_cases": sum(
            bool(item.get("score", {}).get("forbidden_citation_document_ids"))
            for item in values
        ),
        "evidence_only_strict_correct": sum(
            _primary_score_flag(
                item.get("score"),
                semantic="semantic_strict_correct",
            )
            for item in evidence_only
        ),
        "evidence_only_cases": len(evidence_only),
        "domain_inference_strict_correct": sum(
            _primary_score_flag(
                item.get("score"),
                semantic="semantic_strict_correct",
            )
            for item in domain
        ),
        "domain_inference_cases": len(domain),
        "answered_cases": sum(
            item.get("score", {}).get("agent_outcome") == "answered"
            for item in values
        ),
        "answered_precision_cases": sum(
            item.get("score", {}).get("answered_precision_ok") is True
            for item in values
        ),
        "scope_outside_reference_cases": sum(
            bool(item.get("score", {}).get("forbidden_citation_document_ids"))
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


def _primary_score_flag(
    score: object,
    *,
    semantic: str,
) -> bool:
    if not isinstance(score, Mapping):
        return False
    return score.get(semantic) is True


def _legacy_score_flag(value: Mapping[str, Any], field: str) -> bool:
    score = value.get("legacy_score")
    return isinstance(score, Mapping) and score.get(field) is True


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _judge_call_count(value: Mapping[str, Any]) -> int:
    judge = value.get("judge")
    calls = judge.get("calls") if isinstance(judge, Mapping) else None
    if not isinstance(calls, Mapping):
        return 0
    return sum(call is not None for call in calls.values())


def _judge_attempt_count(value: Mapping[str, Any]) -> int:
    judge = value.get("judge")
    calls = judge.get("calls") if isinstance(judge, Mapping) else None
    if not isinstance(calls, Mapping):
        return 0
    total = 0
    for call in calls.values():
        candidate = (
            call.get("transport_attempts") if isinstance(call, Mapping) else None
        )
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            total += candidate
    return total


def _judge_total_tokens(value: Mapping[str, Any]) -> int:
    judge = value.get("judge")
    calls = judge.get("calls") if isinstance(judge, Mapping) else None
    if not isinstance(calls, Mapping):
        return 0
    total = 0
    for call in calls.values():
        usage = call.get("usage") if isinstance(call, Mapping) else None
        candidate = usage.get("total_tokens") if isinstance(usage, Mapping) else None
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            total += candidate
    return total


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
        "agent": _safe_json_value(run.get("agent")),
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


def _default_output(strategy: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"agent-complex-qa-{timestamp}-{strategy}.json"


def _default_rescore_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"agent-complex-qa-{timestamp}-llm-judge.json"


def _answer_checkpoint_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.answers.json")


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
