#!/usr/bin/env python3
"""Provision and observe the frozen Auto-QA off/on evaluation pair."""

from __future__ import annotations

import argparse
import base64
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4

import httpx

from tools.build_document_qa_corpus import validate_corpus
from tools.evaluation_runtime import DEFAULT_RUNTIME_MANIFEST, load_evaluation_runtime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "evaluation/document-qa-v1"
DEFAULT_OUTPUT = ROOT / ".runtime/evaluations/auto-qa-ab-20260904/state.json"
MEDIA_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}
PROFILE_REQUIREMENTS = {
    "chat": ("agentic-v4-mimo-v2.5", "mimo-v2.5"),
    "text_embedding": (
        "agentic-v4-qwen-text-embedding",
        "qwen3.7-text-embedding",
    ),
    "multimodal_embedding": (
        "agentic-v4-tongyi-vision-embedding",
        "tongyi-embedding-vision-flash-2026-03-06",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-runtime", type=Path, default=DEFAULT_RUNTIME_MANIFEST)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name-suffix", default="20260904")
    parser.add_argument("--code-commit")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=21600.0)
    return parser


def _metadata(filename: str) -> str:
    value = json.dumps(
        {"v": 1, "filename": filename, "display_name": filename},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _raise(response: httpx.Response) -> dict[str, Any]:
    if not response.is_success:
        raise RuntimeError(
            f"{response.request.method} {response.request.url} -> "
            f"{response.status_code}: {response.text[:2000]}"
        )
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("API response is not an object")
    return value


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def _profiles(client: httpx.Client) -> dict[str, dict[str, Any]]:
    settings = _raise(client.get("/model-settings"))
    available = settings.get("profiles")
    if not isinstance(available, list):
        raise RuntimeError("model profiles are unavailable")
    result: dict[str, dict[str, Any]] = {}
    for kind, (name, model) in PROFILE_REQUIREMENTS.items():
        matches = [
            item
            for item in available
            if isinstance(item, dict)
            and item.get("name") == name
            and item.get("kind") == kind
            and item.get("model") == model
            and item.get("validation_status") == "valid"
        ]
        if len(matches) != 1 or not matches[0].get("revision_id"):
            raise RuntimeError(f"required valid profile is unavailable: {name}/{model}")
        result[kind] = matches[0]
    return result


def _existing_kb(client: httpx.Client, name: str) -> dict[str, Any] | None:
    payload = _raise(client.get("/knowledge-bases", params={"limit": 100}))
    items = payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("knowledge-base listing is invalid")
    matches = [item for item in items if isinstance(item, dict) and item.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"knowledge-base name is ambiguous: {name}")
    return matches[0] if matches else None


def _create_kb(
    client: httpx.Client,
    *,
    name: str,
    enabled: bool,
    chat_revision_id: str,
    text_revision_id: str,
) -> dict[str, Any]:
    existing = _existing_kb(client, name)
    if existing is not None:
        actual = existing.get("auto_qa") or {}
        if bool(actual.get("enabled")) != enabled:
            raise RuntimeError(f"existing knowledge base has wrong Auto-QA mode: {name}")
        return existing
    payload = {
        "name": name,
        "parsing": {"preset": "text_local_v1"},
        "chunking": {"preset": "structural_balanced_v2"},
        "retrieval_defaults": {
            "strategy": "exact_vector",
            "top_k": 10,
            "rerank_mode": "classic",
        },
        "embedding": {
            "strategy": "text_only",
            "text_profile_revision_id": text_revision_id,
        },
        "auto_qa": {
            "enabled": enabled,
            "model_profile_revision_id": chat_revision_id if enabled else None,
        },
    }
    return _raise(
        client.post(
            "/knowledge-bases",
            headers={"Idempotency-Key": str(uuid4())},
            json=payload,
        )
    )


def _document_files(corpus_root: Path) -> list[Path]:
    return sorted(
        path
        for path in (corpus_root / "documents").rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_TYPES
    )


def _upload_missing(client: httpx.Client, kb_id: str, files: list[Path]) -> None:
    listed = _raise(client.get(f"/knowledge-bases/{kb_id}/documents", params={"limit": 100}))
    items = listed.get("items")
    if not isinstance(items, list):
        raise RuntimeError("document listing is invalid")
    existing = {item.get("display_name") for item in items if isinstance(item, dict)}
    for path in files:
        if path.name in existing:
            continue
        last_error: Exception | None = None
        idempotency_key = str(uuid4())
        for attempt in range(1, 4):
            try:
                response = client.post(
                    f"/knowledge-bases/{kb_id}/documents",
                    headers={
                        "Idempotency-Key": idempotency_key,
                        "X-Document-Metadata": _metadata(path.name),
                        "Content-Type": MEDIA_TYPES[path.suffix.lower()],
                    },
                    content=path.read_bytes(),
                    timeout=180.0,
                )
                _raise(response)
                print(
                    json.dumps(
                        {"event": "uploaded", "kb_id": kb_id, "file": path.name}
                    )
                )
                last_error = None
                break
            except (httpx.HTTPError, RuntimeError) as error:
                last_error = error
                print(
                    json.dumps(
                        {
                            "event": "upload_retry",
                            "kb_id": kb_id,
                            "file": path.name,
                            "attempt": attempt,
                            "error": type(error).__name__,
                        }
                    )
                )
                time.sleep(2 * attempt)
        if last_error is not None:
            raise last_error


def _jobs(client: httpx.Client, kb_id: str) -> list[dict[str, Any]]:
    payload = _raise(
        client.get(f"/knowledge-bases/{kb_id}/indexing-jobs", params={"limit": 100})
    )
    items = payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("indexing job listing is invalid")
    return [item for item in items if isinstance(item, dict)]


def _observe(
    client: httpx.Client,
    pairs: dict[str, dict[str, Any]],
    *,
    expected_jobs: int,
    deadline: float,
    poll_seconds: float,
    state: dict[str, Any],
    output: Path,
) -> None:
    last_summary: object = None
    while time.monotonic() < deadline:
        complete = True
        summary: dict[str, Any] = {}
        for arm, item in pairs.items():
            jobs = _jobs(client, str(item["id"]))
            failed = [job for job in jobs if job.get("status") == "failed"]
            if failed:
                state["status"] = "failed"
                state["failed_arm"] = arm
                state["failed_job"] = failed[0]
                _atomic_write(output, state)
                raise RuntimeError(f"indexing failed for {arm}: {failed[0].get('error')}")
            counts: dict[str, int] = {}
            for job in jobs:
                key = f"{job.get('status')}:{job.get('phase')}"
                counts[key] = counts.get(key, 0) + 1
            ready = sum(
                1
                for job in jobs
                if job.get("status") == "completed"
                and job.get("build_status") == "ready"
                and job.get("serving_status") == "serving"
            )
            summary[arm] = {"job_count": len(jobs), "ready": ready, "counts": counts}
            complete = complete and len(jobs) >= expected_jobs and ready == len(jobs)
        state["last_observed_at"] = datetime.now(UTC).isoformat()
        state["indexing"] = summary
        _atomic_write(output, state)
        if summary != last_summary:
            print(json.dumps({"event": "indexing", "summary": summary}, sort_keys=True))
            last_summary = summary
        if complete:
            return
        time.sleep(poll_seconds)
    raise RuntimeError("Auto-QA A/B indexing timed out")


def main() -> int:
    arguments = _parser().parse_args()
    if (
        arguments.code_commit is not None
        and re.fullmatch(r"[0-9a-f]{40}", arguments.code_commit) is None
    ):
        raise RuntimeError("code commit must be a full lowercase Git revision")
    corpus = validate_corpus(arguments.corpus_root)
    runtime = load_evaluation_runtime(arguments.evaluation_runtime)
    files = _document_files(arguments.corpus_root)
    if len(files) != int(corpus["documents"]):
        raise RuntimeError("corpus document count does not match manifest")
    started_at = datetime.now(UTC)
    state: dict[str, Any] = {
        "schema_version": "auto_qa_ab_state_v1",
        "status": "provisioning",
        "started_at": started_at.isoformat(),
        "corpus_sha256": corpus["dataset_sha256"],
        "document_count": len(files),
        "runtime_owner": runtime.owner,
        "code_commit": arguments.code_commit or runtime.build_revision,
    }
    with httpx.Client(base_url=runtime.api_base_url, timeout=180.0) as client:
        profiles = _profiles(client)
        state["profiles"] = {
            kind: {
                "name": item["name"],
                "model": item["model"],
                "revision_id": item["revision_id"],
            }
            for kind, item in profiles.items()
        }
        pairs = {
            arm: _create_kb(
                client,
                name=f"auto-qa-ab-{arm}-{arguments.name_suffix}",
                enabled=arm == "on",
                chat_revision_id=str(profiles["chat"]["revision_id"]),
                text_revision_id=str(profiles["text_embedding"]["revision_id"]),
            )
            for arm in ("off", "on")
        }
        state["arms"] = {
            arm: {
                "knowledge_base_id": item["id"],
                "index_revision_id": item["active_index_revision_id"],
                "auto_qa": item["auto_qa"],
            }
            for arm, item in pairs.items()
        }
        _atomic_write(arguments.output, state)
        for item in pairs.values():
            _upload_missing(client, str(item["id"]), files)
        _observe(
            client,
            pairs,
            expected_jobs=len(files),
            deadline=time.monotonic() + arguments.timeout_seconds,
            poll_seconds=arguments.poll_seconds,
            state=state,
            output=arguments.output,
        )
        for arm, item in pairs.items():
            refreshed = _raise(client.get(f"/knowledge-bases/{item['id']}"))
            state["arms"][arm]["index_revision_id"] = refreshed["active_index_revision_id"]
    state["status"] = "indexed"
    state["completed_at"] = datetime.now(UTC).isoformat()
    state["elapsed_seconds"] = round(
        (datetime.now(UTC) - started_at).total_seconds(), 3
    )
    _atomic_write(arguments.output, state)
    print(json.dumps(state, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
