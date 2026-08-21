#!/usr/bin/env python3
"""Provision the post-fix routing-rag-v2 corpus in the isolated evaluator."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, UUID, uuid5

from rag_kb.domain import GRAPH_EXTRACTOR_VERSION, GRAPH_RETRIEVAL_PROFILE_VERSION
from tools.evaluation_runtime import (
    AdaptiveGraphIdentity,
    DEFAULT_RUNTIME_MANIFEST,
    EvaluationRuntime,
    EvaluationRuntimeError,
    _runtime_value,
    _write_private,
    load_evaluation_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT / "evaluation/routing-rag-v2/documents"
EXPECTED_DOCUMENT_COUNT = 24
EVALUATION_KB_NAME = "routing-rag-v2-semantic-v4-graphiti-v2"
CONFIRMATION = "PROVISION_ROUTING_RAG_V2_POST_FIX"
MEDIA_TYPES = {".md": "text/markdown", ".txt": "text/plain"}


class ProvisioningError(RuntimeError):
    """The isolated post-fix evaluator cannot be provisioned safely."""


def _corpus_paths() -> tuple[Path, ...]:
    root = CORPUS_ROOT.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ProvisioningError("evaluation corpus directory is invalid")
    paths = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    if (
        len(paths) != EXPECTED_DOCUMENT_COUNT
        or any(
            path.is_symlink()
            or not path.is_file()
            or path.suffix.lower() not in MEDIA_TYPES
            for path in paths
        )
    ):
        raise ProvisioningError("evaluation corpus file set is invalid")
    return paths


def _corpus_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    payload: Mapping[str, object] | None = None,
    body: bytes | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json", **dict(headers or {})}
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        with urlopen(
            Request(url, data=body, headers=request_headers, method=method),
            timeout=timeout_seconds,
        ) as response:
            value = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ProvisioningError("evaluation API request failed") from error
    if not isinstance(value, dict):
        raise ProvisioningError("evaluation API response is invalid")
    return value


def _paged_items(api: str, resource: str) -> tuple[dict[str, Any], ...]:
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        query = urlencode({"limit": 100, **({"cursor": cursor} if cursor else {})})
        value = _request(f"{api}/{resource}?{query}")
        page = value.get("items")
        if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
            raise ProvisioningError("evaluation API page is invalid")
        items.extend(page)
        cursor = value.get("next_cursor")
        if cursor is None:
            return tuple(items)
        if not isinstance(cursor, str) or not cursor:
            raise ProvisioningError("evaluation API cursor is invalid")


def _require_current_runtime(runtime: EvaluationRuntime) -> None:
    capabilities = _request(f"{runtime.api_base_url}/retrieval/capabilities")
    rows = capabilities.get("modes")
    if not isinstance(rows, list) or not any(
        isinstance(row, dict)
        and row.get("mode") == "graph"
        and row.get("profile_version") == GRAPH_RETRIEVAL_PROFILE_VERSION
        and row.get("enabled") is True
        for row in rows
    ):
        raise ProvisioningError("evaluation API is not running the current Graph profile")


def _knowledge_base(runtime: EvaluationRuntime) -> dict[str, Any]:
    matches = tuple(
        row
        for row in _paged_items(runtime.api_base_url, "knowledge-bases")
        if row.get("name") == EVALUATION_KB_NAME and row.get("deleted_at") is None
    )
    if len(matches) > 1:
        raise ProvisioningError("evaluation knowledge base identity is ambiguous")
    if matches:
        knowledge_base = matches[0]
    else:
        key = uuid5(NAMESPACE_URL, EVALUATION_KB_NAME)
        knowledge_base = _request(
            f"{runtime.api_base_url}/knowledge-bases",
            method="POST",
            headers={"Idempotency-Key": str(key)},
            payload={
                "name": EVALUATION_KB_NAME,
                "parsing": {"preset": "text_local_v1"},
                "chunking": {"preset": "semantic_balanced_v1"},
                "retrieval_defaults": {
                    "strategy": "exact_vector",
                    "top_k": 10,
                    "rerank_mode": "classic",
                },
                "embedding": {"strategy": "text_only"},
            },
        )
    if knowledge_base.get("chunking", {}).get("profile") != "semantic_breakpoint_v4":
        raise ProvisioningError("evaluation knowledge base is not semantic v4")
    return knowledge_base


def _upload(runtime: EvaluationRuntime, kb_id: str, path: Path) -> dict[str, Any]:
    metadata = base64.urlsafe_b64encode(
        json.dumps(
            {"v": 1, "filename": path.name, "display_name": path.name},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    key = uuid5(NAMESPACE_URL, f"{EVALUATION_KB_NAME}:{path.name}")
    return _request(
        f"{runtime.api_base_url}/knowledge-bases/{kb_id}/documents",
        method="POST",
        headers={
            "Content-Type": MEDIA_TYPES[path.suffix.lower()],
            "Idempotency-Key": str(key),
            "X-Document-Metadata": metadata,
        },
        body=path.read_bytes(),
    )


def _ensure_documents(
    runtime: EvaluationRuntime,
    kb_id: str,
    paths: tuple[Path, ...],
) -> None:
    resource = f"knowledge-bases/{kb_id}/documents"
    existing = _paged_items(runtime.api_base_url, resource)
    filenames = {
        str(row.get("current_version", {}).get("original_filename", ""))
        for row in existing
        if row.get("deleted_at") is None
    }
    expected = {path.name for path in paths}
    if filenames - expected:
        raise ProvisioningError("evaluation knowledge base contains unexpected documents")
    for path in paths:
        if path.name not in filenames:
            _upload(runtime, kb_id, path)


def _wait_for_indexing(
    runtime: EvaluationRuntime,
    kb_id: str,
    *,
    timeout_seconds: float,
) -> str:
    started = time.monotonic()
    while True:
        jobs = _paged_items(
            runtime.api_base_url,
            f"knowledge-bases/{kb_id}/indexing-jobs",
        )
        if any(job.get("status") in {"failed", "cancelled"} for job in jobs):
            raise ProvisioningError("evaluation indexing failed")
        completed = tuple(job for job in jobs if job.get("status") == "completed")
        revision_ids = {
            str(job.get("index_revision_id", "")) for job in completed
        }
        if len(completed) == EXPECTED_DOCUMENT_COUNT and len(revision_ids) == 1:
            return revision_ids.pop()
        if len(jobs) > EXPECTED_DOCUMENT_COUNT:
            raise ProvisioningError("evaluation indexing job set is ambiguous")
        if time.monotonic() - started >= timeout_seconds:
            raise ProvisioningError("evaluation indexing timed out")
        time.sleep(2.0)


def _wait_for_graph(
    runtime: EvaluationRuntime,
    kb_id: str,
    *,
    answer_profile_revision_id: UUID,
    timeout_seconds: float,
    retry_failed: bool,
    force_rebuild_failed: bool,
) -> dict[str, Any]:
    url = f"{runtime.api_base_url}/knowledge-bases/{kb_id}/graph-config"
    config = _request(url)
    status = config.get("status")
    if status == "disabled":
        config = _request(
            url,
            method="PUT",
            payload={
                "enabled": True,
                "chat_profile_revision_id": str(answer_profile_revision_id),
            },
        )
    elif status == "failed" and force_rebuild_failed:
        config = _request(
            url,
            method="PUT",
            payload={"enabled": True, "retry": True, "force_rebuild": True},
        )
    elif status == "failed" and retry_failed:
        config = _request(
            url,
            method="PUT",
            payload={"enabled": True, "retry": True},
        )
    elif status == "failed":
        raise ProvisioningError("evaluation Graph build failed; retry was not authorized")

    started = time.monotonic()
    while config.get("status") != "ready":
        if config.get("status") == "failed":
            raise ProvisioningError("evaluation Graph build failed")
        if time.monotonic() - started >= timeout_seconds:
            raise ProvisioningError("evaluation Graph build timed out")
        time.sleep(2.0)
        config = _request(url)
    eligible_chunk_count = config.get("eligible_chunk_count")
    if (
        config.get("extractor_version") != GRAPH_EXTRACTOR_VERSION
        or not isinstance(eligible_chunk_count, int)
        or isinstance(eligible_chunk_count, bool)
        or eligible_chunk_count <= 0
        or config.get("processed_chunk_count") != eligible_chunk_count
        or not isinstance(config.get("build_id"), str)
    ):
        raise ProvisioningError("evaluation Graph build is incomplete")
    return config


def _bind_runtime(
    runtime: EvaluationRuntime,
    *,
    knowledge_base_id: UUID,
    index_revision_id: UUID,
    graph_build_id: UUID,
) -> None:
    previous = runtime.adaptive_graph
    if previous is None:
        raise EvaluationRuntimeError("evaluation answer/judge identity is unavailable")
    identity = AdaptiveGraphIdentity(
        workspace_id=previous.workspace_id,
        knowledge_base_id=knowledge_base_id,
        index_revision_id=index_revision_id,
        graph_build_id=graph_build_id,
        answer_profile_revision_id=previous.answer_profile_revision_id,
        judge_profile_revision_id=previous.judge_profile_revision_id,
    )
    payload = (
        json.dumps(
            _runtime_value(
                owner=runtime.owner,
                build_revision=runtime.build_revision,
                adaptive_graph=identity,
            ),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    _write_private(DEFAULT_RUNTIME_MANIFEST, payload, replace=True)


def provision(
    *,
    confirmation: str,
    timeout_seconds: float,
    retry_failed_graph: bool,
    force_rebuild_failed_graph: bool,
) -> dict[str, object]:
    if confirmation != CONFIRMATION:
        raise ProvisioningError("evaluation provisioning confirmation is invalid")
    runtime = load_evaluation_runtime(require_adaptive_graph=True)
    _require_current_runtime(runtime)
    paths = _corpus_paths()
    corpus_digest = _corpus_digest(paths)
    knowledge_base = _knowledge_base(runtime)
    kb_id = str(UUID(str(knowledge_base["id"])))
    _ensure_documents(runtime, kb_id, paths)
    index_revision_id = UUID(
        _wait_for_indexing(
            runtime,
            kb_id,
            timeout_seconds=timeout_seconds,
        )
    )
    config = _wait_for_graph(
        runtime,
        kb_id,
        answer_profile_revision_id=runtime.adaptive_graph.answer_profile_revision_id,
        timeout_seconds=timeout_seconds,
        retry_failed=retry_failed_graph,
        force_rebuild_failed=force_rebuild_failed_graph,
    )
    graph_build_id = UUID(str(config["build_id"]))
    _bind_runtime(
        runtime,
        knowledge_base_id=UUID(kb_id),
        index_revision_id=index_revision_id,
        graph_build_id=graph_build_id,
    )
    return {
        "status": "ready",
        "corpus_digest": corpus_digest,
        "document_count": EXPECTED_DOCUMENT_COUNT,
        "knowledge_base_id": kb_id,
        "index_revision_id": str(index_revision_id),
        "graph_build_id": str(graph_build_id),
        "extractor_version": GRAPH_EXTRACTOR_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--retry-failed-graph", action="store_true")
    parser.add_argument("--force-rebuild-failed-graph", action="store_true")
    arguments = parser.parse_args()
    if not 60.0 <= arguments.timeout_seconds <= 14_400.0:
        parser.error("timeout must be between 60 and 14400 seconds")
    if arguments.retry_failed_graph and arguments.force_rebuild_failed_graph:
        parser.error("resume and force rebuild are mutually exclusive")
    try:
        result = provision(
            confirmation=arguments.confirm,
            timeout_seconds=arguments.timeout_seconds,
            retry_failed_graph=arguments.retry_failed_graph,
            force_rebuild_failed_graph=arguments.force_rebuild_failed_graph,
        )
    except (EvaluationRuntimeError, ProvisioningError, OSError, ValueError):
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "evaluation_provisioning_failed",
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
