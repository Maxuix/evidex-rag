#!/usr/bin/env python3
"""Provision one frozen large-evaluation corpus into an isolated host runtime.

This is deliberately a host-native control plane.  It neither starts nor stops
PostgreSQL, FalkorDB, Docker, or the user's application runtime.  The caller
must supply the already-running isolated API and Worker through the canonical
owner-only evaluation runtime manifest.  Every phase writes a private,
atomic checkpoint so rerunning the same command resumes safely.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, UUID, uuid5

from rag_kb.domain import (
    ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
    ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
    GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST,
    GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
    GRAPH_EXTRACTOR_VERSION,
)
from tools.evaluation_campaign_state import digest, write_private_json
from tools.evaluation_resilience import (
    GRAPH_BUILD_MAX_RESUMES,
    GRAPH_BUILD_RETRY_POLICY,
    HEARTBEAT_INTERVAL_SECONDS,
    RESILIENCE_POLICY,
    RESILIENCE_POLICY_SHA256,
    seconds_until,
    timestamp_after,
)
from tools.evaluation_runtime import (
    EvaluationRuntime,
    EvaluationRuntimeError,
    canonical_evaluation_runtime_manifest,
    load_evaluation_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SCHEMA = "large_evaluation_provisioning_v1"
BINDINGS_SCHEMA = "large_evaluation_runtime_bindings_v1"
# Schema-echo failures are provider-transient and a resumed immutable build has
# repeatedly progressed without re-ingestion.  Keep a generous *bounded*
# control-plane budget so a long frozen corpus does not require manual restarts.
MAX_GRAPH_BUILD_RESUMES = GRAPH_BUILD_MAX_RESUMES
MAX_INDEXING_RETRIES = 3
MAX_PROVISIONING_TIMEOUT_SECONDS = 172_800.0
PUBLIC_DATASET_ID = "public-rag-benchmark-suite-v1"
PARSER_FALLBACK_CHECK = "inline_child_role"


class ProvisioningError(RuntimeError):
    """A stable, content-safe failure while provisioning frozen evaluation data."""


@contextmanager
def _dataset_run_lock(runtime_path: Path, dataset_id: str) -> Iterator[None]:
    """Prevent concurrent provisioners from racing one persisted checkpoint."""
    lock_path = (
        runtime_path.parent
        / "large-evaluation-provisioning"
        / f"{dataset_id}.lock"
    )
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProvisioningError("provisioning_dataset_already_running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class ProvisioningSpec:
    dataset_id: str
    corpus_root: Path
    expected_document_count: int
    knowledge_base_name: str
    confirmation: str
    graph_schema_key: str | None = None
    graph_schema_digest: str | None = None

    @property
    def graph_enabled(self) -> bool:
        return self.graph_schema_key is not None


SPECS = {
    "graph_rag": ProvisioningSpec(
        dataset_id="graph-rag-v1",
        corpus_root=ROOT / "evaluation/graph-rag-v1/documents",
        expected_document_count=16,
        knowledge_base_name="large-evaluation-graph-rag-v1-graphiti-v4",
        confirmation="PROVISION_LARGE_EVALUATION_GRAPH_RAG",
        graph_schema_key=ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
        graph_schema_digest=ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
    ),
    "routing": ProvisioningSpec(
        dataset_id="routing-rag-musique-expanded-v1",
        corpus_root=ROOT / "evaluation/routing-rag-musique-expanded-v1/documents",
        expected_document_count=578,
        knowledge_base_name="large-evaluation-routing-musique-v1-graphiti-v4",
        confirmation="PROVISION_LARGE_EVALUATION_ROUTING",
        graph_schema_key=GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
        graph_schema_digest=GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST,
    ),
    "enterprise": ProvisioningSpec(
        dataset_id="enterprise-profile-qualification-v1",
        corpus_root=ROOT / "evaluation/enterprise-profile-qualification-v1/documents",
        expected_document_count=138,
        knowledge_base_name="large-evaluation-enterprise-profile-v1-graphiti-v4",
        confirmation="PROVISION_LARGE_EVALUATION_ENTERPRISE",
        graph_schema_key=ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
        graph_schema_digest=ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
    ),
    "public": ProvisioningSpec(
        dataset_id="public-rag-benchmark-suite-v1",
        corpus_root=ROOT / "evaluation/public-rag-benchmark-suite-v1/documents",
        expected_document_count=1498,
        knowledge_base_name="large-evaluation-public-rag-v1-text-only",
        confirmation="PROVISION_LARGE_EVALUATION_PUBLIC",
    ),
}

# This profile is provisioned through the same host-native API/Worker and
# model bindings, but has its own corpus identity, KB, checkpoint, and
# optional bindings file.  Keeping it outside SPECS preserves the frozen v1
# four-suite contract used by existing campaigns.
ROUTING_VARIANT_SPEC = ProvisioningSpec(
    dataset_id="routing-rag-musique-short-support-v2",
    corpus_root=ROOT / "evaluation/routing-rag-musique-short-support-v2/documents",
    expected_document_count=1441,
    knowledge_base_name="large-evaluation-routing-musique-short-support-v2-graphiti-v4",
    confirmation="PROVISION_LARGE_EVALUATION_ROUTING_VARIANT",
    graph_schema_key=GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
    graph_schema_digest=GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST,
)
# This case-only variant intentionally points at the completed expanded-v1
# document/index/Graph identity.  It is included in the identity registry so
# the runtime checker can validate an explicit reuse binding without treating
# the evaluator-only case expansion as a new ingestion run.
ROUTING_REUSE_SPEC = ProvisioningSpec(
    dataset_id="routing-rag-musique-reuse-v2",
    corpus_root=ROOT / "evaluation/routing-rag-musique-reuse-v2/documents",
    expected_document_count=578,
    knowledge_base_name="large-evaluation-routing-musique-reuse-v2-graphiti-v4",
    confirmation="PROVISION_LARGE_EVALUATION_ROUTING_REUSE",
    graph_schema_key=GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
    graph_schema_digest=GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST,
)
DATASET_SPECS = {
    **SPECS,
    "routing_variant": ROUTING_VARIANT_SPEC,
    "routing_reuse": ROUTING_REUSE_SPEC,
}


def _spec_for_dataset(dataset: str) -> ProvisioningSpec:
    try:
        return DATASET_SPECS[dataset]
    except KeyError as error:
        raise ProvisioningError("provisioning_dataset_invalid") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=14_400.0)
    parser.add_argument(
        "--evaluation-runtime",
        type=Path,
        default=canonical_evaluation_runtime_manifest(),
    )
    parser.add_argument(
        "--chat-profile-revision-id",
        type=UUID,
        help=(
            "the already validated isolated OpenCode Go mimo-v2.5 profile; "
            "required when provisioning a Graph suite"
        ),
    )
    parser.add_argument(
        "--judge-profile-revision-id",
        type=UUID,
        help="validated isolated chat profile for the final-answer judge; defaults to the chat profile",
    )
    parser.add_argument(
        "--bindings-path",
        type=Path,
        help=(
            "optional owner-only bindings file under the evaluation runtime; "
            "use a separate file for an independent corpus variant"
        ),
    )
    parser.add_argument(
        "--stop-after-indexing",
        action="store_true",
        help=(
            "complete only the indexing phase and return zero; Graph can be "
            "started later from the same immutable checkpoint"
        ),
    )
    return parser


def _paths(spec: ProvisioningSpec) -> tuple[Path, ...]:
    root = spec.corpus_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ProvisioningError("provisioning_corpus_directory_invalid")
    paths = tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.suffix.lower() in {".md", ".txt"}
            ),
            key=lambda value: value.name,
        )
    )
    if (
        len(paths) != spec.expected_document_count
        or any(path.is_symlink() or not path.is_file() for path in paths)
        or len({path.name for path in paths}) != len(paths)
    ):
        raise ProvisioningError("provisioning_corpus_file_set_invalid")
    return paths


def _corpus_digest(paths: tuple[Path, ...]) -> str:
    value = hashlib.sha256()
    for path in paths:
        value.update(path.name.encode("utf-8"))
        value.update(b"\0")
        value.update(hashlib.sha256(path.read_bytes()).digest())
    return value.hexdigest()


def _content_digest(path: Path) -> str:
    """Return the exact upload checksum used by the document API."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    except HTTPError as error:
        raise ProvisioningError(f"provisioning_api_http_{error.code}") from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ProvisioningError("provisioning_api_unavailable") from error
    if not isinstance(value, dict):
        raise ProvisioningError("provisioning_api_response_invalid")
    return value


def _paged_items(api_base_url: str, resource: str) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        query = urlencode({"limit": 100, **({"cursor": cursor} if cursor else {})})
        page = _request(f"{api_base_url}/{resource}?{query}")
        items = page.get("items")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ProvisioningError("provisioning_api_page_invalid")
        result.extend(items)
        cursor = page.get("next_cursor")
        if cursor is None:
            return tuple(result)
        if not isinstance(cursor, str) or not cursor:
            raise ProvisioningError("provisioning_api_cursor_invalid")


def _checkpoint_path(runtime: EvaluationRuntime, spec: ProvisioningSpec) -> Path:
    return runtime.runtime_root / "large-evaluation-provisioning" / f"{spec.dataset_id}.json"


def _bindings_path(
    runtime: EvaluationRuntime, custom_path: Path | None = None
) -> Path:
    default = runtime.runtime_root / "large-evaluation-bindings.json"
    if custom_path is None:
        return default
    path = custom_path if custom_path.is_absolute() else runtime.runtime_root / custom_path
    path = path.absolute()
    if path.parent != runtime.runtime_root.absolute() or path.name in {"", ".", ".."}:
        raise ProvisioningError("provisioning_bindings_path_invalid")
    if path.exists() and (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_mode & 0o077
    ):
        raise ProvisioningError("provisioning_bindings_permissions_invalid")
    return path


def _record_suite_binding(
    runtime: EvaluationRuntime,
    *,
    spec: ProvisioningSpec,
    checkpoint: Mapping[str, Any],
    bindings_path: Path | None = None,
) -> None:
    """Atomically record only a durably completed corpus identity."""
    required = {"knowledge_base_id", "index_revision_id"}
    if spec.graph_enabled:
        required.add("graph_build_id")
    if checkpoint.get("status") != "completed" or not all(
        isinstance(checkpoint.get(name), str) for name in required
    ):
        raise ProvisioningError("provisioning_binding_incomplete")
    path = _bindings_path(runtime, bindings_path)
    bindings: dict[str, Any]
    if path.exists():
        if not path.is_file() or path.stat().st_mode & 0o077:
            raise ProvisioningError("provisioning_bindings_permissions_invalid")
        try:
            bindings = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ProvisioningError("provisioning_bindings_json_invalid") from error
        if not isinstance(bindings, dict) or bindings.get("schema_version") != BINDINGS_SCHEMA:
            raise ProvisioningError("provisioning_bindings_schema_invalid")
    else:
        default_path = _bindings_path(runtime)
        if path != default_path and default_path.is_file():
            try:
                bindings = json.loads(default_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ProvisioningError("provisioning_bindings_json_invalid") from error
            if (
                not isinstance(bindings, dict)
                or bindings.get("schema_version") != BINDINGS_SCHEMA
                or not isinstance(bindings.get("suites"), dict)
                or bindings.get("binding_sha256")
                != digest({"suites": bindings["suites"]})
            ):
                raise ProvisioningError("provisioning_bindings_schema_invalid")
        else:
            bindings = {"schema_version": BINDINGS_SCHEMA, "suites": {}}
    suites = bindings.get("suites")
    if not isinstance(suites, dict):
        raise ProvisioningError("provisioning_bindings_schema_invalid")
    suites[spec.dataset_id] = {
        "corpus_sha256": checkpoint["binding"]["corpus_sha256"],
        "knowledge_base_id": checkpoint["knowledge_base_id"],
        "index_revision_id": checkpoint["index_revision_id"],
        "graph_build_id": checkpoint.get("graph_build_id"),
        "graph_schema_key": spec.graph_schema_key,
        "graph_schema_digest": spec.graph_schema_digest,
        "frozen_document_count": checkpoint.get("frozen_document_count"),
        "indexed_document_count": checkpoint.get("indexed_document_count"),
        "deduplicated_document_count": checkpoint.get(
            "deduplicated_document_count", 0
        ),
        "parser_fallback_document_count": checkpoint.get(
            "parser_fallback_document_count", 0
        ),
    }
    bindings["binding_sha256"] = digest({"suites": suites})
    write_private_json(path, bindings)


def _load_checkpoint(
    path: Path,
    *,
    spec: ProvisioningSpec,
    corpus_digest: str,
) -> dict[str, Any]:
    binding = {"dataset_id": spec.dataset_id, "corpus_sha256": corpus_digest}
    if not path.exists():
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "binding": binding,
            "binding_sha256": digest(binding),
            "status": "started",
            "events": [],
        }
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ProvisioningError("provisioning_checkpoint_permissions_invalid")
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProvisioningError("provisioning_checkpoint_json_invalid") from error
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("binding") != binding
        or checkpoint.get("binding_sha256") != digest(binding)
        or not isinstance(checkpoint.get("events"), list)
    ):
        raise ProvisioningError("provisioning_checkpoint_binding_invalid")
    return checkpoint


def _write_checkpoint(path: Path, checkpoint: dict[str, Any], event: str, **fields: Any) -> None:
    checkpoint["events"].append({"event": event, **fields})
    write_private_json(path, checkpoint)


def _knowledge_base(runtime: EvaluationRuntime, spec: ProvisioningSpec) -> dict[str, Any]:
    matches = tuple(
        item
        for item in _paged_items(runtime.api_base_url, "knowledge-bases")
        if item.get("name") == spec.knowledge_base_name
        and item.get("deleted_at") is None
    )
    if len(matches) > 1:
        raise ProvisioningError("provisioning_knowledge_base_ambiguous")
    if matches:
        knowledge_base = matches[0]
    else:
        knowledge_base = _request(
            f"{runtime.api_base_url}/knowledge-bases",
            method="POST",
            headers={"Idempotency-Key": str(uuid5(NAMESPACE_URL, spec.knowledge_base_name))},
            payload={
                "name": spec.knowledge_base_name,
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
        raise ProvisioningError("provisioning_knowledge_base_chunking_invalid")
    return knowledge_base


def _upload(runtime: EvaluationRuntime, *, kb_id: str, path: Path, spec: ProvisioningSpec) -> None:
    metadata = base64.urlsafe_b64encode(
        json.dumps(
            {"v": 1, "filename": path.name, "display_name": path.name},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    _request(
        f"{runtime.api_base_url}/knowledge-bases/{kb_id}/documents",
        method="POST",
        headers={
            "Content-Type": (
                "text/markdown" if path.suffix.lower() == ".md" else "text/plain"
            ),
            "Idempotency-Key": str(uuid5(NAMESPACE_URL, f"{spec.dataset_id}:{path.name}")),
            "X-Document-Metadata": metadata,
        },
        body=path.read_bytes(),
    )


def _upload_plain_text_version(
    runtime: EvaluationRuntime,
    *,
    document_id: str,
    path: Path,
    parser_filename: str,
    spec: ProvisioningSpec,
) -> dict[str, Any]:
    """Create a same-byte text version for a deterministic Markdown projection defect.

    The frozen path is never rewritten.  The alternate filename is only the
    persisted parser dispatch hint; the checkpoint binds it back to the
    frozen filename and records the exact source checksum.
    """

    metadata = base64.urlsafe_b64encode(
        json.dumps(
            {
                "v": 1,
                "filename": parser_filename,
                "display_name": path.name,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return _request(
        f"{runtime.api_base_url}/documents/{document_id}/versions",
        method="POST",
        headers={
            "Content-Type": "text/plain",
            "Idempotency-Key": str(
                uuid5(
                    NAMESPACE_URL,
                    f"{spec.dataset_id}:parser-fallback:{path.name}",
                )
            ),
            "X-Document-Metadata": metadata,
        },
        body=path.read_bytes(),
    )


def _ensure_documents(
    runtime: EvaluationRuntime,
    *,
    kb_id: str,
    paths: tuple[Path, ...],
    spec: ProvisioningSpec,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> frozenset[str]:
    resource = f"knowledge-bases/{kb_id}/documents"
    existing = _paged_items(runtime.api_base_url, resource)
    existing_names = {
        str(item.get("current_version", {}).get("original_filename", ""))
        for item in existing
        if item.get("deleted_at") is None
    }
    expected_names = {path.name for path in paths}
    parser_aliases = checkpoint.setdefault("parser_content_aliases", {})
    if not isinstance(parser_aliases, dict):
        raise ProvisioningError("provisioning_checkpoint_parser_aliases_invalid")
    parser_targets = set(parser_aliases.values())
    if existing_names - expected_names - parser_targets:
        raise ProvisioningError("provisioning_knowledge_base_contains_unexpected_documents")
    existing_checksums = {
        str(item.get("current_version", {}).get("checksum_sha256", "")): str(
            item.get("current_version", {}).get("original_filename", "")
        )
        for item in existing
        if item.get("deleted_at") is None
        and item.get("current_version", {}).get("checksum_sha256")
        and item.get("current_version", {}).get("original_filename")
    }
    aliases = checkpoint.setdefault("duplicate_content_aliases", {})
    if not isinstance(aliases, dict):
        raise ProvisioningError("provisioning_checkpoint_duplicate_aliases_invalid")
    deduplicated = int(checkpoint.get("deduplicated_document_count", 0))
    uploaded = len(existing_names)
    for path in paths:
        if path.name in parser_aliases:
            continue
        if path.name in existing_names:
            continue
        checksum = _content_digest(path)
        existing_name = existing_checksums.get(checksum)
        if existing_name is not None:
            # The product contract rejects byte-identical documents within one
            # KB.  Keep the frozen file in the corpus binding, but serve one
            # physical document and persist the identity-preserving alias so
            # evaluators can resolve either frozen filename to the same bytes.
            prior = aliases.get(path.name)
            if prior is not None:
                if prior != existing_name:
                    raise ProvisioningError("provisioning_duplicate_alias_changed")
                continue
            aliases[path.name] = existing_name
            deduplicated += 1
            checkpoint["status"] = "uploading"
            checkpoint["deduplicated_document_count"] = deduplicated
            _write_checkpoint(
                checkpoint_path,
                checkpoint,
                "document_deduplicated",
                filename=path.name,
                existing_filename=existing_name,
                count=deduplicated,
            )
            print(
                json.dumps(
                    {
                        "event": "document_deduplicated",
                        "dataset_id": spec.dataset_id,
                        "count": deduplicated,
                    }
                ),
                flush=True,
            )
            continue
        _upload(runtime, kb_id=kb_id, path=path, spec=spec)
        uploaded += 1
        existing_checksums[checksum] = path.name
        checkpoint["status"] = "uploading"
        checkpoint["uploaded_document_count"] = uploaded
        _write_checkpoint(checkpoint_path, checkpoint, "document_uploaded", count=uploaded)
        print(json.dumps({"event": "document_uploaded", "dataset_id": spec.dataset_id, "count": uploaded}), flush=True)
    current = _paged_items(runtime.api_base_url, resource)
    current_by_name = {
        str(item.get("current_version", {}).get("original_filename", "")): item
        for item in current
        if item.get("deleted_at") is None
    }
    if (
        set(current_by_name) | set(aliases) | set(parser_aliases)
        != expected_names
    ):
        raise ProvisioningError("provisioning_document_set_incomplete")
    if any(
        alias not in expected_names
        or target not in current_by_name
        or alias == target
        for alias, target in aliases.items()
    ):
        raise ProvisioningError("provisioning_duplicate_alias_invalid")
    if any(
        alias not in expected_names
        or target not in current_by_name
        or alias == target
        or target in expected_names
        for alias, target in parser_aliases.items()
    ):
        raise ProvisioningError("provisioning_parser_alias_invalid")
    version_ids = frozenset(
        str(item.get("current_version", {}).get("id", ""))
        for item in current_by_name.values()
    )
    if len(version_ids) != len(current_by_name) or "" in version_ids:
        raise ProvisioningError("provisioning_document_versions_invalid")
    return version_ids


def _wait_for_indexing(
    runtime: EvaluationRuntime,
    *,
    kb_id: str,
    expected_version_ids: frozenset[str],
    timeout_seconds: float,
    spec: ProvisioningSpec,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> UUID:
    started = time.monotonic()
    previous: tuple[int, int, int] | None = None
    retry_count = int(checkpoint.get("indexing_retry_count", 0))
    path_by_name = {path.name: path for path in _paths(spec)}
    while True:
        jobs = tuple(
            job
            for job in _paged_items(runtime.api_base_url, f"knowledge-bases/{kb_id}/indexing-jobs")
            if str(job.get("document_version_id", "")) in expected_version_ids
        )
        completed = tuple(job for job in jobs if job.get("status") == "completed")
        failed = tuple(job for job in jobs if job.get("status") in {"failed", "cancelled"})
        snapshot = (len(jobs), len(completed), len(failed))
        if snapshot != previous:
            checkpoint["status"] = "indexing"
            checkpoint["indexing"] = {
                "job_count": snapshot[0],
                "completed_count": snapshot[1],
                "failed_count": snapshot[2],
            }
            _write_checkpoint(checkpoint_path, checkpoint, "indexing_progress", completed=snapshot[1], failed=snapshot[2])
            print(json.dumps({"event": "indexing_progress", "dataset_id": spec.dataset_id, "completed": snapshot[1], "failed": snapshot[2]}), flush=True)
            previous = snapshot
        if failed:
            retryable = tuple(
                job
                for job in failed
                if job.get("can_retry") is not False and job.get("job_id")
            )
            if not retryable:
                raise ProvisioningError("provisioning_indexing_failed")
            if retry_count >= MAX_INDEXING_RETRIES:
                parser_aliases = checkpoint.setdefault(
                    "parser_content_aliases", {}
                )
                if not isinstance(parser_aliases, dict):
                    raise ProvisioningError(
                        "provisioning_checkpoint_parser_aliases_invalid"
                    )
                documents = _paged_items(
                    runtime.api_base_url,
                    f"knowledge-bases/{kb_id}/documents",
                )
                document_by_id = {
                    str(item.get("id")): item
                    for item in documents
                    if item.get("id")
                }
                recovered = False
                for job in retryable:
                    error = job.get("error")
                    detail = error.get("detail") if isinstance(error, dict) else None
                    if (
                        spec.dataset_id != PUBLIC_DATASET_ID
                        or not isinstance(error, dict)
                        or error.get("code") != "PARSER_OUTPUT_INVALID"
                        or not isinstance(detail, dict)
                        or detail.get("check") != PARSER_FALLBACK_CHECK
                    ):
                        continue
                    document = document_by_id.get(str(job.get("document_id")))
                    current_version = (
                        document.get("current_version")
                        if isinstance(document, dict)
                        else None
                    )
                    frozen_name = (
                        current_version.get("original_filename")
                        if isinstance(current_version, dict)
                        else None
                    )
                    path = path_by_name.get(str(frozen_name))
                    if path is None or path.suffix.lower() != ".md":
                        continue
                    parser_filename = f"{path.stem}.txt"
                    if parser_filename in path_by_name or (
                        parser_filename in {
                            str(item.get("current_version", {}).get("original_filename", ""))
                            for item in documents
                        }
                        and parser_aliases.get(path.name) != parser_filename
                    ):
                        raise ProvisioningError(
                            "provisioning_parser_fallback_filename_collision"
                        )
                    prior = parser_aliases.get(path.name)
                    if prior is not None and prior != parser_filename:
                        raise ProvisioningError(
                            "provisioning_parser_fallback_alias_changed"
                        )
                    response = _upload_plain_text_version(
                        runtime,
                        document_id=str(job["document_id"]),
                        path=path,
                        parser_filename=parser_filename,
                        spec=spec,
                    )
                    new_version_id = response.get("document_version_id")
                    if not isinstance(new_version_id, str) or not new_version_id:
                        raise ProvisioningError(
                            "provisioning_parser_fallback_response_invalid"
                        )
                    parser_aliases[path.name] = parser_filename
                    checkpoint["parser_fallback_document_count"] = len(
                        parser_aliases
                    )
                    checkpoint["indexing_retry_count"] = 0
                    _write_checkpoint(
                        checkpoint_path,
                        checkpoint,
                        "document_parser_fallback",
                        filename=path.name,
                        parser_filename=parser_filename,
                        source_checksum_sha256=_content_digest(path),
                        reason=PARSER_FALLBACK_CHECK,
                    )
                    expected_version_ids = frozenset(
                        (*(
                            version_id
                            for version_id in expected_version_ids
                            if version_id != str(job.get("document_version_id"))
                        ), new_version_id)
                    )
                    retry_count = 0
                    recovered = True
                    print(
                        json.dumps(
                            {
                                "event": "document_parser_fallback",
                                "dataset_id": spec.dataset_id,
                                "parser_filename": parser_filename,
                            }
                        ),
                        flush=True,
                    )
                if recovered:
                    previous = None
                    continue
                raise ProvisioningError("provisioning_indexing_retry_exhausted")
            for job in retryable:
                job_id = str(job["job_id"])
                retry_count += 1
                _request(
                    f"{runtime.api_base_url}/indexing-jobs/{job_id}/retry",
                    method="POST",
                    headers={
                        "Idempotency-Key": str(
                            uuid5(
                                NAMESPACE_URL,
                                f"large-evaluation-index-retry:{spec.dataset_id}:{job_id}",
                            )
                        )
                    },
                )
                checkpoint["status"] = "indexing"
                checkpoint["indexing_retry_count"] = retry_count
                _write_checkpoint(
                    checkpoint_path,
                    checkpoint,
                    "indexing_job_retried",
                    job_id=job_id,
                    retry_count=retry_count,
                )
                print(
                    json.dumps(
                        {
                            "event": "indexing_job_retried",
                            "dataset_id": spec.dataset_id,
                            "retry_count": retry_count,
                        }
                    ),
                    flush=True,
                )
            previous = None
            continue
        revision_ids = {str(job.get("index_revision_id", "")) for job in completed}
        if len(completed) == len(expected_version_ids) and len(revision_ids) == 1:
            revision = revision_ids.pop()
            try:
                return UUID(revision)
            except ValueError as error:
                raise ProvisioningError("provisioning_index_revision_invalid") from error
        if len(jobs) > len(expected_version_ids):
            raise ProvisioningError("provisioning_indexing_jobs_ambiguous")
        if time.monotonic() - started >= timeout_seconds:
            raise ProvisioningError("provisioning_indexing_timed_out")
        time.sleep(2.0)


def _graph_config(
    runtime: EvaluationRuntime,
    *,
    kb_id: str,
    spec: ProvisioningSpec,
    chat_profile_revision_id: UUID,
) -> dict[str, Any]:
    if not spec.graph_enabled:
        raise ProvisioningError("provisioning_graph_not_requested")
    url = f"{runtime.api_base_url}/knowledge-bases/{kb_id}/graph-config"
    config = _request(url)
    active_matches = (
        config.get("schema_profile_key") == spec.graph_schema_key
        and config.get("schema_profile_digest") == spec.graph_schema_digest
        and config.get("extractor_version") == GRAPH_EXTRACTOR_VERSION
        and config.get("active_build_schema_profile_key") == spec.graph_schema_key
        and config.get("active_build_schema_profile_digest") == spec.graph_schema_digest
    )
    if config.get("status") == "ready" and active_matches:
        return config
    payload: dict[str, object]
    if config.get("status") == "failed":
        # A failed immutable build may safely resume only its uncommitted
        # chunk.  Do not rotate the build by default: doing so would throw
        # away all successfully mapped Graphiti episodes and would defeat the
        # checkpoint/recovery contract this tool exists to verify.
        payload = {"enabled": True, "retry": True, "force_rebuild": False}
    else:
        payload = {
            "enabled": True,
            "chat_profile_revision_id": str(chat_profile_revision_id),
            "schema_profile_key": spec.graph_schema_key,
            "force_rebuild": config.get("status") not in {"disabled", "building"},
        }
    return _request(url, method="PUT", payload=payload)


def _wait_for_graph(
    runtime: EvaluationRuntime,
    *,
    kb_id: str,
    spec: ProvisioningSpec,
    chat_profile_revision_id: UUID,
    timeout_seconds: float,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    config = _graph_config(
        runtime,
        kb_id=kb_id,
        spec=spec,
        chat_profile_revision_id=chat_profile_revision_id,
    )
    started = time.monotonic()
    previous: tuple[object, ...] | None = None
    resume_count = int(checkpoint.get("graph_resume_count", 0))
    consecutive_failures = int(
        checkpoint.get("graph_consecutive_failure_count", 0)
    )
    previous_processed = int(
        checkpoint.get("graph", {}).get("processed_chunk_count") or 0
    )
    checkpoint_policy = checkpoint.get("resilience_policy")
    if checkpoint_policy is not None and checkpoint_policy != RESILIENCE_POLICY:
        raise ProvisioningError("provisioning_resilience_policy_changed")
    checkpoint["resilience_policy"] = RESILIENCE_POLICY
    checkpoint["resilience_policy_sha256"] = RESILIENCE_POLICY_SHA256
    last_heartbeat = time.monotonic()
    url = f"{runtime.api_base_url}/knowledge-bases/{kb_id}/graph-config"
    while True:
        snapshot = (
            config.get("status"),
            config.get("eligible_chunk_count"),
            config.get("processed_chunk_count"),
            config.get("extracted_chunk_count"),
            config.get("last_error_code"),
        )
        if snapshot != previous:
            processed = snapshot[2] if isinstance(snapshot[2], int) else 0
            if processed > previous_processed:
                consecutive_failures = 0
                checkpoint["graph_consecutive_failure_count"] = 0
                previous_processed = processed
            checkpoint["status"] = "building_graph"
            checkpoint["graph"] = {
                "status": snapshot[0],
                "eligible_chunk_count": snapshot[1],
                "processed_chunk_count": snapshot[2],
                "extracted_chunk_count": snapshot[3],
                "last_error_code": snapshot[4],
            }
            _write_checkpoint(checkpoint_path, checkpoint, "graph_progress", status=snapshot[0], processed=snapshot[2])
            print(json.dumps({"event": "graph_progress", "dataset_id": spec.dataset_id, "status": snapshot[0], "processed": snapshot[2]}), flush=True)
            previous = snapshot
            last_heartbeat = time.monotonic()
        elif time.monotonic() - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
            checkpoint["graph_heartbeat_at"] = datetime.now(UTC).isoformat()
            write_private_json(checkpoint_path, checkpoint)
            last_heartbeat = time.monotonic()
        if config.get("status") == "failed":
            # Graphiti only marks the failed chunk after its PostgreSQL episode
            # mapping has not been committed.  Retrying the unchanged build is
            # therefore safe and preserves every completed episode.  Bound the
            # recovery loop so a deterministic provider/schema incompatibility
            # still fails closed with an inspectable checkpoint.
            if resume_count >= MAX_GRAPH_BUILD_RESUMES:
                raise ProvisioningError("provisioning_graph_build_retry_exhausted")
            resume_count += 1
            consecutive_failures += 1
            delay = GRAPH_BUILD_RETRY_POLICY.delay_after(consecutive_failures)
            next_retry_at = timestamp_after(delay)
            checkpoint["graph_resume_count"] = resume_count
            checkpoint["graph_consecutive_failure_count"] = consecutive_failures
            checkpoint["graph_next_retry_at"] = next_retry_at
            _write_checkpoint(
                checkpoint_path,
                checkpoint,
                "graph_build_resumed",
                resume_count=resume_count,
                consecutive_failure_count=consecutive_failures,
                retry_delay_seconds=delay,
                next_retry_at=next_retry_at,
            )
            print(
                json.dumps(
                    {
                        "event": "graph_build_resumed",
                        "dataset_id": spec.dataset_id,
                        "resume_count": resume_count,
                        "consecutive_failure_count": consecutive_failures,
                        "retry_delay_seconds": delay,
                        "next_retry_at": next_retry_at,
                    }
                ),
                flush=True,
            )
            while seconds_until(checkpoint.get("graph_next_retry_at")) > 0:
                time.sleep(
                    min(
                        HEARTBEAT_INTERVAL_SECONDS,
                        seconds_until(checkpoint.get("graph_next_retry_at")),
                    )
                )
                checkpoint["graph_heartbeat_at"] = datetime.now(UTC).isoformat()
                write_private_json(checkpoint_path, checkpoint)
            config = _graph_config(
                runtime,
                kb_id=kb_id,
                spec=spec,
                chat_profile_revision_id=chat_profile_revision_id,
            )
            continue
        valid = (
            config.get("status") == "ready"
            and config.get("schema_profile_key") == spec.graph_schema_key
            and config.get("schema_profile_digest") == spec.graph_schema_digest
            and config.get("extractor_version") == GRAPH_EXTRACTOR_VERSION
            and config.get("active_build_schema_profile_key") == spec.graph_schema_key
            and config.get("active_build_schema_profile_digest") == spec.graph_schema_digest
            and isinstance(config.get("eligible_chunk_count"), int)
            and config.get("eligible_chunk_count") > 0
            and config.get("processed_chunk_count") == config.get("eligible_chunk_count")
            and isinstance(config.get("build_id"), str)
        )
        if valid:
            checkpoint.pop("graph_next_retry_at", None)
            checkpoint["graph_consecutive_failure_count"] = 0
            return config
        if time.monotonic() - started >= timeout_seconds:
            raise ProvisioningError("provisioning_graph_build_timed_out")
        time.sleep(2.0)
        config = _request(url)


def _bind_graph_runtime(
    runtime: EvaluationRuntime,
    *,
    kb_id: UUID,
    index_revision_id: UUID,
    graph_build_id: UUID,
    spec: ProvisioningSpec,
    answer_profile_revision_id: UUID,
    judge_profile_revision_id: UUID,
) -> None:
    """Bind only a proven complete Graph build to the runtime manifest."""

    identity = runtime.adaptive_graph
    if identity is None or not spec.graph_enabled:
        raise ProvisioningError("provisioning_runtime_binding_unavailable")
    try:
        manifest = json.loads(runtime.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvisioningError("provisioning_runtime_manifest_invalid") from error
    manifest["adaptive_graph"] = {
        "workspace_id": str(identity.workspace_id),
        "knowledge_base_id": str(kb_id),
        "index_revision_id": str(index_revision_id),
        "graph_build_id": str(graph_build_id),
        "answer_profile_revision_id": str(answer_profile_revision_id),
        "judge_profile_revision_id": str(judge_profile_revision_id),
        "schema_profile_key": spec.graph_schema_key,
        "schema_profile_digest": spec.graph_schema_digest,
        "extractor_version": GRAPH_EXTRACTOR_VERSION,
    }
    write_private_json(runtime.manifest, manifest)


def provision(
    *,
    dataset: str,
    confirmation: str,
    timeout_seconds: float,
    runtime_path: Path,
    chat_profile_revision_id: UUID | None,
    judge_profile_revision_id: UUID | None,
    bindings_path: Path | None = None,
    stop_after_indexing: bool = False,
) -> dict[str, Any]:
    spec = _spec_for_dataset(dataset)
    if confirmation != spec.confirmation:
        raise ProvisioningError("provisioning_confirmation_invalid")
    if not 60.0 <= timeout_seconds <= MAX_PROVISIONING_TIMEOUT_SECONDS:
        raise ProvisioningError("provisioning_timeout_invalid")
    if spec.graph_enabled and chat_profile_revision_id is None:
        raise ProvisioningError("provisioning_chat_profile_required")
    runtime = load_evaluation_runtime(
        runtime_path,
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
    resolved_bindings_path = _bindings_path(runtime, bindings_path)
    paths = _paths(spec)
    corpus_sha256 = _corpus_digest(paths)
    checkpoint_path = _checkpoint_path(runtime, spec)
    checkpoint = _load_checkpoint(
        checkpoint_path,
        spec=spec,
        corpus_digest=corpus_sha256,
    )
    if checkpoint.get("status") == "completed":
        _record_suite_binding(
            runtime,
            spec=spec,
            checkpoint=checkpoint,
            bindings_path=resolved_bindings_path,
        )
        return {
            "status": "completed",
            "resumed": True,
            "dataset_id": spec.dataset_id,
            "checkpoint": str(checkpoint_path),
        }
    knowledge_base = _knowledge_base(runtime, spec)
    try:
        kb_id = UUID(str(knowledge_base["id"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ProvisioningError("provisioning_knowledge_base_identity_invalid") from error
    checkpoint["knowledge_base_id"] = str(kb_id)
    checkpoint["status"] = "knowledge_base_ready"
    _write_checkpoint(checkpoint_path, checkpoint, "knowledge_base_ready")
    versions = _ensure_documents(
        runtime,
        kb_id=str(kb_id),
        paths=paths,
        spec=spec,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    checkpoint["frozen_document_count"] = len(paths)
    checkpoint["indexed_document_count"] = len(versions)
    checkpoint.setdefault("deduplicated_document_count", 0)
    _write_checkpoint(checkpoint_path, checkpoint, "document_set_completed")
    index_revision_id = _wait_for_indexing(
        runtime,
        kb_id=str(kb_id),
        expected_version_ids=versions,
        timeout_seconds=timeout_seconds,
        spec=spec,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    checkpoint["index_revision_id"] = str(index_revision_id)
    _write_checkpoint(checkpoint_path, checkpoint, "indexing_completed")
    if stop_after_indexing:
        checkpoint["status"] = "indexed"
        _write_checkpoint(checkpoint_path, checkpoint, "indexing_stage_completed")
        return {
            "status": "indexed",
            "resumed": False,
            "dataset_id": spec.dataset_id,
            "document_count": len(paths),
            "indexed_document_count": len(versions),
            "deduplicated_document_count": checkpoint.get(
                "deduplicated_document_count", 0
            ),
            "parser_fallback_document_count": checkpoint.get(
                "parser_fallback_document_count", 0
            ),
            "knowledge_base_id": str(kb_id),
            "index_revision_id": str(index_revision_id),
            "checkpoint": str(checkpoint_path),
        }
    graph_build_id: UUID | None = None
    if spec.graph_enabled:
        assert chat_profile_revision_id is not None
        config = _wait_for_graph(
            runtime,
            kb_id=str(kb_id),
            spec=spec,
            chat_profile_revision_id=chat_profile_revision_id,
            timeout_seconds=timeout_seconds,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
        )
        try:
            graph_build_id = UUID(str(config["build_id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ProvisioningError("provisioning_graph_build_identity_invalid") from error
        _bind_graph_runtime(
            runtime,
            kb_id=kb_id,
            index_revision_id=index_revision_id,
            graph_build_id=graph_build_id,
            spec=spec,
            answer_profile_revision_id=chat_profile_revision_id,
            judge_profile_revision_id=(
                judge_profile_revision_id or chat_profile_revision_id
            ),
        )
        checkpoint["graph_build_id"] = str(graph_build_id)
    checkpoint["status"] = "completed"
    _write_checkpoint(checkpoint_path, checkpoint, "provisioning_completed")
    _record_suite_binding(
        runtime,
        spec=spec,
        checkpoint=checkpoint,
        bindings_path=resolved_bindings_path,
    )
    return {
        "status": "completed",
        "resumed": False,
        "dataset_id": spec.dataset_id,
        "document_count": len(paths),
        "indexed_document_count": len(versions),
        "deduplicated_document_count": checkpoint.get(
            "deduplicated_document_count", 0
        ),
        "parser_fallback_document_count": checkpoint.get(
            "parser_fallback_document_count", 0
        ),
        "knowledge_base_id": str(kb_id),
        "index_revision_id": str(index_revision_id),
        "graph_build_id": str(graph_build_id) if graph_build_id else None,
        "checkpoint": str(checkpoint_path),
    }


def main() -> int:
    arguments = _parser().parse_args()
    try:
        with _dataset_run_lock(
            arguments.evaluation_runtime.resolve(),
            _spec_for_dataset(arguments.dataset).dataset_id,
        ):
            result = provision(
                dataset=arguments.dataset,
                confirmation=arguments.confirm,
                timeout_seconds=arguments.timeout_seconds,
                runtime_path=arguments.evaluation_runtime,
                chat_profile_revision_id=arguments.chat_profile_revision_id,
                judge_profile_revision_id=arguments.judge_profile_revision_id,
                bindings_path=arguments.bindings_path,
                stop_after_indexing=arguments.stop_after_indexing,
            )
    except (EvaluationRuntimeError, ProvisioningError, OSError, ValueError) as error:
        print(json.dumps({"status": "blocked", "failure_code": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
