#!/usr/bin/env python3
"""Provision one frozen routing RAG corpus in the isolated evaluator."""

from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from apps.worker.dependencies import build_worker_dependencies
from rag_kb.adapters.file_store.local import LocalFileStore
from rag_kb.config import load_settings
from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    GRAPH_RETRIEVAL_PROFILE_VERSION,
    SourceFileDigest,
    SourceFileIdentity,
)
from rag_kb.graph import GraphConfigurationService
from rag_kb.observability import configure_logging
from rag_kb.scheduling.worker import consume_lane
from rag_kb.services.content import CREATE_DOCUMENT_ENDPOINT
from tools.evaluation_runtime import (
    AdaptiveGraphIdentity,
    EvaluationRuntime,
    EvaluationRuntimeError,
    _runtime_value,
    _write_private,
    canonical_evaluation_runtime_manifest,
    load_evaluation_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ProvisioningSpec:
    dataset_id: str
    corpus_root: Path
    expected_document_count: int
    knowledge_base_name: str
    confirmation: str
    media_types: Mapping[str, str]


V2_SPEC = ProvisioningSpec(
    dataset_id="routing-rag-v2",
    corpus_root=PROJECT_ROOT / "evaluation/routing-rag-v2/documents",
    expected_document_count=24,
    knowledge_base_name="routing-rag-v2-semantic-v4-graphiti-v2",
    confirmation="PROVISION_ROUTING_RAG_V2_POST_FIX",
    media_types={".md": "text/markdown", ".txt": "text/plain"},
)
V3_SPEC = ProvisioningSpec(
    dataset_id="routing-rag-v3-open-source",
    corpus_root=PROJECT_ROOT / "evaluation/routing-rag-v3/documents",
    expected_document_count=14,
    knowledge_base_name="routing-rag-v3-open-source-semantic-v4-graphiti-v3",
    confirmation="PROVISION_ROUTING_RAG_V3_OPEN_SOURCE",
    media_types={
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".csv": "text/csv",
    },
)
MUSIQUE_MINI_SPEC = ProvisioningSpec(
    dataset_id="routing-rag-musique-full-mini-v1",
    corpus_root=PROJECT_ROOT / "evaluation/routing-rag-musique-mini/documents",
    expected_document_count=80,
    knowledge_base_name="routing-rag-musique-full-mini-semantic-v4-graphiti-v3",
    confirmation="PROVISION_ROUTING_RAG_MUSIQUE_MINI",
    media_types={".md": "text/markdown"},
)
PROVISIONING_SPECS = {
    spec.dataset_id: spec for spec in (V2_SPEC, V3_SPEC, MUSIQUE_MINI_SPEC)
}

# Backward-compatible names used by the existing v2 unit contract.
CORPUS_ROOT = V2_SPEC.corpus_root
EXPECTED_DOCUMENT_COUNT = V2_SPEC.expected_document_count
EVALUATION_KB_NAME = V2_SPEC.knowledge_base_name
CONFIRMATION = V2_SPEC.confirmation
MEDIA_TYPES = V2_SPEC.media_types


class ProvisioningError(RuntimeError):
    """The isolated post-fix evaluator cannot be provisioned safely."""


class _HostIndexingDriver:
    """Consume only the isolated runtime's indexing/Graph lane on host Python."""

    def __init__(self, runtime: EvaluationRuntime) -> None:
        self._runtime = runtime
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._thread_main,
            name="rag-v3-host-indexing",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=30.0) or self._failed:
            raise ProvisioningError("evaluation host indexing worker failed to start")

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=30.0)
        self._thread = None
        if thread.is_alive() or self._failed:
            raise ProvisioningError("evaluation host indexing worker failed")

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception:
            self._failed = True
            self._ready.set()

    async def _run(self) -> None:
        configure_logging(
            level="INFO",
            process="v3-host-worker",
            log_directory=self._runtime.runtime_root / "logs",
        )
        settings = load_settings(env_file=self._runtime.env_file)
        host_text_artifacts = self._runtime.runtime_root / "host-text-artifacts"
        host_text_artifacts.mkdir(mode=0o700, exist_ok=True)
        host_text_artifacts.chmod(0o700)
        parser_settings = settings.parser.model_copy(
            update={
                "docling_artifacts_path": host_text_artifacts,
                "docling_artifact_manifest_path": (
                    PROJECT_ROOT / "config/docling-artifacts-v1.json"
                ),
            }
        )
        settings = settings.model_copy(update={"parser": parser_settings})
        dependencies = build_worker_dependencies(
            settings,
            worker_id=f"v3-host-{uuid4().hex}",
        )
        stopped = asyncio.Event()
        try:
            await dependencies.start()
            self._ready.set()
            consumer = asyncio.create_task(
                consume_lane(
                    "indexing",
                    dependencies.indexing_scheduler,
                    stopped,
                    poll_interval_seconds=0.05,
                )
            )
            while not self._stop.is_set():
                await asyncio.sleep(0.05)
            stopped.set()
            await consumer
        finally:
            self._ready.set()
            await dependencies.close()


def _corpus_paths(spec: ProvisioningSpec = V2_SPEC) -> tuple[Path, ...]:
    root = spec.corpus_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ProvisioningError("evaluation corpus directory is invalid")
    paths = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    if (
        len(paths) != spec.expected_document_count
        or any(
            path.is_symlink()
            or not path.is_file()
            or path.suffix.lower() not in spec.media_types
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


def _knowledge_base(
    runtime: EvaluationRuntime,
    spec: ProvisioningSpec = V2_SPEC,
) -> dict[str, Any]:
    matches = tuple(
        row
        for row in _paged_items(runtime.api_base_url, "knowledge-bases")
        if row.get("name") == spec.knowledge_base_name
        and row.get("deleted_at") is None
    )
    if len(matches) > 1:
        raise ProvisioningError("evaluation knowledge base identity is ambiguous")
    if matches:
        knowledge_base = matches[0]
    else:
        key = uuid5(NAMESPACE_URL, spec.knowledge_base_name)
        knowledge_base = _request(
            f"{runtime.api_base_url}/knowledge-bases",
            method="POST",
            headers={"Idempotency-Key": str(key)},
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
        raise ProvisioningError("evaluation knowledge base is not semantic v4")
    return knowledge_base


def _upload(
    runtime: EvaluationRuntime,
    kb_id: str,
    path: Path,
    spec: ProvisioningSpec = V2_SPEC,
) -> dict[str, Any]:
    metadata = base64.urlsafe_b64encode(
        json.dumps(
            {"v": 1, "filename": path.name, "display_name": path.name},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    key = uuid5(NAMESPACE_URL, f"{spec.knowledge_base_name}:{path.name}")
    return _request(
        f"{runtime.api_base_url}/knowledge-bases/{kb_id}/documents",
        method="POST",
        headers={
            "Content-Type": spec.media_types[path.suffix.lower()],
            "Idempotency-Key": str(key),
            "X-Document-Metadata": metadata,
        },
        body=path.read_bytes(),
    )


def _ensure_documents(
    runtime: EvaluationRuntime,
    kb_id: str,
    paths: tuple[Path, ...],
    spec: ProvisioningSpec = V2_SPEC,
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
            _upload(runtime, kb_id, path, spec)


def _host_source_identity(
    runtime: EvaluationRuntime,
    kb_id: str,
    path: Path,
    spec: ProvisioningSpec,
) -> SourceFileIdentity:
    identity = runtime.adaptive_graph
    if identity is None:
        raise ProvisioningError("evaluation workspace identity is unavailable")
    idempotency_key = uuid5(NAMESPACE_URL, f"{spec.knowledge_base_name}:{path.name}")
    key_material = hashlib.sha256(
        "\x1f".join(
            (
                str(identity.workspace_id),
                f"eval-{runtime.owner}",
                "local-evaluator",
                CREATE_DOCUMENT_ENDPOINT,
                str(idempotency_key),
                kb_id,
                "new",
            )
        ).encode("utf-8")
    ).hexdigest()
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return SourceFileIdentity(
        workspace_id=identity.workspace_id,
        key=hashlib.sha256(f"{key_material}\x1f{checksum}".encode("utf-8")).hexdigest(),
    )


def _mirror_host_sources(
    runtime: EvaluationRuntime,
    kb_id: str,
    paths: tuple[Path, ...],
    spec: ProvisioningSpec,
) -> None:
    """Mirror the frozen upload bytes into the host-visible eval file store."""
    documents = _paged_items(
        runtime.api_base_url,
        f"knowledge-bases/{kb_id}/documents",
    )
    versions = {
        str(row.get("current_version", {}).get("original_filename", "")): row.get(
            "current_version", {}
        )
        for row in documents
        if row.get("deleted_at") is None
    }
    if set(versions) != {path.name for path in paths}:
        raise ProvisioningError("evaluation host source document set is invalid")
    root = runtime.runtime_root / "source-data"
    store = LocalFileStore(root / "staging", root / "final")

    async def mirror() -> None:
        for path in paths:
            content = path.read_bytes()
            checksum = hashlib.sha256(content).hexdigest()
            version = versions[path.name]
            if (
                version.get("checksum_sha256") != checksum
                or version.get("size_bytes") != len(content)
            ):
                raise ProvisioningError("evaluation host source digest changed")
            source_identity = _host_source_identity(runtime, kb_id, path, spec)
            expected = SourceFileDigest(checksum, len(content))
            staged = await store.stage_at(source_identity, BytesIO(content))
            if staged.digest != expected:
                raise ProvisioningError("evaluation host source mirror is invalid")
            await store.finalize(source_identity, expected)

    asyncio.run(mirror())


def _wait_for_indexing(
    runtime: EvaluationRuntime,
    kb_id: str,
    *,
    timeout_seconds: float,
    expected_document_count: int = EXPECTED_DOCUMENT_COUNT,
    host_retry_spec: ProvisioningSpec | None = None,
) -> str:
    started = time.monotonic()
    retry_rounds: dict[str, int] = {}
    while True:
        jobs = _paged_items(
            runtime.api_base_url,
            f"knowledge-bases/{kb_id}/indexing-jobs",
        )
        failed = tuple(job for job in jobs if job.get("status") == "failed")
        if failed and host_retry_spec is not None:
            for job in failed:
                job_id = str(UUID(str(job.get("job_id"))))
                rounds = retry_rounds.get(job_id, 0)
                if rounds >= 3 or job.get("can_retry") is not True:
                    raise ProvisioningError("evaluation host indexing retry exhausted")
                updated_at = str(job.get("updated_at", ""))
                key = uuid5(
                    NAMESPACE_URL,
                    f"{host_retry_spec.knowledge_base_name}:host-retry:{job_id}:{updated_at}",
                )
                job_url = f"{runtime.api_base_url}/indexing-jobs/{job_id}"
                try:
                    _request(
                        f"{job_url}/retry",
                        method="POST",
                        headers={"Idempotency-Key": str(key)},
                    )
                except ProvisioningError:
                    # A concurrently running isolated/host worker can move a
                    # job after the list snapshot.  Accept only a proven state
                    # transition; a still-failed job remains a hard error.
                    current = _request(job_url)
                    if current.get("status") == "failed":
                        raise
                retry_rounds[job_id] = rounds + 1
            time.sleep(0.1)
            continue
        if failed or any(job.get("status") == "cancelled" for job in jobs):
            raise ProvisioningError("evaluation indexing failed")
        completed = tuple(job for job in jobs if job.get("status") == "completed")
        revision_ids = {
            str(job.get("index_revision_id", "")) for job in completed
        }
        if len(completed) == expected_document_count and len(revision_ids) == 1:
            return revision_ids.pop()
        if len(jobs) > expected_document_count:
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
    host_runtime: EvaluationRuntime | None = None,
) -> dict[str, Any]:
    url = f"{runtime.api_base_url}/knowledge-bases/{kb_id}/graph-config"
    config = _request(url)
    status = config.get("status")
    retry_attempted = False
    if status == "disabled":
        config = (
            _host_graph_mutation(
                host_runtime,
                kb_id,
                answer_profile_revision_id=answer_profile_revision_id,
                retry=False,
                force_rebuild=False,
            )
            if host_runtime is not None
            else _request(
                url,
                method="PUT",
                payload={
                    "enabled": True,
                    "chat_profile_revision_id": str(answer_profile_revision_id),
                },
            )
        )
    elif status == "failed" and force_rebuild_failed:
        config = (
            _host_graph_mutation(
                host_runtime,
                kb_id,
                answer_profile_revision_id=answer_profile_revision_id,
                retry=True,
                force_rebuild=True,
            )
            if host_runtime is not None
            else _request(
                url,
                method="PUT",
                payload={"enabled": True, "retry": True, "force_rebuild": True},
            )
        )
    elif status == "failed" and retry_failed:
        config = (
            _host_graph_mutation(
                host_runtime,
                kb_id,
                answer_profile_revision_id=answer_profile_revision_id,
                retry=True,
                force_rebuild=False,
            )
            if host_runtime is not None
            else _request(
                url,
                method="PUT",
                payload={"enabled": True, "retry": True},
            )
        )
        retry_attempted = True
    elif status == "failed":
        raise ProvisioningError("evaluation Graph build failed; retry was not authorized")

    started = time.monotonic()
    while config.get("status") != "ready":
        if config.get("status") == "failed":
            if retry_failed and not retry_attempted:
                config = (
                    _host_graph_mutation(
                        host_runtime,
                        kb_id,
                        answer_profile_revision_id=answer_profile_revision_id,
                        retry=True,
                        force_rebuild=False,
                    )
                    if host_runtime is not None
                    else _request(
                        url,
                        method="PUT",
                        payload={"enabled": True, "retry": True},
                    )
                )
                retry_attempted = True
                continue
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


def _host_graph_mutation(
    runtime: EvaluationRuntime,
    kb_id: str,
    *,
    answer_profile_revision_id: UUID,
    retry: bool,
    force_rebuild: bool,
) -> dict[str, Any]:
    """Apply current-source Graph configuration against the isolated DB."""

    async def mutate() -> None:
        dependencies = build_worker_dependencies(
            env_file=runtime.env_file,
            worker_id=f"v3-host-graph-config-{uuid4().hex}",
        )
        try:
            await dependencies.start()
            context = dependencies.auth_provider.get_context()
            service = GraphConfigurationService(
                dependencies.unit_of_work,
                dependencies.access_policy,
                dependencies.graphiti_runtime,
            )
            before = await service.get(context, UUID(kb_id))
            if retry:
                after = await service.retry(
                    context,
                    UUID(kb_id),
                    force_rebuild=force_rebuild,
                )
                if not force_rebuild and after.build_id != before.build_id:
                    raise ProvisioningError(
                        "evaluation host Graph retry changed build identity"
                    )
            else:
                await service.configure(
                    context,
                    UUID(kb_id),
                    enabled=True,
                    chat_profile_revision_id=answer_profile_revision_id,
                )
        finally:
            await dependencies.close()

    asyncio.run(mutate())
    return _request(
        f"{runtime.api_base_url}/knowledge-bases/{kb_id}/graph-config"
    )


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
    # Bind the same owner-only manifest that was validated on load.  In a
    # linked worktree this is the primary checkout's canonical eval runtime,
    # never a new worktree-local .runtime directory.
    _write_private(runtime.manifest, payload, replace=True)


def provision(
    *,
    dataset_id: str = V2_SPEC.dataset_id,
    confirmation: str,
    timeout_seconds: float,
    retry_failed_graph: bool,
    force_rebuild_failed_graph: bool,
    host_worker: bool = False,
) -> dict[str, object]:
    try:
        spec = PROVISIONING_SPECS[dataset_id]
    except KeyError as error:
        raise ProvisioningError("evaluation dataset is unsupported") from error
    if confirmation != spec.confirmation:
        raise ProvisioningError("evaluation provisioning confirmation is invalid")
    runtime = load_evaluation_runtime(
        canonical_evaluation_runtime_manifest(),
        require_adaptive_graph=True,
        allow_canonical_checkout=True,
    )
    driver = _HostIndexingDriver(runtime) if host_worker else None
    try:
        _require_current_runtime(runtime)
        paths = _corpus_paths(spec)
        corpus_digest = _corpus_digest(paths)
        knowledge_base = _knowledge_base(runtime, spec)
        kb_id = str(UUID(str(knowledge_base["id"])))
        _ensure_documents(runtime, kb_id, paths, spec)
        if driver is not None:
            _mirror_host_sources(runtime, kb_id, paths, spec)
            driver.start()
        index_revision_id = UUID(
            _wait_for_indexing(
                runtime,
                kb_id,
                timeout_seconds=timeout_seconds,
                expected_document_count=spec.expected_document_count,
                host_retry_spec=spec if host_worker else None,
            )
        )
        config = _wait_for_graph(
            runtime,
            kb_id,
            answer_profile_revision_id=runtime.adaptive_graph.answer_profile_revision_id,
            timeout_seconds=timeout_seconds,
            retry_failed=retry_failed_graph,
            force_rebuild_failed=force_rebuild_failed_graph,
            host_runtime=runtime if host_worker else None,
        )
        graph_build_id = UUID(str(config["build_id"]))
        _bind_runtime(
            runtime,
            knowledge_base_id=UUID(kb_id),
            index_revision_id=index_revision_id,
            graph_build_id=graph_build_id,
        )
    finally:
        if driver is not None:
            driver.close()
    return {
        "status": "ready",
        "dataset_id": spec.dataset_id,
        "corpus_digest": corpus_digest,
        "document_count": spec.expected_document_count,
        "knowledge_base_id": kb_id,
        "index_revision_id": str(index_revision_id),
        "graph_build_id": str(graph_build_id),
        "extractor_version": GRAPH_EXTRACTOR_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(PROVISIONING_SPECS),
        default=V2_SPEC.dataset_id,
    )
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--retry-failed-graph", action="store_true")
    parser.add_argument("--force-rebuild-failed-graph", action="store_true")
    parser.add_argument("--host-worker", action="store_true")
    arguments = parser.parse_args()
    if not 60.0 <= arguments.timeout_seconds <= 14_400.0:
        parser.error("timeout must be between 60 and 14400 seconds")
    if arguments.retry_failed_graph and arguments.force_rebuild_failed_graph:
        parser.error("resume and force rebuild are mutually exclusive")
    try:
        result = provision(
            dataset_id=arguments.dataset,
            confirmation=arguments.confirm,
            timeout_seconds=arguments.timeout_seconds,
            retry_failed_graph=arguments.retry_failed_graph,
            force_rebuild_failed_graph=arguments.force_rebuild_failed_graph,
            host_worker=arguments.host_worker,
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
