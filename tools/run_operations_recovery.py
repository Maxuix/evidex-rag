#!/usr/bin/env python3
"""Run isolated Stage 06 local operations and recovery exercises."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from http.client import HTTPMessage
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import uuid


SENSITIVE_ENVIRONMENT = {
    "POSTGRES_ADMIN_PASSWORD",
    "RAG_KB_MIGRATION_PASSWORD",
    "RAG_KB_RUNTIME_PASSWORD",
}
TERMINAL_JOBS = {"completed", "failed", "cancelled"}
TERMINAL_RUNS = {"completed", "failed", "cancelled"}
WORKSPACE_ID = "01900000-0000-7000-8000-000000000001"
CONFIRMATION = "DESTROY_RAG_KB_LOCAL_DATA"
EVIDENCE_INPUTS = (
    ".dockerignore",
    ".env.example",
    "Dockerfile",
    "compose.yaml",
    "deploy/compose-operations.override.yaml",
    "deploy/operations_provider/Dockerfile",
    "deploy/operations_provider/provider.py",
    "tools/reset_local.py",
    "tools/run_operations_recovery.py",
    "tests/e2e/fixtures/operations-v1.md",
    "tests/e2e/fixtures/operations-v2.md",
    "tests/e2e/fixtures/operations-index-failure.md",
)


def inherited_operations_environment(
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    """Keep host tooling only; never inherit project runtime inputs or secrets."""

    values = os.environ if source is None else source
    return {
        key: value
        for key, value in values.items()
        if not key.startswith("RAG_KB") and key not in SENSITIVE_ENVIRONMENT
    }


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def reset_invocation(root: Path, project: str, confirmation: str) -> list[str]:
    return [
        str(root / ".venv/bin/python"),
        "tools/reset_local.py",
        "--env-file",
        ".env.example",
        "--project-name",
        project,
        "--confirm",
        confirmation,
    ]


@dataclass(frozen=True, slots=True)
class HttpResult:
    status: int
    headers: HTTPMessage
    body: bytes


class OperationsFailure(RuntimeError):
    pass


class OperationsRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = f"rag-kb-s06-w04-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        passwords = tuple(secrets.token_urlsafe(24) for _ in range(3))
        if len(set(passwords)) != 3:
            raise OperationsFailure("per-run database credentials were not distinct")
        self.environment = {
            **inherited_operations_environment(),
            "POSTGRES_ADMIN_PASSWORD": passwords[0],
            "RAG_KB_MIGRATION_PASSWORD": passwords[1],
            "RAG_KB_RUNTIME_PASSWORD": passwords[2],
            "RAG_KB_POSTGRES_PORT": str(available_port()),
            "RAG_KB_API_PORT": str(available_port()),
            "RAG_KB_FRONTEND_PORT": str(available_port()),
            "RAG_KB_ENV_FILE": ".env.example",
        }
        self.base = [
            "docker",
            "compose",
            "-f",
            "compose.yaml",
            "-f",
            "deploy/compose-operations.override.yaml",
            "--env-file",
            ".env.example",
            "--profile",
            "tools",
            "--project-name",
            self.project,
        ]
        self.scenarios: dict[str, str] = {}
        self.metrics: dict[str, int | float | str | bool] = {}
        self.tested_platform = "not-recorded"

    @property
    def api_origin(self) -> str:
        return f"http://127.0.0.1:{self.environment['RAG_KB_API_PORT']}"

    @property
    def password_values(self) -> tuple[str, str, str]:
        return tuple(self.environment[key] for key in sorted(SENSITIVE_ENVIRONMENT))  # type: ignore[return-value]

    def run(
        self,
        *arguments: str,
        check: bool = True,
        timeout: int = 180,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [*self.base, *arguments],
                cwd=self.root,
                env=self.environment,
                check=check,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as error:
            diagnostic = "\n".join(
                value.strip() for value in (error.stdout, error.stderr) if value.strip()
            )
            raise OperationsFailure(
                f"Compose command failed ({' '.join(arguments)}): {diagnostic[-4000:]}"
            ) from error

    def host_run(
        self,
        command: list[str],
        *,
        check: bool = True,
        timeout: int = 180,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=self.root,
            env=self.environment,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        expected_status: int = 200,
        timeout: float = 10,
    ) -> HttpResult:
        request = urllib.request.Request(
            f"{self.api_origin}{path}",
            data=body,
            method=method,
            headers={"Accept": "application/json", **(headers or {})},
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            result = HttpResult(error.code, error.headers, error.read())
        else:
            with response:
                result = HttpResult(response.status, response.headers, response.read())
        if result.status != expected_status:
            raise OperationsFailure(
                f"{method} {path} returned {result.status}, expected {expected_status}"
            )
        return result

    def json_request(
        self,
        method: str,
        path: str,
        *,
        value: object | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        expected_status: int = 200,
        timeout: float = 10,
    ) -> dict[str, Any]:
        merged = dict(headers or {})
        if value is not None:
            body = json.dumps(value, separators=(",", ":")).encode("utf-8")
            merged["Content-Type"] = "application/json"
        result = self.request(
            method,
            path,
            body=body,
            headers=merged,
            expected_status=expected_status,
            timeout=timeout,
        )
        try:
            parsed = json.loads(result.body)
        except json.JSONDecodeError as error:
            raise OperationsFailure(f"{method} {path} returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise OperationsFailure(f"{method} {path} returned a non-object")
        return parsed

    @staticmethod
    def idempotency_headers() -> dict[str, str]:
        return {"Idempotency-Key": str(uuid.uuid4())}

    def wait_http(self, path: str, *, timeout: float = 90) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{self.api_origin}{path}", timeout=3) as response:
                    if response.status == 200:
                        return
            except Exception as error:
                last_error = error
            time.sleep(0.25)
        raise OperationsFailure(f"timed out waiting for {path}: {last_error}")

    def psql(self, statement: str) -> str:
        return self.run(
            "exec",
            "-T",
            "postgres",
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            "postgres",
            "-d",
            "rag_kb",
            "-Atc",
            statement,
        ).stdout.strip()

    def wait_sql(
        self,
        statement: str,
        accepted: Callable[[str], bool],
        *,
        timeout: float = 45,
    ) -> str:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            last = self.psql(statement)
            if accepted(last):
                return last
            time.sleep(0.2)
        raise OperationsFailure(f"SQL state did not converge; last value was {last!r}")

    def wait_job(self, job_id: str, *, timeout: float = 60) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.json_request("GET", f"/api/v1/indexing-jobs/{job_id}")
            if value.get("status") in TERMINAL_JOBS:
                return value
            time.sleep(0.2)
        raise OperationsFailure(f"indexing job {job_id} did not become terminal")

    def wait_run(self, run_id: str, *, timeout: float = 60) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.json_request("GET", f"/api/v1/chat/runs/{run_id}")
            if value.get("status") in TERMINAL_RUNS:
                return value
            time.sleep(0.2)
        raise OperationsFailure(f"ChatRun {run_id} did not become terminal")

    def upload(
        self,
        path: str,
        fixture: Path,
        *,
        display_name: str,
    ) -> dict[str, Any]:
        return self.json_request(
            "POST",
            path,
            body=fixture.read_bytes(),
            headers={
                **self.idempotency_headers(),
                "Content-Type": "text/markdown; charset=utf-8",
                "X-Document-Filename": fixture.name,
                "X-Document-Display-Name": display_name,
            },
            expected_status=202,
        )

    def create_run(self, kb_id: str, session_id: str, message: str) -> dict[str, Any]:
        return self.json_request(
            "POST",
            "/api/v1/chat/runs",
            value={
                "session_id": session_id,
                "knowledge_base_id": kb_id,
                "message": message,
                "answer_policy": {
                    "answer_style": "concise",
                    "insufficiency_policy": "refuse",
                },
                "retrieval": {"mode": "vector", "top_k": 5},
            },
            headers=self.idempotency_headers(),
            expected_status=202,
        )

    def source_file_count(self, service: str) -> int:
        code = (
            "from pathlib import Path; "
            "print(sum(1 for p in Path('/var/lib/rag-kb/sources').rglob('*') "
            "if p.is_file()))"
        )
        return int(
            self.run("exec", "-T", service, "python", "-c", code).stdout.strip()
        )

    def mark(self, name: str) -> None:
        self.scenarios[name] = "passed"


class InterruptedSSE:
    def __init__(self, url: str) -> None:
        self.url = url
        self.connected = threading.Event()
        self.finished = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _read(self) -> None:
        try:
            request = urllib.request.Request(
                self.url, headers={"Accept": "text/event-stream"}
            )
            response = urllib.request.urlopen(request, timeout=20)
            self.connected.set()
            with response:
                response.read()
        except BaseException as error:
            self.error = error
        finally:
            self.finished.set()


def require(condition: object, message: str) -> None:
    if not condition:
        raise OperationsFailure(message)


def _backdate_stale(runtime: OperationsRuntime, table: str, identifier: str) -> None:
    if table not in {"indexing_job", "chat_run"}:
        raise ValueError("unsupported stale table")
    changed = runtime.psql(
        f"UPDATE {table} SET heartbeat_at = now() - interval '1 hour', "
        f"updated_at = now() - interval '1 hour' "
        f"WHERE id = '{identifier}' AND status = 'running'; "
        "SELECT count(*) FROM "
        f"{table} WHERE id = '{identifier}' AND status = 'running' "
        "AND heartbeat_at <= now() - interval '35 seconds'"
    )
    require(changed.splitlines()[-1:] == ["1"], f"failed to make {table} attempt stale")


def _assert_index_facts(runtime: OperationsRuntime, job_id: str, target_id: str) -> None:
    value = runtime.psql(
        "SELECT (SELECT count(*) FROM indexing_job WHERE id = '" + job_id + "') || '|' || "
        "(SELECT count(*) FROM indexed_document_version WHERE id = '" + target_id + "') || '|' || "
        "(SELECT count(*) FROM index_chunk WHERE indexed_document_version_id = '" + target_id + "') || '|' || "
        "(SELECT count(DISTINCT ordinal) FROM index_chunk WHERE indexed_document_version_id = '" + target_id + "') || '|' || "
        "(SELECT count(*) FROM vector_record_1024 vector JOIN index_chunk chunk ON chunk.id = vector.index_chunk_id "
        "WHERE chunk.indexed_document_version_id = '" + target_id + "')"
    )
    job_count, target_count, chunks, ordinals, vectors = (int(item) for item in value.split("|"))
    require(job_count == target_count == 1, "index recovery duplicated its job or target")
    require(chunks > 0 and chunks == ordinals == vectors, "index recovery duplicated or lost chunks/vectors")


def _assert_chat_facts(runtime: OperationsRuntime, run_id: str) -> None:
    value = runtime.psql(
        "SELECT (SELECT count(*) FROM chat_run WHERE id = '" + run_id + "') || '|' || "
        "(SELECT count(*) FROM chat_message message JOIN chat_run run ON run.user_message_id = message.id "
        "WHERE run.id = '" + run_id + "' AND message.role = 'user') || '|' || "
        "(SELECT count(*) FROM chat_message WHERE chat_run_id = '" + run_id + "' AND role = 'assistant') || '|' || "
        "(SELECT count(*) FROM citation citation JOIN chat_message message ON message.id = citation.assistant_message_id "
        "WHERE message.chat_run_id = '" + run_id + "') || '|' || "
        "(SELECT count(DISTINCT citation.ordinal) FROM citation citation JOIN chat_message message "
        "ON message.id = citation.assistant_message_id WHERE message.chat_run_id = '" + run_id + "')"
    )
    run_count, users, assistants, citations, ordinals = (
        int(item) for item in value.split("|")
    )
    require(run_count == users == assistants == 1, "chat recovery duplicated a run or message")
    require(citations > 0 and citations == ordinals, "chat recovery duplicated or lost citations")


def _seed_orphans(runtime: OperationsRuntime) -> None:
    for location, key in (("staging", "a" * 64), ("final", "b" * 64)):
        parent = f"/var/lib/rag-kb/sources/{location}/{WORKSPACE_ID}/{key[:2]}"
        path = f"{parent}/{key}.source"
        runtime.run("exec", "-T", "worker", "mkdir", "-p", parent)
        runtime.run("exec", "-T", "worker", "touch", path)


def _maintenance_result(runtime: OperationsRuntime) -> dict[str, int]:
    output = runtime.run("run", "--rm", "maintenance", timeout=120).stdout.strip()
    try:
        value = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise OperationsFailure("maintenance did not emit its JSON result") from error
    if not isinstance(value, dict) or any(not isinstance(item, int) for item in value.values()):
        raise OperationsFailure("maintenance result did not contain integer counters")
    return value


def execute(runtime: OperationsRuntime) -> None:
    fixtures = runtime.root / "tests/e2e/fixtures"
    print("[1/9] building and starting the isolated operations project", flush=True)
    runtime.run("build", "api", "operations-provider", timeout=600)
    runtime.run("up", "-d", "--wait", "--wait-timeout", "90", "postgres")
    runtime.run("run", "--rm", "--no-deps", "storage-init")
    runtime.run("run", "--rm", "migrate")
    runtime.run(
        "up",
        "-d",
        "--wait",
        "--wait-timeout",
        "120",
        "operations-provider",
        "api",
        "worker",
    )
    runtime.wait_http("/health/ready")
    machine = runtime.run(
        "exec", "-T", "api", "python", "-c", "import platform; print(platform.machine())"
    ).stdout.strip()
    runtime.tested_platform = f"linux/{'arm64' if machine == 'aarch64' else machine}"
    runtime.mark("isolated_random_credential_start")

    kb = runtime.json_request(
        "POST",
        "/api/v1/knowledge-bases",
        value={"name": "S06 W04 Operations"},
        headers=runtime.idempotency_headers(),
        expected_status=201,
    )
    kb_id = str(kb["id"])

    print("[2/9] killing Worker during indexing and reconciling stale ownership", flush=True)
    first = runtime.upload(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        fixtures / "operations-v1.md",
        display_name="Operations recovery fixture",
    )
    first_job_id = str(first["job_id"])
    first_target_id = str(first["indexed_document_version_id"])
    document_id = str(first["document"]["id"])
    runtime.wait_sql(
        f"SELECT status::text || '|' || attempt FROM indexing_job WHERE id = '{first_job_id}'",
        lambda value: value == "running|1",
    )
    runtime.run("kill", "-s", "SIGKILL", "worker")
    _backdate_stale(runtime, "indexing_job", first_job_id)
    runtime.run("up", "-d", "--wait", "--wait-timeout", "60", "worker")
    first_terminal = runtime.wait_job(first_job_id)
    require(
        first_terminal.get("status") == "completed"
        and first_terminal.get("serving_status") == "serving"
        and first_terminal.get("attempt") == 2,
        "stale indexing attempt did not recover on its second finite attempt",
    )
    _assert_index_facts(runtime, first_job_id, first_target_id)
    runtime.metrics["stale_indexing_final_attempt"] = 2
    runtime.mark("abnormal_worker_indexing_stale_reconciliation")

    session = runtime.json_request(
        "POST",
        "/api/v1/chat/sessions",
        value={"knowledge_base_id": kb_id, "title": "Operations recovery"},
        expected_status=201,
    )
    session_id = str(session["id"])

    print("[3/9] killing API during SSE while Worker completes durable chat", flush=True)
    api_interrupted = runtime.create_run(
        kb_id, session_id, "Explain OPS_ALPHA OPS_CHAT_DELAY after API interruption."
    )
    api_run_id = str(api_interrupted["run_id"])
    runtime.wait_sql(
        f"SELECT status::text || '|' || attempt FROM chat_run WHERE id = '{api_run_id}'",
        lambda value: value == "running|1",
    )
    stream = InterruptedSSE(f"{runtime.api_origin}{api_interrupted['events_url']}")
    stream.start()
    require(stream.connected.wait(5), "SSE client did not connect before API interruption")
    runtime.run("kill", "-s", "SIGKILL", "api")
    require(stream.finished.wait(10), "SSE client did not observe API interruption")
    runtime.wait_sql(
        f"SELECT status::text FROM chat_run WHERE id = '{api_run_id}'",
        lambda value: value == "completed",
        timeout=45,
    )
    runtime.run("up", "-d", "--wait", "--wait-timeout", "60", "api")
    runtime.wait_http("/health/ready")
    recovered_api_run = runtime.wait_run(api_run_id)
    require(
        recovered_api_run.get("status") == "completed"
        and recovered_api_run.get("attempt") == 1,
        "API interruption changed or cancelled Worker-owned chat",
    )
    _assert_chat_facts(runtime, api_run_id)
    runtime.mark("abnormal_api_sse_status_recovery")

    print("[4/9] killing Worker during chat and reconciling stale ownership", flush=True)
    stale_chat = runtime.create_run(
        kb_id, session_id, "Explain OPS_ALPHA OPS_CHAT_DELAY after Worker interruption."
    )
    stale_run_id = str(stale_chat["run_id"])
    runtime.wait_sql(
        f"SELECT status::text || '|' || attempt FROM chat_run WHERE id = '{stale_run_id}'",
        lambda value: value == "running|1",
    )
    runtime.run("kill", "-s", "SIGKILL", "worker")
    _backdate_stale(runtime, "chat_run", stale_run_id)
    runtime.run("up", "-d", "--wait", "--wait-timeout", "60", "worker")
    stale_terminal = runtime.wait_run(stale_run_id)
    require(
        stale_terminal.get("status") == "completed"
        and stale_terminal.get("attempt") == 2,
        "stale ChatRun did not recover on its second finite attempt",
    )
    _assert_chat_facts(runtime, stale_run_id)
    runtime.metrics["stale_chat_final_attempt"] = 2
    runtime.mark("abnormal_worker_chat_stale_reconciliation")

    print("[5/9] exhausting automatic indexing and using explicit public retry", flush=True)
    failed = runtime.upload(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        fixtures / "operations-index-failure.md",
        display_name="Operations finite retry fixture",
    )
    failed_job_id = str(failed["job_id"])
    failed_terminal = runtime.wait_job(failed_job_id)
    require(
        failed_terminal.get("status") == "failed"
        and failed_terminal.get("attempt") == 3
        and failed_terminal.get("can_retry") is True,
        "automatic indexing retry did not stop after three attempts",
    )
    retry_key = runtime.idempotency_headers()
    retried = runtime.json_request(
        "POST",
        f"/api/v1/indexing-jobs/{failed_job_id}/retry",
        headers=retry_key,
        expected_status=202,
    )
    replay = runtime.json_request(
        "POST",
        f"/api/v1/indexing-jobs/{failed_job_id}/retry",
        headers=retry_key,
        expected_status=202,
    )
    require(
        retried.get("job_id") == failed_job_id
        and replay.get("job_id") == failed_job_id
        and retried.get("attempt") == 0,
        "explicit retry did not idempotently reset the same durable job",
    )
    explicit_terminal = runtime.wait_job(failed_job_id)
    require(
        explicit_terminal.get("status") == "completed"
        and explicit_terminal.get("serving_status") == "serving",
        "explicit public retry did not recover the exhausted job",
    )
    _assert_index_facts(runtime, failed_job_id, str(failed["indexed_document_version_id"]))
    runtime.metrics["automatic_indexing_attempt_limit"] = 3
    runtime.mark("finite_retry_and_idempotent_explicit_retry")

    print("[6/9] exercising staging/final orphan and retired-derived cleanup", flush=True)
    second = runtime.upload(
        f"/api/v1/documents/{document_id}/versions",
        fixtures / "operations-v2.md",
        display_name="Operations recovery fixture",
    )
    second_terminal = runtime.wait_job(str(second["job_id"]))
    require(second_terminal.get("serving_status") == "serving", "updated target did not serve")
    _seed_orphans(runtime)
    time.sleep(1.2)
    first_cleanup = _maintenance_result(runtime)
    second_cleanup = _maintenance_result(runtime)
    require(
        first_cleanup.get("orphan_files_removed") == 2,
        "maintenance did not remove both staging and final orphans",
    )
    require(
        first_cleanup.get("retired_targets_cleaned", 0) >= 1
        and first_cleanup.get("chunks_deleted", 0) > 0
        and first_cleanup.get("vectors_deleted") == first_cleanup.get("chunks_deleted"),
        "maintenance did not remove retired derived data",
    )
    require(
        second_cleanup.get("orphan_files_removed") == 0
        and second_cleanup.get("retired_targets_cleaned") == 0
        and second_cleanup.get("chunks_deleted") == 0
        and second_cleanup.get("vectors_deleted") == 0,
        "repeated maintenance did not converge to zero changes",
    )
    target_counts = runtime.psql(
        "SELECT (SELECT count(*) FROM index_chunk WHERE indexed_document_version_id = '"
        + first_target_id
        + "') || '|' || (SELECT count(*) FROM index_chunk WHERE indexed_document_version_id = '"
        + str(second["indexed_document_version_id"])
        + "')"
    )
    retired_chunks, serving_chunks = (int(item) for item in target_counts.split("|"))
    require(retired_chunks == 0 and serving_chunks > 0, "maintenance changed serving data")
    runtime.metrics["orphan_files_removed"] = 2
    runtime.metrics["retired_targets_cleaned"] = first_cleanup["retired_targets_cleaned"]
    runtime.mark("bounded_idempotent_file_and_retired_cleanup")

    print("[7/9] confirming the shared source volume remains visible", flush=True)
    api_sources = runtime.source_file_count("api")
    worker_sources = runtime.source_file_count("worker")
    require(api_sources > 0 and api_sources == worker_sources, "API and Worker source views differ")
    runtime.run("kill", "-s", "SIGKILL", "api", "worker")
    runtime.run("up", "-d", "--wait", "--wait-timeout", "90", "api", "worker")
    require(
        runtime.source_file_count("api") == api_sources
        and runtime.source_file_count("worker") == worker_sources,
        "source files changed across abnormal process restart",
    )
    runtime.metrics["persisted_source_files"] = api_sources
    runtime.mark("shared_source_volume_process_restart_persistence")

    print("[8/9] refusing an inexact destructive reset confirmation", flush=True)
    wrong = runtime.host_run(
        reset_invocation(runtime.root, runtime.project, "WRONG_CONFIRMATION"),
        check=False,
    )
    require(wrong.returncode != 0, "inexact reset confirmation unexpectedly succeeded")
    require(
        runtime.host_run(
            ["docker", "volume", "inspect", f"{runtime.project}_postgres-data"],
            check=False,
        ).returncode
        == 0,
        "inexact confirmation changed the disposable project",
    )
    runtime.mark("destructive_reset_exact_confirmation_guard")

    print("[9/9] resetting only disposable volumes and proving a new empty state", flush=True)
    runtime.host_run(reset_invocation(runtime.root, runtime.project, CONFIRMATION))
    for volume in ("postgres-data", "source-data"):
        require(
            runtime.host_run(
                ["docker", "volume", "inspect", f"{runtime.project}_{volume}"],
                check=False,
            ).returncode
            != 0,
            f"destructive reset retained {volume}",
        )
    remaining = runtime.host_run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={runtime.project}",
            "-q",
        ]
    ).stdout.strip()
    require(not remaining, "destructive reset retained project containers")

    runtime.run("up", "-d", "--wait", "--wait-timeout", "90", "postgres")
    runtime.run("run", "--rm", "--no-deps", "storage-init")
    runtime.run("run", "--rm", "migrate")
    runtime.run(
        "up",
        "-d",
        "--wait",
        "--wait-timeout",
        "90",
        "operations-provider",
        "api",
    )
    runtime.wait_http("/health/ready")
    empty = runtime.json_request("GET", "/api/v1/knowledge-bases")
    require(empty.get("items") == [], "reset database was not empty after fresh migration")
    require(runtime.source_file_count("api") == 0, "reset source volume was not empty")
    runtime.metrics["post_reset_knowledge_bases"] = 0
    runtime.metrics["post_reset_source_files"] = 0
    runtime.mark("destructive_reset_and_fresh_empty_state")


def build_report(
    runtime: OperationsRuntime,
    *,
    started_at: datetime,
    duration_seconds: float,
) -> dict[str, Any]:
    hashes = {
        path: hashlib.sha256((runtime.root / path).read_bytes()).hexdigest()
        for path in EVIDENCE_INPUTS
    }
    return {
        "schema_version": "1.0",
        "artifact": "stage06-operations-recovery-exercises",
        "status": "passed",
        "verified_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "duration_seconds": round(duration_seconds, 3),
        "tested_platform": runtime.tested_platform,
        "inputs": {"algorithm": "sha256", "files": hashes},
        "isolation": {
            "disposable_compose_project": True,
            "randomized_loopback_ports": True,
            "per_run_random_database_credentials": True,
            "distinct_database_role_credentials": True,
            "credentials_persisted_or_reported": False,
            "host_project_runtime_inputs_inherited": False,
            "external_model_credentials": False,
            "provider_host_port_published": False,
            "disposable_volumes_removed": True,
        },
        "fault_injection": {
            "api_exit_signal": "SIGKILL",
            "worker_exit_signal": "SIGKILL",
            "stale_heartbeat_backdating": "test-control SQL after process exit",
            "application_business_mutation_by_sql": False,
            "provider": "network-only deterministic operations test double",
        },
        "scenarios": runtime.scenarios,
        "metrics": runtime.metrics,
        "public_boundary": {
            "business_commands_and_results": "/api/v1",
            "database_access": "fault injection and duplicate-fact assertions only",
            "private_application_endpoint": False,
        },
        "recorded_limits": [
            "the shared source volume is local process-restart persistence, not a backup",
            "no host-failure recovery, RPO, RTO, retention, legal-hold, compliance, high-availability, or production recovery claim is made",
            "stale time is backdated only after the owning process is killed to keep the deterministic exercise bounded",
            "the deterministic provider proves orchestration and recovery contracts, not real-provider quality or availability",
            "single-Worker local recovery does not establish multi-runner takeover or execution-epoch fencing",
            "linux/amd64 remains unexecuted unless tested_platform records it",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = OperationsRuntime(root)
    started_at = datetime.now(UTC)
    started = time.monotonic()
    cleanup_error: Exception | None = None
    completed_duration: float | None = None
    try:
        execute(runtime)
    except (OperationsFailure, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"Operations and recovery exercises failed: {error}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
        completed_duration = time.monotonic() - started
    finally:
        try:
            cleanup = runtime.run(
                "down", "--volumes", "--remove-orphans", check=False, timeout=180
            )
            if cleanup.returncode != 0:
                cleanup_error = OperationsFailure(
                    f"Compose cleanup returned {cleanup.returncode}"
                )
        except (OSError, subprocess.SubprocessError) as error:
            cleanup_error = error
    if cleanup_error is not None:
        print(f"Operations cleanup failed: {cleanup_error}", file=sys.stderr)
        return 1
    if return_code == 0:
        if arguments.report is not None:
            assert completed_duration is not None
            arguments.report.parent.mkdir(parents=True, exist_ok=True)
            arguments.report.write_text(
                json.dumps(
                    build_report(
                        runtime,
                        started_at=started_at,
                        duration_seconds=completed_duration,
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(
            "Operations and recovery exercises passed: abnormal API/Worker exits, "
            "stale finite recovery, duplicate-fact checks, bounded maintenance, "
            "shared persistence, exact-confirmation reset, and empty restart",
            flush=True,
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
