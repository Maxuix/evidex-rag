#!/usr/bin/env python3
"""Run the isolated Stage 06 public API upload-to-citation integration suite."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from http.client import HTTPMessage
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import uuid


SENSITIVE_ENVIRONMENT = {
    "POSTGRES_ADMIN_PASSWORD",
    "RAG_KB_MIGRATION_PASSWORD",
    "RAG_KB_RUNTIME_PASSWORD",
}
TERMINAL_RUNS = {"completed", "failed", "cancelled"}
TERMINAL_JOBS = {"completed", "failed", "cancelled"}
EVIDENCE_INPUTS = (
    ".dockerignore",
    "Dockerfile",
    "compose.yaml",
    "deploy/compose-e2e.override.yaml",
    "tools/.dockerignore",
    "tools/model_provider_stub.py",
    "tools/provider-stub.Dockerfile",
    "tools/run_e2e_integration.py",
    "tests/e2e/fixtures/handbook-v1.md",
    "tests/e2e/fixtures/handbook-v2.md",
    "tests/e2e/fixtures/index-failure.md",
)


def inherited_e2e_environment() -> dict[str, str]:
    """Retain host tooling only; never inherit project runtime inputs or secrets."""

    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("RAG_KB") and key not in SENSITIVE_ENVIRONMENT
    }


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def parse_sse(body: bytes) -> list[tuple[str, dict[str, Any]]]:
    normalized = body.decode("utf-8").replace("\r\n", "\n")
    events: list[tuple[str, dict[str, Any]]] = []
    for block in normalized.split("\n\n"):
        event_name: str | None = None
        data: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data.append(line.removeprefix("data:").strip())
        if event_name and data:
            value = json.loads("\n".join(data))
            if not isinstance(value, dict):
                raise RuntimeError("SSE event data is not an object")
            events.append((event_name, value))
    return events


@dataclass(frozen=True, slots=True)
class HttpResult:
    status: int
    headers: HTTPMessage
    body: bytes


class E2EFailure(RuntimeError):
    pass


class PublicE2E:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = f"rag-kb-s06-w02-{os.getpid()}"
        self.environment = {
            **inherited_e2e_environment(),
            "POSTGRES_ADMIN_PASSWORD": "e2e-admin-password",
            "RAG_KB_MIGRATION_PASSWORD": "e2e-migration-password",
            "RAG_KB_RUNTIME_PASSWORD": "e2e-runtime-password",
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
            "deploy/compose-e2e.override.yaml",
            "--env-file",
            ".env.example",
            "--profile",
            "tools",
            "--project-name",
            self.project,
        ]
        self.scenarios: dict[str, str] = {}
        self.tested_platform = "not-recorded"

    @property
    def api_origin(self) -> str:
        return f"http://127.0.0.1:{self.environment['RAG_KB_API_PORT']}"

    @property
    def frontend_origin(self) -> str:
        return f"http://127.0.0.1:{self.environment['RAG_KB_FRONTEND_PORT']}"

    def run(
        self,
        *arguments: str,
        check: bool = True,
        timeout: int = 180,
    ) -> subprocess.CompletedProcess[str]:
        command = [*self.base, *arguments]
        try:
            return subprocess.run(
                command,
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
            raise E2EFailure(
                f"Compose command failed ({' '.join(arguments)}): {diagnostic[-4000:]}"
            ) from error

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
        url = path if path.startswith("http://") else f"{self.api_origin}{path}"
        request = urllib.request.Request(
            url,
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
            raise E2EFailure(
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
            raise E2EFailure(f"{method} {path} returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise E2EFailure(f"{method} {path} returned a non-object")
        return parsed

    def idempotency_headers(self) -> dict[str, str]:
        return {"Idempotency-Key": str(uuid.uuid4())}

    def wait_http(self, origin: str, path: str, *, timeout: float = 90) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{origin}{path}", timeout=3) as response:
                    if response.status == 200:
                        return
            except Exception as error:
                last_error = error
            time.sleep(0.25)
        raise E2EFailure(f"timed out waiting for {origin}{path}: {last_error}")

    def wait_job(self, job_id: str, *, timeout: float = 45) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.json_request("GET", f"/api/v1/indexing-jobs/{job_id}")
            if value.get("status") in TERMINAL_JOBS:
                return value
            time.sleep(0.2)
        raise E2EFailure(f"indexing job {job_id} did not become terminal")

    def wait_run(self, run_id: str, *, timeout: float = 45) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.json_request("GET", f"/api/v1/chat/runs/{run_id}")
            if value.get("status") in TERMINAL_RUNS:
                return value
            time.sleep(0.2)
        raise E2EFailure(f"ChatRun {run_id} did not become terminal")

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

    def retrieval(self, kb_id: str, query: str) -> dict[str, Any]:
        return self.json_request(
            "POST",
            "/api/v1/retrieval/query",
            value={
                "knowledge_base_id": kb_id,
                "query": query,
                "top_k": 10,
                "strategy": "exact_vector",
                "rerank": False,
                "include_debug": True,
            },
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

    def sse_to_terminal(self, events_url: str, *, timeout: float = 20) -> list[tuple[str, dict[str, Any]]]:
        result = self.request(
            "GET",
            events_url,
            headers={"Accept": "text/event-stream"},
            timeout=timeout,
        )
        media_type = result.headers.get("Content-Type", "").split(";", 1)[0]
        if media_type != "text/event-stream":
            raise E2EFailure("ChatRun events endpoint did not return SSE")
        return parse_sse(result.body)

    def disconnect_sse(self, events_url: str) -> None:
        url = f"{self.api_origin}{events_url}"
        request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
        response = urllib.request.urlopen(request, timeout=5)
        response.close()

    def source_file_count(self, service: str) -> int:
        code = (
            "from pathlib import Path; "
            "print(sum(1 for p in Path('/var/lib/rag-kb/sources/final').rglob('*') "
            "if p.is_file()))"
        )
        value = self.run("exec", "-T", service, "python", "-c", code).stdout.strip()
        return int(value)

    def mark(self, scenario: str) -> None:
        self.scenarios[scenario] = "passed"


def require(condition: object, message: str) -> None:
    if not condition:
        raise E2EFailure(message)


def execute(e2e: PublicE2E) -> None:
    fixtures = e2e.root / "tests/e2e/fixtures"

    print("[1/10] building isolated application and frontend images", flush=True)
    e2e.run("build", "api", "frontend", "provider-stub", timeout=600)
    e2e.run("up", "-d", "--wait", "--wait-timeout", "90", "postgres")
    e2e.run("run", "--rm", "--no-deps", "storage-init")
    pre_migration = e2e.run(
        "run",
        "--rm",
        "worker",
        "python",
        "-m",
        "apps.worker.main",
        "--check",
        check=False,
    )
    require(pre_migration.returncode != 0, "runtime was ready before migration")
    e2e.run("run", "--rm", "migrate")
    e2e.run(
        "up",
        "-d",
        "--wait",
        "--wait-timeout",
        "120",
        "provider-stub",
        "api",
        "worker",
        "frontend",
    )
    e2e.wait_http(e2e.api_origin, "/health/ready")
    e2e.wait_http(e2e.frontend_origin, "/health")
    runtime_config = json.loads(
        urllib.request.urlopen(
            f"{e2e.frontend_origin}/runtime-config.json", timeout=5
        ).read()
    )
    require(
        runtime_config == {"api_base_url": f"{e2e.api_origin}/api/v1"},
        "frontend runtime config did not target the isolated API",
    )
    e2e.mark("clean_migrate_start")

    print("[2/10] creating a knowledge base and serving the first upload", flush=True)
    kb = e2e.json_request(
        "POST",
        "/api/v1/knowledge-bases",
        value={"name": "S06 W02 E2E"},
        headers=e2e.idempotency_headers(),
        expected_status=201,
    )
    kb_id = str(kb["id"])
    first = e2e.upload(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        fixtures / "handbook-v1.md",
        display_name="E2E handbook",
    )
    document_id = str(first["document"]["id"])
    first_version_id = str(first["document_version_id"])
    first_job = e2e.wait_job(str(first["job_id"]))
    require(
        first_job.get("status") == "completed"
        and first_job.get("build_status") == "ready"
        and first_job.get("serving_status") == "serving",
        "first upload did not become ready and serving",
    )
    initial_pack = e2e.retrieval(kb_id, "E2E_ALPHA")
    require(
        initial_pack.get("debug", {}).get("query_plan", {}).get("serving_status")
        == "serving",
        "retrieval debug omitted the mandatory serving filter",
    )
    first_hits = [
        item
        for item in initial_pack.get("evidence", [])
        if item.get("document_id") == document_id
    ]
    require(first_hits and first_hits[0].get("document_version_id") == first_version_id, "first serving version was not retrieved")
    e2e.mark("upload_index_retrieval")

    print("[3/10] completing a grounded answer through terminal SSE", flush=True)
    session = e2e.json_request(
        "POST",
        "/api/v1/chat/sessions",
        value={"knowledge_base_id": kb_id, "title": "E2E session"},
        expected_status=201,
    )
    session_id = str(session["id"])
    created_run = e2e.create_run(kb_id, session_id, "What is the E2E_ALPHA fact?")
    events = e2e.sse_to_terminal(str(created_run["events_url"]))
    require(len(events) == 1 and events[0][0] == "answer.completed", "terminal SSE did not deliver one completed event")
    successful = e2e.wait_run(str(created_run["run_id"]))
    require(successful.get("status") == "completed", "grounded ChatRun did not complete")
    citations = successful.get("citations", [])
    require(
        len(citations) == 1
        and citations[0].get("ordinal") == 0
        and citations[0].get("document_id") == document_id
        and citations[0].get("document_version_id") == first_version_id,
        "completed answer did not preserve the ordered first-version citation",
    )
    require("[1]" in str(successful.get("answer")), "rendered answer omitted its citation marker")
    navigated = e2e.json_request("GET", f"/api/v1/documents/{document_id}")
    require(navigated.get("id") == document_id, "citation document navigation failed")
    e2e.mark("answer_sse_citation")

    print("[4/10] proving old-version visibility and atomic update cutover", flush=True)
    e2e.run("stop", "worker")
    second = e2e.upload(
        f"/api/v1/documents/{document_id}/versions",
        fixtures / "handbook-v2.md",
        display_name="E2E handbook",
    )
    second_version_id = str(second["document_version_id"])
    while_worker_stopped = e2e.retrieval(kb_id, "E2E_ALPHA")
    require(
        any(
            item.get("document_id") == document_id
            and item.get("document_version_id") == first_version_id
            for item in while_worker_stopped.get("evidence", [])
        ),
        "old serving version disappeared before update promotion",
    )
    e2e.run("up", "-d", "--wait", "--wait-timeout", "60", "worker")
    second_job = e2e.wait_job(str(second["job_id"]))
    require(second_job.get("serving_status") == "serving", "updated version was not promoted")
    updated_pack = e2e.retrieval(kb_id, "E2E_BETA")
    require(
        any(
            item.get("document_id") == document_id
            and item.get("document_version_id") == second_version_id
            for item in updated_pack.get("evidence", [])
        )
        and not any(
            item.get("document_id") == document_id
            and item.get("document_version_id") == first_version_id
            for item in updated_pack.get("evidence", [])
        ),
        "retrieval did not atomically cut over to the updated version",
    )
    e2e.mark("version_update_cutover")

    print("[5/10] exhausting and explicitly retrying indexing", flush=True)
    failed_upload = e2e.upload(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        fixtures / "index-failure.md",
        display_name="E2E retry fixture",
    )
    failed_job = e2e.wait_job(str(failed_upload["job_id"]))
    require(
        failed_job.get("status") == "failed"
        and failed_job.get("attempt") == 3
        and failed_job.get("can_retry") is True,
        "indexing failure did not exhaust into an eligible public retry",
    )
    retried = e2e.json_request(
        "POST",
        f"/api/v1/indexing-jobs/{failed_upload['job_id']}/retry",
        headers=e2e.idempotency_headers(),
        expected_status=202,
    )
    require(retried.get("status") == "queued" and retried.get("attempt") == 0, "explicit retry did not reset the same durable job")
    recovered_job = e2e.wait_job(str(failed_upload["job_id"]))
    require(
        recovered_job.get("status") == "completed"
        and recovered_job.get("serving_status") == "serving",
        "explicit indexing retry did not recover",
    )
    e2e.mark("indexing_failure_retry")

    print("[6/10] exercising SSE disconnect and timeout status recovery", flush=True)
    disconnected = e2e.create_run(
        kb_id,
        session_id,
        "Explain E2E_BETA E2E_CHAT_DELAY after client disconnect.",
    )
    e2e.disconnect_sse(str(disconnected["events_url"]))
    disconnect_terminal = e2e.wait_run(str(disconnected["run_id"]), timeout=30)
    require(disconnect_terminal.get("status") == "completed", "SSE disconnect cancelled the ChatRun")

    e2e.run("stop", "worker")
    timed = e2e.create_run(kb_id, session_id, "Explain E2E_BETA after stream timeout.")
    started = time.monotonic()
    timeout_events = e2e.sse_to_terminal(str(timed["events_url"]), timeout=8)
    elapsed = time.monotonic() - started
    require(not timeout_events and 1.0 <= elapsed < 6.0, "terminal SSE timeout was not bounded and empty")
    queued = e2e.json_request("GET", f"/api/v1/chat/runs/{timed['run_id']}")
    require(queued.get("status") == "queued", "SSE timeout changed the queued ChatRun")
    e2e.run("up", "-d", "--wait", "--wait-timeout", "60", "worker")
    timeout_terminal = e2e.wait_run(str(timed["run_id"]))
    require(timeout_terminal.get("status") == "completed", "status recovery did not observe the timed-out stream result")
    e2e.mark("sse_disconnect_timeout_recovery")

    print("[7/10] recording a content-safe terminal chat failure", flush=True)
    failed_run = e2e.create_run(kb_id, session_id, "Explain E2E_BETA E2E_CHAT_FAIL.")
    failed_terminal = e2e.wait_run(str(failed_run["run_id"]))
    error = failed_terminal.get("error") or {}
    require(
        failed_terminal.get("status") == "failed"
        and failed_terminal.get("attempt") == 3
        and error.get("code") == "CHAT_PROVIDER_UNAVAILABLE",
        "chat provider failure did not become the expected terminal fact",
    )
    require(set((error.get("detail") or {})) <= {"check", "retry_exhausted", "http_status", "retryable", "operation", "limit"}, "chat failure exposed an unreviewed diagnostic field")
    messages = e2e.json_request(
        "GET",
        f"/api/v1/chat/sessions/{session_id}/messages?limit=100&sort=created_at",
    )
    require(
        len(messages.get("items", [])) == 8
        and messages["items"][-1].get("assistant_status") == "failed",
        "authoritative history did not retain all four runs and terminal failure",
    )
    e2e.mark("chat_failure_history")

    print("[8/10] restarting every local runtime boundary", flush=True)
    source_count = e2e.source_file_count("worker")
    require(source_count >= 3 and e2e.source_file_count("api") == source_count, "API and Worker did not share the final source volume")
    e2e.run("restart", "frontend")
    e2e.run("up", "-d", "--wait", "--wait-timeout", "60", "frontend")
    e2e.wait_http(e2e.frontend_origin, "/health")
    e2e.run("restart", "api")
    e2e.run("up", "-d", "--wait", "--wait-timeout", "60", "api")
    e2e.wait_http(e2e.api_origin, "/health/ready")
    e2e.run("restart", "worker")
    e2e.run("up", "-d", "--wait", "--wait-timeout", "60", "worker")
    e2e.run("restart", "postgres")
    e2e.run("up", "-d", "--wait", "--wait-timeout", "60", "postgres")
    e2e.run("restart", "api", "worker")
    e2e.run("up", "-d", "--wait", "--wait-timeout", "90", "api", "worker")
    persisted_document = e2e.json_request("GET", f"/api/v1/documents/{document_id}")
    persisted_messages = e2e.json_request(
        "GET",
        f"/api/v1/chat/sessions/{session_id}/messages?limit=100&sort=created_at",
    )
    persisted_run = e2e.json_request("GET", f"/api/v1/chat/runs/{created_run['run_id']}")
    require(
        persisted_document.get("current_version", {}).get("id") == second_version_id
        and len(persisted_messages.get("items", [])) == 8
        and persisted_run.get("citations", [{}])[0].get("document_version_id") == first_version_id
        and e2e.source_file_count("worker") == source_count,
        "restart changed business history, citation snapshots, or source-file visibility",
    )
    e2e.mark("runtime_restart_persistence")

    print("[9/10] deleting the document without erasing citation history", flush=True)
    deleted = e2e.json_request(
        "DELETE",
        f"/api/v1/documents/{document_id}",
        headers=e2e.idempotency_headers(),
    )
    require(deleted.get("document", {}).get("deleted_at"), "document delete did not persist its tombstone")
    after_delete = e2e.retrieval(kb_id, "E2E_BETA")
    require(
        not any(item.get("document_id") == document_id for item in after_delete.get("evidence", [])),
        "deleted document remained retrievable",
    )
    citation_after_delete = e2e.json_request("GET", f"/api/v1/chat/runs/{created_run['run_id']}")
    require(
        citation_after_delete.get("citations", [{}])[0].get("document_id") == document_id,
        "delete erased the committed citation snapshot",
    )
    e2e.mark("delete_and_citation_snapshot")

    print("[10/10] checking content-safe logs and graceful stop", flush=True)
    logs = e2e.run(
        "logs", "--no-color", "provider-stub", "frontend", "api", "worker"
    ).stdout
    for forbidden in (
        "ALPHA-1 is the current handbook fact",
        "BETA-2 replaces the earlier handbook fact",
        "e2e-admin-password",
        "e2e-migration-password",
        "e2e-runtime-password",
    ):
        require(forbidden not in logs, f"content-safe log boundary leaked {forbidden}")
    machine = e2e.run(
        "exec",
        "-T",
        "api",
        "python",
        "-c",
        "import platform; print(platform.machine())",
    ).stdout.strip()
    e2e.tested_platform = f"linux/{'arm64' if machine == 'aarch64' else machine}"
    e2e.run("stop", "--timeout", "15", "frontend", "api", "worker", "provider-stub")
    stopped_logs = e2e.run("logs", "--no-color", "api", "worker").stdout
    require(stopped_logs.count('"event":"process_stopped"') >= 2, "API and Worker did not stop gracefully")
    e2e.mark("content_safe_logs_shutdown")


def report(e2e: PublicE2E, *, started_at: datetime, duration_seconds: float) -> dict[str, Any]:
    hashes = {
        path: hashlib.sha256((e2e.root / path).read_bytes()).hexdigest()
        for path in EVIDENCE_INPUTS
    }
    return {
        "schema_version": "1.0",
        "artifact": "stage06-end-to-end-integration",
        "status": "passed",
        "verified_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "duration_seconds": round(duration_seconds, 3),
        "tested_platform": e2e.tested_platform,
        "inputs": {"algorithm": "sha256", "files": hashes},
        "isolation": {
            "disposable_compose_project": True,
            "randomized_loopback_ports": True,
            "temporary_database_credentials": True,
            "host_project_runtime_inputs_inherited": False,
            "external_model_credentials": False,
            "provider_host_port_published": False,
        },
        "provider": {
            "protocol": "OpenAI-compatible deterministic test double",
            "embedding_dimension": 1024,
            "indexing_failures_before_recovery": 3,
            "chat_failure": "stable HTTP 503",
            "delayed_chat_seconds": 2.5,
        },
        "scenarios": e2e.scenarios,
        "public_boundary": {
            "application_business_calls": "/api/v1 only",
            "database_business_mutation": False,
            "private_application_endpoint": False,
        },
        "recorded_limits": [
            "the deterministic provider proves orchestration and contracts, not real-provider quality or latency",
            "browser rendering was manually accepted by Maxui; the reproducible suite uses real HTTP/SSE plus the separately automated frontend components",
            "linux/amd64 remains recorded but was not executed",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    e2e = PublicE2E(root)
    started_at = datetime.now(UTC)
    started = time.monotonic()
    cleanup_error: Exception | None = None
    try:
        execute(e2e)
    except (E2EFailure, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"End-to-end integration failed: {error}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
        if arguments.report is not None:
            arguments.report.write_text(
                json.dumps(
                    report(
                        e2e,
                        started_at=started_at,
                        duration_seconds=time.monotonic() - started,
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        try:
            cleanup = e2e.run(
                "down",
                "--volumes",
                "--remove-orphans",
                check=False,
            )
            if cleanup.returncode != 0:
                cleanup_error = E2EFailure(
                    f"Compose cleanup returned {cleanup.returncode}"
                )
        except (OSError, subprocess.SubprocessError) as error:
            cleanup_error = error
    if cleanup_error is not None:
        print(f"End-to-end cleanup failed: {cleanup_error}", file=sys.stderr)
        return 1
    if return_code == 0:
        print(
            "End-to-end integration passed: public upload, indexing, retrieval, "
            "answer/citation/history, update/delete, retry, SSE recovery, all "
            "runtime restarts, content-safe logs, and clean teardown",
            flush=True,
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
