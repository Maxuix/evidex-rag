#!/usr/bin/env python3
"""Exercise a clean Stage 03 Compose start, upload, storage, and shutdown."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class ComposeSmoke:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = f"rag-kb-s03-w04-{os.getpid()}"
        self.environment = {
            **os.environ,
            "POSTGRES_ADMIN_PASSWORD": "smoke-admin-password",
            "RAG_KB_MIGRATION_PASSWORD": "smoke-migration-password",
            "RAG_KB_RUNTIME_PASSWORD": "smoke-runtime-password",
            "RAG_KB_POSTGRES_PORT": str(available_port()),
            "RAG_KB_API_PORT": str(available_port()),
            "RAG_KB_FRONTEND_PORT": str(available_port()),
        }
        self.base = [
            "docker",
            "compose",
            "--env-file",
            ".env.example",
            "--profile",
            "tools",
            "--project-name",
            self.project,
        ]

    def run(
        self,
        *arguments: str,
        check: bool = True,
        timeout: int = 180,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.base, *arguments],
            cwd=self.root,
            env=self.environment,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def http_bytes(self, port_name: str, path: str) -> bytes:
        port = self.environment[port_name]
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=5
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"{path} returned {response.status}")
            return response.read()

    def http_json(self, port_name: str, path: str) -> dict[str, object]:
        return json.loads(self.http_bytes(port_name, path))

    def request_json(
        self,
        method: str,
        path: str,
        *,
        body: bytes,
        headers: dict[str, str],
        expected_status: int,
    ) -> dict[str, object]:
        port = self.environment["RAG_KB_API_PORT"]
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != expected_status:
                raise RuntimeError(f"{path} returned {response.status}")
            return json.loads(response.read())

    def wait_http(self, port_name: str, path: str, timeout: float = 60) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self.http_bytes(port_name, path)
                return
            except Exception as error:
                last_error = error
                time.sleep(1)
        raise RuntimeError(f"timed out waiting for {path}: {last_error}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    smoke = ComposeSmoke(root)
    try:
        print("[1/8] building the pinned application image", flush=True)
        smoke.run("build", "api", timeout=600)

        print("[2/8] starting clean PostgreSQL and source storage", flush=True)
        smoke.run("up", "-d", "--wait", "--wait-timeout", "90", "postgres")
        smoke.run("run", "--rm", "--no-deps", "storage-init")

        print("[3/8] proving runtime startup fails before explicit migration", flush=True)
        before_migration = smoke.run(
            "run",
            "--rm",
            "worker",
            "python",
            "-m",
            "apps.worker.main",
            "--check",
            check=False,
        )
        if before_migration.returncode == 0:
            raise RuntimeError("Worker unexpectedly became ready before migration")

        print("[4/8] running the one-shot migration role", flush=True)
        smoke.run("run", "--rm", "migrate")

        print("[5/8] starting API, Worker, and frontend shell", flush=True)
        smoke.run(
            "up",
            "-d",
            "--wait",
            "--wait-timeout",
            "120",
            "api",
            "worker",
            "frontend",
        )
        smoke.wait_http("RAG_KB_API_PORT", "/health/live")
        readiness = smoke.http_json("RAG_KB_API_PORT", "/health/ready")
        if readiness.get("status") != "ready":
            raise RuntimeError("API readiness did not report ready")
        smoke.wait_http("RAG_KB_FRONTEND_PORT", "/")
        smoke.run("exec", "-T", "worker", "python", "-m", "apps.worker.parser_check")

        print("[6/8] exercising bounded public upload", flush=True)
        created = smoke.request_json(
            "POST",
            "/api/v1/knowledge-bases",
            body=json.dumps({"name": "Smoke KB"}).encode(),
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": str(uuid.uuid4()),
            },
            expected_status=201,
        )
        uploaded = smoke.request_json(
            "POST",
            f"/api/v1/knowledge-bases/{created['id']}/documents",
            body=b"# Smoke\nrestart-safe upload",
            headers={
                "Content-Type": "text/markdown; charset=utf-8",
                "X-Document-Filename": "smoke.md",
                "X-Document-Display-Name": "Smoke document",
                "Idempotency-Key": str(uuid.uuid4()),
            },
            expected_status=202,
        )
        if uploaded.get("job_status") != "queued":
            raise RuntimeError("upload did not return a queued durable job")

        print("[7/8] verifying shared same-filesystem storage across restart", flush=True)
        devices = {
            smoke.run("exec", "-T", service, "stat", "-c", "%d", path).stdout.strip()
            for service in ("api", "worker")
            for path in (
                "/var/lib/rag-kb/sources/staging",
                "/var/lib/rag-kb/sources/final",
            )
        }
        if len(devices) != 1:
            raise RuntimeError(f"source paths span multiple filesystems: {devices}")
        smoke.run(
            "exec",
            "-T",
            "api",
            "touch",
            "/var/lib/rag-kb/sources/final/s03-w04-smoke-marker",
        )
        document = smoke.http_json(
            "RAG_KB_API_PORT", f"/api/v1/documents/{uploaded['document']['id']}"
        )
        if document.get("current_version", {}).get("source_status") != "available":
            raise RuntimeError("uploaded source was not readable after restart")
        smoke.run("restart", "api", "worker")
        smoke.run("up", "-d", "--wait", "--wait-timeout", "90", "api", "worker")
        smoke.run(
            "exec",
            "-T",
            "worker",
            "test",
            "-f",
            "/var/lib/rag-kb/sources/final/s03-w04-smoke-marker",
        )

        smoke.http_json(
            "RAG_KB_API_PORT",
            "/health/live?request-content-must-not-appear",
        )
        logs = smoke.run("logs", "--no-color", "api", "worker").stdout
        for forbidden in (
            "request-content-must-not-appear",
            "smoke-admin-password",
            "smoke-migration-password",
            "smoke-runtime-password",
        ):
            if forbidden in logs:
                raise RuntimeError(f"content-safe log boundary leaked {forbidden}")
        if '"event":"http_request_completed"' not in logs or '"trace_id"' not in logs:
            raise RuntimeError("structured request correlation fields were not emitted")

        print("[8/8] exercising graceful process shutdown", flush=True)
        smoke.run("stop", "--timeout", "15", "frontend", "api", "worker")
        stopped_logs = smoke.run("logs", "--no-color", "api", "worker").stdout
        if stopped_logs.count('"event":"process_stopped"') < 2:
            raise RuntimeError("API and Worker did not emit structured shutdown events")
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"Compose smoke failed: {error}", file=sys.stderr)
        return_code = 1
    else:
        print(
            "Compose smoke passed: clean start, explicit migration, bounded upload, "
            "shared persistence, content-safe logs, and shutdown",
            flush=True,
        )
        return_code = 0
    finally:
        try:
            smoke.run("down", "--volumes", "--remove-orphans", check=False)
        except (OSError, subprocess.SubprocessError):
            pass
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
