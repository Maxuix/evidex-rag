#!/usr/bin/env python3
"""Exercise a clean local Compose start, frontend, upload, and shutdown."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from http.client import HTTPMessage
import urllib.error
import urllib.request
import uuid
from pathlib import Path


SENSITIVE_COMPOSE_ENVIRONMENT = {
    "POSTGRES_ADMIN_PASSWORD",
    "RAG_KB_MIGRATION_PASSWORD",
    "RAG_KB_RUNTIME_PASSWORD",
}


def inherited_smoke_environment() -> dict[str, str]:
    """Keep host tooling context without inheriting project runtime inputs."""

    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("RAG_KB") and key not in SENSITIVE_COMPOSE_ENVIRONMENT
    }


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class ComposeSmoke:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = f"rag-kb-s06-w01-{os.getpid()}"
        self.embedding_stub = f"{self.project}-embedding-stub"
        self.environment = {
            **inherited_smoke_environment(),
            "POSTGRES_ADMIN_PASSWORD": "smoke-admin-password",
            "RAG_KB_MIGRATION_PASSWORD": "smoke-migration-password",
            "RAG_KB_RUNTIME_PASSWORD": "smoke-runtime-password",
            "RAG_KB_POSTGRES_PORT": str(available_port()),
            "RAG_KB_API_PORT": str(available_port()),
            "RAG_KB_FRONTEND_PORT": str(available_port()),
            "RAG_KB_SMOKE_EMBEDDING_HOST": self.embedding_stub,
            "RAG_KB_ENV_FILE": ".env.example",
        }
        self.base = [
            "docker",
            "compose",
            "-f",
            "compose.yaml",
            "-f",
            "deploy/compose-smoke.override.yaml",
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
        _, body = self.http_response(port_name, path)
        return body

    def http_response(
        self,
        port_name: str,
        path: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        expected_status: int = 200,
    ) -> tuple[HTTPMessage, bytes]:
        port = self.environment[port_name]
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            method=method,
            headers=headers or {},
        )
        try:
            response = urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as error:
            if error.code != expected_status:
                raise RuntimeError(f"{path} returned {error.code}") from error
            return error.headers, error.read()
        with response:
            if response.status != expected_status:
                raise RuntimeError(f"{path} returned {response.status}")
            return response.headers, response.read()

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

    def wait_indexing(self, job_id: str, target_id: str, timeout: float = 60) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.run(
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "postgres",
                "-d",
                "rag_kb",
                "-Atc",
                (
                    "SELECT job.status::text || '|' || target.build_status::text "
                    "|| '|' || target.serving_status::text || '|' || job.attempt "
                    "FROM indexing_job job JOIN indexed_document_version target "
                    "ON target.id = job.indexed_document_version_id "
                    f"WHERE job.id = '{job_id}' AND target.id = '{target_id}'"
                ),
            ).stdout.strip()
            if state == "completed|ready|serving|1":
                return
            if state.startswith("failed|"):
                raise RuntimeError(f"automatic indexing failed: {state}")
            time.sleep(0.25)
        raise RuntimeError("timed out waiting for automatic indexing")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    smoke = ComposeSmoke(root)
    cleanup_error: Exception | None = None
    try:
        print("[1/9] building the pinned backend and frontend images", flush=True)
        smoke.run("build", "api", "frontend", timeout=600)

        print("[2/9] starting clean PostgreSQL and source storage", flush=True)
        smoke.run("up", "-d", "--wait", "--wait-timeout", "90", "postgres")
        smoke.run("run", "--rm", "--no-deps", "storage-init")

        print("[3/9] proving runtime startup fails before explicit migration", flush=True)
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

        print("[4/9] running the one-shot migration role", flush=True)
        smoke.run("run", "--rm", "migrate")

        smoke.run(
            "run",
            "-d",
            "--name",
            smoke.embedding_stub,
            "--no-deps",
            "worker",
            "python",
            "/app/tools/embedding_stub.py",
        )
        time.sleep(0.5)

        print("[5/9] starting API, Worker, and observation frontend", flush=True)
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
        smoke.wait_http("RAG_KB_FRONTEND_PORT", "/health")
        frontend_config = smoke.http_json(
            "RAG_KB_FRONTEND_PORT", "/runtime-config.json"
        )
        expected_api_base = (
            "http://127.0.0.1:"
            f"{smoke.environment['RAG_KB_API_PORT']}/api/v1"
        )
        if frontend_config != {"api_base_url": expected_api_base}:
            raise RuntimeError(f"frontend runtime config mismatch: {frontend_config}")
        config_headers, _ = smoke.http_response(
            "RAG_KB_FRONTEND_PORT", "/runtime-config.json"
        )
        if config_headers.get("Cache-Control") != "no-store":
            raise RuntimeError("frontend runtime config is cacheable")
        alias_headers, alias_body = smoke.http_response(
            "RAG_KB_FRONTEND_PORT", "/runtime-config%2Ejson"
        )
        if (
            json.loads(alias_body) != frontend_config
            or alias_headers.get("Cache-Control") != "no-store"
        ):
            raise RuntimeError("frontend runtime config alias bypassed dynamic delivery")

        frontend_headers, frontend_html = smoke.http_response(
            "RAG_KB_FRONTEND_PORT", "/"
        )
        decoded_frontend = frontend_html.decode("utf-8")
        if "id=\"root\"" not in decoded_frontend or "/assets/" not in decoded_frontend:
            raise RuntimeError("frontend did not serve the compiled application shell")
        if frontend_headers.get("Cache-Control") != "no-cache":
            raise RuntimeError("frontend application shell has an unsafe cache policy")
        content_security_policy = frontend_headers.get("Content-Security-Policy", "")
        expected_api_origin = expected_api_base.removesuffix("/api/v1")
        if f"connect-src 'self' {expected_api_origin};" not in content_security_policy:
            raise RuntimeError("frontend security headers are missing")
        asset_paths = [
            token.split('"', 1)[0]
            for marker in ('src="/assets/', 'href="/assets/')
            for token in decoded_frontend.split(marker)[1:]
        ]
        if not asset_paths:
            raise RuntimeError("frontend application shell has no compiled assets")
        for asset in asset_paths:
            asset_headers, asset_body = smoke.http_response(
                "RAG_KB_FRONTEND_PORT", f"/assets/{asset}"
            )
            if not asset_body:
                raise RuntimeError(f"frontend asset is empty: {asset}")
            if asset_headers.get("Cache-Control") != (
                "public, max-age=31536000, immutable"
            ):
                raise RuntimeError(f"frontend asset cache policy is unsafe: {asset}")
            media_type = asset_headers.get("Content-Type", "").split(";", 1)[0]
            expected_media_types = (
                {"text/javascript", "application/javascript"}
                if asset.endswith(".js")
                else {"text/css"}
                if asset.endswith(".css")
                else set()
            )
            if expected_media_types and media_type not in expected_media_types:
                raise RuntimeError(
                    f"frontend asset MIME type is unsafe: {asset} -> {media_type}"
                )
        deep_link = smoke.http_bytes("RAG_KB_FRONTEND_PORT", "/chat")
        if deep_link != frontend_html:
            raise RuntimeError("frontend SPA deep link did not return the application shell")
        smoke.http_response(
            "RAG_KB_FRONTEND_PORT",
            "/assets/not-present.js",
            expected_status=404,
        )

        frontend_origin = (
            "http://127.0.0.1:"
            f"{smoke.environment['RAG_KB_FRONTEND_PORT']}"
        )
        cors_headers, _ = smoke.http_response(
            "RAG_KB_API_PORT",
            "/api/v1/knowledge-bases",
            method="OPTIONS",
            headers={
                "Origin": frontend_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": (
                    "content-type,idempotency-key,x-document-filename,"
                    "x-document-display-name"
                ),
            },
        )
        if cors_headers.get("Access-Control-Allow-Origin") != frontend_origin:
            raise RuntimeError("API did not echo the configured frontend CORS origin")
        if cors_headers.get("Access-Control-Allow-Credentials") is not None:
            raise RuntimeError("API unexpectedly allowed browser credentials")
        allowed_headers = {
            value.strip().lower()
            for value in cors_headers.get("Access-Control-Allow-Headers", "").split(",")
        }
        required_headers = {
            "content-type",
            "idempotency-key",
            "x-document-filename",
            "x-document-display-name",
        }
        if not required_headers.issubset(allowed_headers):
            raise RuntimeError(f"API CORS upload headers are incomplete: {allowed_headers}")

        frontend_uid = smoke.run("exec", "-T", "frontend", "id", "-u").stdout.strip()
        if frontend_uid != "10001":
            raise RuntimeError(f"frontend runs as unexpected UID {frontend_uid}")
        frontend_environment = smoke.run(
            "exec", "-T", "frontend", "env"
        ).stdout.lower()
        if "rag_kb__" in frontend_environment or "password" in frontend_environment:
            raise RuntimeError("frontend received privileged application environment")
        smoke.run(
            "exec",
            "-T",
            "frontend",
            "sh",
            "-c",
            (
                "test -f /app/dist/index.html "
                "&& test -f /app/server.py "
                "&& test ! -e /app/apps "
                "&& test ! -e /app/src "
                "&& test ! -e /app/node_modules "
                "&& test ! -e /app/dist/runtime-config.json"
            ),
        )
        smoke.run("exec", "-T", "worker", "python", "-m", "apps.worker.parser_check")

        print("[6/9] exercising bounded public upload", flush=True)
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
        smoke.wait_indexing(
            str(uploaded["job_id"]),
            str(uploaded["indexed_document_version_id"]),
        )
        status = smoke.http_json(
            "RAG_KB_API_PORT",
            f"/api/v1/indexing-jobs/{uploaded['job_id']}",
        )
        if status.get("status") != "completed" or status.get("can_retry"):
            raise RuntimeError("public indexing status did not report completion")

        print("[7/9] verifying shared same-filesystem storage across restart", flush=True)
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
            "/var/lib/rag-kb/sources/final/s03-w07-smoke-marker",
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
            "/var/lib/rag-kb/sources/final/s03-w07-smoke-marker",
        )

        print("[8/9] running bounded idempotent maintenance", flush=True)
        maintenance = smoke.run("run", "--rm", "maintenance").stdout
        if '"retired_targets_cleaned"' not in maintenance:
            raise RuntimeError("maintenance did not emit its bounded result")
        preserved = smoke.http_json(
            "RAG_KB_API_PORT",
            f"/api/v1/indexing-jobs/{uploaded['job_id']}",
        )
        if preserved.get("serving_status") != "serving":
            raise RuntimeError("maintenance changed active serving content")

        smoke.http_json(
            "RAG_KB_API_PORT",
            "/health/live?request-content-must-not-appear",
        )
        logs = smoke.run("logs", "--no-color", "frontend", "api", "worker").stdout
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

        print("[9/9] exercising graceful process shutdown", flush=True)
        smoke.run("stop", "--timeout", "15", "frontend", "api", "worker")
        stopped_logs = smoke.run("logs", "--no-color", "api", "worker").stdout
        if stopped_logs.count('"event":"process_stopped"') < 2:
            raise RuntimeError("API and Worker did not emit structured shutdown events")
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"Compose smoke failed: {error}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        try:
            cleanup = smoke.run(
                "down",
                "--volumes",
                "--remove-orphans",
                check=False,
            )
            if cleanup.returncode != 0:
                cleanup_error = RuntimeError(
                    f"Compose cleanup returned {cleanup.returncode}"
                )
        except (OSError, subprocess.SubprocessError) as error:
            cleanup_error = error
    if cleanup_error is not None:
        print(f"Compose smoke cleanup failed: {cleanup_error}", file=sys.stderr)
        return 1
    if return_code == 0:
        print(
            "Compose smoke passed: clean start, explicit migration, automatic "
            "indexing/status, bounded maintenance, shared persistence, "
            "isolated frontend delivery, content-safe logs, and shutdown",
            flush=True,
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
