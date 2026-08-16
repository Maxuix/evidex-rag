#!/usr/bin/env python3
"""Export a bounded, content-safe issue diagnostics bundle."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import platform
import stat
import subprocess
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from rag_kb.observability.logging import SAFE_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIRECTORY = PROJECT_ROOT / ".runtime" / "logs"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / ".runtime" / "diagnostics"
DEFAULT_SINCE_HOURS = 72.0
DEFAULT_MAX_EVENTS = 50_000
MAX_INPUT_LOG_FILE_BYTES = 12 * 1024 * 1024
CORE_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "timestamp",
        "level",
        "logger",
        "event",
        "process",
        "runtime_id",
        "pid",
        "source",
        "exception",
    }
)
ALLOWED_EVENT_FIELDS = CORE_EVENT_FIELDS | SAFE_FIELDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-directory",
        type=Path,
        default=DEFAULT_LOG_DIRECTORY,
        help="directory containing the rotating application JSONL logs",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="directory that receives the diagnostics zip",
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=DEFAULT_SINCE_HOURS,
        help="include application events no older than this many hours",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help="maximum number of newest safe events to include",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://127.0.0.1:8000",
        help="loopback API origin used only for status-code health probes",
    )
    parser.add_argument(
        "--frontend-base-url",
        default="http://127.0.0.1:3000",
        help="loopback user frontend origin used only for a health probe",
    )
    arguments = parser.parse_args()
    if arguments.since_hours <= 0 or arguments.max_events <= 0:
        parser.error("--since-hours and --max-events must be positive")
    _require_loopback_http_origin(arguments.api_base_url)
    _require_loopback_http_origin(arguments.frontend_base_url)

    now = datetime.now(UTC)
    events, event_summary = collect_safe_events(
        arguments.log_directory,
        since=now - timedelta(hours=arguments.since_hours),
        max_events=arguments.max_events,
    )
    docker_state = collect_docker_state()
    health = collect_health(
        arguments.api_base_url,
        arguments.frontend_base_url,
    )
    git_state = collect_git_state()
    manifest = {
        "schema_version": 1,
        "created_at": now.isoformat(),
        "content_policy": (
            "content-safe metadata only; no environment, bodies, messages, "
            "exception text, source lines, locals, or raw unstructured logs"
        ),
        "window_hours": arguments.since_hours,
        "event_summary": event_summary,
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }

    output_directory = _prepare_output_directory(arguments.output_directory)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_path = output_directory / f"rag-kb-diagnostics-{stamp}.zip"
    if output_path.exists():
        raise FileExistsError(f"diagnostics bundle already exists: {output_path}")
    with zipfile.ZipFile(
        output_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        _write_json(archive, "manifest.json", manifest)
        _write_json(archive, "docker.json", docker_state)
        _write_json(archive, "health.json", health)
        _write_json(archive, "git.json", git_state)
        archive.writestr(
            "events.jsonl",
            "".join(
                json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n"
                for event in events
            ),
        )
        archive.writestr(
            "README.txt",
            (
                "RAG KB content-safe issue diagnostics bundle.\n"
                "events.jsonl contains only allowlisted application metadata.\n"
                "Raw Docker output, .env files, request/model content, exception "
                "messages, source lines and local variables are intentionally absent.\n"
            ),
        )
    output_path.chmod(0o600)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "bundle": str(output_path),
                "events": len(events),
                "sha256": digest,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def collect_safe_events(
    log_directory: Path,
    *,
    since: datetime,
    max_events: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    if not log_directory.exists():
        return [], {"included": 0, "log_directory_present": False}
    status = log_directory.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError("log directory must be a real directory")

    for path in sorted(log_directory.glob("*.jsonl*")):
        file_status = path.lstat()
        if stat.S_ISLNK(file_status.st_mode) or not stat.S_ISREG(file_status.st_mode):
            counts["non_regular_files"] += 1
            continue
        if file_status.st_size > MAX_INPUT_LOG_FILE_BYTES:
            counts["oversized_files"] += 1
            continue
        counts["files"] += 1
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                counts["lines"] += 1
                try:
                    candidate = json.loads(line)
                    event = sanitize_event(candidate)
                    timestamp = datetime.fromisoformat(str(event["timestamp"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    counts["invalid_or_unsafe"] += 1
                    continue
                if timestamp.tzinfo is None or timestamp < since:
                    counts["outside_window"] += 1
                    continue
                records.append(event)

    records.sort(key=lambda item: (str(item["timestamp"]), str(item["runtime_id"])))
    if len(records) > max_events:
        counts["truncated"] = len(records) - max_events
        records = records[-max_events:]
    counts["included"] = len(records)
    summary: dict[str, object] = dict(counts)
    summary["log_directory_present"] = True
    if records:
        summary["earliest"] = records[0]["timestamp"]
        summary["latest"] = records[-1]["timestamp"]
        summary["levels"] = dict(Counter(str(item["level"]) for item in records))
        summary["processes"] = dict(
            Counter(str(item["process"]) for item in records)
        )
    return records, summary


def sanitize_event(value: Any) -> dict[str, object]:
    if not isinstance(value, dict) or value.keys() - ALLOWED_EVENT_FIELDS:
        raise ValueError("event contains unsupported fields")
    required = {
        "schema_version",
        "timestamp",
        "level",
        "logger",
        "event",
        "process",
        "runtime_id",
        "pid",
        "source",
    }
    if not required.issubset(value):
        raise ValueError("event is missing required fields")
    result: dict[str, object] = {}
    for key, item in value.items():
        if key == "source":
            result[key] = _sanitize_source(item)
        elif key == "exception":
            result[key] = _sanitize_exception(item)
        else:
            result[key] = _sanitize_scalar(item)
    return result


def _sanitize_source(value: Any) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"module", "function", "line"}:
        raise ValueError("source location is invalid")
    return {
        "module": _sanitize_scalar(value["module"]),
        "function": _sanitize_scalar(value["function"]),
        "line": _sanitize_scalar(value["line"]),
    }


def _sanitize_exception(value: Any) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "type",
        "chain",
        "frames",
        "fingerprint",
    }:
        raise ValueError("exception diagnostics are invalid")
    chain = value["chain"]
    frames = value["frames"]
    if not isinstance(chain, list) or len(chain) > 4:
        raise ValueError("exception chain is invalid")
    if not isinstance(frames, list) or len(frames) > 16:
        raise ValueError("exception frames are invalid")
    return {
        "type": _sanitize_scalar(value["type"]),
        "chain": [_sanitize_scalar(item) for item in chain],
        "frames": [_sanitize_source(item) for item in frames],
        "fingerprint": _sanitize_scalar(value["fingerprint"]),
    }


def _sanitize_scalar(value: Any) -> object:
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("non-finite number")
        return value
    if isinstance(value, str) and len(value) <= 512:
        if any(ord(character) < 32 for character in value):
            raise ValueError("control character")
        return value
    raise ValueError("unsafe non-scalar value")


def collect_docker_state() -> dict[str, object]:
    env_file = PROJECT_ROOT / ".env.local"
    if not env_file.is_file():
        return {"available": False, "reason": "compose_env_missing"}
    try:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "--env-file",
                str(env_file),
                "ps",
                "-a",
                "--format",
                "json",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = _parse_compose_json(result.stdout)
        services = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            services.append(
                {
                    key: _sanitize_scalar(row[key])
                    for key in (
                        "Name",
                        "Service",
                        "State",
                        "Status",
                        "Health",
                        "ExitCode",
                    )
                    if key in row and row[key] is not None
                }
            )
        return {"available": True, "services": services}
    except Exception as error:
        return {"available": False, "error_type": type(error).__name__}


def collect_health(
    api_base_url: str,
    frontend_base_url: str,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    endpoints = (
        ("api", "/health/live", f"{api_base_url}/health/live"),
        ("api", "/health/ready", f"{api_base_url}/health/ready"),
        ("frontend", "/health", f"{frontend_base_url}/health"),
    )
    for component, path, url in endpoints:
        started = perf_counter()
        status_code: int | None = None
        error_type: str | None = None
        try:
            request = Request(url, method="GET")
            with urlopen(request, timeout=3) as response:  # noqa: S310 - loopback only
                status_code = response.status
        except HTTPError as error:
            status_code = error.code
            error_type = type(error).__name__
        except (OSError, URLError, TimeoutError) as error:
            error_type = type(error).__name__
        result: dict[str, object] = {
            "component": component,
            "path": path,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
        }
        if status_code is not None:
            result["status_code"] = status_code
        if error_type is not None:
            result["error_type"] = error_type
        results.append(result)
    return results


def collect_git_state() -> dict[str, object]:
    try:
        revision = _git("rev-parse", "HEAD").strip()
        branch = _git("branch", "--show-current").strip()
        porcelain = _git("status", "--porcelain").splitlines()
        statuses = Counter(line[:2] for line in porcelain if len(line) >= 2)
        return {
            "available": True,
            "revision": revision,
            "branch": branch,
            "dirty": bool(porcelain),
            "change_status_counts": dict(statuses),
        }
    except Exception as error:
        return {"available": False, "error_type": type(error).__name__}


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout


def _parse_compose_json(value: str) -> list[dict[str, Any]]:
    if not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in value.splitlines() if line.strip()]
    rows = parsed if isinstance(parsed, list) else [parsed]
    return [row for row in rows if isinstance(row, dict)]


def _prepare_output_directory(path: Path) -> Path:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError("output directory must be a real directory")
    path.chmod(0o700)
    return path


def _require_loopback_http_origin(value: str) -> None:
    from urllib.parse import urlsplit

    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("api-base-url has an invalid port") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or port is None
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("api-base-url must be an uncredentialed loopback HTTP origin")


def _write_json(
    archive: zipfile.ZipFile,
    name: str,
    value: object,
) -> None:
    archive.writestr(
        name,
        json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
