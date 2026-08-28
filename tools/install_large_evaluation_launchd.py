#!/usr/bin/env python3
"""Install the isolated large-evaluation supervisor as macOS LaunchAgents.

The installer creates one persistent state-machine agent and four supporting
runtime agents (PostgreSQL, FalkorDB, API, and Worker).  It never touches the
Docker Compose project.  Plists, configuration, PID/lock files, and logs are
owner-only; the state-machine agent exits successfully after a controlled
failure so launchd does not retry past a failed gate.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import plistlib
import signal
import stat
import subprocess
import time
from typing import Any, Iterable, Mapping
import uuid

from tools.check_large_evaluation_runtime import _load_private_plan
from tools.evaluation_campaign_state import digest, write_private_json
from tools.evaluation_runtime import (
    EvaluationRuntimeError,
    load_evaluation_runtime,
)
from tools.prepare_large_evaluation import ROOT
from tools.provision_large_evaluation_host import (
    DATASET_SPECS,
    _corpus_digest,
    _paths,
)
from tools.supervise_large_evaluation import (
    CONFIG_SCHEMA,
    DEFAULT_CONFIG,
    DATASET,
    RUN_MARKER_NAME,
    _automation_implementation_sha256,
)
from tools.evaluation_resilience import RESILIENCE_POLICY_SHA256


LABEL_PREFIX = "com.rag.large-evaluation"
LABELS = {
    "postgres": f"{LABEL_PREFIX}.postgres",
    "falkordb": f"{LABEL_PREFIX}.falkordb",
    "api": f"{LABEL_PREFIX}.api",
    "worker": f"{LABEL_PREFIX}.worker",
    "supervisor": f"{LABEL_PREFIX}.supervisor",
}
SUPERVISOR_ROOT = ROOT / ".runtime/evaluations/large-evaluation-route-variant-v1"
RUNTIME_MANIFEST = ROOT / ".runtime/evaluations/graph-schema-profiles-host/runtime.json"
PLAN = SUPERVISOR_ROOT / "preflight.json"
ROUTING_ROOT = ROOT / "evaluation/routing-rag-musique-short-support-v2"
BINDINGS_NAME = "large-evaluation-bindings-route-variant.json"
CHAT_PROFILE = "01a00fcd-cdfc-77cd-9c17-2f04ea515ae2"
TEXT_EMBEDDING_PROFILE = "01a00f9a-1db2-7db7-93c3-73f34b91f296"
MULTIMODAL_EMBEDDING_PROFILE = "01a00f9a-1db8-7c25-8cfc-6ad6ed344f76"


class LaunchdInstallError(RuntimeError):
    """A content-safe launchd installation failure."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--install", action="store_true")
    mode.add_argument("--uninstall", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--runtime-manifest", type=Path, default=RUNTIME_MANIFEST)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--routing-root", type=Path, default=ROUTING_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--handoff-existing", action="store_true")
    parser.add_argument("--reconfigure", action="store_true")
    return parser


def _ensure_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _private_regular(path: Path, *, code: str) -> Path:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise LaunchdInstallError(f"{code}_missing") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise LaunchdInstallError(f"{code}_invalid")
    if stat.S_IMODE(status.st_mode) & 0o077:
        raise LaunchdInstallError(f"{code}_permissions_invalid")
    return path.resolve()


def _atomic_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    _ensure_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(mode)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _prepare_private_output(path: Path) -> None:
    """Create launchd's inherited output targets before bootstrap.

    launchd creates a missing StandardOutPath/StandardErrorPath with the
    process umask, which is not a sufficient owner-only guarantee.  Pre-create
    and chmod the targets so the guarantee holds from the first byte.
    """

    _ensure_directory(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(descriptor)
    path.chmod(0o600)


def _launchctl(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _loaded(label: str) -> bool:
    result = _launchctl("print", f"{_domain()}/{label}")
    return result.returncode == 0


def _bootout(label: str) -> None:
    result = _launchctl("bootout", f"{_domain()}/{label}")
    if result.returncode != 0 and _loaded(label):
        raise LaunchdInstallError(f"launchd_bootout_failed_{label.replace('.', '_')}")
    deadline = time.monotonic() + 30.0
    while _loaded(label) and time.monotonic() < deadline:
        time.sleep(0.25)
    if _loaded(label):
        raise LaunchdInstallError(f"launchd_bootout_timeout_{label.replace('.', '_')}")


def _plist(
    *,
    label: str,
    program_arguments: Iterable[str],
    working_directory: Path,
    stdout_path: Path,
    stderr_path: Path,
    keep_alive: object,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "Label": label,
        "ProgramArguments": list(program_arguments),
        "WorkingDirectory": str(working_directory),
        "RunAtLoad": False,
        "KeepAlive": keep_alive,
        "ThrottleInterval": 10,
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "ExitTimeOut": 30,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        **({"EnvironmentVariables": dict(environment)} if environment else {}),
    }


def _config(arguments: argparse.Namespace) -> tuple[dict[str, Any], str, Any]:
    runtime = load_evaluation_runtime(
        arguments.runtime_manifest,
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
    plan = _load_private_plan(arguments.plan)
    spec = DATASET_SPECS[DATASET]
    routing_root = arguments.routing_root.resolve()
    if routing_root != spec.corpus_root.parent.resolve():
        raise LaunchdInstallError("launchd_routing_root_invalid")
    corpus_paths = _paths(spec)
    campaign_root = _ensure_directory(SUPERVISOR_ROOT)
    stage_root = _ensure_directory(campaign_root / "supervisor-stages")
    stage_log_root = _ensure_directory(campaign_root / "supervisor-logs")
    runtime_root = runtime.runtime_root
    bindings_path = runtime_root / BINDINGS_NAME
    quality_report = stage_root / "quality-report.json"
    quality_markdown = stage_root / "quality-report.md"
    binding: dict[str, Any] = {
        "repo_root": str(ROOT.resolve()),
        # Keep the venv launcher path itself.  Resolving this symlink would
        # make launchd execute uv's bare interpreter without the venv site
        # packages (pydantic, uvicorn, and the evaluator dependencies).
        "python": str(ROOT / ".venv/bin/python"),
        "runtime_manifest": str(arguments.runtime_manifest.resolve()),
        "runtime_root": str(runtime_root.resolve()),
        "plan": str(arguments.plan.resolve()),
        "routing_root": str(routing_root),
        "bindings_path": str(bindings_path),
        "campaign_root": str(campaign_root.resolve()),
        "provider_smoke_checkpoint": str((campaign_root / "provider-smoke.json").resolve()),
        "campaign_checkpoint": str((campaign_root / "campaign-state.json").resolve()),
        "quality_report": str(quality_report.resolve()),
        "quality_markdown": str(quality_markdown.resolve()),
        "locked_report": str((campaign_root / "final-report.json").resolve()),
        "markdown_report": str((ROOT / "docs/test/36-0827-large-evaluation-route-variant-report.md").resolve()),
        "report_checkpoint": str((campaign_root / "report-publish.json").resolve()),
        "supervisor_state": str((campaign_root / "supervisor-state.json").resolve()),
        "supervisor_lock": str((campaign_root / "supervisor.lock").resolve()),
        "supervisor_pid": str((campaign_root / "supervisor.pid").resolve()),
        "stage_root": str(stage_root.resolve()),
        "stage_log_root": str(stage_log_root.resolve()),
        "chat_profile_revision_id": CHAT_PROFILE,
        "judge_profile_revision_id": CHAT_PROFILE,
        "text_embedding_profile_revision_id": TEXT_EMBEDDING_PROFILE,
        "multimodal_embedding_profile_revision_id": MULTIMODAL_EMBEDDING_PROFILE,
        "routing_corpus_sha256": _corpus_digest(corpus_paths),
        "routing_document_count": len(corpus_paths),
        "plan_binding_sha256": str(plan["plan_binding_sha256"]),
        "runtime_build_revision": runtime.build_revision,
    }
    if plan.get("coverage", {}).get("routing", {}).get("dataset_id") != spec.dataset_id:
        raise LaunchdInstallError("launchd_plan_routing_dataset_invalid")
    if arguments.config.exists() and not arguments.reconfigure:
        existing = _private_regular(arguments.config, code="launchd_config")
        try:
            value = json.loads(existing.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise LaunchdInstallError("launchd_config_json_invalid") from error
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != CONFIG_SCHEMA
            or value.get("binding_sha256") != digest(value.get("binding"))
            or value.get("binding") != binding
        ):
            raise LaunchdInstallError("launchd_config_exists_changed")
    config = {
        "schema_version": CONFIG_SCHEMA,
        "binding": binding,
        "binding_sha256": digest(binding),
    }
    return config, digest(binding), runtime


def _write_config(config_path: Path, config: Mapping[str, Any]) -> None:
    _ensure_directory(config_path.parent)
    write_private_json(config_path, dict(config))


def _build_plists(
    config: Mapping[str, Any], runtime: Any, *, config_path: Path
) -> dict[str, dict[str, Any]]:
    binding = config["binding"]
    runtime_root = Path(binding["runtime_root"])
    log_root = _ensure_directory(runtime_root / "logs")
    runtime_env = Path(runtime.env_file)
    python = Path(binding["python"])
    repo_root = Path(binding["repo_root"])
    # The restored data directory is PostgreSQL 18.  Do not use PATH here:
    # Homebrew's unversioned postgres currently points at PostgreSQL 17.
    postgres = Path("/opt/homebrew/opt/postgresql@18/bin/postgres")
    if not postgres.is_file():
        raise LaunchdInstallError("launchd_postgres_executable_missing")
    postgres_data = runtime_root / "restored-p6-20260820/postgres18"
    if not (postgres_data / "PG_VERSION").is_file():
        raise LaunchdInstallError("launchd_postgres_data_invalid")
    falkordb = repo_root / ".venv/lib/python3.12/site-packages/redislite/bin/redis-server"
    falkordb_workdir = runtime_root / "restored-p6-20260820/falkordb-empty"
    if not falkordb.is_file() or not (falkordb_workdir / "falkordb.rdb").exists():
        raise LaunchdInstallError("launchd_falkordb_runtime_invalid")
    python_env = {
        "PYTHONPATH": f"{repo_root}/src:{repo_root}",
        "LANG": "C",
        "LC_ALL": "C",
        "LC_CTYPE": "C",
    }
    postgres_env = {"LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C"}
    common = {
        # Python launched directly by launchd can stall in interpreter
        # initialization while resolving a Documents-backed cwd.  All paths
        # consumed by these processes are absolute, so use a neutral cwd.
        "working_directory": Path("/private/tmp"),
    }
    run_marker = Path(binding["campaign_root"]) / RUN_MARKER_NAME
    dependency_keep_alive = {"PathState": {str(run_marker): True}}
    supervisor_keep_alive = {"PathState": {str(run_marker): True}}
    return {
        "postgres": _plist(
            label=LABELS["postgres"],
            program_arguments=[
                "/usr/bin/env",
                "-i",
                "PATH=/usr/bin:/bin:/usr/sbin:/sbin",
                f"HOME={Path.home()}",
                "LANG=C",
                "LC_ALL=C",
                "LC_CTYPE=C",
                str(postgres),
                "-D",
                str(postgres_data),
                "-p",
                str(runtime.ports["postgres"]),
                "-h",
                "127.0.0.1",
            ],
            working_directory=Path("/private/tmp"),
            stdout_path=log_root / "launchd-postgres.log",
            stderr_path=log_root / "launchd-postgres.err.log",
            keep_alive=dependency_keep_alive,
            environment=postgres_env,
        ),
        "falkordb": _plist(
            label=LABELS["falkordb"],
            program_arguments=[
                str(falkordb),
                "--bind",
                "127.0.0.1",
                "--port",
                str(runtime.ports["falkordb"]),
                "--dir",
                str(falkordb_workdir),
                "--dbfilename",
                "falkordb.rdb",
                "--loadmodule",
                str(falkordb.parent / "falkordb.so"),
            ],
            working_directory=falkordb_workdir,
            stdout_path=log_root / "launchd-falkordb.log",
            stderr_path=log_root / "launchd-falkordb.err.log",
            keep_alive=dependency_keep_alive,
        ),
        "api": _plist(
            label=LABELS["api"],
            program_arguments=[
                str(python),
                "-m",
                "apps.api.main",
                "--env-file",
                str(runtime_env),
            ],
            **common,
            keep_alive=dependency_keep_alive,
            stdout_path=log_root / "launchd-api.log",
            stderr_path=log_root / "launchd-api.err.log",
            environment=python_env,
        ),
        "worker": _plist(
            label=LABELS["worker"],
            program_arguments=[
                str(python),
                "-m",
                "apps.worker.main",
                "--env-file",
                str(runtime_env),
            ],
            **common,
            keep_alive=dependency_keep_alive,
            stdout_path=log_root / "launchd-worker.log",
            stderr_path=log_root / "launchd-worker.err.log",
            environment=python_env,
        ),
        "supervisor": _plist(
            label=LABELS["supervisor"],
            program_arguments=[
                str(python),
                str(repo_root / "tools/supervise_large_evaluation.py"),
                "--config",
                str(config_path.resolve()),
            ],
            **common,
            keep_alive=supervisor_keep_alive,
            stdout_path=log_root / "launchd-supervisor.log",
            stderr_path=log_root / "launchd-supervisor.err.log",
            environment=python_env,
        ),
    }


def _write_plists(
    plists: Mapping[str, Mapping[str, Any]],
    *,
    runtime_root: Path,
) -> dict[str, Path]:
    launchd_root = _ensure_directory(runtime_root / "launchd")
    user_root = Path.home() / "Library/LaunchAgents"
    user_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, plist in plists.items():
        label = LABELS[name]
        payload = plistlib.dumps(dict(plist), fmt=plistlib.FMT_XML, sort_keys=False)
        runtime_path = launchd_root / f"{label}.plist"
        user_path = user_root / f"{label}.plist"
        _prepare_private_output(Path(str(plist["StandardOutPath"])))
        _prepare_private_output(Path(str(plist["StandardErrorPath"])))
        _atomic_bytes(runtime_path, payload)
        _atomic_bytes(user_path, payload)
        paths[name] = user_path
    return paths


def _process_rows() -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        rows.append((pid, parts[1]))
    return rows


def _handoff_patterns(config: Mapping[str, Any], runtime: Any) -> tuple[str, ...]:
    binding = config["binding"]
    runtime_env = Path(runtime.env_file)
    return (
        f"apps.api.main --env-file {runtime_env}",
        f"apps.api.main --env-file {runtime_env.name}",
        f"apps.worker.main --env-file {runtime_env}",
        f"apps.worker.main --env-file {runtime_env.name}",
        f"tools/provision_large_evaluation_host.py --dataset routing_variant",
        f"tools/run_evaluation_provider_smoke.py",
        f"tools/run_large_evaluation.py",
        f"-D {binding['runtime_root']}/restored-p6-20260820/postgres18 -p {runtime.ports['postgres']}",
        f"--bind 127.0.0.1 --port {runtime.ports['falkordb']}",
        f"redis-server 127.0.0.1:{runtime.ports['falkordb']}",
    )


def _handoff_existing(config: Mapping[str, Any], runtime: Any) -> list[int]:
    patterns = _handoff_patterns(config, runtime)
    own_pid = os.getpid()
    matched = [
        pid
        for pid, command in _process_rows()
        if pid != own_pid and any(pattern in command for pattern in patterns)
        and "install_large_evaluation_launchd.py" not in command
    ]
    for pid in sorted(set(matched)):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        alive = {pid for pid, _ in _process_rows()}
        remaining = sorted(set(matched) & alive)
        if not remaining:
            return sorted(set(matched))
        time.sleep(1.0)
    raise LaunchdInstallError("launchd_handoff_timeout")


def _ensure_ports_free(runtime: Any) -> None:
    for name, port in runtime.ports.items():
        if name == "frontend":
            continue
        with _socket_probe(port):
            raise LaunchdInstallError(f"launchd_port_still_in_use_{name}")


class _socket_probe:
    def __init__(self, port: int) -> None:
        self.port = port
        self.socket: Any = None

    def __enter__(self):
        import socket

        self.socket = socket.socket()
        self.socket.settimeout(0.2)
        try:
            self.socket.connect(("127.0.0.1", self.port))
        except OSError:
            self.socket.close()
            self.socket = None
        return self

    def __exit__(self, *_args: object) -> None:
        if self.socket is not None:
            self.socket.close()


def _bootstrap(paths: Mapping[str, Path]) -> None:
    order = ("postgres", "falkordb", "api", "worker", "supervisor")
    for name in order:
        label = LABELS[name]
        _bootout(label)
        result = _launchctl("bootstrap", _domain(), str(paths[name]))
        if result.returncode != 0:
            raise LaunchdInstallError(f"launchd_bootstrap_failed_{name}")
        if not _loaded(label):
            raise LaunchdInstallError(f"launchd_agent_not_loaded_{name}")


def install(arguments: argparse.Namespace) -> dict[str, Any]:
    config, config_sha256, runtime = _config(arguments)
    config_path = arguments.config.resolve()
    _write_config(config_path, config)
    plists = _build_plists(config, runtime, config_path=config_path)
    paths = _write_plists(plists, runtime_root=runtime.runtime_root)
    if not arguments.handoff_existing:
        occupied = [
            name
            for name, port in runtime.ports.items()
            if name != "frontend" and _port_in_use(port)
        ]
        if occupied:
            raise LaunchdInstallError("launchd_existing_runtime_processes_require_handoff")
    else:
        # Stop the already managed agents before looking for handoff targets.
        # KeepAlive would otherwise respawn a process immediately after the
        # handoff signal, leaving its port occupied and making a reconfigure
        # fail even though the old process itself exited cleanly.
        for label in LABELS.values():
            _bootout(label)
        _handoff_existing(config, runtime)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if not any(
                _port_in_use(port)
                for name, port in runtime.ports.items()
                if name != "frontend"
            ):
                break
            time.sleep(1.0)
        else:
            raise LaunchdInstallError("launchd_runtime_ports_not_released")
    run_marker = Path(config["binding"]["campaign_root"]) / RUN_MARKER_NAME
    write_private_json(
        run_marker,
        {
            "schema_version": "large_evaluation_run_marker_v1",
            "status": "enabled",
            "config_sha256": config_sha256,
            "resilience_policy_sha256": RESILIENCE_POLICY_SHA256,
            "implementation_sha256": _automation_implementation_sha256(
                config["binding"]
            ),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    try:
        _bootstrap(paths)
    except BaseException:
        try:
            run_marker.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "status": "installed",
        "config_sha256": config_sha256,
        "labels": LABELS,
        "config": str(config_path),
        "state": config["binding"]["supervisor_state"],
    }


def _port_in_use(port: int) -> bool:
    import socket

    with socket.socket() as probe:
        probe.settimeout(0.2)
        try:
            probe.connect(("127.0.0.1", port))
        except OSError:
            return False
        return True


def uninstall() -> dict[str, Any]:
    marker = SUPERVISOR_ROOT / RUN_MARKER_NAME
    try:
        marker.unlink()
    except FileNotFoundError:
        pass
    for label in LABELS.values():
        _bootout(label)
    user_root = Path.home() / "Library/LaunchAgents"
    removed: list[str] = []
    for label in LABELS.values():
        path = user_root / f"{label}.plist"
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(str(path))
    return {"status": "uninstalled", "removed": removed}


def status() -> dict[str, Any]:
    return {
        "status": "status",
        "run_marker_present": (SUPERVISOR_ROOT / RUN_MARKER_NAME).exists(),
        "agents": {
            name: {"label": label, "loaded": _loaded(label)}
            for name, label in LABELS.items()
        },
    }


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.install:
            result = install(arguments)
        elif arguments.uninstall:
            result = uninstall()
        else:
            result = status()
    except (EvaluationRuntimeError, LaunchdInstallError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "failure_code": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
