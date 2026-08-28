#!/usr/bin/env python3
"""Pause or resume the isolated launchd-managed large evaluation.

This is a short operator control command.  It never runs an evaluation stage
in the foreground: the long-running state machine remains owned by launchd.
Pause first asks the supervisor to stop its supervised process group, waits for
the durable ``paused`` checkpoint, and only then unloads the isolated runtime
agents.  Resume re-arms exactly the paused stage and bootstraps the same
owner-only plists in dependency order.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import plistlib
import stat
import subprocess
import time
from typing import Any, Mapping

from tools.evaluation_campaign_state import digest, write_private_json
from tools.evaluation_resilience import RESILIENCE_POLICY_SHA256
from tools.evaluation_runtime import load_evaluation_runtime
from tools.install_large_evaluation_launchd import (
    LABELS,
    _bootout,
    _build_plists,
    _domain,
    _launchctl,
    _write_plists,
)
from tools.prepare_large_evaluation import ROOT
from tools.supervise_large_evaluation import (
    DEFAULT_CONFIG,
    RUN_MARKER_NAME,
    _automation_implementation_sha256,
    _config_path,
    _load_config,
    _load_json,
    _load_or_create_state,
    _resume_paused_stage,
    _validate_stage_checkpoints,
    _validate_static_config,
)


CONTROL_SCHEMA = "large_evaluation_control_v1"
PAUSE_TIMEOUT_SECONDS = 90.0
PORT_STOP_TIMEOUT_SECONDS = 30.0
AGENT_ORDER = ("postgres", "falkordb", "api", "worker", "supervisor")


class EvaluationControlError(RuntimeError):
    """A safe control operation could not complete."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pause", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def _state_and_binding(config_path: Path) -> tuple[Mapping[str, Any], str, dict[str, Any], Path]:
    binding, config_sha256 = _load_config(config_path.resolve())
    _validate_static_config(binding)
    state_path = _config_path(binding, "supervisor_state")
    state = _load_or_create_state(state_path, config_sha256)
    _validate_stage_checkpoints(binding, state)
    return binding, config_sha256, state, state_path


def _target(label: str) -> str:
    return f"{_domain()}/{label}"


def _private_plist_paths(*, runtime_root: Path | None = None) -> dict[str, Path]:
    roots = [Path.home() / "Library/LaunchAgents"]
    if runtime_root is not None:
        roots.append(runtime_root / "launchd")
    paths: dict[str, Path] = {}
    for name, label in LABELS.items():
        path = next(
            (
                candidate / f"{label}.plist"
                for candidate in roots
                if (candidate / f"{label}.plist").exists()
            ),
            None,
        )
        if path is None:
            raise EvaluationControlError(f"control_plist_missing_{name}")
        try:
            mode = stat.S_IMODE(path.lstat().st_mode)
        except FileNotFoundError as error:
            raise EvaluationControlError(f"control_plist_missing_{name}") from error
        if stat.S_ISLNK(path.lstat().st_mode) or not stat.S_ISREG(path.lstat().st_mode):
            raise EvaluationControlError(f"control_plist_invalid_{name}")
        if mode & 0o077:
            raise EvaluationControlError(f"control_plist_permissions_invalid_{name}")
        try:
            with path.open("rb") as handle:
                plist = plistlib.load(handle)
        except (OSError, plistlib.InvalidFileException) as error:
            raise EvaluationControlError(f"control_plist_unreadable_{name}") from error
        if not isinstance(plist, dict) or plist.get("Label") != label:
            raise EvaluationControlError(f"control_plist_binding_invalid_{name}")
        paths[name] = path
    return paths


def _port_in_use(port: int) -> bool:
    import socket

    with socket.socket() as probe:
        probe.settimeout(0.2)
        try:
            probe.connect(("127.0.0.1", int(port)))
        except OSError:
            return False
        return True


def _runtime_ports(binding: Mapping[str, Any]) -> dict[str, int]:
    runtime = load_evaluation_runtime(
        _config_path(binding, "runtime_manifest"),
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
    return {
        name: int(port)
        for name, port in runtime.ports.items()
        if name in {"postgres", "falkordb", "api"}
    }


def _run_marker(binding: Mapping[str, Any]) -> Path:
    return _config_path(binding, "campaign_root") / RUN_MARKER_NAME


def _remove_run_marker(binding: Mapping[str, Any]) -> None:
    try:
        _run_marker(binding).unlink()
    except FileNotFoundError:
        pass


def _quarantine_user_plists(binding: Mapping[str, Any]) -> None:
    user_root = Path.home() / "Library/LaunchAgents"
    quarantine = _config_path(binding, "runtime_root") / "paused-launchd"
    quarantine.mkdir(mode=0o700, parents=True, exist_ok=True)
    quarantine.chmod(0o700)
    for label in LABELS.values():
        source = user_root / f"{label}.plist"
        if not source.exists():
            continue
        destination = quarantine / source.name
        if destination.exists():
            if destination.read_bytes() == source.read_bytes():
                source.unlink()
                continue
            destination = quarantine / f"{label}.{int(time.time())}.plist"
        os.replace(source, destination)
        destination.chmod(0o600)


def _wait_state(
    state_path: Path,
    config_sha256: str,
    expected_status: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _load_json(state_path, code="supervisor_state")
        if state.get("config_sha256") != config_sha256:
            raise EvaluationControlError("control_state_config_changed")
        if state.get("status") == expected_status:
            return state
        if state.get("status") == "failed":
            raise EvaluationControlError("control_pause_stage_failed")
        time.sleep(0.25)
    raise EvaluationControlError("control_pause_timeout")


def _wait_ports_stopped(ports: Mapping[str, int]) -> None:
    deadline = time.monotonic() + PORT_STOP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not any(_port_in_use(port) for port in ports.values()):
            return
        time.sleep(0.25)
    raise EvaluationControlError("control_isolated_runtime_still_listening")


def _write_control_checkpoint(
    binding: Mapping[str, Any],
    config_sha256: str,
    state: Mapping[str, Any],
    *,
    action: str,
    ports_stopped: bool,
) -> Path:
    state_path = _config_path(binding, "supervisor_state")
    state_bytes = state_path.read_bytes()
    path = state_path.parent / "control-checkpoint.json"
    write_private_json(
        path,
        {
            "schema_version": CONTROL_SCHEMA,
            "action": action,
            "status": state.get("status"),
            "current_stage": state.get("current_stage"),
            "config_sha256": config_sha256,
            "state_sha256": digest(json.loads(state_bytes.decode("utf-8"))),
            "state_path": str(state_path),
            "runtime_root": str(binding["runtime_root"]),
            "ports_stopped": ports_stopped,
            "recorded_at": _timestamp(),
        },
    )
    return path


def pause(config_path: Path) -> dict[str, Any]:
    binding, config_sha256, state, state_path = _state_and_binding(config_path)
    ports = _runtime_ports(binding)
    _private_plist_paths(runtime_root=_config_path(binding, "runtime_root"))
    if state.get("status") == "completed":
        raise EvaluationControlError("control_evaluation_already_completed")
    if state.get("status") == "failed":
        raise EvaluationControlError("control_evaluation_failed_requires_repair")
    if state.get("status") == "paused":
        paused = _wait_state(
            state_path,
            config_sha256,
            "paused",
            timeout=1.0,
        )
    else:
        supervisor_label = LABELS["supervisor"]
        if _launchctl("print", _target(supervisor_label)).returncode != 0:
            raise EvaluationControlError("control_supervisor_not_loaded")
        result = _launchctl("kill", "SIGUSR1", _target(supervisor_label))
        if result.returncode != 0:
            raise EvaluationControlError("control_pause_signal_failed")
        paused = _wait_state(
            state_path,
            config_sha256,
            "paused",
            timeout=PAUSE_TIMEOUT_SECONDS,
        )
    _remove_run_marker(binding)
    for name in reversed(AGENT_ORDER):
        _bootout(LABELS[name])
    _wait_ports_stopped(ports)
    _quarantine_user_plists(binding)
    checkpoint = _write_control_checkpoint(
        binding,
        config_sha256,
        paused,
        action="pause",
        ports_stopped=True,
    )
    return {
        "status": "paused",
        "current_stage": paused.get("current_stage"),
        "config_sha256": config_sha256,
        "control_checkpoint": str(checkpoint),
        "runtime_root": str(binding["runtime_root"]),
        "ports_stopped": True,
    }


def _bootstrap_agent(name: str, path: Path) -> None:
    _bootout(LABELS[name])
    result = _launchctl("bootstrap", _domain(), str(path))
    if result.returncode != 0:
        raise EvaluationControlError(f"control_bootstrap_failed_{name}")
    if _launchctl("print", _target(LABELS[name])).returncode != 0:
        raise EvaluationControlError(f"control_agent_not_loaded_{name}")


def _resume_state(
    binding: Mapping[str, Any],
    config_path: Path,
    state_path: Path,
    config_sha256: str,
) -> dict[str, Any]:
    command = [
        str(binding["python"]),
        str(ROOT / "tools/supervise_large_evaluation.py"),
        "--config",
        str(config_path.resolve()),
        "--resume-paused",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{binding['repo_root']}/src:{binding['repo_root']}"
    result = subprocess.run(
        command,
        cwd="/private/tmp",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EvaluationControlError("control_resume_state_failed")
    state = _load_json(state_path, code="supervisor_state")
    if state.get("status") != "running" or state.get("config_sha256") != config_sha256:
        raise EvaluationControlError("control_resume_state_invalid")
    return state


def _resume_failed_state(
    binding: Mapping[str, Any],
    config_path: Path,
    state_path: Path,
    config_sha256: str,
    stage: str,
) -> dict[str, Any]:
    command = [
        str(binding["python"]),
        str(ROOT / "tools/supervise_large_evaluation.py"),
        "--config",
        str(config_path.resolve()),
        "--resume-failed-stage",
        stage,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{binding['repo_root']}/src:{binding['repo_root']}"
    result = subprocess.run(
        command,
        cwd="/private/tmp",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EvaluationControlError("control_retry_failed_state_failed")
    state = _load_json(state_path, code="supervisor_state")
    if (
        state.get("status") != "running"
        or state.get("current_stage") != stage
        or state.get("config_sha256") != config_sha256
    ):
        raise EvaluationControlError("control_retry_failed_state_invalid")
    return state


def _prepare_activation(
    binding: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
) -> dict[str, Path]:
    config = _load_json(config_path.resolve(), code="supervisor_config")
    runtime = load_evaluation_runtime(
        _config_path(binding, "runtime_manifest"),
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
    plists = _build_plists(config, runtime, config_path=config_path.resolve())
    paths = _write_plists(plists, runtime_root=runtime.runtime_root)
    write_private_json(
        _run_marker(binding),
        {
            "schema_version": "large_evaluation_run_marker_v1",
            "status": "enabled",
            "config_sha256": config_sha256,
            "resilience_policy_sha256": RESILIENCE_POLICY_SHA256,
            "implementation_sha256": _automation_implementation_sha256(binding),
            "created_at": _timestamp(),
        },
    )
    return paths


def _activation_failed(binding: Mapping[str, Any]) -> None:
    try:
        _remove_run_marker(binding)
    except OSError:
        pass
    for name in reversed(("supervisor", "worker", "api", "falkordb", "postgres")):
        try:
            _bootout(LABELS[name])
        except (OSError, RuntimeError):
            pass
    try:
        _quarantine_user_plists(binding)
    except OSError:
        pass


def resume(config_path: Path) -> dict[str, Any]:
    binding, config_sha256, state, state_path = _state_and_binding(config_path)
    if state.get("status") != "paused":
        raise EvaluationControlError("control_resume_target_not_paused")
    try:
        paths = _prepare_activation(binding, config_path, config_sha256)
        for name in ("postgres", "falkordb", "api", "worker"):
            _bootstrap_agent(name, paths[name])
        resumed = _resume_state(binding, config_path, state_path, config_sha256)
        _bootstrap_agent("supervisor", paths["supervisor"])
    except BaseException:
        _activation_failed(binding)
        raise
    checkpoint = _write_control_checkpoint(
        binding,
        config_sha256,
        resumed,
        action="resume",
        ports_stopped=False,
    )
    return {
        "status": "resumed",
        "current_stage": resumed.get("current_stage"),
        "config_sha256": config_sha256,
        "control_checkpoint": str(checkpoint),
        "runtime_root": str(binding["runtime_root"]),
    }


def retry_failed(config_path: Path) -> dict[str, Any]:
    binding, config_sha256, state, state_path = _state_and_binding(config_path)
    stage = state.get("current_stage")
    if state.get("status") != "failed" or not isinstance(stage, str):
        raise EvaluationControlError("control_retry_target_not_failed")
    try:
        paths = _prepare_activation(binding, config_path, config_sha256)
        for name in ("postgres", "falkordb", "api", "worker"):
            _bootstrap_agent(name, paths[name])
        resumed = _resume_failed_state(
            binding,
            config_path,
            state_path,
            config_sha256,
            stage,
        )
        _bootstrap_agent("supervisor", paths["supervisor"])
    except BaseException:
        _activation_failed(binding)
        raise
    checkpoint = _write_control_checkpoint(
        binding,
        config_sha256,
        resumed,
        action="retry_failed",
        ports_stopped=False,
    )
    return {
        "status": "resumed",
        "current_stage": stage,
        "config_sha256": config_sha256,
        "control_checkpoint": str(checkpoint),
        "runtime_root": str(binding["runtime_root"]),
    }


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.pause:
            result = pause(arguments.config)
        elif arguments.resume:
            result = resume(arguments.config)
        else:
            result = retry_failed(arguments.config)
    except (EvaluationControlError, OSError, ValueError, TypeError) as error:
        print(json.dumps({"status": "failed", "failure_code": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
