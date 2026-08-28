#!/usr/bin/env python3
"""Persistently supervise the frozen large-evaluation state machine.

The supervisor is deliberately separate from Codex's foreground terminal.  A
macOS LaunchAgent starts it, the supervisor owns one lock and PID file, and
every stage is bound to one owner-only, atomically replaced checkpoint.

The state machine is fail-closed:

    indexing -> graph -> gate -> provider smoke -> quality evaluation -> report

Controlled child failures are persisted as ``failed`` and stop the state
machine.  Unexpected supervisor crashes leave the active stage as ``running``
and exit non-zero so launchd restarts the process; the next invocation adopts
an active child or resumes from its durable checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.error import URLError
from urllib.request import urlopen

from tools.evaluation_campaign_state import canonical_bytes, digest, write_private_json
from tools.evaluation_resilience import (
    HEARTBEAT_INTERVAL_SECONDS,
    LONG_STAGE_STALL_TIMEOUT_SECONDS,
    MAX_STALL_RESTARTS,
    RESILIENCE_POLICY_SHA256,
    SHORT_STAGE_STALL_TIMEOUT_SECONDS,
)
from tools.evaluation_runtime import (
    EvaluationRuntimeError,
    load_evaluation_runtime,
)
from tools.prepare_large_evaluation import ROOT
from tools.provision_large_evaluation_host import (
    DATASET_SPECS,
    GRAPH_EXTRACTOR_VERSION,
    GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST,
    GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
    _corpus_digest,
    _paths,
)
from tools.run_evaluation_provider_smoke import (
    EXPECTED_MODELS,
    SCHEMA as PROVIDER_SMOKE_SCHEMA,
)
from tools.check_large_evaluation_runtime import _load_private_plan


CONFIG_SCHEMA = "large_evaluation_supervisor_config_v1"
STATE_SCHEMA = "large_evaluation_supervisor_state_v1"
STAGE_SCHEMA = "large_evaluation_supervisor_stage_v1"
EXIT_SCHEMA = "supervised_command_exit_v1"
REPORT_SCHEMA = "large_evaluation_final_report_v1"
CONFIRM_PROVISION = "PROVISION_LARGE_EVALUATION_ROUTING_VARIANT"
CONFIRM_SMOKE = "RUN_EVALUATION_PROVIDER_SMOKE"
CONFIRM_EVALUATION = "RUN_LARGE_EVALUATION_EXTERNAL_CALLS"
DATASET = "routing_variant"
DATASET_ID = DATASET_SPECS[DATASET].dataset_id
STAGES = (
    "indexing",
    "graph",
    "gate",
    "provider_smoke",
    "quality_evaluation",
    "report",
)
POLL_SECONDS = 5.0
CHILD_START_GRACE_SECONDS = 2.0
CHILD_PAUSE_TIMEOUT_SECONDS = 30.0
API_READY_PATH = "/health/ready"
RUNTIME_READY_TIMEOUT_SECONDS = 900.0
RUNTIME_SETTLE_TIMEOUT_SECONDS = 60.0
RUN_MARKER_NAME = ".evaluation-enabled"
SUPPORT_LABELS = (
    "com.rag.large-evaluation.worker",
    "com.rag.large-evaluation.api",
    "com.rag.large-evaluation.falkordb",
    "com.rag.large-evaluation.postgres",
)
DEFAULT_CONFIG = ROOT / ".runtime/evaluations/large-evaluation-route-variant-v1/supervisor-config.json"
AUTOMATION_IMPLEMENTATION_FILES = (
    "tools/evaluation_campaign_state.py",
    "tools/evaluation_resilience.py",
    "tools/run_supervised_command.py",
    "tools/provision_large_evaluation_host.py",
    "tools/check_large_evaluation_runtime.py",
    "tools/run_evaluation_provider_smoke.py",
    "tools/run_large_evaluation.py",
    "tools/finalize_large_evaluation_report.py",
    "tools/supervise_large_evaluation.py",
    "tools/install_large_evaluation_launchd.py",
    "tools/control_large_evaluation.py",
)


class SupervisorError(RuntimeError):
    """A content-safe, controlled supervisor failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ControlledStageFailure(SupervisorError):
    """A stage failed and the state machine must stop without skipping ahead."""


class SupervisorPaused(SupervisorError):
    """The operator requested a durable pause at the current stage."""


class SupervisorShutdown(SupervisorError):
    """The host is shutting down; keep the campaign armed for next login."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _automation_implementation_sha256(
    binding: Mapping[str, Any] | None = None,
) -> str:
    repo_root = (
        Path(str(binding["repo_root"])).resolve()
        if binding is not None and isinstance(binding.get("repo_root"), str)
        else ROOT.resolve()
    )
    values: dict[str, str] = {}
    for relative in AUTOMATION_IMPLEMENTATION_FILES:
        path = repo_root / relative
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise SupervisorError("supervisor_implementation_file_missing") from error
        values[relative] = hashlib.sha256(payload).hexdigest()
    return digest(values)


def _private_regular_file(path: Path, *, code: str) -> Path:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise SupervisorError(f"{code}_missing") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise SupervisorError(f"{code}_invalid")
    if stat.S_IMODE(status.st_mode) & 0o077:
        raise SupervisorError(f"{code}_permissions_invalid")
    return path.resolve()


def _load_json(path: Path, *, code: str) -> dict[str, Any]:
    private = _private_regular_file(path, code=code)
    try:
        value = json.loads(private.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SupervisorError(f"{code}_json_invalid") from error
    if not isinstance(value, dict):
        raise SupervisorError(f"{code}_shape_invalid")
    return value


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = _load_json(path, code="supervisor_config")
    binding = config.get("binding")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not isinstance(binding, dict)
        or config.get("binding_sha256") != digest(binding)
    ):
        raise SupervisorError("supervisor_config_binding_invalid")
    required = {
        "repo_root",
        "python",
        "runtime_manifest",
        "runtime_root",
        "plan",
        "routing_root",
        "bindings_path",
        "campaign_root",
        "provider_smoke_checkpoint",
        "campaign_checkpoint",
        "quality_report",
        "quality_markdown",
        "locked_report",
        "markdown_report",
        "report_checkpoint",
        "supervisor_state",
        "supervisor_lock",
        "supervisor_pid",
        "stage_root",
        "stage_log_root",
        "chat_profile_revision_id",
        "judge_profile_revision_id",
        "text_embedding_profile_revision_id",
        "multimodal_embedding_profile_revision_id",
        "routing_corpus_sha256",
        "routing_document_count",
        "plan_binding_sha256",
        "runtime_build_revision",
    }
    if set(binding) != required:
        raise SupervisorError("supervisor_config_fields_invalid")
    repo_root = Path(str(binding["repo_root"])).resolve()
    if repo_root != ROOT.resolve():
        raise SupervisorError("supervisor_config_repo_invalid")
    for key in (
        "python",
        "runtime_manifest",
        "runtime_root",
        "plan",
        "routing_root",
        "bindings_path",
        "campaign_root",
        "provider_smoke_checkpoint",
        "campaign_checkpoint",
        "quality_report",
        "quality_markdown",
        "locked_report",
        "markdown_report",
        "report_checkpoint",
        "supervisor_state",
        "supervisor_lock",
        "supervisor_pid",
        "stage_root",
        "stage_log_root",
    ):
        if not isinstance(binding[key], str) or not binding[key]:
            raise SupervisorError("supervisor_config_path_invalid")
    for key in (
        "chat_profile_revision_id",
        "judge_profile_revision_id",
        "text_embedding_profile_revision_id",
        "multimodal_embedding_profile_revision_id",
        "routing_corpus_sha256",
        "plan_binding_sha256",
        "runtime_build_revision",
    ):
        if not isinstance(binding[key], str) or not binding[key]:
            raise SupervisorError("supervisor_config_identity_invalid")
    if (
        isinstance(binding["routing_document_count"], bool)
        or not isinstance(binding["routing_document_count"], int)
        or binding["routing_document_count"] <= 0
    ):
        raise SupervisorError("supervisor_config_document_count_invalid")
    return binding, str(config["binding_sha256"])


def _config_path(binding: Mapping[str, Any], name: str) -> Path:
    return Path(str(binding[name])).resolve()


def _validate_static_config(binding: Mapping[str, Any]) -> None:
    plan_path = _config_path(binding, "plan")
    plan = _load_private_plan(plan_path)
    if plan.get("plan_binding_sha256") != binding["plan_binding_sha256"]:
        raise SupervisorError("supervisor_plan_binding_changed")
    provider_contract = plan.get("plan_binding", {}).get("provider_contract")
    expected_contract = {
        "provider": "OpenCode Go",
        "chat_model": "mimo-v2.5",
        "text_embedding_model": "qwen3.7-text-embedding",
        "multimodal_embedding_model": "tongyi-embedding-vision-flash-2026-03-06",
    }
    if provider_contract != expected_contract:
        raise SupervisorError("supervisor_provider_contract_invalid")
    runtime = load_evaluation_runtime(
        _config_path(binding, "runtime_manifest"),
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
    if runtime.build_revision != binding["runtime_build_revision"]:
        raise SupervisorError("supervisor_runtime_build_changed")
    routing_root = _config_path(binding, "routing_root")
    spec = DATASET_SPECS[DATASET]
    paths = _paths(spec)
    if routing_root != spec.corpus_root.parent.resolve():
        raise SupervisorError("supervisor_routing_root_changed")
    if len(paths) != binding["routing_document_count"]:
        raise SupervisorError("supervisor_routing_document_count_changed")
    if _corpus_digest(paths) != binding["routing_corpus_sha256"]:
        raise SupervisorError("supervisor_routing_corpus_changed")
    if binding["routing_document_count"] != spec.expected_document_count:
        raise SupervisorError("supervisor_routing_document_contract_invalid")
    for key in (
        "campaign_root",
        "stage_root",
        "stage_log_root",
    ):
        path = _config_path(binding, key)
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)
    runtime_root = _config_path(binding, "runtime_root")
    if runtime.runtime_root != runtime_root:
        raise SupervisorError("supervisor_runtime_root_changed")
    if _config_path(binding, "bindings_path").parent != runtime_root:
        raise SupervisorError("supervisor_bindings_path_invalid")
    if _config_path(binding, "python").resolve() != (ROOT / ".venv/bin/python").resolve():
        raise SupervisorError("supervisor_python_invalid")


def _new_state(config_sha256: str) -> dict[str, Any]:
    now = _timestamp()
    return {
        "schema_version": STATE_SCHEMA,
        "config_sha256": config_sha256,
        "resilience_policy_sha256": RESILIENCE_POLICY_SHA256,
        "implementation_sha256": _automation_implementation_sha256(),
        "status": "running",
        "current_stage": STAGES[0],
        "created_at": now,
        "updated_at": now,
        "stages": {
            stage: {
                "status": "pending",
                "attempt_count": 0,
                "config_sha256": config_sha256,
            }
            for stage in STAGES
        },
        "events": [{"event": "supervisor_created", "at": now}],
    }


def _validate_state(state: dict[str, Any], config_sha256: str) -> None:
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("config_sha256") != config_sha256
        or state.get("status") not in {"running", "paused", "failed", "completed"}
        or not isinstance(state.get("stages"), dict)
        or not isinstance(state.get("events"), list)
    ):
        raise SupervisorError("supervisor_state_binding_invalid")
    stages = state["stages"]
    if set(stages) != set(STAGES) or any(
        not isinstance(stages.get(stage), dict) for stage in STAGES
    ):
        raise SupervisorError("supervisor_state_stage_set_invalid")
    for stage in STAGES:
        if stages[stage].get("config_sha256") != config_sha256:
            raise SupervisorError("supervisor_state_stage_config_changed")
    _validate_stage_order(state)


def _validate_stage_order(state: Mapping[str, Any]) -> None:
    """Reject skipped gates and inconsistent terminal state."""

    stages = state["stages"]
    statuses = [stages[stage].get("status") for stage in STAGES]
    allowed = {"pending", "running", "paused", "failed", "completed"}
    if any(status not in allowed for status in statuses):
        raise SupervisorError("supervisor_state_stage_status_invalid")
    incomplete_seen = False
    for status in statuses:
        if status != "completed":
            incomplete_seen = True
        elif incomplete_seen:
            raise SupervisorError("supervisor_state_stage_order_invalid")
    active = [
        stage
        for stage, status in zip(STAGES, statuses, strict=True)
        if status in {"running", "paused", "failed"}
    ]
    if len(active) > 1:
        raise SupervisorError("supervisor_state_multiple_active_stages")
    current = state.get("current_stage")
    if current not in STAGES:
        raise SupervisorError("supervisor_state_current_stage_invalid")
    state_status = state.get("status")
    if state_status in {"paused", "failed"}:
        if active != [current] or stages[current].get("status") != state_status:
            raise SupervisorError("supervisor_state_terminal_stage_invalid")
    elif state_status == "completed":
        if any(status != "completed" for status in statuses):
            raise SupervisorError("supervisor_state_completion_invalid")
    elif state_status == "running":
        if active and active != [current]:
            raise SupervisorError("supervisor_state_active_stage_invalid")
        current_index = STAGES.index(current)
        if any(
            stages[stage].get("status") != "completed"
            for stage in STAGES[:current_index]
        ):
            raise SupervisorError("supervisor_state_previous_stage_incomplete")
        if any(
            stages[stage].get("status") != "pending"
            for stage in STAGES[current_index + 1 :]
        ):
            raise SupervisorError("supervisor_state_following_stage_not_pending")


def _load_or_create_state(path: Path, config_sha256: str) -> dict[str, Any]:
    if not path.exists():
        state = _new_state(config_sha256)
        write_private_json(path, state)
        return state
    state = _load_json(path, code="supervisor_state")
    try:
        _validate_state(state, config_sha256)
    except SupervisorError:
        # A reconfiguration can legitimately change the executable identity
        # (for example, from a resolved interpreter to the venv launcher).
        # Only migrate a completely pristine state; any real progress or
        # failure remains bound to its original configuration and fails closed.
        old_config_sha256 = state.get("config_sha256")
        stages = state.get("stages")
        pristine = (
            isinstance(old_config_sha256, str)
            and old_config_sha256 != config_sha256
            and state.get("schema_version") == STATE_SCHEMA
            and state.get("status") == "running"
            and state.get("current_stage") == STAGES[0]
            and isinstance(state.get("events"), list)
            and len(state["events"]) == 1
            and isinstance(stages, dict)
            and set(stages) == set(STAGES)
            and all(
                isinstance(stages[stage], dict)
                and stages[stage].get("status") == "pending"
                and stages[stage].get("attempt_count") == 0
                and stages[stage].get("config_sha256") == old_config_sha256
                for stage in STAGES
            )
        )
        if not pristine:
            raise
        state["config_sha256"] = config_sha256
        for stage in STAGES:
            stages[stage]["config_sha256"] = config_sha256
        _append_event(
            state,
            "supervisor_config_migrated",
            previous_config_sha256=old_config_sha256,
            config_sha256=config_sha256,
        )
        state["updated_at"] = _timestamp()
        write_private_json(path, state)
        _validate_state(state, config_sha256)
    return state


def _validate_stage_checkpoints(
    binding: Mapping[str, Any], state: Mapping[str, Any]
) -> None:
    """Verify every published stage checkpoint before trusting its state."""

    stages = state.get("stages")
    if not isinstance(stages, Mapping):
        raise SupervisorError("supervisor_state_stage_set_invalid")
    for stage in STAGES:
        record = stages.get(stage)
        if not isinstance(record, Mapping):
            raise SupervisorError("supervisor_state_stage_shape_invalid")
        checkpoint_path = record.get("checkpoint_path")
        checkpoint_sha256 = record.get("checkpoint_sha256")
        if checkpoint_path is None or checkpoint_sha256 is None:
            if record.get("status") == "pending":
                continue
            raise SupervisorError("supervisor_stage_checkpoint_missing")
        expected_path = _stage_checkpoint_path(binding, stage)
        if Path(str(checkpoint_path)).resolve() != expected_path:
            raise SupervisorError("supervisor_stage_checkpoint_path_invalid")
        actual_path = _private_regular_file(
            expected_path,
            code="supervisor_stage_checkpoint",
        )
        payload = actual_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != checkpoint_sha256:
            raise SupervisorError("supervisor_stage_checkpoint_digest_invalid")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SupervisorError("supervisor_stage_checkpoint_json_invalid") from error
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != STAGE_SCHEMA
            or value.get("stage") != stage
            or value.get("config_sha256") != state.get("config_sha256")
            or not isinstance(value.get("record"), dict)
        ):
            raise SupervisorError("supervisor_stage_checkpoint_schema_invalid")


def _write_pid(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _remove_pid(path: Path) -> None:
    try:
        if path.read_text(encoding="ascii").strip() == str(os.getpid()):
            path.unlink()
    except (FileNotFoundError, OSError):
        pass


@contextmanager
def _supervisor_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    with path.open("a+", encoding="ascii") as handle:
        path.chmod(0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SupervisorError("supervisor_already_running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _emit(event: str, **fields: Any) -> None:
    payload = {"event": event, "at": _timestamp(), **fields}
    print(json.dumps(payload, sort_keys=True), flush=True)


def _append_event(state: dict[str, Any], event: str, **fields: Any) -> None:
    state["events"].append({"event": event, "at": _timestamp(), **fields})


def _stage_checkpoint_path(binding: Mapping[str, Any], stage: str) -> Path:
    return _config_path(binding, "stage_root") / f"{stage}.json"


def _persist(
    state_path: Path,
    state: dict[str, Any],
    *,
    event: str | None = None,
    event_fields: Mapping[str, Any] | None = None,
) -> None:
    if event is not None:
        _append_event(state, event, **dict(event_fields or {}))
    state["updated_at"] = _timestamp()
    write_private_json(state_path, state)


def _persist_stage(
    binding: Mapping[str, Any],
    state_path: Path,
    state: dict[str, Any],
    stage: str,
    *,
    event: str | None = None,
    event_fields: Mapping[str, Any] | None = None,
) -> None:
    record = state["stages"][stage]
    stage_path = _stage_checkpoint_path(binding, stage)
    stage_payload = {
        "schema_version": STAGE_SCHEMA,
        "stage": stage,
        "config_sha256": state["config_sha256"],
        "record": {
            key: value
            for key, value in record.items()
            if key != "checkpoint_sha256"
        },
    }
    stage_sha256 = write_private_json(stage_path, stage_payload)
    record["checkpoint_path"] = str(stage_path)
    record["checkpoint_sha256"] = stage_sha256
    _persist(state_path, state, event=event, event_fields=event_fields)


def _stage_log_path(binding: Mapping[str, Any], stage: str, attempt: int) -> Path:
    return _config_path(binding, "stage_log_root") / f"{stage}-attempt-{attempt}.log"


def _ensure_private_log(path: Path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.chmod(path, 0o600)
    return os.fdopen(descriptor, "ab", buffering=0)


def _command_digest(command: Sequence[str]) -> str:
    return hashlib.sha256(canonical_bytes(list(command))).hexdigest()


def _base_environment(binding: Mapping[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{binding['repo_root']}/src:{binding['repo_root']}"
    return environment


def _tool_path(binding: Mapping[str, Any], name: str) -> str:
    return str(Path(str(binding["repo_root"])) / "tools" / name)


def _command_for_stage(binding: Mapping[str, Any], stage: str) -> list[str]:
    python = str(binding["python"])
    common = [
        "--evaluation-runtime",
        str(binding["runtime_manifest"]),
    ]
    if stage == "indexing":
        return [
            python,
            _tool_path(binding, "provision_large_evaluation_host.py"),
            "--dataset",
            DATASET,
            "--confirm",
            CONFIRM_PROVISION,
            "--timeout-seconds",
            "172800",
            *common,
            "--chat-profile-revision-id",
            str(binding["chat_profile_revision_id"]),
            "--judge-profile-revision-id",
            str(binding["judge_profile_revision_id"]),
            "--bindings-path",
            str(binding["bindings_path"]),
            "--stop-after-indexing",
        ]
    if stage == "graph":
        return [
            python,
            _tool_path(binding, "provision_large_evaluation_host.py"),
            "--dataset",
            DATASET,
            "--confirm",
            CONFIRM_PROVISION,
            "--timeout-seconds",
            "172800",
            *common,
            "--chat-profile-revision-id",
            str(binding["chat_profile_revision_id"]),
            "--judge-profile-revision-id",
            str(binding["judge_profile_revision_id"]),
            "--bindings-path",
            str(binding["bindings_path"]),
        ]
    if stage == "gate":
        return [
            python,
            _tool_path(binding, "check_large_evaluation_runtime.py"),
            "--plan",
            str(binding["plan"]),
            *common,
            "--bindings-path",
            str(binding["bindings_path"]),
        ]
    if stage == "provider_smoke":
        return [
            python,
            _tool_path(binding, "run_evaluation_provider_smoke.py"),
            *common,
            "--checkpoint",
            str(binding["provider_smoke_checkpoint"]),
            "--chat-profile-revision-id",
            str(binding["chat_profile_revision_id"]),
            "--text-embedding-profile-revision-id",
            str(binding["text_embedding_profile_revision_id"]),
            "--multimodal-embedding-profile-revision-id",
            str(binding["multimodal_embedding_profile_revision_id"]),
            "--confirm",
            CONFIRM_SMOKE,
        ]
    if stage == "quality_evaluation":
        return [
            python,
            _tool_path(binding, "run_large_evaluation.py"),
            "--plan",
            str(binding["plan"]),
            *common,
            "--provider-smoke-checkpoint",
            str(binding["provider_smoke_checkpoint"]),
            "--campaign-checkpoint",
            str(binding["campaign_checkpoint"]),
            "--locked-output",
            str(binding["quality_report"]),
            "--markdown-output",
            str(binding["quality_markdown"]),
            "--routing-root",
            str(binding["routing_root"]),
            "--bindings-path",
            str(binding["bindings_path"]),
            "--confirm",
            CONFIRM_EVALUATION,
        ]
    if stage == "report":
        return [
            python,
            _tool_path(binding, "finalize_large_evaluation_report.py"),
            "--source-report",
            str(binding["quality_report"]),
            "--source-markdown",
            str(binding["quality_markdown"]),
            "--locked-output",
            str(binding["locked_report"]),
            "--markdown-output",
            str(binding["markdown_report"]),
            "--checkpoint",
            str(binding["report_checkpoint"]),
        ]
    raise SupervisorError("supervisor_stage_invalid")


def _provision_checkpoint(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _config_path(binding, "runtime_root") / "large-evaluation-provisioning" / f"{DATASET_ID}.json"
    return _load_json(path, code="provisioning_checkpoint")


def _validate_index(binding: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _provision_checkpoint(binding)
    expected = int(binding["routing_document_count"])
    indexing = checkpoint.get("indexing")
    binding_value = checkpoint.get("binding")
    if (
        checkpoint.get("schema_version") != "large_evaluation_provisioning_v1"
        or not isinstance(binding_value, dict)
        or binding_value.get("dataset_id") != DATASET_ID
        or binding_value.get("corpus_sha256") != binding["routing_corpus_sha256"]
        or checkpoint.get("frozen_document_count") != expected
        or checkpoint.get("indexed_document_count") != expected
        or not isinstance(indexing, dict)
        or indexing.get("job_count") != expected
        or indexing.get("completed_count") != expected
        or indexing.get("failed_count") != 0
        or not isinstance(checkpoint.get("index_revision_id"), str)
        or not any(
            isinstance(event, dict) and event.get("event") == "indexing_stage_completed"
            for event in checkpoint.get("events", [])
        )
    ):
        raise ControlledStageFailure("supervisor_index_validation_failed")
    return {
        "expected_document_count": expected,
        "frozen_document_count": checkpoint["frozen_document_count"],
        "indexed_document_count": checkpoint["indexed_document_count"],
        "job_count": indexing["job_count"],
        "completed_count": indexing["completed_count"],
        "failed_count": indexing["failed_count"],
        "index_revision_id": checkpoint["index_revision_id"],
        "corpus_sha256": binding_value["corpus_sha256"],
    }


def _validate_graph(binding: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _provision_checkpoint(binding)
    graph = checkpoint.get("graph")
    if (
        checkpoint.get("status") != "completed"
        or not isinstance(graph, dict)
        or graph.get("status") != "ready"
        or graph.get("eligible_chunk_count") != binding["routing_document_count"]
        or graph.get("processed_chunk_count") != graph.get("eligible_chunk_count")
        or graph.get("last_error_code") is not None
        or not isinstance(checkpoint.get("knowledge_base_id"), str)
        or not isinstance(checkpoint.get("index_revision_id"), str)
        or not isinstance(checkpoint.get("graph_build_id"), str)
    ):
        raise ControlledStageFailure("supervisor_graph_validation_failed")
    runtime = load_evaluation_runtime(
        _config_path(binding, "runtime_manifest"),
        require_adaptive_graph=True,
        allow_canonical_checkout=True,
    )
    identity = runtime.adaptive_graph
    if (
        identity is None
        or str(identity.knowledge_base_id) != checkpoint["knowledge_base_id"]
        or str(identity.index_revision_id) != checkpoint["index_revision_id"]
        or str(identity.graph_build_id) != checkpoint["graph_build_id"]
        or str(identity.answer_profile_revision_id) != binding["chat_profile_revision_id"]
        or identity.schema_profile_key != GENERIC_GRAPH_SCHEMA_PROFILE_KEY
        or identity.schema_profile_digest != GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST
        or identity.extractor_version != GRAPH_EXTRACTOR_VERSION
    ):
        raise ControlledStageFailure("supervisor_graph_runtime_binding_failed")
    return {
        "knowledge_base_id": checkpoint["knowledge_base_id"],
        "index_revision_id": checkpoint["index_revision_id"],
        "graph_build_id": checkpoint["graph_build_id"],
        "eligible_chunk_count": graph["eligible_chunk_count"],
        "processed_chunk_count": graph["processed_chunk_count"],
        "extracted_chunk_count": graph.get("extracted_chunk_count"),
        "last_error_code": graph.get("last_error_code"),
        "schema_profile_key": identity.schema_profile_key,
        "schema_profile_digest": identity.schema_profile_digest,
        "extractor_version": identity.extractor_version,
    }


def _validate_gate_output(binding: Mapping[str, Any], path: Path) -> dict[str, Any]:
    result = _load_last_json_line(path, code="gate_output")
    if result.get("status") != "ready_for_smoke_revalidation":
        raise ControlledStageFailure("supervisor_gate_output_invalid")
    if result.get("plan_binding_sha256") != binding["plan_binding_sha256"]:
        raise ControlledStageFailure("supervisor_gate_binding_changed")
    checks = result.get("checks")
    if not isinstance(checks, dict) or any(
        checks.get(name) != "ok" for name in ("plan_binding", "host_dependencies", "chat_profile")
    ):
        raise ControlledStageFailure("supervisor_gate_checks_incomplete")
    return {
        "status": result["status"],
        "plan_binding_sha256": result["plan_binding_sha256"],
        "runtime_bundle_present": bool(result.get("runtime_bundle_present")),
        "checks": {name: checks.get(name) for name in ("plan_binding", "host_dependencies", "chat_profile", "provider_call")},
    }


def _validate_smoke(binding: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _load_json(_config_path(binding, "provider_smoke_checkpoint"), code="provider_smoke_checkpoint")
    providers = checkpoint.get("providers")
    steps = checkpoint.get("steps")
    smoke_binding = checkpoint.get("binding")
    expected_smoke_binding = {
        "runtime_build_revision": binding["runtime_build_revision"],
        "chat_profile_revision_id": binding["chat_profile_revision_id"],
        "text_embedding_profile_revision_id": binding[
            "text_embedding_profile_revision_id"
        ],
        "multimodal_embedding_profile_revision_id": binding[
            "multimodal_embedding_profile_revision_id"
        ],
        "models": dict(EXPECTED_MODELS),
        "resilience_policy_sha256": RESILIENCE_POLICY_SHA256,
    }
    if (
        checkpoint.get("schema_version") != PROVIDER_SMOKE_SCHEMA
        or checkpoint.get("status") != "completed"
        or smoke_binding != expected_smoke_binding
        or checkpoint.get("binding_sha256") != digest(expected_smoke_binding)
        or not isinstance(providers, dict)
        or set(providers) != set(EXPECTED_MODELS)
        or not isinstance(steps, list)
        or any(
            not isinstance(providers.get(name), dict)
            or providers[name].get("status") != "completed"
            or providers[name].get("model") != model
            for name, model in EXPECTED_MODELS.items()
        )
        or not any(
            isinstance(step, dict)
            and step.get("name") == "runner"
            and step.get("status") == "ok"
            for step in steps
        )
    ):
        raise ControlledStageFailure("supervisor_provider_smoke_validation_failed")
    return {
        "provider_count": len(providers),
        "models": {name: providers[name]["model"] for name in sorted(providers)},
        "completed_providers": sorted(providers),
        "binding_sha256": checkpoint["binding_sha256"],
        "attempt_count": {
            name: int(checkpoint.get("attempts", {}).get(name, {}).get("attempt_count", 0))
            for name in sorted(providers)
        },
    }


def _phase_progress(campaign: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    phases = campaign.get("phases")
    if not isinstance(phases, dict):
        return {}
    result: dict[str, dict[str, int]] = {}
    for phase, records in phases.items():
        if not isinstance(records, dict):
            continue
        completed = sum(
            1
            for record in records.values()
            if isinstance(record, dict) and record.get("status") == "completed"
        )
        result[str(phase)] = {"started_or_completed": len(records), "completed": completed}
    return result


def _validate_quality(binding: Mapping[str, Any]) -> dict[str, Any]:
    campaign = _load_json(_config_path(binding, "campaign_checkpoint"), code="campaign_checkpoint")
    campaign_binding = campaign.get("binding")
    smoke_path = _private_regular_file(
        _config_path(binding, "provider_smoke_checkpoint"),
        code="provider_smoke_checkpoint",
    )
    if (
        campaign.get("status") != "completed"
        or not isinstance(campaign_binding, dict)
        or campaign.get("binding_sha256") != digest(campaign_binding)
        or campaign_binding.get("plan_binding_sha256")
        != binding["plan_binding_sha256"]
        or campaign_binding.get("runtime_build_revision")
        != binding["runtime_build_revision"]
        or campaign_binding.get("provider_smoke_sha256")
        != hashlib.sha256(smoke_path.read_bytes()).hexdigest()
        or campaign_binding.get("resilience_policy_sha256")
        != RESILIENCE_POLICY_SHA256
    ):
        raise ControlledStageFailure("supervisor_quality_campaign_not_completed")
    progress = _phase_progress(campaign)
    summaries = campaign.get("phase_summaries")
    if (
        not progress
        or not isinstance(summaries, dict)
        or set(summaries) != set(progress)
        or any(
            not isinstance(summaries.get(phase), dict)
            or summaries[phase].get("status") != "completed"
            or summaries[phase].get("planned") != values["completed"]
            or values["started_or_completed"] != values["completed"]
            for phase, values in progress.items()
        )
    ):
        raise ControlledStageFailure("supervisor_quality_phase_counts_invalid")
    report = _load_json(_config_path(binding, "quality_report"), code="quality_report")
    if report.get("schema_version") != REPORT_SCHEMA or report.get("status") != "completed":
        raise ControlledStageFailure("supervisor_quality_report_invalid")
    if report.get("plan_binding_sha256") != binding["plan_binding_sha256"]:
        raise ControlledStageFailure("supervisor_quality_plan_binding_changed")
    quality_markdown = _config_path(binding, "quality_markdown")
    try:
        quality_markdown.chmod(0o600)
    except OSError as error:
        raise ControlledStageFailure("supervisor_quality_markdown_permissions_failed") from error
    return {
        "campaign_binding_sha256": campaign.get("binding_sha256"),
        "phase_progress": progress,
        "report_artifact_sha256": report.get("artifact_sha256"),
        "case_retry_count": report.get("resilience", {}).get("case_retry_count"),
    }


def _validate_report(binding: Mapping[str, Any]) -> dict[str, Any]:
    report = _load_json(_config_path(binding, "locked_report"), code="locked_report")
    if report.get("schema_version") != REPORT_SCHEMA or report.get("status") != "completed":
        raise ControlledStageFailure("supervisor_report_validation_failed")
    publish = _load_json(_config_path(binding, "report_checkpoint"), code="report_checkpoint")
    if (
        publish.get("schema_version") != "large_evaluation_report_publish_v1"
        or publish.get("status") != "completed"
        or publish.get("report_artifact_sha256") != report.get("artifact_sha256")
        or publish.get("locked_report_path") != str(_config_path(binding, "locked_report"))
        or publish.get("markdown_report_path") != str(_config_path(binding, "markdown_report"))
    ):
        raise ControlledStageFailure("supervisor_report_checkpoint_invalid")
    markdown = _private_regular_file(_config_path(binding, "markdown_report"), code="markdown_report")
    if not markdown.read_bytes().strip():
        raise ControlledStageFailure("supervisor_markdown_report_empty")
    supervisor_state = _load_json(
        _config_path(binding, "supervisor_state"), code="supervisor_state"
    )
    transition_audit = _validate_transition_evidence(supervisor_state)
    bundle = _write_analysis_bundle(binding)
    return {
        "artifact_sha256": report.get("artifact_sha256"),
        "locked_report_sha256": publish.get("locked_report_sha256"),
        "markdown_report_sha256": publish.get("markdown_report_sha256"),
        "analysis_bundle_path": bundle["path"],
        "analysis_bundle_sha256": bundle["sha256"],
        "analysis_archive_path": bundle["archive_path"],
        "analysis_archive_sha256": bundle["archive_sha256"],
        "transition_audit": transition_audit,
    }


def _validate_transition_evidence(state: Mapping[str, Any]) -> dict[str, Any]:
    events = state.get("events")
    stages = state.get("stages")
    if not isinstance(events, list) or not isinstance(stages, Mapping):
        raise ControlledStageFailure("supervisor_transition_evidence_invalid")
    started_indexes: dict[str, list[int]] = {stage: [] for stage in STAGES}
    completed_indexes: dict[str, list[int]] = {stage: [] for stage in STAGES}
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        stage = event.get("stage")
        if stage not in STAGES:
            continue
        if event.get("event") == "stage_started":
            started_indexes[str(stage)].append(index)
        elif event.get("event") == "stage_completed":
            completed_indexes[str(stage)].append(index)
    for index, stage in enumerate(STAGES):
        record = stages.get(stage)
        if not isinstance(record, Mapping) or not started_indexes[stage]:
            raise ControlledStageFailure("supervisor_stage_start_evidence_missing")
        if stage == STAGES[-1]:
            # Report validation runs immediately before the supervisor writes
            # report=completed, so its durable start is the required evidence.
            continue
        if record.get("status") != "completed" or not completed_indexes[stage]:
            raise ControlledStageFailure(
                "supervisor_stage_completion_evidence_missing"
            )
        completion_index = completed_indexes[stage][-1]
        next_stage = STAGES[index + 1]
        if not any(value > completion_index for value in started_indexes[next_stage]):
            raise ControlledStageFailure("supervisor_next_stage_start_missing")
    return {
        "status": "ok",
        "stage_started_count": {
            stage: len(started_indexes[stage]) for stage in STAGES
        },
        "completed_transition_count": len(STAGES) - 1,
    }


def _write_analysis_bundle(binding: Mapping[str, Any]) -> dict[str, str]:
    """Publish one content-safe manifest for later Agent-side analysis."""

    artifact_paths = [
        _config_path(binding, "locked_report"),
        _config_path(binding, "markdown_report"),
        _config_path(binding, "quality_report"),
        _config_path(binding, "quality_markdown"),
        _config_path(binding, "campaign_checkpoint"),
        _config_path(binding, "provider_smoke_checkpoint"),
        _config_path(binding, "report_checkpoint"),
        _config_path(binding, "supervisor_state"),
        _provision_checkpoint_path(binding),
    ]
    artifact_paths.extend(
        sorted(_config_path(binding, "stage_root").glob("*.json"))
    )
    artifact_paths.extend(
        sorted(_config_path(binding, "stage_log_root").glob("*.log"))
    )
    control_checkpoint = _config_path(binding, "campaign_root") / "control-checkpoint.json"
    if control_checkpoint.is_file():
        artifact_paths.append(control_checkpoint)
    artifacts: list[dict[str, Any]] = []
    private_paths: list[Path] = []
    for path in sorted(set(artifact_paths)):
        private = _private_regular_file(path, code="analysis_bundle_artifact")
        try:
            relative = private.relative_to(ROOT.resolve())
        except ValueError as error:
            raise ControlledStageFailure(
                "analysis_bundle_artifact_outside_repo"
            ) from error
        payload = private.read_bytes()
        private_paths.append(private)
        artifacts.append(
            {
                "path": str(private),
                "archive_path": str(Path("large-evaluation") / relative),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    bundle_path = _config_path(binding, "campaign_root") / "analysis-bundle.json"
    payload = {
        "schema_version": "large_evaluation_analysis_bundle_v1",
        "status": "completed",
        "created_at": _timestamp(),
        "config_sha256": digest(dict(binding)),
        "provider_contract": {
            "provider": "OpenCode Go",
            "chat_model": "mimo-v2.5",
            "text_embedding_model": "qwen3.7-text-embedding",
            "multimodal_embedding_model": "tongyi-embedding-vision-flash-2026-03-06",
        },
        "artifacts": artifacts,
    }
    bundle_sha256 = write_private_json(bundle_path, payload)
    archive_path = _config_path(binding, "campaign_root") / "analysis-bundle.tar.gz"
    archive_sha256 = _write_analysis_archive(
        archive_path,
        (*private_paths, bundle_path),
    )
    return {
        "path": str(bundle_path),
        "sha256": bundle_sha256,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
    }


def _write_analysis_archive(
    destination: Path,
    paths: Sequence[Path],
) -> str:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.parent.chmod(0o700)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mode = 0o600
        info.mtime = 0
        return info

    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                ) as archive:
                    for path in sorted(set(paths)):
                        private = _private_regular_file(
                            path, code="analysis_archive_artifact"
                        )
                        try:
                            relative = private.relative_to(ROOT.resolve())
                        except ValueError as error:
                            raise ControlledStageFailure(
                                "analysis_archive_artifact_outside_repo"
                            ) from error
                        archive.add(
                            private,
                            arcname=str(Path("large-evaluation") / relative),
                            recursive=False,
                            filter=normalize,
                        )
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
        destination.chmod(0o600)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def _provision_checkpoint_path(binding: Mapping[str, Any]) -> Path:
    return (
        _config_path(binding, "runtime_root")
        / "large-evaluation-provisioning"
        / f"{DATASET_ID}.json"
    )


def _validator_for_stage(binding: Mapping[str, Any], stage: str) -> Callable[[Path | None], dict[str, Any]]:
    if stage == "indexing":
        return lambda _path: _validate_index(binding)
    if stage == "graph":
        return lambda _path: _validate_graph(binding)
    if stage == "gate":
        return lambda path: _validate_gate_output(binding, path or Path(""))
    if stage == "provider_smoke":
        return lambda _path: _validate_smoke(binding)
    if stage == "quality_evaluation":
        return lambda _path: _validate_quality(binding)
    if stage == "report":
        return lambda _path: _validate_report(binding)
    raise SupervisorError("supervisor_stage_invalid")


def _read_exit(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _load_json(path, code="supervised_command_exit")
    if (
        value.get("schema_version") != EXIT_SCHEMA
        or value.get("status") not in {"completed", "failed"}
        or isinstance(value.get("exit_code"), bool)
        or not isinstance(value.get("exit_code"), int)
        or not isinstance(value.get("completed_at"), str)
    ):
        raise SupervisorError("supervised_command_exit_invalid")
    return value


def _load_last_json_line(path: Path, *, code: str) -> dict[str, Any]:
    private = _private_regular_file(path, code=code)
    try:
        lines = private.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SupervisorError(f"{code}_read_failed") from error
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise SupervisorError(f"{code}_json_invalid")


def _pid_command(pid: int) -> str | None:
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    result = subprocess.run(
        ["ps", "-p", str(pid), "-ww", "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    command = result.stdout.strip()
    return command or None


def _child_is_supervised(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int):
        return False
    command = _pid_command(pid)
    return command is not None and "tools/run_supervised_command.py" in command


def _stop_child_for_pause(record: Mapping[str, Any]) -> dict[str, Any] | None:
    pid = record.get("child_pid")
    exit_value: dict[str, Any] | None = None
    exit_file_value = record.get("exit_file")
    exit_path = Path(str(exit_file_value)) if exit_file_value else None
    if not _child_is_supervised(pid):
        if exit_path is not None and exit_path.exists():
            exit_value = _read_exit(exit_path)
        return exit_value
    assert isinstance(pid, int)
    try:
        process_group = os.getpgid(pid)
    except ProcessLookupError:
        process_group = pid
    if process_group != pid:
        raise SupervisorError("supervisor_child_process_group_invalid")
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + CHILD_PAUSE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if exit_path is not None and exit_path.exists():
            exit_value = _read_exit(exit_path)
        if _pid_command(pid) is None:
            return exit_value
        time.sleep(0.25)
    if _pid_command(pid) is not None:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _pid_command(pid) is not None:
        time.sleep(0.1)
    if _pid_command(pid) is not None:
        raise SupervisorError("supervisor_child_pause_timeout")
    if exit_path is not None and exit_path.exists():
        exit_value = _read_exit(exit_path)
    return exit_value


def _mark_stage_paused(
    binding: Mapping[str, Any],
    state_path: Path,
    state: dict[str, Any],
    stage: str,
    *,
    exit_value: Mapping[str, Any] | None = None,
    reason: str = "operator_request",
) -> None:
    record = state["stages"][stage]
    child_pid = record.get("child_pid")
    exit_file = record.get("exit_file")
    command_sha256 = record.get("command_sha256")
    if child_pid is not None:
        record["paused_child_pid"] = child_pid
    if exit_file is not None:
        record["paused_exit_file"] = exit_file
    if exit_value is not None:
        record["paused_exit_code"] = exit_value.get("exit_code")
    if command_sha256 is not None:
        record["paused_command_sha256"] = command_sha256
    for key in (
        "child_pid",
        "exit_file",
        "log_path",
        "command_sha256",
        "command_wrapper_sha256",
        "child_completed_at",
        "exit_code",
    ):
        record.pop(key, None)
    now = _timestamp()
    record["status"] = "paused"
    record["paused_at"] = now
    record["pause_reason"] = reason
    state["status"] = "paused"
    state["current_stage"] = stage
    state["paused_at"] = now
    state["pause_reason"] = reason
    _persist_stage(
        binding,
        state_path,
        state,
        stage,
        event="stage_paused",
        event_fields={
            "stage": stage,
            "child_exit_code": exit_value.get("exit_code") if exit_value else None,
            "reason": reason,
        },
    )
    _emit("stage_paused", stage=stage, reason=reason)


def _pause_pending_stage(
    binding: Mapping[str, Any],
    state_path: Path,
    state: dict[str, Any],
    stage: str,
    *,
    reason: str = "operator_request",
) -> None:
    record = state["stages"][stage]
    now = _timestamp()
    record["status"] = "paused"
    record["paused_at"] = now
    record["pause_reason"] = reason
    state["status"] = "paused"
    state["current_stage"] = stage
    state["paused_at"] = now
    state["pause_reason"] = reason
    _persist_stage(
        binding,
        state_path,
        state,
        stage,
        event="stage_paused",
        event_fields={"stage": stage, "child_exit_code": None, "reason": reason},
    )
    _emit("stage_paused", stage=stage, reason=reason)


def _interrupt_stage_for_shutdown(
    binding: Mapping[str, Any],
    state_path: Path,
    state: dict[str, Any],
    stage: str,
    *,
    exit_value: Mapping[str, Any] | None = None,
) -> None:
    record = state["stages"][stage]
    for key in (
        "child_pid",
        "exit_file",
        "log_path",
        "command_sha256",
        "command_wrapper_sha256",
        "child_completed_at",
        "exit_code",
        "heartbeat_at",
        "child_alive",
        "log_bytes",
        "stalled_seconds",
    ):
        record.pop(key, None)
    now = _timestamp()
    record["status"] = "pending"
    record["interrupted_at"] = now
    record["interruption_reason"] = "host_shutdown"
    if exit_value is not None:
        record["interrupted_exit_code"] = exit_value.get("exit_code")
    state["status"] = "running"
    state["current_stage"] = stage
    _persist_stage(
        binding,
        state_path,
        state,
        stage,
        event="stage_interrupted",
        event_fields={
            "stage": stage,
            "reason": "host_shutdown",
            "child_exit_code": exit_value.get("exit_code")
            if exit_value
            else None,
        },
    )
    _emit("stage_interrupted", stage=stage, reason="host_shutdown")


def _start_child(
    binding: Mapping[str, Any],
    state_path: Path,
    state: dict[str, Any],
    stage: str,
    command: Sequence[str],
) -> None:
    record = state["stages"][stage]
    attempt = int(record.get("attempt_count", 0)) + 1
    exit_path = _config_path(binding, "stage_root") / f"{stage}-attempt-{attempt}-exit.json"
    log_path = _stage_log_path(binding, stage, attempt)
    wrapper = [
        str(binding["python"]),
        _tool_path(binding, "run_supervised_command.py"),
        "--exit-file",
        str(exit_path),
        "--",
        *command,
    ]
    log_handle = _ensure_private_log(log_path)
    try:
        child = subprocess.Popen(
            wrapper,
            cwd="/private/tmp",
            env=_base_environment(binding),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        raise
    finally:
        log_handle.close()
    time.sleep(CHILD_START_GRACE_SECONDS)
    if child.poll() is not None and not exit_path.exists():
        raise ControlledStageFailure("supervisor_child_failed_to_start")
    record.update(
        {
            "status": "running",
            "attempt_count": attempt,
            "started_at": _timestamp(),
            "child_pid": child.pid,
            "exit_file": str(exit_path),
            "log_path": str(log_path),
            "command_sha256": _command_digest(command),
            "command_wrapper_sha256": _command_digest(wrapper),
            "resilience_policy_sha256": RESILIENCE_POLICY_SHA256,
            "implementation_sha256": _automation_implementation_sha256(binding),
            "observed": {},
        }
    )
    _persist_stage(
        binding,
        state_path,
        state,
        stage,
        event="stage_started",
        event_fields={
            "stage": stage,
            "attempt": attempt,
            "child_pid": child.pid,
            "command_sha256": record["command_sha256"],
            "next_stage_started": bool(record.get("next_stage_of")),
        },
    )
    _emit("stage_started", stage=stage, attempt=attempt, pid=child.pid)


def _refresh_observation(
    binding: Mapping[str, Any],
    stage: str,
    record: dict[str, Any],
) -> bool:
    previous = record.get("observed")
    observed: dict[str, Any] | None = None
    if stage in {"indexing", "graph"}:
        try:
            checkpoint = _provision_checkpoint(binding)
        except SupervisorError:
            return False
        if stage == "indexing":
            indexing = checkpoint.get("indexing")
            if isinstance(indexing, dict):
                observed = {
                    "status": checkpoint.get("status"),
                    "job_count": indexing.get("job_count"),
                    "completed_count": indexing.get("completed_count"),
                    "failed_count": indexing.get("failed_count"),
                }
        else:
            graph = checkpoint.get("graph")
            if isinstance(graph, dict):
                observed = {
                    "status": graph.get("status"),
                    "eligible_chunk_count": graph.get("eligible_chunk_count"),
                    "processed_chunk_count": graph.get("processed_chunk_count"),
                    "extracted_chunk_count": graph.get("extracted_chunk_count"),
                    "last_error_code": graph.get("last_error_code"),
                }
    elif stage == "provider_smoke":
        try:
            checkpoint = _load_json(_config_path(binding, "provider_smoke_checkpoint"), code="provider_smoke_checkpoint")
        except SupervisorError:
            return False
        providers = checkpoint.get("providers")
        if isinstance(providers, dict):
            observed = {
                "status": checkpoint.get("status"),
                "completed_providers": sorted(
                    name
                    for name, value in providers.items()
                    if isinstance(value, dict) and value.get("status") == "completed"
                ),
                "attempts": {
                    name: {
                        key: attempt.get(key)
                        for key in (
                            "status",
                            "attempt_count",
                            "next_retry_at",
                            "heartbeat_at",
                        )
                        if attempt.get(key) is not None
                    }
                    for name, attempt in checkpoint.get("attempts", {}).items()
                    if isinstance(name, str) and isinstance(attempt, dict)
                },
                "heartbeat_at": checkpoint.get("heartbeat_at"),
            }
    elif stage == "quality_evaluation":
        try:
            checkpoint = _load_json(_config_path(binding, "campaign_checkpoint"), code="campaign_checkpoint")
        except SupervisorError:
            return False
        observed = {
            "status": checkpoint.get("status"),
            "phase_progress": _phase_progress(checkpoint),
            "active_phase": checkpoint.get("active_phase"),
            "active_case": checkpoint.get("active_case"),
            "heartbeat_at": checkpoint.get("runner_heartbeat_at"),
        }
    if observed is None or observed == previous:
        return False
    record["observed"] = observed
    return True


def _progress_signature(stage: str, observed: object) -> object:
    if not isinstance(observed, Mapping):
        return observed
    if stage == "provider_smoke":
        attempts = observed.get("attempts")
        return {
            "status": observed.get("status"),
            "completed_providers": observed.get("completed_providers"),
            "attempts": {
                name: {
                    key: value.get(key)
                    for key in ("status", "attempt_count", "next_retry_at")
                }
                for name, value in attempts.items()
                if isinstance(name, str) and isinstance(value, Mapping)
            }
            if isinstance(attempts, Mapping)
            else {},
        }
    if stage == "quality_evaluation":
        active = observed.get("active_case")
        return {
            "status": observed.get("status"),
            "phase_progress": observed.get("phase_progress"),
            "active_phase": observed.get("active_phase"),
            "active_case": {
                key: active.get(key)
                for key in ("phase", "case_id", "attempt_count")
            }
            if isinstance(active, Mapping)
            else None,
        }
    return dict(observed)


def _stage_stall_timeout(stage: str) -> float:
    if stage in {"gate", "report"}:
        return SHORT_STAGE_STALL_TIMEOUT_SECONDS
    return LONG_STAGE_STALL_TIMEOUT_SECONDS


def _wait_for_stage(
    binding: Mapping[str, Any],
    state_path: Path,
    state: dict[str, Any],
    stage: str,
) -> tuple[int, dict[str, Any]]:
    record = state["stages"][stage]
    command = _command_for_stage(binding, stage)
    expected_command_sha256 = _command_digest(command)
    stored_command = record.get("command_sha256")
    if record.get("status") == "running" and stored_command not in {None, expected_command_sha256}:
        raise ControlledStageFailure("supervisor_stage_command_changed")
    exit_path = Path(str(record["exit_file"])) if record.get("exit_file") else None
    exit_record = _read_exit(exit_path) if exit_path is not None else None
    if exit_record is None:
        pid = record.get("child_pid")
        if not _child_is_supervised(pid):
            _start_child(binding, state_path, state, stage, command)
            record = state["stages"][stage]
            exit_path = Path(str(record["exit_file"]))
        else:
            _emit("stage_adopted", stage=stage, pid=pid)
    else:
        _emit("stage_exit_recovered", stage=stage, exit_code=exit_record["exit_code"])

    last_persisted = record.get("observed")
    last_progress_signature = _progress_signature(stage, last_persisted)
    last_heartbeat = time.monotonic()
    last_activity = time.monotonic()
    last_log_size = int(record.get("log_bytes", 0))
    while True:
        if _refresh_observation(binding, stage, record):
            if record.get("observed") != last_persisted:
                _persist_stage(
                    binding,
                    state_path,
                    state,
                    stage,
                    event="stage_progress",
                    event_fields={"stage": stage},
                )
                last_persisted = record.get("observed")
                _emit("stage_progress", stage=stage, observed=record["observed"])
                signature = _progress_signature(stage, last_persisted)
                if signature != last_progress_signature:
                    last_progress_signature = signature
                    last_activity = time.monotonic()
        if time.monotonic() - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
            record["heartbeat_at"] = _timestamp()
            record["child_alive"] = _child_is_supervised(record.get("child_pid"))
            log_value = record.get("log_path")
            if isinstance(log_value, str):
                try:
                    record["log_bytes"] = Path(log_value).stat().st_size
                except OSError:
                    pass
            current_log_size = int(record.get("log_bytes", 0))
            if current_log_size != last_log_size:
                last_log_size = current_log_size
                last_activity = time.monotonic()
            record["stalled_seconds"] = int(time.monotonic() - last_activity)
            _persist_stage(binding, state_path, state, stage)
            _emit(
                "stage_heartbeat",
                stage=stage,
                child_alive=record["child_alive"],
                log_bytes=record.get("log_bytes"),
                stalled_seconds=record["stalled_seconds"],
            )
            last_heartbeat = time.monotonic()
        exit_record = _read_exit(exit_path)
        if exit_record is not None:
            break
        if _pause_requested:
            exit_record = _stop_child_for_pause(record)
            _mark_stage_paused(
                binding,
                state_path,
                state,
                stage,
                exit_value=exit_record,
            )
            raise SupervisorPaused("supervisor_paused")
        if _shutdown_requested:
            exit_record = _stop_child_for_pause(record)
            _interrupt_stage_for_shutdown(
                binding,
                state_path,
                state,
                stage,
                exit_value=exit_record,
            )
            raise SupervisorShutdown("supervisor_host_shutdown")
        if time.monotonic() - last_activity >= _stage_stall_timeout(stage):
            restart_count = int(record.get("stall_restart_count", 0))
            _stop_child_for_pause(record)
            if restart_count >= MAX_STALL_RESTARTS:
                raise ControlledStageFailure(
                    f"supervisor_{stage}_stall_restart_exhausted"
                )
            record["stall_restart_count"] = restart_count + 1
            _persist_stage(
                binding,
                state_path,
                state,
                stage,
                event="stage_stall_restart",
                event_fields={
                    "stage": stage,
                    "stall_restart_count": record["stall_restart_count"],
                },
            )
            _emit(
                "stage_stall_restart",
                stage=stage,
                stall_restart_count=record["stall_restart_count"],
            )
            _start_child(binding, state_path, state, stage, command)
            record = state["stages"][stage]
            exit_path = Path(str(record["exit_file"]))
            exit_record = None
            last_persisted = record.get("observed")
            last_progress_signature = _progress_signature(stage, last_persisted)
            last_heartbeat = time.monotonic()
            last_activity = time.monotonic()
            last_log_size = 0
            continue
        time.sleep(POLL_SECONDS)

    exit_code = int(exit_record["exit_code"])
    record["exit_code"] = exit_code
    record["child_completed_at"] = exit_record.get("completed_at")
    if exit_code != 0:
        record["status"] = "failed"
        record["failure_code"] = f"child_exit_{exit_code}"
        raise ControlledStageFailure(f"supervisor_{stage}_child_exit_{exit_code}")
    try:
        observed = _validator_for_stage(binding, stage)(
            _stage_log_path(binding, stage, int(record["attempt_count"]))
            if stage == "gate"
            else None
        )
    except SupervisorError:
        record["status"] = "failed"
        record["failure_code"] = f"{stage}_validation_failed"
        raise
    record["status"] = "completed"
    record["completed_at"] = _timestamp()
    record["observed"] = observed
    _persist_stage(
        binding,
        state_path,
        state,
        stage,
        event="stage_completed",
        event_fields={"stage": stage, "exit_code": 0},
    )
    _emit("stage_completed", stage=stage, exit_code=0)
    return 0, observed


def _mark_failed(
    binding: Mapping[str, Any],
    state_path: Path,
    state: dict[str, Any],
    stage: str,
    error: BaseException,
) -> None:
    record = state["stages"][stage]
    record["status"] = "failed"
    record["failure_code"] = getattr(error, "code", str(error))
    record["failure_type"] = type(error).__name__
    record["failed_at"] = _timestamp()
    state["status"] = "failed"
    state["current_stage"] = stage
    state["last_failure"] = {
        "stage": stage,
        "failure_code": record["failure_code"],
        "failure_type": record["failure_type"],
        "at": record["failed_at"],
    }
    _persist_stage(
        binding,
        state_path,
        state,
        stage,
        event="stage_failed",
        event_fields={"stage": stage, "failure_code": record["failure_code"]},
    )
    _emit("stage_failed", stage=stage, failure_code=record["failure_code"])


def _mark_next_started(
    binding: Mapping[str, Any],
    state_path: Path,
    state: dict[str, Any],
    previous_stage: str,
    next_stage: str,
) -> None:
    next_record = state["stages"][next_stage]
    state["current_stage"] = next_stage
    next_record["status"] = "pending"
    next_record["next_stage_of"] = previous_stage
    _persist(
        state_path,
        state,
        event="next_stage_ready",
        event_fields={"previous_stage": previous_stage, "next_stage": next_stage},
    )


def _resume_failed_stage(
    binding: Mapping[str, Any],
    state_path: Path,
    config_sha256: str,
    stage: str,
) -> dict[str, Any]:
    """Re-arm exactly one failed stage without fabricating completion.

    A repair/retry is an explicit operator action.  It preserves the failed
    attempt and its exit sidecar, clears only the transient child fields, and
    leaves every later stage pending.  The normal launchd supervisor then
    starts a new attempt and applies the same exit/count/config validation.
    """

    if stage not in STAGES:
        raise SupervisorError("supervisor_resume_stage_invalid")
    state = _load_or_create_state(state_path, config_sha256)
    _validate_stage_checkpoints(binding, state)
    if state.get("status") != "failed" or state.get("current_stage") != stage:
        raise SupervisorError("supervisor_resume_target_invalid")
    record = state["stages"][stage]
    if record.get("status") != "failed":
        raise SupervisorError("supervisor_resume_record_invalid")
    stage_index = STAGES.index(stage)
    if any(
        state["stages"][previous].get("status") != "completed"
        for previous in STAGES[:stage_index]
    ) or any(
        state["stages"][following].get("status") != "pending"
        for following in STAGES[stage_index + 1 :]
    ):
        raise SupervisorError("supervisor_resume_stage_order_invalid")
    previous_failure = {
        key: record.get(key)
        for key in ("failure_code", "failure_type", "failed_at", "exit_code")
        if record.get(key) is not None
    }
    _clear_stage_execution_fields(record)
    record["status"] = "pending"
    record["resume_count"] = int(record.get("resume_count", 0)) + 1
    state["status"] = "running"
    state["current_stage"] = stage
    state["resilience_policy_sha256"] = RESILIENCE_POLICY_SHA256
    state["implementation_sha256"] = _automation_implementation_sha256(binding)
    state.pop("last_failure", None)
    _persist_stage(
        binding,
        state_path,
        state,
        stage,
        event="stage_resume_requested",
        event_fields={
            "stage": stage,
            "resume_count": record["resume_count"],
            "previous_failure": previous_failure,
        },
    )
    return {
        "status": "resumed",
        "stage": stage,
        "resume_count": record["resume_count"],
        "config_sha256": config_sha256,
    }


def _clear_stage_execution_fields(record: dict[str, Any]) -> None:
    for key in (
        "child_pid",
        "exit_file",
        "log_path",
        "command_sha256",
        "command_wrapper_sha256",
        "started_at",
        "child_completed_at",
        "completed_at",
        "failed_at",
        "failure_code",
        "failure_type",
        "exit_code",
        "observed",
        "paused_at",
        "pause_reason",
        "paused_child_pid",
        "paused_exit_file",
        "paused_exit_code",
        "heartbeat_at",
        "child_alive",
        "log_bytes",
        "stalled_seconds",
        "stall_restart_count",
        "interrupted_at",
        "interruption_reason",
        "interrupted_exit_code",
    ):
        record.pop(key, None)


def _resume_paused_stage(
    binding: Mapping[str, Any],
    state_path: Path,
    config_sha256: str,
) -> dict[str, Any]:
    """Re-arm the one paused stage while preserving all prior attempts."""

    state = _load_or_create_state(state_path, config_sha256)
    _validate_stage_checkpoints(binding, state)
    if state.get("status") != "paused":
        raise SupervisorError("supervisor_resume_paused_target_invalid")
    stage = state.get("current_stage")
    if stage not in STAGES:
        raise SupervisorError("supervisor_resume_paused_stage_invalid")
    record = state["stages"][stage]
    if record.get("status") != "paused":
        raise SupervisorError("supervisor_resume_paused_record_invalid")
    stage_index = STAGES.index(stage)
    if any(
        state["stages"][previous].get("status") != "completed"
        for previous in STAGES[:stage_index]
    ) or any(
        state["stages"][following].get("status") != "pending"
        for following in STAGES[stage_index + 1 :]
    ):
        raise SupervisorError("supervisor_resume_paused_stage_order_invalid")
    previous_pause = {
        key: record.get(key)
        for key in ("pause_reason", "paused_at", "paused_exit_code")
        if record.get(key) is not None
    }
    resume_count = int(record.get("resume_count", 0)) + 1
    _clear_stage_execution_fields(record)
    record["status"] = "pending"
    record["resume_count"] = resume_count
    state["status"] = "running"
    state["current_stage"] = stage
    state["resilience_policy_sha256"] = RESILIENCE_POLICY_SHA256
    state["implementation_sha256"] = _automation_implementation_sha256(binding)
    state.pop("pause_reason", None)
    state.pop("paused_at", None)
    _persist_stage(
        binding,
        state_path,
        state,
        stage,
        event="stage_resume_requested",
        event_fields={
            "stage": stage,
            "resume_count": resume_count,
            "previous_pause": previous_pause,
        },
    )
    return {
        "status": "resumed",
        "stage": stage,
        "resume_count": resume_count,
        "config_sha256": config_sha256,
    }


def _wait_for_runtime_ready(binding: Mapping[str, Any]) -> None:
    runtime = load_evaluation_runtime(
        _config_path(binding, "runtime_manifest"),
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
    health_url = runtime.api_base_url.rsplit("/api/v1", 1)[0] + API_READY_PATH
    deadline = time.monotonic() + RUNTIME_READY_TIMEOUT_SECONDS
    _emit("runtime_readiness_wait_started", health_url=health_url)
    while time.monotonic() < deadline:
        if _pause_requested:
            raise SupervisorPaused("supervisor_paused")
        if _shutdown_requested:
            raise SupervisorShutdown("supervisor_host_shutdown")
        try:
            with urlopen(health_url, timeout=2.0) as response:
                if 200 <= int(response.status) < 300:
                    _emit("runtime_ready", health_url=health_url)
                    return
        except (OSError, URLError, ValueError):
            pass
        time.sleep(POLL_SECONDS)
    raise ControlledStageFailure("supervisor_runtime_not_ready")


def _run_marker_path(binding: Mapping[str, Any]) -> Path:
    return _config_path(binding, "campaign_root") / RUN_MARKER_NAME


def _validate_run_marker(
    binding: Mapping[str, Any], config_sha256: str
) -> None:
    marker = _load_json(_run_marker_path(binding), code="evaluation_run_marker")
    if (
        marker.get("schema_version") != "large_evaluation_run_marker_v1"
        or marker.get("status") != "enabled"
        or marker.get("config_sha256") != config_sha256
        or marker.get("resilience_policy_sha256") != RESILIENCE_POLICY_SHA256
        or marker.get("implementation_sha256")
        != _automation_implementation_sha256(binding)
    ):
        raise ControlledStageFailure("supervisor_run_marker_invalid")


def _validate_automation_binding(
    binding: Mapping[str, Any], state: Mapping[str, Any]
) -> None:
    if (
        state.get("resilience_policy_sha256") != RESILIENCE_POLICY_SHA256
        or state.get("implementation_sha256")
        != _automation_implementation_sha256(binding)
    ):
        raise ControlledStageFailure("supervisor_automation_binding_invalid")


def _runtime_ports_stopped(binding: Mapping[str, Any]) -> bool:
    runtime = load_evaluation_runtime(
        _config_path(binding, "runtime_manifest"),
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
    ports = [
        int(port)
        for name, port in runtime.ports.items()
        if name in {"postgres", "falkordb", "api"}
    ]
    deadline = time.monotonic() + RUNTIME_SETTLE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        listening = False
        for port in ports:
            with socket.socket() as probe:
                probe.settimeout(0.2)
                try:
                    probe.connect(("127.0.0.1", port))
                except OSError:
                    continue
                listening = True
                break
        if not listening:
            return True
        time.sleep(0.25)
    return False


def _settle_isolated_runtime(
    binding: Mapping[str, Any],
    state_path: Path,
    state: dict[str, Any],
    *,
    reason: str,
) -> None:
    """Stop supporting LaunchAgents after a terminal evaluation state."""

    marker = _run_marker_path(binding)
    marker_removed = True
    try:
        marker.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        marker_removed = False
    failures: list[str] = []
    domain = f"gui/{os.getuid()}"
    for label in SUPPORT_LABELS:
        target = f"{domain}/{label}"
        loaded = subprocess.run(
            ["launchctl", "print", target],
            check=False,
            capture_output=True,
            text=True,
        ).returncode == 0
        if not loaded:
            continue
        result = subprocess.run(
            ["launchctl", "bootout", target],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failures.append(label)
    ports_stopped = _runtime_ports_stopped(binding)
    status = (
        "completed"
        if marker_removed and not failures and ports_stopped
        else "failed"
    )
    state["runtime_settlement"] = {
        "status": status,
        "reason": reason,
        "marker_removed": marker_removed,
        "ports_stopped": ports_stopped,
        "failed_agent_count": len(failures),
        "at": _timestamp(),
    }
    _persist(
        state_path,
        state,
        event=(
            "runtime_settlement_completed"
            if status == "completed"
            else "runtime_settlement_failed"
        ),
        event_fields={"reason": reason, "ports_stopped": ports_stopped},
    )
    _emit(
        "runtime_settlement",
        status=status,
        reason=reason,
        ports_stopped=ports_stopped,
        failed_agent_count=len(failures),
    )


def _run_state_machine(binding: Mapping[str, Any], config_sha256: str) -> int:
    _validate_static_config(binding)
    state_path = _config_path(binding, "supervisor_state")
    state = _load_or_create_state(state_path, config_sha256)
    _validate_stage_checkpoints(binding, state)
    if state.get("status") == "completed":
        _settle_isolated_runtime(
            binding, state_path, state, reason="evaluation_completed"
        )
        _emit("supervisor_already_completed")
        return 0
    if state.get("status") == "paused":
        _emit(
            "supervisor_paused_waiting_for_resume",
            stage=state.get("current_stage"),
        )
        return 0
    if state.get("status") == "failed":
        _settle_isolated_runtime(
            binding, state_path, state, reason="evaluation_failed"
        )
        _emit("supervisor_stopped_after_failure", stage=state.get("current_stage"))
        return 0
    try:
        _validate_automation_binding(binding, state)
        _validate_run_marker(binding, config_sha256)
    except SupervisorError as error:
        stage = str(state.get("current_stage"))
        _mark_failed(binding, state_path, state, stage, error)
        _settle_isolated_runtime(
            binding, state_path, state, reason="run_marker_invalid"
        )
        return 0
    runtime_ready = False
    for index, stage in enumerate(STAGES):
        record = state["stages"][stage]
        if record.get("status") == "completed":
            continue
        state["current_stage"] = stage
        if not runtime_ready:
            _persist(
                state_path,
                state,
                event="runtime_readiness_waiting",
                event_fields={"stage": stage},
            )
            try:
                _wait_for_runtime_ready(binding)
            except ControlledStageFailure as error:
                _mark_failed(binding, state_path, state, stage, error)
                _settle_isolated_runtime(
                    binding, state_path, state, reason="runtime_readiness_failed"
                )
                return 0
            except SupervisorPaused:
                _pause_pending_stage(binding, state_path, state, stage)
                return 0
            except SupervisorShutdown:
                _persist(
                    state_path,
                    state,
                    event="supervisor_shutdown",
                    event_fields={"stage": stage, "reason": "host_shutdown"},
                )
                return 0
            runtime_ready = True
        if record.get("status") == "pending":
            record["status"] = "running"
            record["attempt_count"] = int(record.get("attempt_count", 0))
            _persist_stage(
                binding,
                state_path,
                state,
                stage,
                event="stage_armed",
                event_fields={"stage": stage},
            )
        try:
            _wait_for_stage(binding, state_path, state, stage)
        except ControlledStageFailure as error:
            _mark_failed(binding, state_path, state, stage, error)
            _settle_isolated_runtime(
                binding, state_path, state, reason=f"stage_failed:{stage}"
            )
            return 0
        except SupervisorPaused:
            return 0
        except SupervisorShutdown:
            return 0
        except SupervisorError:
            raise
        if index + 1 < len(STAGES):
            _mark_next_started(binding, state_path, state, stage, STAGES[index + 1])
            if _pause_requested:
                _pause_pending_stage(
                    binding,
                    state_path,
                    state,
                    STAGES[index + 1],
                )
                return 0
    state["status"] = "completed"
    state["current_stage"] = "report"
    state["completed_at"] = _timestamp()
    _persist(state_path, state, event="supervisor_completed")
    _settle_isolated_runtime(
        binding, state_path, state, reason="evaluation_completed"
    )
    _emit("supervisor_completed")
    return 0


def _status(arguments: argparse.Namespace) -> int:
    binding, config_sha256 = _load_config(arguments.config)
    state_path = _config_path(binding, "supervisor_state")
    state = _load_or_create_state(state_path, config_sha256)
    _validate_stage_checkpoints(binding, state)
    runtime = load_evaluation_runtime(
        _config_path(binding, "runtime_manifest"),
        require_adaptive_graph=False,
        allow_canonical_checkout=True,
    )
    ports: dict[str, bool] = {}
    for name, port in runtime.ports.items():
        if name not in {"postgres", "falkordb", "api"}:
            continue
        with socket.socket() as probe:
            probe.settimeout(0.2)
            try:
                probe.connect(("127.0.0.1", int(port)))
            except OSError:
                ports[name] = False
            else:
                ports[name] = True
    current_record = state["stages"].get(state.get("current_stage"), {})
    bundle_path = _config_path(binding, "campaign_root") / "analysis-bundle.json"
    archive_path = _config_path(binding, "campaign_root") / "analysis-bundle.tar.gz"
    current_implementation_sha256 = _automation_implementation_sha256(binding)
    summary = {
        "schema_version": state["schema_version"],
        "status": state["status"],
        "current_stage": state.get("current_stage"),
        "config_sha256": state["config_sha256"],
        "updated_at": state.get("updated_at"),
        "stages": {
            stage: {
                key: record.get(key)
                for key in (
                    "status",
                    "attempt_count",
                    "exit_code",
                    "started_at",
                    "completed_at",
                    "failed_at",
                    "failure_code",
                    "observed",
                    "heartbeat_at",
                    "child_alive",
                    "log_bytes",
                )
                if (record := state["stages"][stage]).get(key) is not None
            }
            for stage in STAGES
        },
        "last_failure": state.get("last_failure"),
        "runtime": {
            "run_marker_present": _run_marker_path(binding).exists(),
            "ports_listening": ports,
            "settlement": state.get("runtime_settlement"),
        },
        "supervision": {
            "supervisor_pid_present": _config_path(
                binding, "supervisor_pid"
            ).exists(),
            "current_child_alive": _child_is_supervised(
                current_record.get("child_pid")
            )
            if isinstance(current_record, Mapping)
            else False,
            "checkpoint_integrity": "ok",
            "automation_binding": {
                "resilience_policy_sha256": state.get(
                    "resilience_policy_sha256"
                ),
                "current_resilience_policy_sha256": RESILIENCE_POLICY_SHA256,
                "implementation_sha256": state.get("implementation_sha256"),
                "current_implementation_sha256": current_implementation_sha256,
                "matches_current": (
                    state.get("resilience_policy_sha256")
                    == RESILIENCE_POLICY_SHA256
                    and state.get("implementation_sha256")
                    == current_implementation_sha256
                ),
            },
        },
        "analysis_bundle": {
            "manifest": str(bundle_path) if bundle_path.is_file() else None,
            "archive": str(archive_path) if archive_path.is_file() else None,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


_pause_requested = False
_shutdown_requested = False


def _handle_signal(signum: int, _frame: Any) -> None:
    global _pause_requested, _shutdown_requested
    if signum == signal.SIGTERM:
        _shutdown_requested = True
        _emit("supervisor_shutdown_requested", signal=signum)
        return
    _pause_requested = True
    _emit("supervisor_pause_requested", signal=signum)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--resume-failed-stage", choices=STAGES)
    mode.add_argument("--resume-paused", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        binding, config_sha256 = _load_config(arguments.config.resolve())
        if arguments.status:
            return _status(arguments)
        _validate_static_config(binding)
        state_path = _config_path(binding, "supervisor_state")
        lock_path = _config_path(binding, "supervisor_lock")
        pid_path = _config_path(binding, "supervisor_pid")
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, _handle_signal)
        with _supervisor_lock(lock_path):
            _write_pid(pid_path)
            try:
                if arguments.resume_failed_stage:
                    result = _resume_failed_stage(
                        binding,
                        state_path,
                        config_sha256,
                        arguments.resume_failed_stage,
                    )
                    print(json.dumps(result, sort_keys=True))
                    return 0
                if arguments.resume_paused:
                    result = _resume_paused_stage(
                        binding,
                        state_path,
                        config_sha256,
                    )
                    print(json.dumps(result, sort_keys=True))
                    return 0
                _emit("supervisor_started", config_sha256=config_sha256, pid=os.getpid())
                return _run_state_machine(binding, config_sha256)
            finally:
                _remove_pid(pid_path)
    except ControlledStageFailure as error:
        _emit("supervisor_controlled_failure", failure_code=error.code)
        return 0
    except SupervisorPaused as error:
        _emit("supervisor_paused", reason=error.code)
        return 0
    except SupervisorShutdown as error:
        _emit("supervisor_shutdown", reason=error.code)
        return 0
    except SupervisorError as error:
        _emit("supervisor_blocked", failure_code=error.code)
        return 0
    except (EvaluationRuntimeError, OSError, ValueError, TypeError) as error:
        _emit("supervisor_crashed", failure_type=type(error).__name__)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
