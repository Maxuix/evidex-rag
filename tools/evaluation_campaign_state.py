"""Atomic, resumable state for a multi-phase locked evaluation campaign.

The module deliberately contains no provider, database, Docker, or process
management code.  Runners use it to bind a campaign to its frozen corpus plan
and persist every case transition before proceeding to the next case.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


SCHEMA_VERSION = "large_evaluation_campaign_state_v1"


class CampaignStateError(RuntimeError):
    """Raised when a checkpoint cannot safely be resumed."""


def canonical_bytes(value: object) -> bytes:
    """Return a deterministic JSON representation suitable for hashing."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def digest(value: object) -> str:
    """Return the SHA-256 digest for canonical JSON data."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_private_json(path: Path, value: Mapping[str, Any]) -> str:
    """Durably replace a private checkpoint and return its payload digest."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    payload = canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _validate_state(value: object, binding: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise CampaignStateError("campaign_state_schema_invalid")
    if value.get("binding") != dict(binding):
        raise CampaignStateError("campaign_state_binding_mismatch")
    if value.get("binding_sha256") != digest(binding):
        raise CampaignStateError("campaign_state_binding_digest_invalid")
    if not isinstance(value.get("phases"), dict) or not isinstance(value.get("events"), list):
        raise CampaignStateError("campaign_state_shape_invalid")
    return value


def load_or_create(path: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Load a bound campaign state or create its initial durable checkpoint."""

    if not path.exists():
        state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "binding": dict(binding),
            "binding_sha256": digest(binding),
            "status": "running",
            "events": [{"event": "campaign_created", "at": _timestamp()}],
            "phases": {},
        }
        write_private_json(path, state)
        return state
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise CampaignStateError("campaign_state_permissions_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CampaignStateError("campaign_state_json_invalid") from error
    return _validate_state(value, binding)


def begin_case(
    path: Path,
    state: dict[str, Any],
    *,
    phase: str,
    case_id: str,
) -> bool:
    """Persist a case attempt and return whether the caller must execute it.

    A completed case is never rerun.  A previously interrupted ``started``
    case is deliberately retried, with the interruption retained in its event
    history; it is not misreported as a completed observation.
    """

    if not phase or not case_id:
        raise CampaignStateError("campaign_case_identity_invalid")
    phases = state["phases"]
    if not isinstance(phases, dict):
        raise CampaignStateError("campaign_state_shape_invalid")
    phase_cases = phases.setdefault(phase, {})
    if not isinstance(phase_cases, dict):
        raise CampaignStateError("campaign_phase_shape_invalid")
    record = phase_cases.get(case_id)
    if record is not None and not isinstance(record, dict):
        raise CampaignStateError("campaign_case_shape_invalid")
    if record is not None and record.get("status") == "completed":
        return False
    attempts = int(record.get("attempt_count", 0)) if record else 0
    if record is not None and record.get("status") == "started":
        state["events"].append(
            {
                "event": "case_resumed_after_interruption",
                "phase": phase,
                "case_id": case_id,
                "at": _timestamp(),
            }
        )
    elif record is not None and record.get("status") == "retry_wait":
        state["events"].append(
            {
                "event": "case_retry_started",
                "phase": phase,
                "case_id": case_id,
                "at": _timestamp(),
            }
        )
    retained = {
        key: record[key]
        for key in ("retry_count", "last_failure", "last_retry_at")
        if record is not None and key in record
    }
    phase_cases[case_id] = {
        "status": "started",
        "attempt_count": attempts + 1,
        "started_at": _timestamp(),
        **retained,
    }
    state["events"].append(
        {"event": "case_started", "phase": phase, "case_id": case_id, "at": _timestamp()}
    )
    write_private_json(path, state)
    return True


def schedule_case_retry(
    path: Path,
    state: dict[str, Any],
    *,
    phase: str,
    case_id: str,
    failure: Mapping[str, Any],
    next_retry_at: str,
) -> None:
    """Persist a bounded retry without marking the case complete."""

    phase_cases = state.get("phases", {}).get(phase)
    if not isinstance(phase_cases, dict) or not isinstance(
        phase_cases.get(case_id), dict
    ):
        raise CampaignStateError("campaign_case_not_started")
    record = phase_cases[case_id]
    if record.get("status") != "started":
        raise CampaignStateError("campaign_case_not_running")
    try:
        json.dumps(failure, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise CampaignStateError("campaign_failure_not_json") from error
    now = _timestamp()
    record["status"] = "retry_wait"
    record["retry_count"] = int(record.get("retry_count", 0)) + 1
    record["last_failure"] = dict(failure)
    record["last_retry_at"] = now
    record["next_retry_at"] = next_retry_at
    record["heartbeat_at"] = now
    state["events"].append(
        {
            "event": "case_retry_scheduled",
            "phase": phase,
            "case_id": case_id,
            "retry_count": record["retry_count"],
            "at": now,
        }
    )
    write_private_json(path, state)


def complete_case(
    path: Path,
    state: dict[str, Any],
    *,
    phase: str,
    case_id: str,
    observation: Mapping[str, Any],
) -> None:
    """Durably mark one fully materialized observation as completed."""

    try:
        json.dumps(observation, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise CampaignStateError("campaign_observation_not_json") from error
    phase_cases = state.get("phases", {}).get(phase)
    if not isinstance(phase_cases, dict) or not isinstance(phase_cases.get(case_id), dict):
        raise CampaignStateError("campaign_case_not_started")
    record = phase_cases[case_id]
    if record.get("status") != "started":
        raise CampaignStateError("campaign_case_not_running")
    record["status"] = "completed"
    record["completed_at"] = _timestamp()
    record["observation"] = dict(observation)
    record.pop("next_retry_at", None)
    record.pop("heartbeat_at", None)
    state["events"].append(
        {"event": "case_completed", "phase": phase, "case_id": case_id, "at": _timestamp()}
    )
    write_private_json(path, state)


def completed_case_ids(state: Mapping[str, Any], *, phase: str) -> frozenset[str]:
    """Return only durably complete case identifiers for the named phase."""

    phase_cases = state.get("phases", {}).get(phase, {})
    if not isinstance(phase_cases, Mapping):
        raise CampaignStateError("campaign_phase_shape_invalid")
    return frozenset(
        str(case_id)
        for case_id, record in phase_cases.items()
        if isinstance(record, Mapping) and record.get("status") == "completed"
    )


def phase_progress(state: Mapping[str, Any], *, phase: str) -> dict[str, int]:
    """Summarize durable completion without treating interrupted work as done."""

    phase_cases = state.get("phases", {}).get(phase, {})
    if not isinstance(phase_cases, Mapping):
        raise CampaignStateError("campaign_phase_shape_invalid")
    completed = sum(
        1
        for record in phase_cases.values()
        if isinstance(record, Mapping) and record.get("status") == "completed"
    )
    return {"started_or_completed": len(phase_cases), "completed": completed}
