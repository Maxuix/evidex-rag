"""Content-safe JSON logging with an explicit metadata allowlist."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Final
from uuid import UUID


SAFE_FIELDS: Final = frozenset(
    {
        "client_id",
        "cleanup_completed",
        "cleanup_failed",
        "component",
        "database",
        "duration_ms",
        "error_type",
        "attempt",
        "failed",
        "indexed_document_version_id",
        "job_id",
        "lane",
        "method",
        "missing_compensated",
        "orphans_removed",
        "path",
        "pending_activated",
        "principal_id",
        "process",
        "queue",
        "queue_backend",
        "reason_code",
        "requeued",
        "status_code",
        "trace_id",
        "workspace_id",
    }
)


class ContentSafeJsonFormatter(logging.Formatter):
    """Serialize only stable event names and explicitly permitted metadata."""

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "safe_event", "external_log")
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": event,
        }
        payload.update(getattr(record, "safe_fields", {}))
        if event == "external_log":
            error_type = _external_error_type(record.exc_info)
            if error_type is not None:
                payload["error_type"] = error_type
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def configure_logging(*, level: str) -> None:
    """Install one process-wide content-safe structured handler."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ContentSafeJsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    logging.captureWarnings(True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _external_error_type(exc_info: object) -> str | None:
    if isinstance(exc_info, BaseException):
        return type(exc_info).__name__
    if not isinstance(exc_info, tuple) or not exc_info:
        return None
    exception_type = exc_info[0]
    if isinstance(exception_type, type) and issubclass(
        exception_type,
        BaseException,
    ):
        return exception_type.__name__
    if len(exc_info) > 1 and isinstance(exc_info[1], BaseException):
        return type(exc_info[1]).__name__
    return None


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: object,
) -> None:
    """Emit one event after rejecting non-allowlisted metadata keys."""

    unknown = fields.keys() - SAFE_FIELDS
    if unknown:
        raise ValueError(f"unsafe structured log fields: {sorted(unknown)}")
    normalized = {
        key: str(value) if isinstance(value, UUID) else value
        for key, value in fields.items()
        if value is not None
    }
    logger.log(
        level,
        event,
        extra={"safe_event": event, "safe_fields": normalized},
    )
