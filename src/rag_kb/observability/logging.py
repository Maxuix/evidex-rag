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
        "component",
        "database",
        "duration_ms",
        "method",
        "path",
        "principal_id",
        "process",
        "queue",
        "queue_backend",
        "reason_code",
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
