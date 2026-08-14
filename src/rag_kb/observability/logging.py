"""Content-safe structured logging with correlation and local rotation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import re
import stat
import sys
import traceback
from types import TracebackType
from typing import Final
from uuid import UUID, uuid4


LOG_SCHEMA_VERSION: Final = 1
DEFAULT_LOG_MAX_BYTES: Final = 10 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT: Final = 5
MAX_EXCEPTION_FRAMES: Final = 16
MAX_SAFE_STRING_LENGTH: Final = 512
_EVENT_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_PROCESS_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_RUNTIME_ID: Final = str(uuid4())


SAFE_FIELDS: Final = frozenset(
    {
        "assets_deleted",
        "attempt",
        "chunks_deleted",
        "client_id",
        "cleanup_completed",
        "cleanup_failed",
        "completion_tokens",
        "component",
        "database",
        "document_id",
        "document_version_id",
        "duration_ms",
        "error_code",
        "error_type",
        "failed",
        "file_cleanup_completed",
        "file_cleanup_failed",
        "file_cleanup_records_deleted",
        "indexed_document_version_id",
        "index_revision_id",
        "job_id",
        "jobs_deleted",
        "knowledge_base_id",
        "lane",
        "manifests_deleted",
        "method",
        "missing_compensated",
        "operation",
        "orphan_files_removed",
        "orphans_removed",
        "outcome",
        "path",
        "pending_activated",
        "prompt_tokens",
        "phase",
        "plans_deleted",
        "principal_id",
        "queue",
        "queue_backend",
        "reason_code",
        "requeued",
        "retryable",
        "run_id",
        "session_id",
        "status_code",
        "trace_id",
        "total_tokens",
        "retired_targets_cleaned",
        "vectors_deleted",
        "workspace_id",
    }
)
CONTEXT_FIELDS: Final = frozenset(
    {
        "attempt",
        "client_id",
        "indexed_document_version_id",
        "job_id",
        "lane",
        "principal_id",
        "run_id",
        "trace_id",
        "workspace_id",
    }
)
_LOG_CONTEXT: ContextVar[dict[str, object]] = ContextVar(
    "rag_kb_log_context",
    default={},
)


class ContentSafeJsonFormatter(logging.Formatter):
    """Serialize stable events and diagnostics without content or messages."""

    def __init__(
        self,
        *,
        process: str = "application",
        runtime_id: str = _RUNTIME_ID,
    ) -> None:
        super().__init__()
        self._process = _validate_process(process)
        self._runtime_id = runtime_id

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "safe_event", "external_log")
        payload: dict[str, object] = {
            "schema_version": LOG_SCHEMA_VERSION,
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": event,
            "process": self._process,
            "runtime_id": self._runtime_id,
            "pid": record.process,
            "source": _record_source(record),
        }
        payload.update(getattr(record, "safe_fields", {}))
        exception = getattr(record, "safe_exception", None)
        if exception is None and event == "external_log":
            exception = _external_exception(record.exc_info)
        if exception is not None:
            payload["exception"] = exception
            payload.setdefault("error_type", exception["type"])
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class _ApplicationOrWarningFilter(logging.Filter):
    """Keep application events and actionable third-party warnings/errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, "safe_event") or record.levelno >= logging.WARNING


class _PrivateRotatingFileHandler(RotatingFileHandler):
    """Rotate a process-owned JSONL file while preserving mode 0600."""

    def __init__(
        self,
        filename: Path,
        *,
        max_bytes: int,
        backup_count: int,
    ) -> None:
        _prepare_private_log_file(filename)
        super().__init__(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        os.chmod(filename, 0o600)

    def doRollover(self) -> None:  # noqa: N802 - logging handler API
        super().doRollover()
        os.chmod(self.baseFilename, 0o600)


def configure_logging(
    *,
    level: str,
    process: str = "application",
    log_directory: Path | None = None,
    max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> None:
    """Install stdout and optional private rotating JSONL handlers."""

    process = _validate_process(process)
    if max_bytes <= 0 or backup_count <= 0:
        raise ValueError("log rotation bounds must be positive")
    formatter = ContentSafeJsonFormatter(process=process)
    event_filter = _ApplicationOrWarningFilter()
    handlers: list[logging.Handler] = []

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(event_filter)
    handlers.append(stream)
    try:
        if log_directory is not None:
            directory = _validate_log_directory(log_directory)
            file_handler = _PrivateRotatingFileHandler(
                directory / f"{process}.jsonl",
                max_bytes=max_bytes,
                backup_count=backup_count,
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(event_filter)
            handlers.append(file_handler)
    except Exception:
        for handler in handlers:
            handler.close()
        raise

    root = logging.getLogger()
    previous = tuple(root.handlers)
    root.handlers[:] = handlers
    root.setLevel(level)
    logging.captureWarnings(True)
    for handler in previous:
        handler.close()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def bind_log_context(**fields: object) -> Iterator[None]:
    """Bind approved correlation fields for the current async execution context."""

    unknown = fields.keys() - CONTEXT_FIELDS
    if unknown:
        raise ValueError(f"unsupported log context fields: {sorted(unknown)}")
    normalized = _normalize_fields(fields)
    combined = {**_LOG_CONTEXT.get(), **normalized}
    token = _LOG_CONTEXT.set(combined)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: object,
) -> None:
    """Emit one event after validating its content-safe scalar metadata."""

    _validate_event(event)
    normalized = _event_fields(fields)
    logger.log(
        level,
        event,
        extra={"safe_event": event, "safe_fields": normalized},
        stacklevel=2,
    )


def log_exception(
    logger: logging.Logger,
    event: str,
    error: BaseException,
    *,
    level: int = logging.ERROR,
    **fields: object,
) -> None:
    """Emit an exception class, safe stack locations, and stable fingerprint."""

    _validate_event(event)
    normalized = _event_fields({**fields, "error_type": type(error).__name__})
    logger.log(
        level,
        event,
        extra={
            "safe_event": event,
            "safe_fields": normalized,
            "safe_exception": _safe_exception(error),
        },
        stacklevel=2,
    )


def _event_fields(fields: dict[str, object]) -> dict[str, object]:
    unknown = fields.keys() - SAFE_FIELDS
    if unknown:
        raise ValueError(f"unsafe structured log fields: {sorted(unknown)}")
    normalized = _normalize_fields(fields)
    context = _LOG_CONTEXT.get()
    for key, value in context.items():
        existing = normalized.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"structured log field conflicts with context: {key}")
        normalized.setdefault(key, value)
    return normalized


def _normalize_fields(fields: dict[str, object]) -> dict[str, object]:
    return {
        key: _normalize_scalar(key, value)
        for key, value in fields.items()
        if value is not None
    }


def _normalize_scalar(key: str, value: object) -> object:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return _normalize_scalar(key, value.value)
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"structured log field must be finite: {key}")
        return value
    if isinstance(value, str):
        if len(value) > MAX_SAFE_STRING_LENGTH:
            raise ValueError(f"structured log field is too long: {key}")
        if any(ord(character) < 32 for character in value):
            raise ValueError(f"structured log field contains control characters: {key}")
        return value
    raise TypeError(f"structured log field must be a safe scalar: {key}")


def _safe_exception(error: BaseException) -> dict[str, object]:
    frames = _safe_frames(error.__traceback__)
    chain = _exception_chain(error)
    identity = json.dumps(
        {
            "chain": chain,
            "frames": frames,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "type": type(error).__name__,
        "chain": chain,
        "frames": frames,
        "fingerprint": hashlib.sha256(identity).hexdigest()[:20],
    }


def _external_exception(exc_info: object) -> dict[str, object] | None:
    error: BaseException | None = None
    if isinstance(exc_info, BaseException):
        error = exc_info
    elif isinstance(exc_info, tuple) and len(exc_info) > 1:
        candidate = exc_info[1]
        if isinstance(candidate, BaseException):
            error = candidate
    return _safe_exception(error) if error is not None else None


def _exception_chain(error: BaseException) -> list[str]:
    result: list[str] = []
    observed: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in observed and len(result) < 4:
        observed.add(id(current))
        result.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    return result


def _safe_frames(trace: TracebackType | None) -> list[dict[str, object]]:
    if trace is None:
        return []
    extracted = traceback.extract_tb(trace)
    return [
        {
            "module": _safe_module_path(frame.filename),
            "function": frame.name[:128],
            "line": frame.lineno,
        }
        for frame in extracted[-MAX_EXCEPTION_FRAMES:]
    ]


def _safe_module_path(filename: str) -> str:
    path = Path(filename)
    parts = path.parts
    for anchor in ("apps", "src", "tools", "tests"):
        if anchor in parts:
            position = parts.index(anchor)
            return "/".join(parts[position:])[-256:]
    return path.name[-256:]


def _record_source(record: logging.LogRecord) -> dict[str, object]:
    return {
        "module": _safe_module_path(record.pathname),
        "function": (record.funcName or "<unknown>")[:128],
        "line": record.lineno,
    }


def _validate_event(event: str) -> None:
    if not _EVENT_PATTERN.fullmatch(event):
        raise ValueError("structured log event name is invalid")


def _validate_process(process: str) -> str:
    if not _PROCESS_PATTERN.fullmatch(process):
        raise ValueError("logging process name is invalid")
    return process


def _validate_log_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("log_directory must be absolute")
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError("log_directory must be a real directory")
    if not os.access(path, os.W_OK):
        raise PermissionError("log_directory is not writable")
    return path


def _prepare_private_log_file(path: Path) -> None:
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("log target must be a regular file")
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)
