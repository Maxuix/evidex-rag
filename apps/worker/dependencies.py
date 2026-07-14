"""Worker composition root for process-wide foundation dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_kb.config import (
    Settings,
    StartupValidation,
    load_settings,
    validate_startup_environment,
)


@dataclass(frozen=True)
class WorkerDependencies:
    """Dependencies currently safe to construct before task contracts exist."""

    settings: Settings
    startup: StartupValidation


def build_worker_dependencies(
    settings: Settings | None = None,
    *,
    env_file: str | Path | None = ".env",
) -> WorkerDependencies:
    """Load configuration explicitly and fail before starting task polling."""

    resolved_settings = settings or load_settings(env_file=env_file)
    startup = validate_startup_environment(resolved_settings)
    return WorkerDependencies(settings=resolved_settings, startup=startup)
