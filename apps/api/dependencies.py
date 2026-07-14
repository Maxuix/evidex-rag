"""API composition root for process-wide foundation dependencies."""

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
class ApiDependencies:
    """Dependencies currently safe to construct before API contracts exist."""

    settings: Settings
    startup: StartupValidation


def build_api_dependencies(
    settings: Settings | None = None,
    *,
    env_file: str | Path | None = ".env",
) -> ApiDependencies:
    """Load configuration explicitly and fail before constructing an API app."""

    resolved_settings = settings or load_settings(env_file=env_file)
    startup = validate_startup_environment(resolved_settings)
    return ApiDependencies(settings=resolved_settings, startup=startup)
