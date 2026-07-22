"""Read-only startup checks for configured local runtime resources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import which

from rag_kb.config.settings import Settings


class StartupConfigurationError(RuntimeError):
    """The configured process cannot safely start in the current environment."""


@dataclass(frozen=True)
class StartupValidation:
    profile: str
    bind_host: str
    storage_root: Path
    storage_device: int
    configured_pool_capacity: int
    application_connection_budget: int


def validate_startup_environment(settings: Settings) -> StartupValidation:
    """Validate shared local storage without provisioning or repairing it."""

    assert settings.file_store.asset_staging_path is not None
    assert settings.file_store.asset_final_path is not None
    assert settings.file_store.parser_temp_path is not None
    configured_paths = (
        settings.file_store.root_path,
        settings.file_store.staging_path,
        settings.file_store.final_path,
    )
    if settings.model_provider.multimodal_embedding is not None:
        missing_tools = tuple(
            executable for executable in ("pdftoppm", "tesseract") if which(executable) is None
        )
        if missing_tools:
            raise StartupConfigurationError(
                "multimodal local parsing requires executables: "
                + ", ".join(missing_tools)
            )
        configured_paths += (
            settings.file_store.asset_staging_path,
            settings.file_store.asset_final_path,
            settings.file_store.parser_temp_path,
        )
    resolved_paths: list[Path] = []
    device_ids: set[int] = set()

    for path in configured_paths:
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise StartupConfigurationError(
                f"configured file-store path does not exist: {path}"
            ) from error
        if not resolved.is_dir():
            raise StartupConfigurationError(
                f"configured file-store path is not a directory: {path}"
            )
        resolved_paths.append(resolved)
        device_ids.add(resolved.stat().st_dev)

    resolved_root, resolved_staging, resolved_final, *derived_paths = resolved_paths
    if len({resolved_staging, resolved_final, *derived_paths}) != len(resolved_paths) - 1:
        raise StartupConfigurationError(
            "resolved staging and final file-store paths must be different"
        )
    if not resolved_staging.is_relative_to(
        resolved_root
    ) or not all(
        path.is_relative_to(resolved_root)
        for path in (resolved_final, *derived_paths)
    ):
        raise StartupConfigurationError(
            "resolved staging and final paths must remain beneath root_path"
        )
    if len(device_ids) != 1:
        raise StartupConfigurationError(
            "root, staging, and final file-store paths must share one filesystem"
        )

    return StartupValidation(
        profile=settings.app.deployment_profile.value,
        bind_host=str(settings.app.bind_host),
        storage_root=resolved_root,
        storage_device=device_ids.pop(),
        configured_pool_capacity=settings.database.configured_pool_capacity,
        application_connection_budget=(
            settings.database.application_connection_budget
        ),
    )
