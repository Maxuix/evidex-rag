#!/usr/bin/env python3
"""Preview or apply the one-time merge from legacy local env files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from urllib.parse import quote

from pydantic import ValidationError

from rag_kb.config import load_settings
from tools.local_runtime import (
    CANONICAL_COMPOSE_PROJECT,
    DEFAULT_PORTS,
    LOCAL_MANIFEST_NAME,
    LocalRuntimeError,
    _parse_manifest,
    resolve_local_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_APP_ENV_NAME = ".env"
TEMPLATE_NAME = ".env.example"
BACKUP_NAME = ".env.local.before-single-manifest-20260820"
COMPOSE_SECRET_KEYS = (
    "POSTGRES_ADMIN_PASSWORD",
    "RAG_KB_MIGRATION_PASSWORD",
    "RAG_KB_RUNTIME_PASSWORD",
)
IDENTITY_KEYS = (
    "COMPOSE_PROJECT_NAME",
    *DEFAULT_PORTS,
    *COMPOSE_SECRET_KEYS,
)


@dataclass(frozen=True, slots=True)
class ManifestMigration:
    root: Path
    manifest: Path
    legacy_app_env: Path
    backup: Path
    before_sha256: str
    state_key_count: int
    app_key_count: int
    merged: dict[str, str]

    def safe_summary(self, *, applied: bool) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "applied" if applied else "ready",
            "applied": applied,
            "state_key_count": self.state_key_count,
            "legacy_app_key_count": self.app_key_count,
            "result_key_count": len(self.merged),
            "removed_state_indirection": True,
            "dsn_credentials_aligned": True,
            "backup_created": applied,
        }


def build_migration(
    root: Path = PROJECT_ROOT,
    *,
    canonical_checkout: Path | None = None,
) -> ManifestMigration:
    root = root.resolve()
    runtime = resolve_local_runtime(
        checkout=root,
        canonical_checkout=canonical_checkout,
        env_file=root / LOCAL_MANIFEST_NAME,
        require_manifest=True,
    )
    if not runtime.is_primary_checkout or not runtime.is_canonical_manifest:
        raise LocalRuntimeError("manifest migration requires the primary checkout")
    if runtime.manifest_mode != 0o600:
        raise LocalRuntimeError("local manifest must be owner-only")

    legacy_app_env = root / LEGACY_APP_ENV_NAME
    template = root / TEMPLATE_NAME
    _require_private_regular_file(legacy_app_env)
    _require_regular_file(template)

    state_values = _parse_manifest(runtime.manifest)
    app_values = _parse_manifest(legacy_app_env)
    template_values = _parse_manifest(template)
    if any(not key.startswith("RAG_KB__") for key in app_values):
        raise LocalRuntimeError("legacy app env contains non-application keys")
    allowed_state_keys = {"RAG_KB_ENV_FILE", *IDENTITY_KEYS}
    if set(state_values) - allowed_state_keys:
        raise LocalRuntimeError("legacy state env contains unsupported keys")

    merged = dict(template_values)
    merged.update(app_values)
    for key in IDENTITY_KEYS:
        if key in state_values:
            merged[key] = state_values[key]
    merged["COMPOSE_PROJECT_NAME"] = CANONICAL_COMPOSE_PROJECT
    for name, default in DEFAULT_PORTS.items():
        merged.setdefault(name, str(default))
    for key in COMPOSE_SECRET_KEYS:
        value = merged.get(key)
        if not value or value == "replace-locally":
            raise LocalRuntimeError("database startup credentials are incomplete")

    merged.pop("RAG_KB_ENV_FILE", None)
    merged["RAG_KB__DATABASE__RUNTIME_DSN"] = _database_dsn(
        "rag_kb_runtime",
        merged["RAG_KB_RUNTIME_PASSWORD"],
    )
    merged["RAG_KB__DATABASE__MIGRATION_DSN"] = _database_dsn(
        "rag_kb_migration",
        merged["RAG_KB_MIGRATION_PASSWORD"],
    )
    _validate_candidate(root, merged)

    return ManifestMigration(
        root=root,
        manifest=runtime.manifest,
        legacy_app_env=legacy_app_env,
        backup=root / BACKUP_NAME,
        before_sha256=hashlib.sha256(runtime.manifest.read_bytes()).hexdigest(),
        state_key_count=len(state_values),
        app_key_count=len(app_values),
        merged=merged,
    )


def apply_migration(migration: ManifestMigration) -> None:
    if migration.backup.exists():
        raise FileExistsError("local manifest migration backup already exists")
    current_hash = hashlib.sha256(migration.manifest.read_bytes()).hexdigest()
    if current_hash != migration.before_sha256:
        raise RuntimeError("local manifest changed after preview")

    backup_descriptor = os.open(
        migration.backup,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(backup_descriptor, "wb") as stream:
            stream.write(migration.manifest.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        migration.backup.unlink(missing_ok=True)
        raise

    temporary = migration.manifest.with_name(
        f"{LOCAL_MANIFEST_NAME}.tmp.{os.getpid()}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_render_manifest(migration.merged))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, migration.manifest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    migration.manifest.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="create a private backup and atomically replace .env.local",
    )
    arguments = parser.parse_args()
    try:
        migration = build_migration()
        if arguments.apply:
            apply_migration(migration)
        summary = migration.safe_summary(applied=arguments.apply)
    except (LocalRuntimeError, OSError, RuntimeError, ValidationError):
        summary = {
            "schema_version": 1,
            "status": "blocked",
            "applied": False,
            "reason": "manifest_migration_precondition_failed",
        }
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    return 0 if summary["status"] in {"ready", "applied"} else 1


def _database_dsn(username: str, password: str) -> str:
    encoded = quote(password, safe="")
    return f"postgresql+asyncpg://{username}:{encoded}@postgres:5432/rag_kb"


def _validate_candidate(root: Path, values: dict[str, str]) -> None:
    temporary = root / f".env.local.validate.{os.getpid()}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_render_manifest(values))
        load_settings(env_file=temporary)
    finally:
        temporary.unlink(missing_ok=True)


def _render_manifest(values: dict[str, str]) -> str:
    identity = [key for key in IDENTITY_KEYS if key in values]
    remaining = sorted(set(values) - set(identity))
    return "".join(f"{key}={values[key]}\n" for key in (*identity, *remaining))


def _require_private_regular_file(path: Path) -> None:
    _require_regular_file(path)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise LocalRuntimeError("legacy app env must be owner-only")


def _require_regular_file(path: Path) -> None:
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise LocalRuntimeError("manifest input is not a regular file")


if __name__ == "__main__":
    raise SystemExit(main())
