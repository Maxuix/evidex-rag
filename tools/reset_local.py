#!/usr/bin/env python3
"""Precisely reset development business volumes for one Compose project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

from rag_kb.config import DeploymentProfile, load_settings
from tools.local_runtime import (
    CANONICAL_COMPOSE_PROJECT,
    LocalRuntimeError,
    resolve_local_runtime,
)


CONFIRMATION = "DESTROY_RAG_KB_LOCAL_DATA"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUSINESS_VOLUME_KEYS = ("postgres-data", "source-data")
MODEL_CACHE_VOLUME_KEY = "inference-model-cache"


@dataclass(frozen=True, slots=True)
class VolumeTargets:
    project_name: str
    remove: tuple[str, ...]
    preserve: tuple[str, ...]


def compose_down_command(
    *,
    env_file: str,
    project_name: str,
) -> list[str]:
    return [
        "docker",
        "compose",
        "--env-file",
        env_file,
        "--profile",
        "tools",
        "--project-name",
        project_name,
        "down",
        "--remove-orphans",
    ]


def volume_remove_command(volume_names: tuple[str, ...]) -> list[str]:
    if not volume_names:
        raise ValueError("at least one resolved volume is required")
    return ["docker", "volume", "rm", *volume_names]


def resolve_volume_targets(project_name: str) -> VolumeTargets:
    if not project_name.strip():
        raise ValueError("Compose project name must not be empty")
    resolved = {
        key: _resolve_project_volume(project_name, key)
        for key in (*BUSINESS_VOLUME_KEYS, MODEL_CACHE_VOLUME_KEY)
    }
    return VolumeTargets(
        project_name=project_name,
        remove=tuple(
            resolved[key] for key in BUSINESS_VOLUME_KEYS if resolved[key] is not None
        ),
        preserve=(
            (resolved[MODEL_CACHE_VOLUME_KEY],)
            if resolved[MODEL_CACHE_VOLUME_KEY] is not None
            else ()
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "DESTROYS one development RAG KB project's PostgreSQL and source "
            "volumes while preserving its inference model cache."
        )
    )
    parser.add_argument("--confirm", required=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="single local runtime manifest (defaults to canonical .env.local)",
    )
    parser.add_argument(
        "--project-name",
        default=CANONICAL_COMPOSE_PROJECT,
        help="exact Compose project target (defaults to canonical rag)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="resolve and print exact volume targets without changing anything",
    )
    arguments = parser.parse_args()
    if arguments.confirm != CONFIRMATION:
        parser.error(f"--confirm must equal {CONFIRMATION}")
    try:
        runtime = resolve_local_runtime(
            env_file=arguments.env_file,
            require_manifest=True,
        )
    except (LocalRuntimeError, OSError, subprocess.SubprocessError):
        parser.error("local runtime manifest is invalid")
    if not runtime.is_primary_checkout:
        parser.error("local reset must run from the primary checkout")
    if not runtime.is_canonical_manifest:
        parser.error("local reset requires the canonical .env.local manifest")
    if arguments.project_name != runtime.compose_project:
        parser.error("--project-name does not match the canonical local runtime")
    settings = load_settings(env_file=runtime.manifest)
    if settings.app.deployment_profile is not DeploymentProfile.DEVELOPMENT:
        parser.error("local reset is restricted to the development profile")

    targets = resolve_volume_targets(arguments.project_name)
    print(
        json.dumps(
            {
                "project_name": targets.project_name,
                "remove": list(targets.remove),
                "preserve": list(targets.preserve),
                "inspect_only": arguments.inspect_only,
            },
            separators=(",", ":"),
        )
    )
    if arguments.inspect_only:
        return 0

    subprocess.run(
        compose_down_command(
            env_file=str(runtime.manifest),
            project_name=arguments.project_name,
        ),
        cwd=PROJECT_ROOT,
        check=True,
    )
    if targets.remove:
        subprocess.run(
            volume_remove_command(targets.remove),
            cwd=PROJECT_ROOT,
            check=True,
        )
    observed = resolve_volume_targets(arguments.project_name)
    if observed.remove:
        raise RuntimeError("one or more resolved business volumes still exist")
    if observed.preserve != targets.preserve:
        raise RuntimeError("the inference model cache preservation check failed")
    print(
        json.dumps(
            {
                "project_name": targets.project_name,
                "deleted": list(targets.remove),
                "preserved": list(targets.preserve),
            },
            separators=(",", ":"),
        )
    )
    return 0


def _resolve_project_volume(project_name: str, volume_key: str) -> str | None:
    result = subprocess.run(
        [
            "docker",
            "volume",
            "ls",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
            "--filter",
            f"label=com.docker.compose.volume={volume_key}",
            "--format",
            "{{.Name}}",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    names = tuple(
        line.strip() for line in result.stdout.splitlines() if line.strip()
    )
    if len(names) > 1:
        raise RuntimeError(
            f"multiple Compose volumes resolved for project={project_name!r}, "
            f"volume={volume_key!r}"
        )
    if not names:
        return None
    name = names[0]
    inspected = subprocess.run(
        [
            "docker",
            "volume",
            "inspect",
            "--format",
            (
                '{{index .Labels "com.docker.compose.project"}}'
                '\t{{index .Labels "com.docker.compose.volume"}}'
            ),
            name,
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if inspected.stdout.strip() != f"{project_name}\t{volume_key}":
        raise RuntimeError("resolved volume labels do not match the requested target")
    return name


if __name__ == "__main__":
    raise SystemExit(main())
