#!/usr/bin/env python3
"""Clearly destructive, development-only reset of local Compose data."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rag_kb.config import DeploymentProfile, load_settings


CONFIRMATION = "DESTROY_RAG_KB_LOCAL_DATA"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def reset_command(
    *,
    env_file: str,
    project_name: str | None,
) -> list[str]:
    command = ["docker", "compose", "--env-file", env_file, "--profile", "tools"]
    if project_name is not None:
        command.extend(("--project-name", project_name))
    command.extend(("down", "--volumes", "--remove-orphans"))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DESTROYS the local RAG KB database and source-file volumes."
    )
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--project-name")
    arguments = parser.parse_args()
    if arguments.confirm != CONFIRMATION:
        parser.error(f"--confirm must equal {CONFIRMATION}")
    settings = load_settings(env_file=arguments.env_file)
    if settings.app.deployment_profile is not DeploymentProfile.DEVELOPMENT:
        parser.error("local reset is restricted to the development profile")
    subprocess.run(
        reset_command(
            env_file=arguments.env_file,
            project_name=arguments.project_name,
        ),
        cwd=PROJECT_ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
