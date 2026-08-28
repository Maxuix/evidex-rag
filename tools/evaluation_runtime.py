"""Validate the owner-only host runtime used by evaluation tools.

Evaluation tools may read a user-provided, already-running host-Python test
runtime. This module deliberately owns no processes and has no Docker
lifecycle commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import stat
import subprocess
from typing import Mapping
from urllib.parse import urlparse
from uuid import UUID

from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY,
)
from rag_kb.graph.schema_profiles import get_graph_schema_registry
from tools.local_runtime import LocalRuntimeError, discover_canonical_checkout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST_TEST_PROJECT = "python-host-test"
RUNTIME_SCHEMA_VERSION = 1
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / ".runtime/evaluations/graph-schema-profiles-host"
DEFAULT_RUNTIME_MANIFEST = DEFAULT_RUNTIME_ROOT / "runtime.json"
_RUNTIME_PORTS = frozenset({"api", "frontend", "postgres", "falkordb"})
_CANONICAL_PORTS = frozenset({8000, 3000, 5432, 6379})
_OWNER_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class EvaluationRuntimeError(ValueError):
    """The evaluator runtime cannot be proven isolated."""


def canonical_evaluation_runtime_manifest() -> Path:
    """Return the primary checkout's host-Python evaluator manifest path."""
    try:
        checkout = discover_canonical_checkout(PROJECT_ROOT)
    except (LocalRuntimeError, OSError, subprocess.SubprocessError):
        return DEFAULT_RUNTIME_MANIFEST
    return checkout / ".runtime/evaluations/graph-schema-profiles-host/runtime.json"


@dataclass(frozen=True, slots=True)
class AdaptiveGraphIdentity:
    workspace_id: UUID
    knowledge_base_id: UUID
    index_revision_id: UUID
    graph_build_id: UUID
    answer_profile_revision_id: UUID
    judge_profile_revision_id: UUID
    schema_profile_key: str = SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY
    schema_profile_digest: str = SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST
    extractor_version: str = GRAPH_EXTRACTOR_VERSION


@dataclass(frozen=True, slots=True)
class EvaluationRuntime:
    manifest: Path
    runtime_root: Path
    env_file: Path
    compose_env_file: Path
    owner: str
    build_revision: str
    api_base_url: str
    ports: Mapping[str, int]
    adaptive_graph: AdaptiveGraphIdentity | None
    compose_project: str = HOST_TEST_PROJECT


def load_evaluation_runtime(
    path: Path = DEFAULT_RUNTIME_MANIFEST,
    *,
    require_adaptive_graph: bool = False,
    allow_canonical_checkout: bool = False,
) -> EvaluationRuntime:
    """Load a canonical, owner-only runtime without changing external state."""
    runtime_directory = path.absolute().parent
    allowed_directory = DEFAULT_RUNTIME_ROOT.absolute()
    if allow_canonical_checkout:
        allowed_directory = canonical_evaluation_runtime_manifest().absolute().parent
    if runtime_directory != allowed_directory:
        raise EvaluationRuntimeError("evaluation runtime directory is not canonical")
    try:
        runtime_status = runtime_directory.lstat()
    except FileNotFoundError as error:
        raise EvaluationRuntimeError("evaluation runtime directory is missing") from error
    if stat.S_ISLNK(runtime_status.st_mode) or not stat.S_ISDIR(runtime_status.st_mode):
        raise EvaluationRuntimeError("evaluation runtime directory is invalid")
    if stat.S_IMODE(runtime_status.st_mode) != 0o700:
        raise EvaluationRuntimeError("evaluation runtime directory must be owner-only")
    runtime_root = runtime_directory.resolve()
    manifest = _private_regular_file(path.absolute(), reason="evaluation runtime manifest")
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationRuntimeError("evaluation runtime manifest is invalid") from error
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "compose_project",
        "owner",
        "build_revision",
        "env_file",
        "compose_env_file",
        "api_base_url",
        "ports",
        "adaptive_graph",
    }:
        raise EvaluationRuntimeError("evaluation runtime schema is invalid")
    if value["schema_version"] != RUNTIME_SCHEMA_VERSION:
        raise EvaluationRuntimeError("evaluation runtime schema version is unsupported")
    if value["compose_project"] != HOST_TEST_PROJECT:
        raise EvaluationRuntimeError("evaluation runtime project is invalid")
    owner = value["owner"]
    if not isinstance(owner, str) or _OWNER_PATTERN.fullmatch(owner) is None:
        raise EvaluationRuntimeError("evaluation owner is invalid")
    build_revision = value["build_revision"]
    if not isinstance(build_revision, str) or _REVISION_PATTERN.fullmatch(build_revision) is None:
        raise EvaluationRuntimeError("evaluation build revision is invalid")
    env_file = _runtime_private_file(runtime_root, value["env_file"], reason="evaluation runtime env")
    compose_env_file = _runtime_private_file(
        runtime_root,
        value["compose_env_file"],
        reason="evaluation runtime Compose env",
    )
    ports = _validate_ports(value["ports"])
    api_base_url = _validated_api_base(value["api_base_url"], ports["api"])
    adaptive_graph = _adaptive_graph_identity(value["adaptive_graph"])
    if require_adaptive_graph and adaptive_graph is None:
        raise EvaluationRuntimeError("adaptive Graph identity is unavailable")
    return EvaluationRuntime(
        manifest=manifest,
        runtime_root=runtime_root,
        env_file=env_file,
        compose_env_file=compose_env_file,
        owner=owner,
        build_revision=build_revision,
        api_base_url=api_base_url,
        ports=ports,
        adaptive_graph=adaptive_graph,
    )


def _private_regular_file(path: Path, *, reason: str) -> Path:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise EvaluationRuntimeError(f"{reason} is missing") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise EvaluationRuntimeError(f"{reason} is not a regular file")
    if stat.S_IMODE(status.st_mode) != 0o600:
        raise EvaluationRuntimeError(f"{reason} must be owner-only")
    return path.resolve()


def _runtime_private_file(runtime_root: Path, name: object, *, reason: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise EvaluationRuntimeError("evaluation runtime file identity is invalid")
    return _private_regular_file(runtime_root / name, reason=reason)


def _validate_ports(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != _RUNTIME_PORTS:
        raise EvaluationRuntimeError("evaluation ports are invalid")
    ports: dict[str, int] = {}
    for name, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int) or not 1024 <= item <= 65535:
            raise EvaluationRuntimeError("evaluation port is invalid")
        ports[name] = item
    if len(set(ports.values())) != len(ports) or set(ports.values()) & _CANONICAL_PORTS:
        raise EvaluationRuntimeError("evaluation ports are not isolated")
    return ports


def _validated_api_base(value: object, api_port: int) -> str:
    if not isinstance(value, str):
        raise EvaluationRuntimeError("evaluation API identity is invalid")
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != api_port
        or parsed.path.rstrip("/") != "/api/v1"
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise EvaluationRuntimeError("evaluation API is not isolated")
    return f"http://127.0.0.1:{api_port}/api/v1"


def _adaptive_graph_identity(value: object) -> AdaptiveGraphIdentity | None:
    if value is None:
        return None
    fields = {
        "knowledge_base_id",
        "workspace_id",
        "index_revision_id",
        "graph_build_id",
        "answer_profile_revision_id",
        "judge_profile_revision_id",
        "schema_profile_key",
        "schema_profile_digest",
        "extractor_version",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise EvaluationRuntimeError("adaptive Graph identity is invalid")
    try:
        identifiers = {name: UUID(str(value[name])) for name in fields if name.endswith("_id")}
    except (TypeError, ValueError) as error:
        raise EvaluationRuntimeError("adaptive Graph identity is invalid") from error
    profile_key = value["schema_profile_key"]
    profile_digest = value["schema_profile_digest"]
    extractor_version = value["extractor_version"]
    if not all(isinstance(item, str) for item in (profile_key, profile_digest, extractor_version)):
        raise EvaluationRuntimeError("adaptive Graph identity is invalid")
    try:
        get_graph_schema_registry().resolve(
            profile_key,
            digest=profile_digest,
            extractor_version=extractor_version,
        )
    except ValueError as error:
        raise EvaluationRuntimeError("adaptive Graph identity is invalid") from error
    return AdaptiveGraphIdentity(
        **identifiers,
        schema_profile_key=profile_key,
        schema_profile_digest=profile_digest,
        extractor_version=extractor_version,
    )
