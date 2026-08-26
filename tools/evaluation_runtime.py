#!/usr/bin/env python3
"""Create and validate the one isolated local evaluator runtime."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from uuid import UUID, uuid4

from tools.local_runtime import (
    LocalRuntimeError,
    _parse_manifest,
    discover_canonical_checkout,
)
from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY,
)
from rag_kb.graph.schema_profiles import get_graph_schema_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# `rag-eval` is retained below only for legacy lifecycle compatibility.  The
# current evaluator identity is a host-Python runtime and must never create a
# second Compose project.
EVALUATION_PROJECT = "rag-eval"
HOST_TEST_PROJECT = "python-host-test"
ACCEPTED_RUNTIME_PROJECTS = frozenset({EVALUATION_PROJECT, HOST_TEST_PROJECT})
EVALUATION_OWNER_LABEL = "rag-kb.evaluation-owner"
RUNTIME_SCHEMA_VERSION = 1
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / ".runtime/evaluations/graph-schema-profiles-host"
DEFAULT_RUNTIME_MANIFEST = DEFAULT_RUNTIME_ROOT / "runtime.json"
DEFAULT_RUNTIME_ENV = DEFAULT_RUNTIME_ROOT / "runtime.env"
DEFAULT_COMPOSE_ENV = DEFAULT_RUNTIME_ROOT / "compose.env"
DEFAULT_SEED_BACKUP = PROJECT_ROOT / ".runtime/backups/local-workflow-20260820"
CANONICAL_MANIFEST = PROJECT_ROOT / ".env.local"
COMPOSE_FILES = (PROJECT_ROOT / "compose.yaml", PROJECT_ROOT / "compose.eval.yaml")
CREATE_CONFIRMATION = "CREATE_ISOLATED_RAG_EVAL"
DESTROY_CONFIRMATION = "DESTROY_ISOLATED_RAG_EVAL"
EVALUATION_PASSWORD = "isolated-evaluation-only"
LOCAL_IMAGE_REUSE = (
    (
        "rag-kb-app:local",
        "rag-kb-app:eval",
        (
            "Dockerfile",
            "requirements.lock",
            "alembic.ini",
            "apps",
            "config/docling-artifacts-v1.json",
            "config/local-reranker-artifacts-v1.json",
            "src",
            "tools/prepare_docling_artifacts.py",
            "tools/prepare_local_reranker_artifacts.py",
        ),
    ),
    (
        "rag-kb-user-frontend:local",
        "rag-kb-user-frontend:eval",
        ("apps/web-chat",),
    ),
)
EVALUATION_PORTS = {
    "api": 28000,
    "frontend": 23000,
    "postgres": 25432,
    "falkordb": 26379,
}
CANONICAL_PORTS = frozenset({8000, 3000, 5432, 6379})
REQUIRED_SEED_FILES = (
    "p6-postgres.dump",
    "p6-source-data.tar.gz",
    "p6-model-secrets.tar.gz",
    "p6-falkordb.rdb",
    "adaptive-graph-route-v2.tar.gz",
    "SHA256SUMS",
)
FROZEN_ADAPTIVE_RESULT = (
    "adaptive-graph-route-v2/stage-a-20260820/answers.json"
)
MAX_FROZEN_ADAPTIVE_RESULT_BYTES = 1024 * 1024
EVALUATION_RUNTIME_GRANTS = (
    "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
    "TO rag_kb_runtime; "
    "REVOKE INSERT, UPDATE, DELETE ON TABLE alembic_version FROM rag_kb_runtime; "
    "ALTER DEFAULT PRIVILEGES FOR ROLE rag_kb_migration IN SCHEMA public "
    "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag_kb_runtime; "
    "REVOKE UPDATE ON TABLE index_chunk_plan FROM rag_kb_runtime; "
    "REVOKE ALL ON FUNCTION enforce_provisioned_kb_active_revision() FROM PUBLIC; "
    "REVOKE ALL ON FUNCTION enforce_document_version_source_immutability() FROM PUBLIC; "
    "REVOKE ALL ON FUNCTION enforce_source_change_immutability() FROM PUBLIC"
)
EVALUATION_RUNTIME_GRANTS_CHECK = (
    "WITH runtime_tables AS ("
    "SELECT table_schema, table_name FROM information_schema.tables "
    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    ") SELECT COALESCE(bool_and("
    "has_table_privilege('rag_kb_runtime', format('%I.%I', table_schema, table_name), 'SELECT') "
    "AND (table_name = 'alembic_version' OR "
    "has_table_privilege('rag_kb_runtime', format('%I.%I', table_schema, table_name), 'INSERT')) "
    "AND (table_name IN ('alembic_version', 'index_chunk_plan') OR "
    "has_table_privilege('rag_kb_runtime', format('%I.%I', table_schema, table_name), 'UPDATE')) "
    "AND (table_name = 'alembic_version' OR "
    "has_table_privilege('rag_kb_runtime', format('%I.%I', table_schema, table_name), 'DELETE'))"
    "), false) FROM runtime_tables"
)
EXPECTED_SERVICES = frozenset(
    {"postgres", "falkordb", "storage-init", "migrate", "maintenance", "api", "worker", "frontend"}
)
READY_SERVICES = frozenset(
    {"postgres", "falkordb", "storage-init", "api", "worker", "frontend"}
)
EXPECTED_VOLUMES = frozenset(
    {"postgres-data", "source-data", "inference-model-cache", "model-secrets", "falkordb-data"}
)
EXPECTED_NETWORKS = frozenset({"default"})
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
class FrozenAdaptiveGraphIdentity:
    knowledge_base_id: UUID
    index_revision_id: UUID
    graph_build_id: UUID
    answer_profile_revision_id: UUID
    judge_profile_revision_id: UUID


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


def load_evaluation_runtime(
    path: Path = DEFAULT_RUNTIME_MANIFEST,
    *,
    require_adaptive_graph: bool = False,
    allow_canonical_checkout: bool = False,
) -> EvaluationRuntime:
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
    if value["compose_project"] not in ACCEPTED_RUNTIME_PROJECTS:
        raise EvaluationRuntimeError("evaluation runtime project is invalid")
    owner = value["owner"]
    if not isinstance(owner, str) or _OWNER_PATTERN.fullmatch(owner) is None:
        raise EvaluationRuntimeError("evaluation owner is invalid")
    build_revision = value["build_revision"]
    if not isinstance(build_revision, str) or _REVISION_PATTERN.fullmatch(build_revision) is None:
        raise EvaluationRuntimeError("evaluation build revision is invalid")
    env_name = value["env_file"]
    if not isinstance(env_name, str) or Path(env_name).name != env_name:
        raise EvaluationRuntimeError("evaluation env identity is invalid")
    env_file = _private_regular_file(
        runtime_root / env_name,
        reason="evaluation runtime env",
    )
    compose_env_name = value["compose_env_file"]
    if not isinstance(compose_env_name, str) or Path(compose_env_name).name != compose_env_name:
        raise EvaluationRuntimeError("evaluation Compose env identity is invalid")
    compose_env_file = _private_regular_file(
        runtime_root / compose_env_name,
        reason="evaluation Compose env",
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


def compose_command(runtime: EvaluationRuntime, *arguments: str) -> list[str]:
    command = ["docker", "compose", "--env-file", str(runtime.compose_env_file)]
    for compose_file in COMPOSE_FILES:
        command.extend(("--file", str(compose_file)))
    command.extend(("--project-name", EVALUATION_PROJECT, *arguments))
    return command


def compose_environment(runtime: EvaluationRuntime) -> dict[str, str]:
    return {
        **os.environ,
        "RAG_KB_EVAL_ENV_FILE": str(runtime.compose_env_file),
        "RAG_KB_EVAL_OWNER": runtime.owner,
        "RAG_KB_EVAL_RUNTIME_ROOT": str(runtime.runtime_root),
        "RAG_KB_BUILD_REVISION": runtime.build_revision,
    }


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


def _validate_ports(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(EVALUATION_PORTS):
        raise EvaluationRuntimeError("evaluation ports are invalid")
    ports: dict[str, int] = {}
    for name, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int) or not 1024 <= item <= 65535:
            raise EvaluationRuntimeError("evaluation port is invalid")
        ports[name] = item
    if len(set(ports.values())) != len(ports) or set(ports.values()) & CANONICAL_PORTS:
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
        identifiers = {
            name: UUID(str(value[name]))
            for name in fields
            if name.endswith("_id")
        }
    except (TypeError, ValueError) as error:
        raise EvaluationRuntimeError("adaptive Graph identity is invalid") from error
    profile_key = value["schema_profile_key"]
    profile_digest = value["schema_profile_digest"]
    extractor_version = value["extractor_version"]
    if (
        not isinstance(profile_key, str)
        or not isinstance(profile_digest, str)
        or not isinstance(extractor_version, str)
    ):
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


def _runtime_value(
    *,
    owner: str,
    build_revision: str,
    adaptive_graph: AdaptiveGraphIdentity | None,
    compose_project: str = EVALUATION_PROJECT,
) -> dict[str, object]:
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "compose_project": compose_project,
        "owner": owner,
        "build_revision": build_revision,
        "env_file": DEFAULT_RUNTIME_ENV.name,
        "compose_env_file": DEFAULT_COMPOSE_ENV.name,
        "api_base_url": f"http://127.0.0.1:{EVALUATION_PORTS['api']}/api/v1",
        "ports": dict(EVALUATION_PORTS),
        "adaptive_graph": (
            {
                "knowledge_base_id": str(adaptive_graph.knowledge_base_id),
                "workspace_id": str(adaptive_graph.workspace_id),
                "index_revision_id": str(adaptive_graph.index_revision_id),
                "graph_build_id": str(adaptive_graph.graph_build_id),
                "answer_profile_revision_id": str(adaptive_graph.answer_profile_revision_id),
                "judge_profile_revision_id": str(adaptive_graph.judge_profile_revision_id),
                "schema_profile_key": adaptive_graph.schema_profile_key,
                "schema_profile_digest": adaptive_graph.schema_profile_digest,
                "extractor_version": adaptive_graph.extractor_version,
            }
            if adaptive_graph is not None
            else None
        ),
    }


def _write_private(path: Path, payload: bytes, *, replace: bool = False) -> None:
    if path.exists() or path.is_symlink():
        if not replace:
            raise EvaluationRuntimeError("evaluation runtime artifact already exists")
        _private_regular_file(path, reason="evaluation runtime artifact")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _evaluation_env(
    *,
    owner: str,
    workspace_id: UUID | None = None,
    host_access: bool,
) -> bytes:
    if _OWNER_PATTERN.fullmatch(owner) is None:
        raise EvaluationRuntimeError("evaluation owner is invalid")
    canonical = _private_regular_file(CANONICAL_MANIFEST, reason="canonical manifest")
    values = _parse_manifest(canonical)
    retained = {
        name: value
        for name, value in values.items()
        if name.startswith("RAG_KB__")
        and name not in {
            "RAG_KB__DATABASE__RUNTIME_DSN",
            "RAG_KB__DATABASE__MIGRATION_DSN",
            "RAG_KB__IDENTITY__PRINCIPAL_ID",
            "RAG_KB__IDENTITY__CLIENT_ID",
            "RAG_KB__IDENTITY__WORKSPACE_ID",
        }
        and not name.startswith("RAG_KB__MODEL_PROVIDER__")
    }
    database_host = (
        f"127.0.0.1:{EVALUATION_PORTS['postgres']}"
        if host_access
        else "postgres:5432"
    )
    retained.update(
        {
            "COMPOSE_PROJECT_NAME": EVALUATION_PROJECT,
            "RAG_KB_API_PORT": str(EVALUATION_PORTS["api"]),
            "RAG_KB_FRONTEND_PORT": str(EVALUATION_PORTS["frontend"]),
            "RAG_KB_POSTGRES_PORT": str(EVALUATION_PORTS["postgres"]),
            "RAG_KB_FALKORDB_PORT": str(EVALUATION_PORTS["falkordb"]),
            "POSTGRES_ADMIN_PASSWORD": EVALUATION_PASSWORD,
            "RAG_KB_MIGRATION_PASSWORD": EVALUATION_PASSWORD,
            "RAG_KB_RUNTIME_PASSWORD": EVALUATION_PASSWORD,
            "RAG_KB__IDENTITY__PRINCIPAL_ID": f"eval-{owner}",
            "RAG_KB__IDENTITY__CLIENT_ID": "local-evaluator",
            "RAG_KB__DATABASE__RUNTIME_DSN": (
                f"postgresql+asyncpg://rag_kb_runtime:{EVALUATION_PASSWORD}@{database_host}/rag_kb"
            ),
            "RAG_KB__DATABASE__MIGRATION_DSN": (
                f"postgresql+asyncpg://rag_kb_migration:{EVALUATION_PASSWORD}@{database_host}/rag_kb"
            ),
        }
    )
    if host_access:
        source_root = DEFAULT_RUNTIME_ROOT / "source-data"
        retained.update(
            {
                "RAG_KB__FILE_STORE__ROOT_PATH": str(source_root),
                "RAG_KB__FILE_STORE__STAGING_PATH": str(source_root / "staging"),
                "RAG_KB__FILE_STORE__FINAL_PATH": str(source_root / "final"),
                "RAG_KB__FILE_STORE__ASSET_STAGING_PATH": str(source_root / "asset-staging"),
                "RAG_KB__FILE_STORE__ASSET_FINAL_PATH": str(source_root / "assets"),
                "RAG_KB__FILE_STORE__PARSER_TEMP_PATH": str(source_root / "parser-temp"),
                "RAG_KB__MODEL_SECRETS__ROOT_PATH": str(DEFAULT_RUNTIME_ROOT / "model-secrets"),
                "RAG_KB__GRAPHITI__HOST": "127.0.0.1",
                "RAG_KB__GRAPHITI__PORT": str(EVALUATION_PORTS["falkordb"]),
                "RAG_KB__OBSERVABILITY__LOG_DIRECTORY": str(DEFAULT_RUNTIME_ROOT / "logs"),
            }
        )
    if workspace_id is not None:
        retained["RAG_KB__IDENTITY__WORKSPACE_ID"] = str(workspace_id)
    return "".join(f"{name}={retained[name]}\n" for name in sorted(retained)).encode("utf-8")


def _reuse_local_images() -> str:
    revisions: list[str] = []
    for source, _, paths in LOCAL_IMAGE_REUSE:
        revision = _run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                '{{index .Config.Labels "org.opencontainers.image.revision"}}',
                source,
            ],
            capture=True,
        )
        if _REVISION_PATTERN.fullmatch(revision) is None:
            raise EvaluationRuntimeError("local image revision is invalid")
        try:
            _run(
                [
                    "git",
                    "-C",
                    str(PROJECT_ROOT),
                    "diff",
                    "--quiet",
                    revision,
                    "HEAD",
                    "--",
                    *paths,
                ]
            )
        except subprocess.CalledProcessError as error:
            raise EvaluationRuntimeError("local image source is stale") from error
        revisions.append(revision)
    if len(set(revisions)) != 1:
        raise EvaluationRuntimeError("local image revisions do not match")
    for source, target, _ in LOCAL_IMAGE_REUSE:
        _run(["docker", "image", "tag", source, target])
    return revisions[0]


def _assert_primary_checkout() -> None:
    try:
        canonical = discover_canonical_checkout(PROJECT_ROOT)
    except (LocalRuntimeError, OSError, subprocess.SubprocessError) as error:
        raise EvaluationRuntimeError("evaluation lifecycle requires the primary checkout") from error
    if canonical != PROJECT_ROOT.resolve():
        raise EvaluationRuntimeError("evaluation lifecycle requires the primary checkout")


def _extract_private_archive(archive_path: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise EvaluationRuntimeError("evaluation host seed target already exists")
    target.mkdir(mode=0o700)
    target_root = target.resolve()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise EvaluationRuntimeError("evaluation seed archive is unsafe")
                destination = (target_root / relative).resolve()
                if not destination.is_relative_to(target_root):
                    raise EvaluationRuntimeError("evaluation seed archive is unsafe")
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
                    os.chmod(destination, 0o700)
                    continue
                if not member.isfile():
                    raise EvaluationRuntimeError("evaluation seed archive is unsafe")
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = archive.extractfile(member)
                if source is None:
                    raise EvaluationRuntimeError("evaluation seed archive is unreadable")
                descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with source, os.fdopen(descriptor, "wb") as output:
                    shutil.copyfileobj(source, output)
                os.chmod(destination, 0o600)
    except BaseException:
        if target.exists() and not target.is_symlink():
            shutil.rmtree(target)
        raise


def _prepare_host_runtime(seed: Path) -> None:
    _extract_private_archive(seed / "p6-source-data.tar.gz", DEFAULT_RUNTIME_ROOT / "source-data")
    _extract_private_archive(seed / "p6-model-secrets.tar.gz", DEFAULT_RUNTIME_ROOT / "model-secrets")
    logs = DEFAULT_RUNTIME_ROOT / "logs"
    logs.mkdir(mode=0o700)
    os.chmod(logs, 0o700)


def _run(
    command: Sequence[str],
    *,
    runtime: EvaluationRuntime | None = None,
    capture: bool = False,
    cwd: Path = PROJECT_ROOT,
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=compose_environment(runtime) if runtime is not None else None,
        check=True,
        capture_output=capture,
        text=True,
    )
    return completed.stdout.strip() if capture else ""


def _verify_seed(seed: Path) -> Path:
    seed = seed.resolve()
    if seed != DEFAULT_SEED_BACKUP.resolve() or not seed.is_dir() or seed.is_symlink():
        raise EvaluationRuntimeError("evaluation seed directory is invalid")
    for name in REQUIRED_SEED_FILES:
        _private_regular_file(seed / name, reason="evaluation seed payload")
    _run(
        ["shasum", "-a", "256", "-c", "SHA256SUMS"],
        capture=True,
        cwd=seed,
    )
    return seed


def _seed_adaptive_identity(seed: Path) -> FrozenAdaptiveGraphIdentity:
    archive_path = _private_regular_file(
        seed / "adaptive-graph-route-v2.tar.gz",
        reason="evaluation adaptive Graph seed",
    )
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name == FROZEN_ADAPTIVE_RESULT
        ]
        if (
            len(members) != 1
            or not members[0].isfile()
            or not 0 < members[0].size <= MAX_FROZEN_ADAPTIVE_RESULT_BYTES
        ):
            raise EvaluationRuntimeError("evaluation adaptive Graph seed is invalid")
        source = archive.extractfile(members[0])
        if source is None:
            raise EvaluationRuntimeError("evaluation adaptive Graph seed is unreadable")
        with source:
            try:
                value = json.loads(source.read(MAX_FROZEN_ADAPTIVE_RESULT_BYTES + 1))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EvaluationRuntimeError(
                    "evaluation adaptive Graph seed is invalid"
                ) from error
    if not isinstance(value, dict) or value.get("status") != "completed":
        raise EvaluationRuntimeError("evaluation adaptive Graph seed is invalid")
    runtime = value.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("schema_version")
        not in {
            "adaptive_graph_r7_stage_a_v2",
            "adaptive_graph_r7_stage_a_v3",
        }
        or runtime.get("dataset_id") != "routing-rag-v2"
    ):
        raise EvaluationRuntimeError("evaluation adaptive Graph seed is invalid")
    fields = {
        "knowledge_base_id",
        "index_revision_id",
        "graph_build_id",
        "answer_profile_revision_id",
        "judge_source_profile_revision_id",
    }
    if any(name not in runtime for name in fields):
        raise EvaluationRuntimeError("evaluation adaptive Graph seed is invalid")
    try:
        identifiers = {name: UUID(str(runtime[name])) for name in fields}
    except (TypeError, ValueError) as error:
        raise EvaluationRuntimeError("evaluation adaptive Graph seed is invalid") from error
    return FrozenAdaptiveGraphIdentity(
        knowledge_base_id=identifiers["knowledge_base_id"],
        index_revision_id=identifiers["index_revision_id"],
        graph_build_id=identifiers["graph_build_id"],
        answer_profile_revision_id=identifiers["answer_profile_revision_id"],
        judge_profile_revision_id=identifiers["judge_source_profile_revision_id"],
    )


def _docker_project_objects() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    containers = tuple(
        line
        for line in _run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"label=com.docker.compose.project={EVALUATION_PROJECT}",
                "--format",
                "{{.ID}}",
            ],
            capture=True,
        ).splitlines()
        if line
    )
    volumes = tuple(
        line
        for line in _run(
            [
                "docker",
                "volume",
                "ls",
                "--filter",
                f"label=com.docker.compose.project={EVALUATION_PROJECT}",
                "--format",
                "{{.Name}}",
            ],
            capture=True,
        ).splitlines()
        if line
    )
    networks = tuple(
        line
        for line in _run(
            [
                "docker",
                "network",
                "ls",
                "--filter",
                f"label=com.docker.compose.project={EVALUATION_PROJECT}",
                "--format",
                "{{.ID}}",
            ],
            capture=True,
        ).splitlines()
        if line
    )
    return containers, volumes, networks


def _assert_no_existing_runtime() -> None:
    if DEFAULT_RUNTIME_ROOT.exists() or DEFAULT_RUNTIME_ROOT.is_symlink():
        raise EvaluationRuntimeError("evaluation runtime already exists")
    containers, volumes, networks = _docker_project_objects()
    if containers or volumes or networks:
        raise EvaluationRuntimeError("evaluation Docker objects already exist")


def _volume_name(runtime: EvaluationRuntime, key: str) -> str:
    output = _run(
        compose_command(runtime, "config", "--format", "json"),
        runtime=runtime,
        capture=True,
    )
    value = json.loads(output)
    volume = value.get("volumes", {}).get(key, {})
    name = volume.get("name") if isinstance(volume, dict) else None
    if not isinstance(name, str) or not name.startswith(f"{EVALUATION_PROJECT}_"):
        raise EvaluationRuntimeError("evaluation volume identity is invalid")
    return name


def _restore_volume(
    runtime: EvaluationRuntime,
    *,
    seed: Path,
    archive: str,
    volume_key: str,
) -> None:
    volume = _volume_name(runtime, volume_key)
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--label",
            f"{EVALUATION_OWNER_LABEL}={runtime.owner}",
            "--mount",
            f"type=bind,src={seed},dst=/seed,readonly",
            "--mount",
            f"type=volume,src={volume},dst=/target",
            "alpine:latest",
            "tar",
            "-xzf",
            f"/seed/{archive}",
            "-C",
            "/target",
        ]
    )


def _restore_falkor(runtime: EvaluationRuntime, *, seed: Path) -> None:
    volume = _volume_name(runtime, "falkordb-data")
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--label",
            f"{EVALUATION_OWNER_LABEL}={runtime.owner}",
            "--mount",
            f"type=bind,src={seed},dst=/seed,readonly",
            "--mount",
            f"type=volume,src={volume},dst=/target",
            "alpine:latest",
            "cp",
            "/seed/p6-falkordb.rdb",
            "/target/dump.rdb",
        ]
    )


def _container_id(runtime: EvaluationRuntime, service: str) -> str:
    value = _run(
        compose_command(runtime, "ps", "-q", service),
        runtime=runtime,
        capture=True,
    )
    if not value or "\n" in value:
        raise EvaluationRuntimeError("evaluation container identity is invalid")
    owner = _run(
        [
            "docker",
            "inspect",
            "--format",
            f'{{{{index .Config.Labels "{EVALUATION_OWNER_LABEL}"}}}}',
            value,
        ],
        capture=True,
    )
    if owner != runtime.owner:
        raise EvaluationRuntimeError("evaluation container owner is invalid")
    return value


def _restore_database(runtime: EvaluationRuntime, *, seed: Path) -> None:
    container = _container_id(runtime, "postgres")
    target = "/tmp/rag-eval-seed.dump"
    _run(["docker", "cp", str(seed / "p6-postgres.dump"), f"{container}:{target}"])
    try:
        _run(
            [
                "docker",
                "exec",
                container,
                "pg_restore",
                "--clean",
                "--if-exists",
                "--username",
                "postgres",
                "--dbname",
                "rag_kb",
                target,
            ]
        )
    finally:
        _run(["docker", "exec", container, "rm", "-f", target])
    _run(["docker", "exec", container, "/docker-entrypoint-initdb.d/10-init-runtime.sh"])


def _reconcile_runtime_grants(runtime: EvaluationRuntime) -> None:
    """Restore the immutable 0001 runtime ACL omitted by the frozen dump."""

    container = _container_id(runtime, "postgres")
    _run(
        [
            "docker",
            "exec",
            container,
            "psql",
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            "postgres",
            "-d",
            "rag_kb",
            "-c",
            EVALUATION_RUNTIME_GRANTS,
        ]
    )
    valid = _run(
        [
            "docker",
            "exec",
            container,
            "psql",
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            "postgres",
            "-d",
            "rag_kb",
            "-At",
            "-c",
            EVALUATION_RUNTIME_GRANTS_CHECK,
        ],
        capture=True,
    )
    if valid != "t":
        raise EvaluationRuntimeError("evaluation runtime grants are invalid")


def _adaptive_identity(
    runtime: EvaluationRuntime,
    *,
    frozen: FrozenAdaptiveGraphIdentity,
) -> AdaptiveGraphIdentity:
    container = _container_id(runtime, "postgres")
    query = (
        "SELECT config.workspace_id, config.kb_id, kb.active_index_revision_id, config.active_build_id, "
        "answer.id, judge.id, build.schema_profile_key, build.schema_profile_digest, "
        "build.extractor_version "
        "FROM knowledge_base_graph_config config "
        "JOIN knowledge_base kb ON kb.workspace_id = config.workspace_id AND kb.id = config.kb_id "
        "JOIN graphiti_graph_build build ON build.workspace_id = config.workspace_id "
        "AND build.kb_id = config.kb_id AND build.build_id = config.active_build_id "
        "JOIN model_selection selection ON selection.workspace_id = config.workspace_id "
        f"AND selection.chat_profile_revision_id = '{frozen.answer_profile_revision_id}'::uuid "
        "JOIN model_profile_revision answer ON answer.workspace_id = config.workspace_id "
        f"AND answer.id = '{frozen.answer_profile_revision_id}'::uuid "
        "JOIN model_profile answer_profile ON answer_profile.workspace_id = answer.workspace_id "
        "AND answer_profile.id = answer.profile_id "
        "JOIN model_profile_revision judge ON judge.workspace_id = config.workspace_id "
        f"AND judge.id = '{frozen.judge_profile_revision_id}'::uuid "
        "JOIN model_profile judge_profile ON judge_profile.workspace_id = judge.workspace_id "
        "AND judge_profile.id = judge.profile_id "
        "WHERE config.status = 'ready' AND build.status = 'ready' "
        "AND build.completed_at IS NOT NULL "
        f"AND config.kb_id = '{frozen.knowledge_base_id}'::uuid "
        f"AND kb.active_index_revision_id = '{frozen.index_revision_id}'::uuid "
        f"AND config.active_build_id = '{frozen.graph_build_id}'::uuid "
        "AND build.index_revision_id = kb.active_index_revision_id "
        "AND answer.validation_status = 'valid' AND judge.validation_status = 'valid' "
        "AND answer_profile.kind = 'chat' AND answer_profile.enabled "
        "AND judge_profile.kind = 'chat' AND judge_profile.enabled"
    )
    output = _run(
        [
            "docker",
            "exec",
            container,
            "psql",
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            "postgres",
            "-d",
            "rag_kb",
            "-At",
            "-F",
            "\t",
            "-c",
            query,
        ],
        capture=True,
    )
    rows = [line.split("\t") for line in output.splitlines() if line]
    if len(rows) != 1 or len(rows[0]) != 9:
        raise EvaluationRuntimeError("evaluation adaptive Graph identity is unavailable")
    try:
        workspace_id, kb_id, revision_id, build_id, answer_profile_id, judge_profile_id = (
            UUID(item) for item in rows[0][:6]
        )
    except ValueError as error:
        raise EvaluationRuntimeError("evaluation adaptive Graph identity is invalid") from error
    schema_profile_key, schema_profile_digest, extractor_version = rows[0][6:]
    try:
        get_graph_schema_registry().resolve(
            schema_profile_key,
            digest=schema_profile_digest,
            extractor_version=extractor_version,
        )
    except ValueError as error:
        raise EvaluationRuntimeError("evaluation adaptive Graph identity is invalid") from error
    return AdaptiveGraphIdentity(
        workspace_id=workspace_id,
        knowledge_base_id=kb_id,
        index_revision_id=revision_id,
        graph_build_id=build_id,
        answer_profile_revision_id=answer_profile_id,
        judge_profile_revision_id=judge_profile_id,
        schema_profile_key=schema_profile_key,
        schema_profile_digest=schema_profile_digest,
        extractor_version=extractor_version,
    )


def create_runtime(*, seed: Path, confirmation: str) -> dict[str, object]:
    if confirmation != CREATE_CONFIRMATION:
        raise EvaluationRuntimeError("evaluation create confirmation is invalid")
    _assert_primary_checkout()
    seed = _verify_seed(seed)
    frozen_identity = _seed_adaptive_identity(seed)
    _assert_no_existing_runtime()
    build_revision = _reuse_local_images()
    DEFAULT_RUNTIME_ROOT.mkdir(parents=True, mode=0o700)
    os.chmod(DEFAULT_RUNTIME_ROOT, 0o700)
    owner = uuid4().hex
    _write_private(DEFAULT_RUNTIME_ENV, _evaluation_env(owner=owner, host_access=True))
    _write_private(DEFAULT_COMPOSE_ENV, _evaluation_env(owner=owner, host_access=False))
    _write_private(
        DEFAULT_RUNTIME_MANIFEST,
        (
            json.dumps(
                _runtime_value(
                    owner=owner,
                    build_revision=build_revision,
                    adaptive_graph=None,
                ),
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    runtime = load_evaluation_runtime()
    _run(compose_command(runtime, "up", "-d", "--wait", "postgres"), runtime=runtime)
    _run(compose_command(runtime, "create", "storage-init", "falkordb"), runtime=runtime)
    _restore_database(runtime, seed=seed)
    _restore_volume(runtime, seed=seed, archive="p6-source-data.tar.gz", volume_key="source-data")
    _restore_volume(runtime, seed=seed, archive="p6-model-secrets.tar.gz", volume_key="model-secrets")
    _restore_falkor(runtime, seed=seed)
    _prepare_host_runtime(seed)
    _run(compose_command(runtime, "--profile", "tools", "run", "--rm", "migrate"), runtime=runtime)
    _reconcile_runtime_grants(runtime)
    identity = _adaptive_identity(runtime, frozen=frozen_identity)
    _write_private(
        DEFAULT_RUNTIME_ENV,
        _evaluation_env(
            owner=owner,
            workspace_id=identity.workspace_id,
            host_access=True,
        ),
        replace=True,
    )
    _write_private(
        DEFAULT_COMPOSE_ENV,
        _evaluation_env(
            owner=owner,
            workspace_id=identity.workspace_id,
            host_access=False,
        ),
        replace=True,
    )
    _write_private(
        DEFAULT_RUNTIME_MANIFEST,
        (
            json.dumps(
                _runtime_value(
                    owner=owner,
                    build_revision=build_revision,
                    adaptive_graph=identity,
                ),
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        replace=True,
    )
    runtime = load_evaluation_runtime(require_adaptive_graph=True)
    _run(compose_command(runtime, "up", "-d", "--wait", "api", "worker", "frontend"), runtime=runtime)
    result = inspect_runtime(runtime)
    if result["status"] != "ready":
        raise EvaluationRuntimeError("evaluation runtime did not become ready")
    return result


def _owned_project_objects(
    runtime: EvaluationRuntime,
    *,
    require_complete: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    containers, volumes, networks = _docker_project_objects()
    services: list[str] = []
    for resource in containers:
        output = _run(
            [
                "docker",
                "inspect",
                "--format",
                f'{{{{index .Config.Labels "{EVALUATION_OWNER_LABEL}"}}}}\t'
                '{{index .Config.Labels "com.docker.compose.service"}}',
                resource,
            ],
            capture=True,
        )
        owner, separator, service = output.partition("\t")
        if not separator or owner != runtime.owner or service not in EXPECTED_SERVICES:
            raise EvaluationRuntimeError("evaluation Docker owner is invalid")
        services.append(service)
    if len(services) != len(set(services)):
        raise EvaluationRuntimeError("evaluation service identity is ambiguous")
    volume_keys: list[str] = []
    for resource in volumes:
        output = _run(
            [
                "docker",
                "volume",
                "inspect",
                "--format",
                f'{{{{index .Labels "{EVALUATION_OWNER_LABEL}"}}}}\t'
                '{{index .Labels "com.docker.compose.volume"}}',
                resource,
            ],
            capture=True,
        )
        owner, separator, volume_key = output.partition("\t")
        if not separator or owner != runtime.owner or volume_key not in EXPECTED_VOLUMES:
            raise EvaluationRuntimeError("evaluation Docker owner is invalid")
        volume_keys.append(volume_key)
    if len(volume_keys) != len(set(volume_keys)):
        raise EvaluationRuntimeError("evaluation volume identity is ambiguous")
    if require_complete and set(volume_keys) != EXPECTED_VOLUMES:
        raise EvaluationRuntimeError("evaluation volume set is incomplete")
    network_keys: list[str] = []
    for resource in networks:
        output = _run(
            [
                "docker",
                "network",
                "inspect",
                "--format",
                f'{{{{index .Labels "{EVALUATION_OWNER_LABEL}"}}}}\t'
                '{{index .Labels "com.docker.compose.network"}}',
                resource,
            ],
            capture=True,
        )
        owner, separator, network_key = output.partition("\t")
        if not separator or owner != runtime.owner or network_key not in EXPECTED_NETWORKS:
            raise EvaluationRuntimeError("evaluation Docker owner is invalid")
        network_keys.append(network_key)
    if len(network_keys) != len(set(network_keys)):
        raise EvaluationRuntimeError("evaluation network identity is ambiguous")
    if require_complete and set(network_keys) != EXPECTED_NETWORKS:
        raise EvaluationRuntimeError("evaluation network set is incomplete")
    return containers, volumes, networks


def _services_ready(runtime: EvaluationRuntime, containers: Sequence[str]) -> bool:
    states: dict[str, tuple[str, str, str]] = {}
    for resource in containers:
        output = _run(
            [
                "docker",
                "inspect",
                "--format",
                f'{{{{index .Config.Labels "{EVALUATION_OWNER_LABEL}"}}}}\t'
                '{{index .Config.Labels "com.docker.compose.service"}}\t'
                "{{.State.Status}}\t"
                "{{if .State.Health}}{{.State.Health.Status}}{{end}}\t"
                "{{.State.ExitCode}}",
                resource,
            ],
            capture=True,
        )
        fields = output.split("\t")
        if len(fields) != 5:
            raise EvaluationRuntimeError("evaluation service state is invalid")
        owner, service, state, health, exit_code = fields
        if owner != runtime.owner or service not in EXPECTED_SERVICES or service in states:
            raise EvaluationRuntimeError("evaluation service state is invalid")
        states[service] = (state, health, exit_code)
    if set(states) != READY_SERVICES:
        return False
    for service in READY_SERVICES - {"storage-init"}:
        state, health, _ = states[service]
        if state != "running" or health != "healthy":
            return False
    storage_state, _, storage_exit = states["storage-init"]
    return storage_state == "exited" and storage_exit == "0"


def _falkor_restored(runtime: EvaluationRuntime) -> bool:
    container = _container_id(runtime, "falkordb")
    output = _run(
        ["docker", "exec", container, "redis-cli", "--raw", "DBSIZE"],
        capture=True,
    )
    try:
        return int(output) > 0
    except ValueError:
        return False


def inspect_runtime(runtime: EvaluationRuntime) -> dict[str, object]:
    containers, volumes, networks = _owned_project_objects(
        runtime,
        require_complete=False,
    )
    ready = (
        runtime.adaptive_graph is not None
        and len(volumes) == len(EXPECTED_VOLUMES)
        and len(networks) == len(EXPECTED_NETWORKS)
        and _services_ready(runtime, containers)
        and _falkor_restored(runtime)
    )
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "status": "ready" if ready else "incomplete",
        "compose_project": EVALUATION_PROJECT,
        "container_count": len(containers),
        "volume_count": len(volumes),
        "network_count": len(networks),
        "adaptive_graph_identity": runtime.adaptive_graph is not None,
        "ports": dict(runtime.ports),
    }


def destroy_runtime(*, confirmation: str) -> dict[str, object]:
    if confirmation != DESTROY_CONFIRMATION:
        raise EvaluationRuntimeError("evaluation destroy confirmation is invalid")
    _assert_primary_checkout()
    runtime = load_evaluation_runtime()
    containers, volumes, networks = _owned_project_objects(
        runtime,
        require_complete=False,
    )
    _run(compose_command(runtime, "down", "--volumes"), runtime=runtime)
    remaining_containers, remaining_volumes, remaining_networks = _docker_project_objects()
    if remaining_containers or remaining_volumes or remaining_networks:
        raise EvaluationRuntimeError("evaluation Docker cleanup is incomplete")
    expected_entries = {
        runtime.manifest.name,
        runtime.env_file.name,
        runtime.compose_env_file.name,
        "source-data",
        "model-secrets",
        "logs",
    }
    observed_entries = {path.name for path in runtime.runtime_root.iterdir()}
    if not observed_entries.issubset(expected_entries):
        raise EvaluationRuntimeError("evaluation runtime file set is unexpected")
    for path in runtime.runtime_root.rglob("*"):
        if path.is_symlink():
            raise EvaluationRuntimeError("evaluation runtime contains a symbolic link")
    shutil.rmtree(runtime.runtime_root)
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "status": "destroyed",
        "container_count": len(containers),
        "volume_count": len(volumes),
        "network_count": len(networks),
    }


def preview_runtime(seed: Path) -> dict[str, object]:
    seed = _verify_seed(seed)
    del seed
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "status": "ready",
        "compose_project": EVALUATION_PROJECT,
        "ports": dict(EVALUATION_PORTS),
        "seed_payload_count": len(REQUIRED_SEED_FILES) - 1,
        "create_confirmation": CREATE_CONFIRMATION,
        "destroy_confirmation": DESTROY_CONFIRMATION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview")
    preview.add_argument("--seed", type=Path, default=DEFAULT_SEED_BACKUP)
    create = commands.add_parser("create")
    create.add_argument("--seed", type=Path, default=DEFAULT_SEED_BACKUP)
    create.add_argument("--confirm", required=True)
    commands.add_parser("inspect")
    destroy = commands.add_parser("destroy")
    destroy.add_argument("--confirm", required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "preview":
            result = preview_runtime(arguments.seed)
        elif arguments.command == "create":
            result = create_runtime(seed=arguments.seed, confirmation=arguments.confirm)
        elif arguments.command == "inspect":
            result = inspect_runtime(load_evaluation_runtime())
        else:
            result = destroy_runtime(confirmation=arguments.confirm)
    except (
        EvaluationRuntimeError,
        LocalRuntimeError,
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        tarfile.TarError,
    ):
        print(
            json.dumps(
                {"schema_version": RUNTIME_SCHEMA_VERSION, "status": "blocked", "reason": "evaluation_runtime_invalid"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
