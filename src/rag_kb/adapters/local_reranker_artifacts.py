"""Content-addressed verification for the fixed local reranker bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path, PurePosixPath
import platform
from typing import Any


_MANIFEST_SCHEMA_VERSION = 1
_HASH_BUFFER_BYTES = 1024 * 1024
_PROFILE = "local_minilm_v1"
_REPOSITORY = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class LocalRerankerArtifactError(RuntimeError):
    """The fixed local reranker bundle is missing or does not match."""


@dataclass(frozen=True, slots=True)
class LocalRerankerArtifactEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LocalRerankerArtifactManifest:
    profile: str
    repository: str
    revision: str
    architectures: tuple[str, ...]
    onnxruntime_version: str
    tokenizers_version: str
    entries: tuple[LocalRerankerArtifactEntry, ...]

    @classmethod
    def load(cls, manifest_path: Path) -> LocalRerankerArtifactManifest:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise LocalRerankerArtifactError("manifest_unreadable") from error
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "profile",
            "repository",
            "revision",
            "runtime",
            "artifacts",
        }:
            raise LocalRerankerArtifactError("manifest_shape")
        if payload.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
            raise LocalRerankerArtifactError("manifest_schema_version")
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict) or set(runtime) != {
            "architecture",
            "onnxruntime_version",
            "tokenizers_version",
        }:
            raise LocalRerankerArtifactError("manifest_runtime")
        raw_architectures = runtime.get("architecture")
        if (
            not isinstance(raw_architectures, list)
            or not raw_architectures
            or any(
                not isinstance(value, str) or not value
                for value in raw_architectures
            )
            or len(set(raw_architectures)) != len(raw_architectures)
        ):
            raise LocalRerankerArtifactError("manifest_architecture")
        entries_payload = payload.get("artifacts")
        if not isinstance(entries_payload, list) or not entries_payload:
            raise LocalRerankerArtifactError("manifest_artifacts")
        entries: list[LocalRerankerArtifactEntry] = []
        seen: set[str] = set()
        for raw_entry in entries_payload:
            entry = _parse_entry(raw_entry)
            if entry.path in seen:
                raise LocalRerankerArtifactError("manifest_duplicate_path")
            seen.add(entry.path)
            entries.append(entry)
        identity = (
            payload.get("profile"),
            payload.get("repository"),
            payload.get("revision"),
            runtime.get("onnxruntime_version"),
            runtime.get("tokenizers_version"),
        )
        if not all(isinstance(value, str) and value for value in identity):
            raise LocalRerankerArtifactError("manifest_identity")
        return cls(
            profile=identity[0],
            repository=identity[1],
            revision=identity[2],
            architectures=tuple(raw_architectures),
            onnxruntime_version=identity[3],
            tokenizers_version=identity[4],
            entries=tuple(entries),
        )

    def verify(
        self,
        artifacts_path: Path,
        *,
        verify_runtime: bool = True,
    ) -> None:
        if self.profile != _PROFILE or self.repository != _REPOSITORY:
            raise LocalRerankerArtifactError("manifest_profile")
        if verify_runtime:
            if platform.machine().lower() not in self.architectures:
                raise LocalRerankerArtifactError("runtime_architecture")
            _require_package_version("onnxruntime", self.onnxruntime_version)
            _require_package_version("tokenizers", self.tokenizers_version)
        try:
            root = artifacts_path.resolve(strict=True)
        except OSError as error:
            raise LocalRerankerArtifactError("artifact_root") from error
        if not root.is_dir():
            raise LocalRerankerArtifactError("artifact_root")
        for entry in self.entries:
            destination = root.joinpath(*PurePosixPath(entry.path).parts)
            try:
                resolved = destination.resolve(strict=True)
            except OSError as error:
                raise LocalRerankerArtifactError("artifact_missing") from error
            if (
                not resolved.is_relative_to(root)
                or destination.is_symlink()
                or not resolved.is_file()
            ):
                raise LocalRerankerArtifactError("artifact_path")
            try:
                observed_size = resolved.stat().st_size
            except OSError as error:
                raise LocalRerankerArtifactError("artifact_unreadable") from error
            if observed_size != entry.size:
                raise LocalRerankerArtifactError("artifact_size")
            if _sha256_file(resolved) != entry.sha256:
                raise LocalRerankerArtifactError("artifact_digest")


def verify_local_reranker_artifacts(
    artifacts_path: Path,
    manifest_path: Path,
    *,
    verify_runtime: bool = True,
) -> LocalRerankerArtifactManifest:
    """Load and verify the complete fixed local reranker bundle."""

    manifest = LocalRerankerArtifactManifest.load(manifest_path)
    manifest.verify(artifacts_path, verify_runtime=verify_runtime)
    return manifest


def _parse_entry(payload: Any) -> LocalRerankerArtifactEntry:
    if not isinstance(payload, dict) or set(payload) != {"path", "size", "sha256"}:
        raise LocalRerankerArtifactError("manifest_entry_shape")
    path = payload.get("path")
    size = payload.get("size")
    sha256 = payload.get("sha256")
    if not isinstance(path, str) or not path:
        raise LocalRerankerArtifactError("manifest_entry_path")
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or ".." in pure_path.parts or "." in pure_path.parts:
        raise LocalRerankerArtifactError("manifest_entry_path")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise LocalRerankerArtifactError("manifest_entry_size")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise LocalRerankerArtifactError("manifest_entry_digest")
    return LocalRerankerArtifactEntry(path=path, size=size, sha256=sha256)


def _require_package_version(package: str, expected: str) -> None:
    try:
        observed = version(package)
    except PackageNotFoundError as error:
        raise LocalRerankerArtifactError("dependency_missing") from error
    if observed != expected:
        raise LocalRerankerArtifactError("dependency_version")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(_HASH_BUFFER_BYTES):
                digest.update(block)
    except OSError as error:
        raise LocalRerankerArtifactError("artifact_unreadable") from error
    return digest.hexdigest()
