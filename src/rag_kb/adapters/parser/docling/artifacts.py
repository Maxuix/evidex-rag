"""Content-addressed verification for the offline Docling model bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path, PurePosixPath
from typing import Any


_MANIFEST_SCHEMA_VERSION = 1
_HASH_BUFFER_BYTES = 1024 * 1024


class ArtifactManifestError(RuntimeError):
    """The configured offline artifact bundle is missing or does not match."""


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DoclingArtifactManifest:
    profile: str
    docling_version: str
    docling_core_version: str
    docling_document_version: str
    entries: tuple[ArtifactEntry, ...]

    @classmethod
    def load(cls, manifest_path: Path) -> DoclingArtifactManifest:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ArtifactManifestError("manifest_unreadable") from error
        if not isinstance(payload, dict):
            raise ArtifactManifestError("manifest_shape")
        if payload.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
            raise ArtifactManifestError("manifest_schema_version")
        entries_payload = payload.get("artifacts")
        if not isinstance(entries_payload, list) or not entries_payload:
            raise ArtifactManifestError("manifest_artifacts")
        entries: list[ArtifactEntry] = []
        seen: set[str] = set()
        for raw_entry in entries_payload:
            entry = _parse_entry(raw_entry)
            if entry.path in seen:
                raise ArtifactManifestError("manifest_duplicate_path")
            seen.add(entry.path)
            entries.append(entry)
        profile = payload.get("profile")
        docling_version = payload.get("docling_version")
        docling_core_version = payload.get("docling_core_version")
        docling_document_version = payload.get("docling_document_version")
        if not all(
            isinstance(value, str) and value
            for value in (
                profile,
                docling_version,
                docling_core_version,
                docling_document_version,
            )
        ):
            raise ArtifactManifestError("manifest_identity")
        return cls(
            profile=profile,
            docling_version=docling_version,
            docling_core_version=docling_core_version,
            docling_document_version=docling_document_version,
            entries=tuple(entries),
        )

    def verify(self, artifacts_path: Path) -> None:
        if self.profile != "docling_native_v1":
            raise ArtifactManifestError("manifest_profile")
        _require_package_version("docling", self.docling_version)
        _require_package_version("docling-core", self.docling_core_version)
        try:
            root = artifacts_path.resolve(strict=True)
        except OSError as error:
            raise ArtifactManifestError("artifact_root") from error
        if not root.is_dir():
            raise ArtifactManifestError("artifact_root")
        for entry in self.entries:
            destination = root.joinpath(*PurePosixPath(entry.path).parts)
            try:
                resolved = destination.resolve(strict=True)
            except OSError as error:
                raise ArtifactManifestError("artifact_missing") from error
            if (
                not resolved.is_relative_to(root)
                or destination.is_symlink()
                or not resolved.is_file()
            ):
                raise ArtifactManifestError("artifact_path")
            try:
                observed_size = resolved.stat().st_size
            except OSError as error:
                raise ArtifactManifestError("artifact_unreadable") from error
            if observed_size != entry.size:
                raise ArtifactManifestError("artifact_size")
            if _sha256_file(resolved) != entry.sha256:
                raise ArtifactManifestError("artifact_digest")


def verify_docling_artifacts(
    artifacts_path: Path,
    manifest_path: Path,
) -> DoclingArtifactManifest:
    """Load and verify one complete local artifact bundle."""

    manifest = DoclingArtifactManifest.load(manifest_path)
    manifest.verify(artifacts_path)
    return manifest


def _parse_entry(payload: Any) -> ArtifactEntry:
    if not isinstance(payload, dict) or set(payload) != {"path", "size", "sha256"}:
        raise ArtifactManifestError("manifest_entry_shape")
    path = payload.get("path")
    size = payload.get("size")
    sha256 = payload.get("sha256")
    if not isinstance(path, str) or not path:
        raise ArtifactManifestError("manifest_entry_path")
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or ".." in pure_path.parts or "." in pure_path.parts:
        raise ArtifactManifestError("manifest_entry_path")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ArtifactManifestError("manifest_entry_size")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ArtifactManifestError("manifest_entry_digest")
    return ArtifactEntry(path=path, size=size, sha256=sha256)


def _require_package_version(package: str, expected: str) -> None:
    try:
        observed = version(package)
    except PackageNotFoundError as error:
        raise ArtifactManifestError("dependency_missing") from error
    if observed != expected:
        raise ArtifactManifestError("dependency_version")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(_HASH_BUFFER_BYTES):
                digest.update(block)
    except OSError as error:
        raise ArtifactManifestError("artifact_unreadable") from error
    return digest.hexdigest()
