"""Source FileStore contract and local implementation."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from rag_kb.domain import (
    FileLocation,
    InvalidStorageIdentityError,
    SourceFileDigest,
    SourceFileIdentity,
    SourceFileIntegrityError,
    SourceFileMissingError,
    StagedSourceFile,
    StoredSourceFile,
)


_KEY = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 1024 * 1024


class LocalFileStore:
    """Same-filesystem staged source storage with no caller-controlled paths."""

    def __init__(self, staging_path: Path, final_path: Path) -> None:
        self._staging = staging_path.resolve(strict=True)
        self._final = final_path.resolve(strict=True)
        if self._staging.stat().st_dev != self._final.stat().st_dev:
            raise ValueError("staging and final paths must share one filesystem")

    async def stage(
        self,
        workspace_id: UUID,
        key_material: str,
        source: BinaryIO,
    ) -> StagedSourceFile:
        return await asyncio.to_thread(
            self._stage, workspace_id, key_material, source
        )

    async def finalize(
        self,
        identity: SourceFileIdentity,
        expected: SourceFileDigest,
    ) -> None:
        await asyncio.to_thread(self._finalize, identity, expected)

    async def inspect(
        self,
        identity: SourceFileIdentity,
        location: FileLocation,
    ) -> SourceFileDigest | None:
        return await asyncio.to_thread(self._inspect, identity, location)

    async def read_final(self, identity: SourceFileIdentity) -> bytes:
        return await asyncio.to_thread(self._read_final, identity)

    async def delete(self, identity: SourceFileIdentity) -> None:
        await asyncio.to_thread(self._delete, identity)

    async def discard_staged(self, identity: SourceFileIdentity) -> None:
        await asyncio.to_thread(self._discard_staged, identity)

    async def list_files(self) -> tuple[StoredSourceFile, ...]:
        return await asyncio.to_thread(self._list_files)

    async def delete_stored(self, stored: StoredSourceFile) -> None:
        await asyncio.to_thread(self._delete_stored, stored)

    @staticmethod
    def parse_uri(storage_uri: str) -> SourceFileIdentity:
        prefix = "local-source://"
        if not storage_uri.startswith(prefix):
            raise InvalidStorageIdentityError("unsupported source storage identity")
        remainder = storage_uri.removeprefix(prefix)
        parts = remainder.split("/")
        if len(parts) != 2 or not _KEY.fullmatch(parts[1]):
            raise InvalidStorageIdentityError("invalid source storage identity")
        try:
            workspace_id = UUID(parts[0])
        except ValueError as error:
            raise InvalidStorageIdentityError("invalid source storage identity") from error
        return SourceFileIdentity(workspace_id=workspace_id, key=parts[1])

    def _stage(
        self,
        workspace_id: UUID,
        key_material: str,
        source: BinaryIO,
    ) -> StagedSourceFile:
        incoming = self._staging / str(workspace_id) / ".incoming"
        incoming.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not incoming.resolve(strict=True).is_relative_to(self._staging):
            raise InvalidStorageIdentityError("source file escaped configured root")
        temporary = incoming / f"{uuid4().hex}.part"
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("xb") as handle:
                os.chmod(temporary, 0o600)
                while True:
                    block = source.read(_CHUNK_SIZE)
                    if not block:
                        break
                    if not isinstance(block, bytes):
                        raise TypeError("source file must yield bytes")
                    handle.write(block)
                    digest.update(block)
                    size += len(block)
                handle.flush()
                os.fsync(handle.fileno())
            checksum = digest.hexdigest()
            key = hashlib.sha256(
                f"{key_material}\x1f{checksum}".encode("utf-8")
            ).hexdigest()
            identity = SourceFileIdentity(workspace_id=workspace_id, key=key)
            target = self._path(identity, FileLocation.STAGING, create_parent=True)
            os.replace(temporary, target)
            self._fsync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return StagedSourceFile(
            identity=identity,
            digest=SourceFileDigest(checksum, size),
        )

    def _finalize(self, identity: SourceFileIdentity, expected: SourceFileDigest) -> None:
        final = self._path(identity, FileLocation.FINAL, create_parent=True)
        staging = self._path(identity, FileLocation.STAGING)
        if final.exists():
            self._require_digest(final, expected)
            staging.unlink(missing_ok=True)
            return
        if not staging.is_file():
            raise SourceFileMissingError("staged source file is missing")
        self._require_digest(staging, expected)
        os.replace(staging, final)
        self._fsync_directory(final.parent)
        self._require_digest(final, expected)

    def _inspect(
        self,
        identity: SourceFileIdentity,
        location: FileLocation,
    ) -> SourceFileDigest | None:
        path = self._path(identity, location)
        if not path.is_file():
            return None
        return self._digest(path)

    def _read_final(self, identity: SourceFileIdentity) -> bytes:
        path = self._path(identity, FileLocation.FINAL)
        if not path.is_file():
            raise SourceFileMissingError("final source file is missing")
        return path.read_bytes()

    def _delete(self, identity: SourceFileIdentity) -> None:
        for location in (FileLocation.STAGING, FileLocation.FINAL):
            path = self._path(identity, location)
            path.unlink(missing_ok=True)

    def _discard_staged(self, identity: SourceFileIdentity) -> None:
        self._path(identity, FileLocation.STAGING).unlink(missing_ok=True)

    def _list_files(self) -> tuple[StoredSourceFile, ...]:
        values: list[StoredSourceFile] = []
        for location, root in (
            (FileLocation.STAGING, self._staging),
            (FileLocation.FINAL, self._final),
        ):
            for path in root.rglob("*"):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(root)
                identity = self._identity_from_relative(relative)
                stat = path.stat()
                values.append(
                    StoredSourceFile(
                        identity=identity,
                        location=location,
                        opaque_name=relative.as_posix(),
                        modified_at=datetime.fromtimestamp(stat.st_mtime, UTC),
                        size_bytes=stat.st_size,
                    )
                )
        return tuple(values)

    def _delete_stored(self, stored: StoredSourceFile) -> None:
        root = self._staging if stored.location is FileLocation.STAGING else self._final
        candidate = (root / stored.opaque_name).resolve(strict=False)
        if not candidate.is_relative_to(root):
            raise InvalidStorageIdentityError("stored file escaped configured root")
        candidate.unlink(missing_ok=True)

    def _path(
        self,
        identity: SourceFileIdentity,
        location: FileLocation,
        *,
        create_parent: bool = False,
    ) -> Path:
        if not _KEY.fullmatch(identity.key):
            raise InvalidStorageIdentityError("invalid source file key")
        root = self._staging if location is FileLocation.STAGING else self._final
        parent = root / str(identity.workspace_id) / identity.key[:2]
        if create_parent:
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        candidate = parent / f"{identity.key}.source"
        resolved_parent = parent.resolve(strict=create_parent)
        if not resolved_parent.is_relative_to(root):
            raise InvalidStorageIdentityError("source file escaped configured root")
        return candidate

    @staticmethod
    def _digest(path: Path) -> SourceFileDigest:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while block := handle.read(_CHUNK_SIZE):
                digest.update(block)
                size += len(block)
        return SourceFileDigest(digest.hexdigest(), size)

    @classmethod
    def _require_digest(cls, path: Path, expected: SourceFileDigest) -> None:
        actual = cls._digest(path)
        if actual != expected:
            raise SourceFileIntegrityError("source file checksum or size differs")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _identity_from_relative(relative: Path) -> SourceFileIdentity | None:
        parts = relative.parts
        if len(parts) != 3 or not parts[2].endswith(".source"):
            return None
        key = parts[2].removesuffix(".source")
        if parts[1] != key[:2] or not _KEY.fullmatch(key):
            return None
        try:
            workspace_id = UUID(parts[0])
        except ValueError:
            return None
        return SourceFileIdentity(workspace_id=workspace_id, key=key)
