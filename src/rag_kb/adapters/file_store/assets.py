"""Atomic local storage for indexed-version-scoped derived assets."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
from pathlib import Path
from uuid import UUID, uuid4

from rag_kb.domain import (
    IndexAssetIdentity,
    InvalidStorageIdentityError,
    SourceFileIntegrityError,
    SourceFileMissingError,
)


_KEY = re.compile(r"^[0-9a-f]{64}$")


class LocalIndexAssetStore:
    def __init__(self, staging_path: Path, final_path: Path) -> None:
        self._staging = staging_path.resolve(strict=True)
        self._final = final_path.resolve(strict=True)
        if self._staging.stat().st_dev != self._final.stat().st_dev:
            raise ValueError("asset staging and final paths must share one filesystem")

    async def put(
        self, identity: IndexAssetIdentity, content: bytes, checksum_sha256: str
    ) -> None:
        await asyncio.to_thread(self._put, identity, content, checksum_sha256)

    async def read(self, identity: IndexAssetIdentity) -> bytes:
        return await asyncio.to_thread(self._read, identity)

    async def delete(self, identity: IndexAssetIdentity) -> None:
        await asyncio.to_thread(self._delete, identity)

    async def discard_target(
        self,
        workspace_id: UUID,
        indexed_document_version_id: UUID,
    ) -> None:
        await asyncio.to_thread(
            self._discard_target,
            workspace_id,
            indexed_document_version_id,
        )

    @staticmethod
    def parse_uri(storage_uri: str) -> IndexAssetIdentity:
        prefix = "local-index-asset://"
        if not storage_uri.startswith(prefix):
            raise InvalidStorageIdentityError("unsupported index asset identity")
        parts = storage_uri.removeprefix(prefix).split("/")
        if len(parts) != 3 or not _KEY.fullmatch(parts[2]):
            raise InvalidStorageIdentityError("invalid index asset identity")
        try:
            return IndexAssetIdentity(UUID(parts[0]), UUID(parts[1]), parts[2])
        except ValueError as error:
            raise InvalidStorageIdentityError("invalid index asset identity") from error

    def _put(
        self, identity: IndexAssetIdentity, content: bytes, checksum_sha256: str
    ) -> None:
        if hashlib.sha256(content).hexdigest() != checksum_sha256:
            raise SourceFileIntegrityError("asset checksum differs")
        final = self._path(self._final, identity, create_parent=True)
        if final.exists():
            if hashlib.sha256(final.read_bytes()).hexdigest() != checksum_sha256:
                raise SourceFileIntegrityError("stable asset identity content differs")
            return
        staging = self._path(self._staging, identity, create_parent=True)
        temporary = staging.with_name(f".{uuid4().hex}.part")
        try:
            with temporary.open("xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, staging)
            os.replace(staging, final)
            self._fsync(final.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _read(self, identity: IndexAssetIdentity) -> bytes:
        path = self._path(self._final, identity)
        if not path.is_file():
            raise SourceFileMissingError("index asset is missing")
        return path.read_bytes()

    def _delete(self, identity: IndexAssetIdentity) -> None:
        self._path(self._staging, identity).unlink(missing_ok=True)
        self._path(self._final, identity).unlink(missing_ok=True)

    def _discard_target(
        self,
        workspace_id: UUID,
        indexed_document_version_id: UUID,
    ) -> None:
        for root in (self._staging, self._final):
            candidate = (
                root / str(workspace_id) / str(indexed_document_version_id)
            )
            target = candidate.resolve(strict=False)
            if (
                target != candidate
                or not target.is_relative_to(root)
                or target == root
            ):
                raise InvalidStorageIdentityError(
                    "index asset target escaped configured root"
                )
            if target.exists():
                shutil.rmtree(target)

    @staticmethod
    def _path(root: Path, identity: IndexAssetIdentity, *, create_parent: bool = False) -> Path:
        if not _KEY.fullmatch(identity.asset_key):
            raise InvalidStorageIdentityError("invalid asset key")
        parent = root / str(identity.workspace_id) / str(identity.indexed_document_version_id)
        if create_parent:
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved = parent.resolve(strict=create_parent)
        if not resolved.is_relative_to(root):
            raise InvalidStorageIdentityError("index asset escaped configured root")
        return resolved / f"{identity.asset_key}.asset"

    @staticmethod
    def _fsync(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
