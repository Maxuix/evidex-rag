"""Application-facing source FileStore contract."""

from __future__ import annotations

from typing import BinaryIO, Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain import (
    FileLocation,
    SourceFileDigest,
    SourceFileIdentity,
    StagedSourceFile,
    StoredSourceFile,
)


@runtime_checkable
class SourceFileStore(Protocol):
    async def stage(
        self,
        workspace_id: UUID,
        key_material: str,
        source: BinaryIO,
    ) -> StagedSourceFile: ...

    async def finalize(
        self,
        identity: SourceFileIdentity,
        expected: SourceFileDigest,
    ) -> None: ...

    async def inspect(
        self,
        identity: SourceFileIdentity,
        location: FileLocation,
    ) -> SourceFileDigest | None: ...

    async def read_final(self, identity: SourceFileIdentity) -> bytes: ...

    async def delete(self, identity: SourceFileIdentity) -> None: ...

    async def discard_staged(self, identity: SourceFileIdentity) -> None: ...

    async def list_files(self) -> tuple[StoredSourceFile, ...]: ...

    async def delete_stored(self, stored: StoredSourceFile) -> None: ...

    def parse_uri(self, storage_uri: str) -> SourceFileIdentity: ...
