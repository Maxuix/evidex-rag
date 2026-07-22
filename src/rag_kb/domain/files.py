"""Framework-independent local source-file consistency facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from rag_kb.domain.idempotency import IdempotencyScope


class FileLocation(StrEnum):
    STAGING = "staging"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class SourceFileIdentity:
    workspace_id: UUID
    key: str

    @property
    def storage_uri(self) -> str:
        return f"local-source://{self.workspace_id}/{self.key}"


@dataclass(frozen=True, slots=True)
class IndexAssetIdentity:
    workspace_id: UUID
    indexed_document_version_id: UUID
    asset_key: str

    @property
    def storage_uri(self) -> str:
        return (
            f"local-index-asset://{self.workspace_id}/"
            f"{self.indexed_document_version_id}/{self.asset_key}"
        )


@dataclass(frozen=True, slots=True)
class StoredSourceFile:
    identity: SourceFileIdentity | None
    location: FileLocation
    opaque_name: str
    modified_at: datetime
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SourceFileDigest:
    checksum_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class StagedSourceFile:
    identity: SourceFileIdentity
    digest: SourceFileDigest


@dataclass(frozen=True, slots=True)
class SourceFileReference:
    document_id: UUID
    document_version_id: UUID
    source_status: str
    storage_uri: str
    checksum_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class PendingFileMutation:
    scope: IdempotencyScope
    document_id: UUID
    document_version_id: UUID
    storage_uri: str
    checksum_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SourceFileCleanup:
    id: UUID
    document_version_id: UUID
    storage_uri: str
    reason: str
    status: str
    attempt_count: int
    next_attempt_at: datetime


@dataclass(frozen=True, slots=True)
class FileReconciliationResult:
    pending_activated: int = 0
    missing_compensated: int = 0
    cleanup_completed: int = 0
    cleanup_failed: int = 0
    orphans_removed: int = 0


class FileStoreError(RuntimeError):
    """Base class for content-safe local file-store failures."""


class InvalidStorageIdentityError(FileStoreError):
    pass


class SourceFileMissingError(FileStoreError):
    pass


class SourceFileIntegrityError(FileStoreError):
    pass
