"""Framework-independent indexing execution facts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid5

from rag_kb.domain.content import EmbeddingSpaceDefinition
from rag_kb.domain.errors import ErrorCode


CHUNK_ID_NAMESPACE = UUID("bfa48c2a-6d99-5b0c-94df-0f7bb462c704")
VECTOR_ID_NAMESPACE = UUID("263db84c-f438-5bd1-b9ca-666752fc2e92")


class IndexingPhase(StrEnum):
    SOURCE_READ = "source_read"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    PERSISTING = "persisting"
    VALIDATING = "validating"
    COMPLETED = "completed"


class PromotionStatus(StrEnum):
    SERVING = "serving"
    RETIRED = "retired"
    NOT_READY = "not_ready"


class PromotionReason(StrEnum):
    PROMOTED = "promoted"
    ALREADY_SERVING = "already_serving"
    ALREADY_RETIRED = "already_retired"
    NOT_READY = "not_ready"
    JOB_INCOMPLETE = "job_incomplete"
    DOCUMENT_DELETED = "document_deleted"
    SUPERSEDED = "superseded"
    LATER_SOURCE_CHANGE = "later_source_change"
    REVISION_INACTIVE = "revision_inactive"


class WorkLane(StrEnum):
    CHAT = "chat"
    INDEXING = "indexing"


@dataclass(frozen=True, slots=True)
class IndexingCommand:
    job_id: UUID
    indexed_document_version_id: UUID


@dataclass(frozen=True, slots=True)
class PromotionCommand:
    job_id: UUID
    indexed_document_version_id: UUID


@dataclass(frozen=True, slots=True)
class PromotionResult:
    job_id: UUID
    indexed_document_version_id: UUID
    status: PromotionStatus
    reason: PromotionReason
    previous_serving_target_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class IndexingLease:
    job_id: UUID
    indexed_document_version_id: UUID
    claimed_by: str
    attempt: int
    claimed_at: datetime


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    requeued: int
    failed: int


@dataclass(frozen=True, slots=True)
class IndexingJobSnapshot:
    job_id: UUID
    kb_id: UUID
    document_id: UUID
    document_version_id: UUID
    indexed_document_version_id: UUID
    index_revision_id: UUID
    job_status: str
    phase: str
    attempt: int
    build_status: str
    serving_status: str
    claimed_at: datetime | None
    heartbeat_at: datetime | None
    next_attempt_at: datetime | None
    error_code: str | None
    error_detail: dict[str, Any] | None
    can_retry: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class IndexCleanupResult:
    retired_targets_cleaned: int = 0
    vectors_deleted: int = 0
    chunks_deleted: int = 0
    jobs_deleted: int = 0
    file_cleanup_records_deleted: int = 0


@dataclass(frozen=True, slots=True)
class IndexingTarget:
    job_id: UUID
    indexed_document_version_id: UUID
    workspace_id: UUID
    kb_id: UUID
    document_id: UUID
    document_version_id: UUID
    index_revision_id: UUID
    embedding_space_id: UUID
    source_change_seq: int
    storage_uri: str
    checksum_sha256: str
    size_bytes: int
    original_filename: str
    media_type: str
    parser_config: dict[str, Any]
    chunking_config: dict[str, Any]
    embedding_space: EmbeddingSpaceDefinition
    already_complete: bool = False


@dataclass(frozen=True, slots=True)
class IndexChunkWrite:
    id: UUID
    ordinal: int
    content: str
    content_hash: str
    token_count: int
    source_location: dict[str, Any]
    hierarchy: dict[str, Any]
    source_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VectorRecordWrite:
    id: UUID
    index_chunk_id: UUID
    embedding_space_id: UUID
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class IndexingResult:
    job_id: UUID
    indexed_document_version_id: UUID
    status: str
    chunk_count: int
    replayed: bool = False
    serving_status: str = "candidate"


class IndexingExecutionError(RuntimeError):
    """Stable, content-safe indexing failure persisted by the coordinator."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        phase: IndexingPhase,
        diagnostic: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.phase = phase
        self.diagnostic = dict(diagnostic or {})


class IndexingCancelled(RuntimeError):
    """The durable target became cancelled or retired during execution."""


def stable_chunk_id(indexed_document_version_id: UUID, ordinal: int) -> UUID:
    if ordinal < 0:
        raise ValueError("chunk ordinal must be non-negative")
    return uuid5(CHUNK_ID_NAMESPACE, f"{indexed_document_version_id}:{ordinal}")


def stable_vector_id(embedding_space_id: UUID, index_chunk_id: UUID) -> UUID:
    return uuid5(VECTOR_ID_NAMESPACE, f"{embedding_space_id}:{index_chunk_id}")


def validate_embedding_vector(
    vector: tuple[float, ...],
    definition: EmbeddingSpaceDefinition,
    *,
    normalization_tolerance: float = 0.001,
) -> None:
    if len(vector) != definition.dimension:
        raise IndexingExecutionError(
            ErrorCode.EMBEDDING_RESPONSE_INVALID,
            phase=IndexingPhase.EMBEDDING,
            diagnostic={
                "check": "dimension",
                "expected": definition.dimension,
                "observed": len(vector),
            },
        )
    if definition.vector_data_type != "float32" or not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and abs(float(value)) <= 3.4028235e38
        for value in vector
    ):
        raise IndexingExecutionError(
            ErrorCode.EMBEDDING_RESPONSE_INVALID,
            phase=IndexingPhase.EMBEDDING,
            diagnostic={"check": "finite_float32"},
        )
    if definition.normalization == "l2":
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if abs(norm - 1.0) > normalization_tolerance:
            raise IndexingExecutionError(
                ErrorCode.EMBEDDING_RESPONSE_INVALID,
                phase=IndexingPhase.EMBEDDING,
                diagnostic={
                    "check": "l2_normalization",
                    "tolerance": normalization_tolerance,
                },
            )
