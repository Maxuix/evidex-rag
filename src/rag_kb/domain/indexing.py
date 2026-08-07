"""Framework-independent indexing execution facts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid5

from rag_kb.domain.content import EmbeddingSpaceDefinition
from rag_kb.domain.composite import ChunkAssetRelationDraft
from rag_kb.domain.errors import ErrorCode
from rag_kb.domain.parsing import ContentModality


VECTOR_ID_NAMESPACE = UUID("263db84c-f438-5bd1-b9ca-666752fc2e92")
ASSET_ID_NAMESPACE = UUID("fef9ec6a-9ec5-58ac-9ceb-487db6cbeb79")
RELATION_ID_NAMESPACE = UUID("a758db33-f59e-5ed5-a68b-8cd4e0fbbe55")


class IndexingPhase(StrEnum):
    SOURCE_READ = "source_read"
    PARSING = "parsing"
    ASSET_EXTRACTION = "asset_extraction"
    ENRICHMENT = "enrichment"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    EMBEDDING = "embedding"
    MULTIMODAL_EMBEDDING = "multimodal_embedding"
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
    plans_deleted: int = 0
    manifests_deleted: int = 0
    assets_deleted: int = 0
    relations_deleted: int = 0
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
    enrichment_config: dict[str, Any] = field(default_factory=dict)
    representation_config: dict[str, Any] = field(default_factory=dict)
    embedding_space_ids: dict[str, UUID] = field(default_factory=dict)
    embedding_spaces: dict[str, EmbeddingSpaceDefinition] = field(default_factory=dict)


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
    unit_key: str
    modality: ContentModality = ContentModality.TEXT
    index_asset_id: UUID | None = None
    evidence_group_key: str | None = None
    relations: dict[str, Any] | None = None
    embedding_text: str | None = None
    embedding_text_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.unit_key.strip():
            raise ValueError("index chunk unit key must not be empty")


@dataclass(frozen=True, slots=True)
class IndexChunkLexicalWrite:
    index_chunk_id: UUID
    analyzer_version: str
    lexical_text: str
    lexical_text_hash: str


@dataclass(frozen=True, slots=True)
class IndexLexicalManifest:
    indexed_document_version_id: UUID
    analyzer_version: str
    lexical_chunk_count: int
    lexical_manifest_hash: str


@dataclass(frozen=True, slots=True)
class VectorRecordWrite:
    id: UUID
    index_chunk_id: UUID
    embedding_space_id: UUID
    embedding_dimension: int
    embedding: tuple[float, ...]
    representation_kind: str = "text"


@dataclass(frozen=True, slots=True)
class EvidenceUnitDraft:
    unit_key: str
    ordinal: int
    modality: ContentModality
    content: str
    token_count: int
    asset_key: str | None
    evidence_group_key: str | None
    related_unit_keys: tuple[str, ...]
    source_location: dict[str, Any]
    hierarchy: dict[str, Any]
    processing_metadata: dict[str, Any]
    required_representations: tuple[str, ...]
    embedding_text: str | None = None
    embedding_text_hash: str | None = None


@dataclass(frozen=True, slots=True)
class CompositeEvidenceDraft:
    units: tuple[EvidenceUnitDraft, ...]
    relations: tuple[ChunkAssetRelationDraft, ...]


class EmbeddingSpaceRole(StrEnum):
    TEXT_RETRIEVAL = "text_retrieval"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    CROSS_MODAL_RETRIEVAL = "cross_modal_retrieval"


class RepresentationKind(StrEnum):
    TEXT = "text"
    NATIVE_IMAGE = "native_image"
    CAPTION_TEXT = "caption_text"
    OCR_TEXT = "ocr_text"
    TABLE_TEXT = "table_text"
    TABLE_IMAGE = "table_image"


@dataclass(frozen=True, slots=True)
class IndexAssetWrite:
    id: UUID
    asset_key: str
    kind: str
    storage_uri: str
    media_type: str
    checksum_sha256: str
    width: int | None
    height: int | None
    source_location: dict[str, Any]
    processing_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IndexAssetSnapshot:
    id: UUID
    workspace_id: UUID
    kb_id: UUID
    document_id: UUID
    document_version_id: UUID
    indexed_document_version_id: UUID
    storage_uri: str
    media_type: str
    checksum_sha256: str
    size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class RetiredIndexTargetAssets:
    indexed_document_version_id: UUID
    assets: tuple[IndexAssetSnapshot, ...]


@dataclass(frozen=True, slots=True)
class IndexAssetContent:
    snapshot: IndexAssetSnapshot
    content: bytes


@dataclass(frozen=True, slots=True)
class IndexArtifactManifest:
    indexed_document_version_id: UUID
    source_checksum_sha256: str
    profile_fingerprint: str
    element_sequence_hash: str
    asset_manifest_hash: str
    unit_plan: tuple[dict[str, Any], ...]
    representation_matrix: tuple[dict[str, Any], ...]
    unit_count: int
    asset_count: int
    representation_count: int
    relation_plan: tuple[dict[str, Any], ...]
    relation_count: int
    relation_manifest_hash: str
    manifest_hash: str

    def __post_init__(self) -> None:
        if min(
            self.unit_count,
            self.asset_count,
            self.representation_count,
            self.relation_count,
        ) < 0:
            raise ValueError("artifact manifest counts must be non-negative")
        if self.unit_count != len(self.unit_plan):
            raise ValueError("artifact manifest unit count differs from its plan")
        if self.representation_count != len(self.representation_matrix):
            raise ValueError(
                "artifact manifest representation count differs from its matrix"
            )
        if self.relation_count != len(self.relation_plan):
            raise ValueError(
                "artifact manifest relation count differs from its plan"
            )
        for value in (
            self.source_checksum_sha256,
            self.profile_fingerprint,
            self.element_sequence_hash,
            self.asset_manifest_hash,
            self.relation_manifest_hash,
            self.manifest_hash,
        ):
            if len(value) != 64:
                raise ValueError("artifact manifest hash is invalid")


@dataclass(frozen=True, slots=True)
class IndexChunkAssetRelationWrite:
    id: UUID
    chunk_id: UUID
    visual_unit_id: UUID
    asset_id: UUID
    relation_type: str
    confidence_micros: int
    figure_label: str | None
    ordinal: int
    provenance: str
    evidence_group_key: str


@dataclass(frozen=True, slots=True)
class IndexChunkAssetRelationSnapshot:
    id: UUID
    workspace_id: UUID
    kb_id: UUID
    indexed_document_version_id: UUID
    index_revision_id: UUID
    chunk_id: UUID
    visual_unit_id: UUID
    asset_id: UUID
    relation_type: str
    confidence_micros: int
    figure_label: str | None
    ordinal: int
    provenance: str
    evidence_group_key: str
    document_id: UUID
    document_version_id: UUID
    chunk_ordinal: int
    chunk_content: str
    chunk_modality: str
    chunk_source_location: dict[str, Any]
    chunk_hierarchy: dict[str, Any]
    chunk_source_metadata: dict[str, Any]
    visual_ordinal: int
    visual_content: str
    visual_modality: str
    visual_source_location: dict[str, Any]
    visual_hierarchy: dict[str, Any]
    visual_source_metadata: dict[str, Any]
    asset_media_type: str
    asset_checksum_sha256: str
    asset_width: int | None
    asset_height: int | None


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class ImageEmbeddingInput:
    content: bytes
    media_type: str
    content_sha256: str


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


def stable_chunk_id(
    indexed_document_version_id: UUID,
    *,
    profile_fingerprint: str,
    unit_key: str,
) -> UUID:
    if not profile_fingerprint or not unit_key:
        raise ValueError("profile fingerprint and unit key must not be empty")
    return uuid5(indexed_document_version_id, f"{profile_fingerprint}:{unit_key}")


def stable_vector_id(
    embedding_space_id: UUID,
    index_chunk_id: UUID,
    representation_kind: str = "text",
) -> UUID:
    if not representation_kind:
        raise ValueError("representation kind must not be empty")
    identity = f"{embedding_space_id}:{index_chunk_id}"
    if representation_kind != "text":
        identity = f"{identity}:{representation_kind}"
    return uuid5(VECTOR_ID_NAMESPACE, identity)


def stable_asset_id(indexed_document_version_id: UUID, asset_key: str) -> UUID:
    if not asset_key:
        raise ValueError("asset key must not be empty")
    return uuid5(ASSET_ID_NAMESPACE, f"{indexed_document_version_id}:{asset_key}")


def stable_relation_id(
    indexed_document_version_id: UUID,
    chunk_id: UUID,
    asset_id: UUID,
    relation_type: str,
) -> UUID:
    if not relation_type:
        raise ValueError("relation type must not be empty")
    return uuid5(
        RELATION_ID_NAMESPACE,
        f"{indexed_document_version_id}:{chunk_id}:{asset_id}:{relation_type}",
    )


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
    if definition.normalization in {"l2", "client_l2_v1"}:
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


def normalize_embedding_vector(
    value: object,
    definition: EmbeddingSpaceDefinition,
    *,
    normalization_tolerance: float = 0.001,
    minimum_norm: float = 1e-12,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        not isinstance(item, bool) and isinstance(item, (int, float))
        for item in value
    ):
        raise IndexingExecutionError(
            ErrorCode.EMBEDDING_RESPONSE_INVALID,
            phase=IndexingPhase.EMBEDDING,
            diagnostic={"check": "numeric_sequence"},
        )
    vector = tuple(float(item) for item in value)
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
    if not all(math.isfinite(item) and abs(item) <= 3.4028235e38 for item in vector):
        raise IndexingExecutionError(
            ErrorCode.EMBEDDING_RESPONSE_INVALID,
            phase=IndexingPhase.EMBEDDING,
            diagnostic={"check": "finite_float32"},
        )
    norm = math.sqrt(sum(item * item for item in vector))
    if not math.isfinite(norm) or norm <= minimum_norm:
        raise IndexingExecutionError(
            ErrorCode.EMBEDDING_RESPONSE_INVALID,
            phase=IndexingPhase.EMBEDDING,
            diagnostic={"check": "nonzero_norm"},
        )
    if definition.normalization == "client_l2_v1":
        normalized = tuple(item / norm for item in vector)
        validate_embedding_vector(
            normalized,
            definition,
            normalization_tolerance=normalization_tolerance,
        )
        return normalized
    if definition.normalization == "l2":
        validate_embedding_vector(
            vector,
            definition,
            normalization_tolerance=normalization_tolerance,
        )
        return vector
    raise IndexingExecutionError(
        ErrorCode.EMBEDDING_RESPONSE_INVALID,
        phase=IndexingPhase.EMBEDDING,
        diagnostic={"check": "normalization_policy"},
    )
