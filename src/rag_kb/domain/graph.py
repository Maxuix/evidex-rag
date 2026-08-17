"""Framework-independent Graphiti build and retrieval facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


GRAPH_EXTRACTOR_VERSION = "graphiti_v1"
GRAPH_RETRIEVAL_PROFILE_VERSION = "graphiti_edge_augmented_v1"
GRAPH_AUGMENTATION_VERSION = "graphiti_edge_v1"
GRAPH_MAX_PATHS = 20


class GraphConfigStatus(StrEnum):
    DISABLED = "disabled"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class GraphitiBuildStatus(StrEnum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class GraphWorkKind(StrEnum):
    PREFLIGHT = "preflight"
    CHUNK = "chunk"
    FINALIZE = "finalize"




@dataclass(frozen=True, slots=True)
class GraphPathHop:
    subject_entity_key: str
    object_entity_key: str
    predicate: str
    normalized_predicate: str
    relation_id: UUID
    source_chunk_id: UUID
    source_index_revision_id: UUID
    source_location: dict[str, Any]
    support_count: int = 1

    def __post_init__(self) -> None:
        if self.support_count < 1:
            raise ValueError("graph support count must be positive")
        object.__setattr__(self, "source_location", dict(self.source_location))


@dataclass(frozen=True, slots=True)
class GraphChunkEvidence:
    """Current serving chunk facts used to materialize a graph path."""

    workspace_id: UUID
    knowledge_base_id: UUID
    index_revision_id: UUID
    index_chunk_id: UUID
    indexed_document_version_id: UUID
    document_id: UUID
    document_version_id: UUID
    ordinal: int
    text: str
    source_location: dict[str, Any]
    hierarchy: dict[str, Any]
    source_metadata: dict[str, Any]
    modality: str
    evidence_group_key: str | None
    document_display_name: str | None
    document_original_filename: str | None
    excluded: bool = False

    def __post_init__(self) -> None:
        if self.ordinal < 0 or not self.text:
            raise ValueError("graph evidence chunk is invalid")
        if self.modality not in {"text", "table"}:
            raise ValueError("graph evidence chunk modality is unsupported")
        if self.excluded:
            raise ValueError("excluded graph evidence cannot be materialized")
        object.__setattr__(self, "source_location", dict(self.source_location))
        object.__setattr__(self, "hierarchy", dict(self.hierarchy))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))


@dataclass(frozen=True, slots=True)
class GraphPathCandidate:
    path_id: str
    entry_entity_key: str
    hops: tuple[GraphPathHop, ...]
    anchor_chunk_id: UUID
    rank: int
    seed_entry: bool

    def __post_init__(self) -> None:
        if not self.path_id or not 1 <= len(self.hops) <= 2:
            raise ValueError("graph path hop count is invalid")
        if self.rank < 1:
            raise ValueError("graph path rank must be positive")

    @property
    def hop_count(self) -> int:
        return len(self.hops)

    @property
    def source_chunk_ids(self) -> tuple[UUID, ...]:
        return tuple(
            dict.fromkeys(
                (self.anchor_chunk_id,)
                + tuple(hop.source_chunk_id for hop in self.hops)
            )
        )

    @property
    def support_counts(self) -> tuple[int, ...]:
        return tuple(hop.support_count for hop in self.hops)


@dataclass(frozen=True, slots=True)
class GraphEvidenceBundle:
    path: GraphPathCandidate
    chunk_ids: tuple[UUID, ...]

    def __post_init__(self) -> None:
        if not self.chunk_ids:
            raise ValueError("graph evidence bundle must contain chunks")
        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("graph evidence bundle chunks must be unique")
        if not set(self.path.source_chunk_ids).issubset(self.chunk_ids):
            raise ValueError("graph evidence bundle is missing a path source")

    @property
    def path_id(self) -> str:
        return self.path.path_id


@dataclass(frozen=True, slots=True)
class GraphConfigSnapshot:
    workspace_id: UUID
    knowledge_base_id: UUID
    status: GraphConfigStatus
    build_id: UUID
    chat_profile_revision_id: UUID | None
    extractor_version: str
    preflight_extractor_version: str | None
    last_error_code: str | None
    eligible_chunk_count: int = 0
    processed_chunk_count: int = 0
    extracted_chunk_count: int = 0
    empty_chunk_count: int = 0
    protocol_skipped_count: int = 0
    resource_skipped_count: int = 0
    active_build_id: UUID | None = None
    group_id: str | None = None
    index_revision_id: UUID | None = None
    embedding_profile_revision_id: UUID | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", GraphConfigStatus(self.status))
        counts = (
            self.eligible_chunk_count,
            self.processed_chunk_count,
            self.extracted_chunk_count,
            self.empty_chunk_count,
            self.protocol_skipped_count,
            self.resource_skipped_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("graph config counts must be non-negative")
        if self.processed_chunk_count > self.eligible_chunk_count:
            raise ValueError("graph processed count exceeds eligible count")


@dataclass(frozen=True, slots=True)
class GraphitiBuildSnapshot:
    workspace_id: UUID
    knowledge_base_id: UUID
    build_id: UUID
    group_id: str
    status: GraphitiBuildStatus
    index_revision_id: UUID
    serving_chunk_digest: str
    expected_episode_count: int
    chat_profile_revision_id: UUID
    embedding_profile_revision_id: UUID
    embedding_model: str
    embedding_dimension: int
    extractor_version: str
    superseded_by: UUID | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", GraphitiBuildStatus(self.status))
        if not self.group_id or not self.serving_chunk_digest:
            raise ValueError("Graphiti build identity is incomplete")
        if self.expected_episode_count < 0:
            raise ValueError("Graphiti expected episode count is invalid")
        if not 64 <= self.embedding_dimension <= 4096:
            raise ValueError("Graphiti embedding dimension is invalid")


@dataclass(frozen=True, slots=True)
class GraphitiSearchQuery:
    workspace_id: UUID
    knowledge_base_id: UUID
    build_id: UUID
    group_id: str
    query: str
    limit: int = 10

    def __post_init__(self) -> None:
        if not self.group_id or not self.query.strip() or not 1 <= self.limit <= 40:
            raise ValueError("Graphiti search query is invalid")


@dataclass(frozen=True, slots=True)
class GraphitiEdgeResult:
    edge_uuid: str
    fact: str
    episode_uuids: tuple[str, ...]
    rank: int

    def __post_init__(self) -> None:
        if not self.edge_uuid or self.rank < 1:
            raise ValueError("Graphiti edge result is invalid")
        object.__setattr__(
            self,
            "episode_uuids",
            tuple(dict.fromkeys(self.episode_uuids)),
        )


@dataclass(frozen=True, slots=True)
class GraphChunkSource:
    workspace_id: UUID
    knowledge_base_id: UUID
    build_id: UUID
    index_chunk_id: UUID
    index_revision_id: UUID
    indexed_document_version_id: UUID
    document_id: UUID
    document_version_id: UUID
    ordinal: int
    modality: str
    content: str
    content_hash: str
    source_location: dict[str, Any]
    hierarchy: dict[str, Any]
    source_metadata: dict[str, Any]
    reference_time: datetime | None = None
    excluded: bool = False

    def __post_init__(self) -> None:
        if self.modality not in {"text", "table"}:
            raise ValueError("graph chunk source modality is unsupported")
        if self.ordinal < 0 or not self.content:
            raise ValueError("graph chunk source is invalid")
        object.__setattr__(self, "source_location", dict(self.source_location))
        object.__setattr__(self, "hierarchy", dict(self.hierarchy))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))


@dataclass(frozen=True, slots=True)
class GraphWorkItem:
    kind: GraphWorkKind
    config: GraphConfigSnapshot
    chunk: GraphChunkSource | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", GraphWorkKind(self.kind))
        if self.kind is GraphWorkKind.CHUNK and self.chunk is None:
            raise ValueError("chunk work item requires a chunk")
        if self.kind is not GraphWorkKind.CHUNK and self.chunk is not None:
            raise ValueError("non-chunk graph work item cannot carry a chunk")




@dataclass(frozen=True, slots=True)
class GraphTraversalResult:
    resolved_active_revision_id: UUID
    paths: tuple[GraphPathCandidate, ...] = ()
    chunks: tuple[GraphChunkEvidence, ...] = ()
    rejected_path_count: int = 0

    def __post_init__(self) -> None:
        if len(self.paths) > GRAPH_MAX_PATHS or self.rejected_path_count < 0:
            raise ValueError("graph traversal result bound is invalid")
        chunk_ids = {item.index_chunk_id for item in self.chunks}
        if any(
            source_chunk_id not in chunk_ids
            for path in self.paths
            for source_chunk_id in path.source_chunk_ids
        ):
            raise ValueError("graph traversal result is missing a source chunk")


@dataclass(frozen=True, slots=True)
class GraphDebug:
    dense_seed_count: int = 0
    lexical_seed_count: int = 0
    fused_seed_count: int = 0
    query_entity_count: int = 0
    one_hop_path_count: int = 0
    two_hop_path_count: int = 0
    rejected_path_count: int = 0
    bundle_count: int = 0
    protocol_skipped_count: int = 0
    resource_skipped_count: int = 0
    paths: tuple[GraphPathCandidate, ...] = ()
    bundles: tuple[GraphEvidenceBundle, ...] = ()

    def __post_init__(self) -> None:
        counters = (
            self.dense_seed_count,
            self.lexical_seed_count,
            self.fused_seed_count,
            self.query_entity_count,
            self.one_hop_path_count,
            self.two_hop_path_count,
            self.rejected_path_count,
            self.bundle_count,
            self.protocol_skipped_count,
            self.resource_skipped_count,
        )
        if any(value < 0 for value in counters):
            raise ValueError("graph debug counters must be non-negative")
        if len(self.paths) > GRAPH_MAX_PATHS or len(self.bundles) > GRAPH_MAX_PATHS:
            raise ValueError("graph debug path bound exceeded")
