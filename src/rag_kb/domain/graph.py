"""Framework-independent Graphiti build and retrieval facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


GRAPH_EXTRACTOR_VERSION = "graphiti_v4"
# Migration 0014 uses this marker for disabled legacy Graph configuration
# rows.  It is readable for configuration display only; it is not a serving
# or build-compatible extractor version.
GRAPH_LEGACY_EXTRACTOR_VERSION = "graphiti_v1"
GRAPH_HISTORICAL_EXTRACTOR_VERSIONS = frozenset({"graphiti_v3"})
GRAPH_SUPPORTED_EXTRACTOR_VERSIONS = frozenset(
    {GRAPH_EXTRACTOR_VERSION, *GRAPH_HISTORICAL_EXTRACTOR_VERSIONS}
)
GENERIC_GRAPH_SCHEMA_PROFILE_KEY = "generic_open_domain_v1"
SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY = "software_knowledge_v1"
ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY = "enterprise_knowledge_v1"
# These literals are the persistence identity contract.  Generic and Software
# also appear in migration/backfill history.  Registry tests keep every value
# aligned with its canonical manifest without making the domain import the
# graph package during application bootstrap.
GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST = (
    "3b351f4e2c601226f922d12b60d4c9f98a4770f4ec04e94b08f5a3f0d021eaf0"
)
SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST = (
    "6cae93809f060d21f0c85ba04cde955abdc5259fd93e1b1757fb7445d51eaf38"
)
ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST = (
    "478c9f04b819a158977a2e90f397960478d77c41a125c8254519491001a6903c"
)
GRAPH_RETRIEVAL_PROFILE_VERSION = "graphiti_path_augmented_v3"
GRAPH_AUGMENTATION_VERSION = "graphiti_path_v3"
GRAPH_MAX_PATHS = 20
GRAPH_MAX_HOPS = 3
GRAPH_WORK_LEASE_SECONDS = 180
GRAPH_WORK_HEARTBEAT_SECONDS = 30


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
        if not self.path_id or not 1 <= len(self.hops) <= GRAPH_MAX_HOPS:
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
    schema_profile_key: str = GENERIC_GRAPH_SCHEMA_PROFILE_KEY
    schema_profile_digest: str = GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST
    active_build_schema_profile_key: str | None = None
    active_build_schema_profile_digest: str | None = None

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
    schema_profile_key: str = SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY
    schema_profile_digest: str = SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", GraphitiBuildStatus(self.status))
        if not self.group_id or not self.serving_chunk_digest:
            raise ValueError("Graphiti build identity is incomplete")
        if not self.schema_profile_key or len(self.schema_profile_digest) != 64:
            raise ValueError("Graphiti schema profile identity is incomplete")
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
        # Keep the production default at K=8 while allowing the evaluator's
        # bounded 8/16/32/64 layer comparisons through the same domain port.
        if not self.group_id or not self.query.strip() or not 1 <= self.limit <= 64:
            raise ValueError("Graphiti search query is invalid")


@dataclass(frozen=True, slots=True)
class GraphitiEdgeResult:
    edge_uuid: str
    fact: str
    episode_uuids: tuple[str, ...]
    rank: int
    source_entity_uuid: str = ""
    source_entity_name: str = ""
    target_entity_uuid: str = ""
    target_entity_name: str = ""
    relation_type: str = ""

    def __post_init__(self) -> None:
        if not self.edge_uuid or self.rank < 1:
            raise ValueError("Graphiti edge result is invalid")
        object.__setattr__(
            self,
            "episode_uuids",
            tuple(dict.fromkeys(self.episode_uuids)),
        )

    @property
    def endpoint_uuids(self) -> tuple[str, str]:
        return self.source_entity_uuid, self.target_entity_uuid


@dataclass(frozen=True, slots=True)
class GraphitiPathResult:
    """One bounded Graphiti path before source chunks are hydrated."""

    path_id: str
    entry_entity_uuid: str
    hops: tuple[GraphitiEdgeResult, ...]
    rank: int
    seed_entry: bool

    def __post_init__(self) -> None:
        if (
            not self.path_id
            or not self.entry_entity_uuid
            or not 1 <= len(self.hops) <= GRAPH_MAX_HOPS
            or self.rank < 1
        ):
            raise ValueError("Graphiti path result is invalid")
        if any(
            not hop.source_entity_uuid
            or not hop.target_entity_uuid
            or hop.source_entity_uuid == hop.target_entity_uuid
            for hop in self.hops
        ):
            raise ValueError("Graphiti path endpoints are invalid")
        first_endpoints = set(self.hops[0].endpoint_uuids)
        if self.entry_entity_uuid not in first_endpoints:
            raise ValueError("Graphiti path entry is not grounded")
        visited = set(first_endpoints)
        previous = first_endpoints
        for hop in self.hops[1:]:
            endpoints = set(hop.endpoint_uuids)
            if len(previous & endpoints) != 1 or len(visited & endpoints) != 1:
                raise ValueError("Graphiti path is disconnected or cyclic")
            visited.update(endpoints)
            previous = endpoints


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
    lease_token: UUID | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", GraphWorkKind(self.kind))
        if self.kind is GraphWorkKind.CHUNK and self.chunk is None:
            raise ValueError("chunk work item requires a chunk")
        if self.kind is not GraphWorkKind.CHUNK and self.chunk is not None:
            raise ValueError("non-chunk graph work item cannot carry a chunk")
        if self.lease_token is not None and not self.lease_owner:
            raise ValueError("leased graph work item requires an owner")


@dataclass(frozen=True, slots=True)
class GraphTraversalResult:
    resolved_active_revision_id: UUID
    paths: tuple[GraphPathCandidate, ...] = ()
    chunks: tuple[GraphChunkEvidence, ...] = ()
    rejected_path_count: int = 0
    mapped_episode_ids: tuple[str, ...] = ()
    mapped_episode_chunks: tuple[tuple[str, UUID], ...] = ()

    def __post_init__(self) -> None:
        if len(self.paths) > GRAPH_MAX_PATHS or self.rejected_path_count < 0:
            raise ValueError("graph traversal result bound is invalid")
        object.__setattr__(
            self,
            "mapped_episode_ids",
            tuple(dict.fromkeys(str(item) for item in self.mapped_episode_ids)),
        )
        mapped_episode_chunks = tuple(dict.fromkeys(self.mapped_episode_chunks))
        if (
            any(
                not episode_uuid
                or not isinstance(chunk_id, UUID)
                for episode_uuid, chunk_id in mapped_episode_chunks
            )
            or len({episode_uuid for episode_uuid, _ in mapped_episode_chunks})
            != len(mapped_episode_chunks)
            or {episode_uuid for episode_uuid, _ in mapped_episode_chunks}
            != set(self.mapped_episode_ids)
        ):
            raise ValueError("graph traversal episode mapping is invalid")
        object.__setattr__(self, "mapped_episode_chunks", mapped_episode_chunks)
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
    three_hop_path_count: int = 0
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
            self.three_hop_path_count,
            self.rejected_path_count,
            self.bundle_count,
            self.protocol_skipped_count,
            self.resource_skipped_count,
        )
        if any(value < 0 for value in counters):
            raise ValueError("graph debug counters must be non-negative")
        if len(self.paths) > GRAPH_MAX_PATHS or len(self.bundles) > GRAPH_MAX_PATHS:
            raise ValueError("graph debug path bound exceeded")
