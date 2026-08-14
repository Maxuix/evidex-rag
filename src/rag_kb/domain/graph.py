"""Framework-independent entity-graph extraction and retrieval facts."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID


GRAPH_EXTRACTOR_VERSION = "entity_graph_v4"
GRAPH_RETRIEVAL_PROFILE_VERSION = "graph_augmented_v1"
GRAPH_AUGMENTATION_VERSION = "entity_graph_v1"
GRAPH_MAX_ENTITIES = 64
GRAPH_MAX_RELATIONS = 128
GRAPH_MAX_PATHS = 20
GRAPH_MAX_NEIGHBORS_PER_ENTRY = 8
GRAPH_MAX_INTERMEDIATE_DEGREE = 16
GRAPH_MAX_QUERY_ENTITY_CANDIDATES = 8


class GraphConfigStatus(StrEnum):
    DISABLED = "disabled"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class GraphWorkKind(StrEnum):
    PREFLIGHT = "preflight"
    CHUNK = "chunk"
    FINALIZE = "finalize"


class GraphChunkResultStatus(StrEnum):
    EXTRACTED = "extracted"
    EMPTY = "empty"
    SKIPPED_PROTOCOL = "skipped_protocol"
    SKIPPED_RESOURCE = "skipped_resource"


class GraphEntityType(StrEnum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    PRODUCT = "product"
    SYSTEM = "system"
    DOCUMENT = "document"
    EVENT = "event"
    CONCEPT = "concept"


GRAPH_RELATION_GROUNDING_CODES = frozenset(
    {
        "relation_support_not_locatable",
        "relation_support_missing_subject",
        "relation_support_missing_object",
        "relation_support_missing_both",
    }
)


@dataclass(frozen=True, slots=True)
class GraphAdmissionStats:
    """Content-safe item-admission counters for one extractor response."""

    dropped_entity_count: int = 0
    dropped_relation_count: int = 0
    dropped_relation_grounding_count: int = 0
    entity_drop_codes: tuple[tuple[str, int], ...] = ()
    relation_drop_codes: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if (
            self.dropped_entity_count < 0
            or self.dropped_relation_count < 0
            or self.dropped_relation_grounding_count < 0
        ):
            raise ValueError("graph admission counts must be non-negative")
        if self.dropped_relation_grounding_count > self.dropped_relation_count:
            raise ValueError("graph grounding drops cannot exceed relation drops")
        if sum(count for _, count in self.entity_drop_codes) != self.dropped_entity_count:
            raise ValueError("graph entity drop codes do not match the entity count")
        if sum(count for _, count in self.relation_drop_codes) != self.dropped_relation_count:
            raise ValueError("graph relation drop codes do not match the relation count")
        if (
            sum(
                count
                for code, count in self.relation_drop_codes
                if code in GRAPH_RELATION_GROUNDING_CODES
            )
            != self.dropped_relation_grounding_count
        ):
            raise ValueError("graph grounding drop codes do not match the grounding count")
        if any(count < 1 for _, count in (*self.entity_drop_codes, *self.relation_drop_codes)):
            raise ValueError("graph admission drop codes must be positive")


class GraphProtocolError(ValueError):
    """The extractor returned a response outside the fixed graph protocol."""

    def __init__(
        self,
        code: str = "schema_invalid",
        *,
        admission: GraphAdmissionStats | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.admission = admission or GraphAdmissionStats()


class GraphResourceLimitError(ValueError):
    """A deterministic graph input or output resource limit was exceeded."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_SPACE_RE = re.compile(r"\s+")
_CONNECTOR_RE = re.compile(r"[\-_\u2010\u2011\u2012\u2013\u2014\u2212]+")
_CONNECTOR_CHARS = frozenset("-_\u2010\u2011\u2012\u2013\u2014\u2212")
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
    }
)
_EDGE_PUNCTUATION_RE = re.compile(
    r"^[\s\.,;:!?，。；：！？()\[\]{}\-]+|[\s\.,;:!?，。；：！？()\[\]{}\-]+$"
)


def normalize_entity_surface(value: str) -> str:
    """Apply the v2 identity normalization used by extraction and lookup."""

    if not isinstance(value, str):
        raise ValueError("entity surface must be text")
    normalized, _ = _normalized_surface_stream(value)
    if not normalized:
        raise ValueError("entity surface must not be empty")
    return normalized


def normalize_predicate(value: str) -> str:
    """Normalize predicate spelling without attempting synonym resolution."""

    if not isinstance(value, str):
        raise ValueError("predicate must be text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _CONNECTOR_RE.sub("-", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = _EDGE_PUNCTUATION_RE.sub("", normalized)
    if not normalized:
        raise ValueError("predicate must not be empty")
    return normalized


def entity_key(
    entity_type: GraphEntityType | str,
    surface: str,
    disambiguator: str | None = None,
) -> str:
    """Return the deterministic identity for one type/surface/disambiguator."""

    try:
        resolved_type = GraphEntityType(entity_type)
    except (TypeError, ValueError) as error:
        raise ValueError("entity type is unsupported") from error
    normalized_surface = normalize_entity_surface(surface)
    normalized_disambiguator = (
        normalize_entity_surface(disambiguator) if disambiguator else ""
    )
    material = "\x1f".join(
        (resolved_type.value, normalized_surface, normalized_disambiguator)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _NormalizedCharacter:
    value: str
    source_start: int
    source_end: int
    source_token: int


def _normalize_graph_token(value: str) -> str:
    """Shared character/token primitive; callers retain identity/location duties."""

    return unicodedata.normalize("NFKC", value).casefold().translate(_QUOTE_TRANSLATION)


def _normalized_surface_stream(
    value: str,
) -> tuple[str, tuple[_NormalizedCharacter, ...]]:
    characters: list[_NormalizedCharacter] = []
    index = 0
    token_id = 0
    while index < len(value):
        start = index
        if value[index].isspace():
            index += 1
            while index < len(value) and value[index].isspace():
                index += 1
        else:
            index += 1
            while index < len(value) and unicodedata.combining(value[index]):
                index += 1
        token = _normalize_graph_token(value[start:index])
        for character in token:
            mapped = "-" if character in _CONNECTOR_CHARS else character
            characters.append(_NormalizedCharacter(mapped, start, index, token_id))
        token_id += 1

    collapsed: list[_NormalizedCharacter] = []
    for character in characters:
        mapped = " " if character.value.isspace() else character.value
        current = _NormalizedCharacter(
            mapped,
            character.source_start,
            character.source_end,
            character.source_token,
        )
        if collapsed and mapped == " " and collapsed[-1].value == " ":
            previous = collapsed[-1]
            collapsed[-1] = _NormalizedCharacter(
                " ", previous.source_start, current.source_end, previous.source_token
            )
        elif collapsed and mapped == "-" and collapsed[-1].value == "-":
            previous = collapsed[-1]
            collapsed[-1] = _NormalizedCharacter(
                "-", previous.source_start, current.source_end, previous.source_token
            )
        else:
            collapsed.append(current)

    compact = [
        character
        for position, character in enumerate(collapsed)
        if not (
            character.value == " "
            and (
                (position > 0 and collapsed[position - 1].value == "-")
                or (position + 1 < len(collapsed) and collapsed[position + 1].value == "-")
            )
        )
    ]
    while compact and compact[0].value == " ":
        compact.pop(0)
    while compact and compact[-1].value == " ":
        compact.pop()
    return "".join(character.value for character in compact), tuple(compact)


def first_grounded_span(text: str, value: str, *, start_at: int = 0) -> tuple[int, int]:
    """Locate a narrow display-equivalent string and map it to the original text."""

    if not isinstance(value, str) or not value:
        raise GraphProtocolError("support_text_empty")
    normalized_text, mapping = _normalized_surface_stream(text)
    normalized_value, _ = _normalized_surface_stream(value)
    if not normalized_value:
        raise GraphProtocolError("support_text_empty")
    search_at = 0
    while True:
        match_at = normalized_text.find(normalized_value, search_at)
        if match_at < 0:
            raise GraphProtocolError("support_text_not_locatable")
        match_end = match_at + len(normalized_value)
        begins_token = match_at == 0 or (
            mapping[match_at - 1].source_token != mapping[match_at].source_token
        )
        ends_token = match_end == len(mapping) or (
            mapping[match_end - 1].source_token != mapping[match_end].source_token
        )
        source_start = mapping[match_at].source_start
        source_end = mapping[match_end - 1].source_end
        if begins_token and ends_token and source_start >= start_at:
            return source_start, source_end
        search_at = match_at + 1


@dataclass(frozen=True, slots=True)
class GraphEntityMention:
    mention_id: str
    ordinal: int
    entity_type: GraphEntityType
    surface: str
    normalized_surface: str
    disambiguator: str | None
    disambiguator_support_start: int | None
    disambiguator_support_end: int | None
    surface_start: int
    surface_end: int
    entity_key: str

    def __post_init__(self) -> None:
        if not self.mention_id.strip() or len(self.mention_id) > 64:
            raise ValueError("graph mention id is invalid")
        if self.ordinal < 0:
            raise ValueError("graph mention ordinal must be non-negative")
        if self.surface_start < 0 or self.surface_end <= self.surface_start:
            raise ValueError("graph surface span is invalid")
        if len(self.entity_key) != 64:
            raise ValueError("graph entity key is invalid")
        if (self.disambiguator_support_start is None) != (
            self.disambiguator_support_end is None
        ):
            raise ValueError("graph disambiguator span is incomplete")


@dataclass(frozen=True, slots=True)
class GraphRelationAssertion:
    relation_id: str
    ordinal: int
    subject_mention_id: str
    object_mention_id: str
    subject_entity_key: str
    object_entity_key: str
    predicate: str
    normalized_predicate: str
    support_start: int
    support_end: int

    def __post_init__(self) -> None:
        if not self.relation_id.strip() or len(self.relation_id) > 64:
            raise ValueError("graph relation id is invalid")
        if self.ordinal < 0:
            raise ValueError("graph relation ordinal must be non-negative")
        if self.subject_entity_key == self.object_entity_key:
            raise ValueError("graph self relations are not supported")
        if self.support_start < 0 or self.support_end <= self.support_start:
            raise ValueError("graph relation support span is invalid")
        if len(self.subject_entity_key) != 64 or len(self.object_entity_key) != 64:
            raise ValueError("graph relation entity key is invalid")


@dataclass(frozen=True, slots=True)
class GraphChunkExtraction:
    """Validated, locatable extraction ready for one short DB transaction."""

    result_status: GraphChunkResultStatus
    mentions: tuple[GraphEntityMention, ...] = ()
    relations: tuple[GraphRelationAssertion, ...] = ()
    result_hash: str = ""
    error_code: str | None = None
    admission: GraphAdmissionStats = GraphAdmissionStats()

    def __post_init__(self) -> None:
        if len(self.mentions) > GRAPH_MAX_ENTITIES:
            raise ValueError("graph entity limit exceeded")
        if len(self.relations) > GRAPH_MAX_RELATIONS:
            raise ValueError("graph relation limit exceeded")
        if self.result_status is GraphChunkResultStatus.EXTRACTED:
            if not self.result_hash or len(self.result_hash) != 64:
                raise ValueError("extracted graph result hash is invalid")
        elif self.mentions or self.relations:
            raise ValueError("skipped graph result cannot contain rows")


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
class GraphEntityCandidate:
    entity_key: str
    entity_type: GraphEntityType
    surface: str
    normalized_surface: str
    index_chunk_id: UUID
    indexed_document_version_id: UUID
    index_revision_id: UUID

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_type", GraphEntityType(self.entity_type))


@dataclass(frozen=True, slots=True)
class GraphEntityLookupQuery:
    workspace_id: UUID
    knowledge_base_id: UUID
    build_id: UUID
    normalized_surface: str
    prefix: bool = False
    limit: int = GRAPH_MAX_QUERY_ENTITY_CANDIDATES

    def __post_init__(self) -> None:
        if not self.normalized_surface:
            raise ValueError("graph lookup surface must not be empty")
        if not 1 <= self.limit <= GRAPH_MAX_QUERY_ENTITY_CANDIDATES:
            raise ValueError("graph lookup limit is invalid")


@dataclass(frozen=True, slots=True)
class GraphTraversalQuery:
    workspace_id: UUID
    knowledge_base_id: UUID
    build_id: UUID
    index_revision_id: UUID
    entry_entity_keys: tuple[str, ...]
    seed_chunk_ids: tuple[UUID, ...] = ()
    max_hops: int = 1
    max_paths: int = GRAPH_MAX_PATHS

    def __post_init__(self) -> None:
        if not self.entry_entity_keys and not self.seed_chunk_ids:
            raise ValueError("graph traversal needs an entry entity")
        if self.max_hops not in {1, 2}:
            raise ValueError("graph traversal supports one or two hops")
        if not 1 <= self.max_paths <= GRAPH_MAX_PATHS:
            raise ValueError("graph traversal path limit is invalid")


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


def allowed_graph_skips(eligible_chunk_count: int) -> int:
    if eligible_chunk_count < 0:
        raise ValueError("eligible chunk count must be non-negative")
    if eligible_chunk_count == 0:
        return 0
    return max(1, math.floor(eligible_chunk_count * 0.05))


def graph_extraction_hash(
    mentions: tuple[GraphEntityMention, ...],
    relations: tuple[GraphRelationAssertion, ...],
) -> str:
    """Hash only validated deterministic fields, never raw provider payload."""

    rows = [
        "|".join(
            (
                item.mention_id,
                str(item.ordinal),
                item.entity_type.value,
                item.surface,
                item.normalized_surface,
                item.disambiguator or "",
                str(item.surface_start),
                str(item.surface_end),
                item.entity_key,
            )
        )
        for item in mentions
    ]
    rows.append("--relations--")
    rows.extend(
        "|".join(
            (
                item.relation_id,
                str(item.ordinal),
                item.subject_mention_id,
                item.object_mention_id,
                item.subject_entity_key,
                item.object_entity_key,
                item.normalized_predicate,
                str(item.support_start),
                str(item.support_end),
            )
        )
        for item in relations
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
