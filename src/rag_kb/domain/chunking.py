"""Framework-independent semantic chunking facts and invariants."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID



_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ChunkingPreset(StrEnum):
    STRUCTURAL_BALANCED_V2 = "structural_balanced_v2"
    SEMANTIC_BALANCED_V1 = "semantic_balanced_v1"


class ChunkingStrategyKind(StrEnum):
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


class ChunkBoundaryReason(StrEnum):
    SEMANTIC = "semantic"
    PAGE = "page"
    TABLE = "table"
    SECTION = "section"
    #: A non-prose structural block such as code or a formula.
    BLOCK = "block"
    MAX_TOKENS = "max_tokens"


@dataclass(frozen=True, slots=True)
class ChunkBoundary:
    after_unit_ordinal: int
    reason: ChunkBoundaryReason
    score_micros: int | None = None

    def __post_init__(self) -> None:
        if self.after_unit_ordinal < 0:
            raise ValueError("boundary ordinal must be non-negative")
        if self.reason is ChunkBoundaryReason.SEMANTIC:
            if self.score_micros is None or self.score_micros < 0:
                raise ValueError("semantic boundary requires a non-negative score")
        elif self.score_micros is not None:
            raise ValueError("only semantic boundaries may contain a score")


@dataclass(frozen=True, slots=True)
class IndexChunkPlan:
    indexed_document_version_id: UUID
    source_checksum_sha256: str
    profile_fingerprint: str
    unit_sequence_hash: str
    unit_count: int
    chunk_count: int
    boundaries: tuple[ChunkBoundary, ...]
    plan_hash: str

    def __post_init__(self) -> None:
        for value in (
            self.source_checksum_sha256,
            self.profile_fingerprint,
            self.unit_sequence_hash,
            self.plan_hash,
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError("chunk plan hashes must be lowercase SHA-256 hex")
        if self.unit_count < 1:
            raise ValueError("chunk plan must describe at least one unit")
        if self.chunk_count != len(self.boundaries) + 1:
            raise ValueError("chunk count must equal boundary count plus one")
        ordinals = tuple(item.after_unit_ordinal for item in self.boundaries)
        if ordinals != tuple(sorted(set(ordinals))):
            raise ValueError("boundary ordinals must be strictly increasing")
        if ordinals and ordinals[-1] >= self.unit_count - 1:
            raise ValueError("the final unit cannot be followed by a boundary")
