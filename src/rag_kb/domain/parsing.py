"""Framework-independent file-admission and parser facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_kb.domain.errors import ErrorCode


@dataclass(frozen=True, slots=True)
class AdmissionLimits:
    max_bytes: int = 10 * 1024 * 1024
    max_lines: int = 200_000
    max_archive_entries: int = 10_000
    max_expanded_bytes: int = 100 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ParserLimits:
    max_chunks: int = 20_000
    max_extracted_characters: int = 5_000_000
    max_metadata_bytes: int = 65_536
    wall_seconds: float = 60.0
    cpu_seconds: int = 45
    memory_bytes: int = 4 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AdmittedFile:
    original_filename: str
    extension: str
    media_type: str
    size_bytes: int
    line_count: int | None


@dataclass(frozen=True, slots=True)
class ParserSource:
    original_filename: str
    media_type: str
    content: bytes


@dataclass(frozen=True, slots=True)
class IndexChunkDraft:
    ordinal: int
    text: str
    token_count: int
    source_location: dict[str, Any]
    hierarchy: dict[str, Any]
    processing_metadata: dict[str, Any]
    content_sha256: str


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    chunks: tuple[IndexChunkDraft, ...]
    extracted_character_count: int


class FileAdmissionError(ValueError):
    """A safe, stable synchronous upload rejection."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        limit: int | None = None,
        observed: int | None = None,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.limit = limit
        self.observed = observed


class ParserExecutionError(RuntimeError):
    """A content-safe isolated parser failure for later durable persistence."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        phase: str = "parsing",
        diagnostic: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.phase = phase
        self.diagnostic = dict(diagnostic or {})
