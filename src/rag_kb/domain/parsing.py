"""Framework-independent file-admission and parser facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_kb.domain.errors import ErrorCode


@dataclass(frozen=True, slots=True)
class AdmissionLimits:
    max_bytes: int = 10 * 1024 * 1024
    max_lines: int = 200_000


@dataclass(frozen=True, slots=True)
class ParserLimits:
    max_chunks: int = 20_000
    wall_seconds: float = 30.0
    cpu_seconds: int = 20
    memory_bytes: int = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AdmittedFile:
    original_filename: str
    extension: str
    media_type: str
    size_bytes: int
    line_count: int


@dataclass(frozen=True, slots=True)
class ParserSource:
    original_filename: str
    media_type: str
    content: bytes


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    text: str
    start_character: int
    end_character: int
    heading_hierarchy: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    canonical_text: str
    blocks: tuple[ParsedBlock, ...]


@dataclass(frozen=True, slots=True)
class IndexChunkDraft:
    ordinal: int
    text: str
    start_character: int
    end_character: int
    heading_hierarchy: tuple[str, ...]
    content_sha256: str


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    parsed: ParsedDocument
    chunks: tuple[IndexChunkDraft, ...]


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
