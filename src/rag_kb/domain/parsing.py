"""Framework-independent file-admission and parser facts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rag_kb.domain.errors import ErrorCode


@dataclass(frozen=True, slots=True)
class AdmissionLimits:
    max_bytes: int = 10 * 1024 * 1024
    max_markdown_bundle_bytes: int = 20 * 1024 * 1024
    max_lines: int = 200_000
    max_csv_columns: int = 1_024
    max_csv_cells: int = 200_000
    max_archive_entries: int = 10_000
    max_expanded_bytes: int = 100 * 1024 * 1024
    max_assets: int = 1_000
    max_image_width: int = 16_384
    max_image_height: int = 16_384
    max_image_pixels: int = 40_000_000
    max_total_image_pixels: int = 80_000_000


@dataclass(frozen=True, slots=True)
class ParserLimits:
    max_file_size: int = 10 * 1024 * 1024
    max_markdown_bundle_size: int = 20 * 1024 * 1024
    max_num_pages: int = 500
    document_timeout_seconds: float = 600.0
    max_csv_columns: int = 1_024
    max_csv_cells: int = 200_000
    max_docling_items: int = 20_000
    max_chunks: int = 20_000
    max_extracted_characters: int = 5_000_000
    max_metadata_bytes: int = 65_536
    max_assets: int = 1_000
    max_total_asset_bytes: int = 100 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_total_image_pixels: int = 80_000_000
    max_image_width: int = 16_384
    max_image_height: int = 16_384
    max_ocr_characters: int = 2_000_000
    max_ocr_tokens: int = 500_000
    max_caption_tokens: int = 512
    max_table_html_bytes: int = 1_048_576
    max_units: int = 20_000
    max_representations: int = 60_000
    max_relations: int = 50_000
    max_relations_per_chunk: int = 32


class ParsingPreset(StrEnum):
    TEXT_LOCAL_V1 = "text_local_v1"
    MULTIMODAL_LOCAL_V2 = "multimodal_local_v2"


class ContentModality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


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
class ParsedAssetDraft:
    asset_key: str
    kind: str
    media_type: str
    content: bytes
    content_sha256: str
    width: int | None
    height: int | None
    source_location: dict[str, Any]
    processing_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChunkAssemblyDraft:
    """One assembled chunk, positioned by its index in the assembly tuple.

    ``item_refs`` records which Docling items the chunk consumed; it never holds
    Docling item objects, so an assembly cannot extend a ``DoclingDocument``
    lifetime into planning or embedding.
    """

    text: str
    token_count: int
    item_refs: tuple[str, ...]
    source_location: dict[str, Any]
    hierarchy: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SemanticUnit:
    """A bounded analysis unit derived directly from Docling items."""

    ordinal: int
    text: str
    token_count: int
    item_refs: tuple[str, ...]
    source_location: dict[str, Any]
    hard_boundary_before: str | None


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
        check: str | None = None,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.limit = limit
        self.observed = observed
        self.check = check


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
