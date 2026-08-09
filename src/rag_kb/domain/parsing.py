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
    pdf_num_threads: int = 1
    pdf_ocr_batch_size: int = 1
    pdf_layout_batch_size: int = 1
    pdf_table_batch_size: int = 1
    pdf_segment_pages: int = 20
    pdf_segment_timeout_seconds: float = 180.0
    pdf_total_timeout_seconds: float = 1_800.0

    def __post_init__(self) -> None:
        if (
            self.pdf_num_threads <= 0
            or self.pdf_ocr_batch_size <= 0
            or self.pdf_layout_batch_size <= 0
            or self.pdf_table_batch_size <= 0
            or self.pdf_segment_pages <= 0
            or self.pdf_segment_timeout_seconds <= 0
            or self.pdf_total_timeout_seconds < self.pdf_segment_timeout_seconds
        ):
            raise ValueError("PDF parser limits are invalid")


class ParsingPreset(StrEnum):
    TEXT_LOCAL_V1 = "text_local_v1"
    MULTIMODAL_LOCAL_V2 = "multimodal_local_v2"


class ParserProfile(StrEnum):
    DOCLING_TEXT_LOCAL_V1 = "docling_text_local_v1"
    DOCLING_MULTIMODAL_LOCAL_V2 = "docling_multimodal_local_v2"
    DOCLING_TEXT_LOCAL_V2 = "docling_text_local_v2"
    DOCLING_MULTIMODAL_LOCAL_V3 = "docling_multimodal_local_v3"

    @property
    def preset(self) -> ParsingPreset:
        if self in {
            self.DOCLING_MULTIMODAL_LOCAL_V2,
            self.DOCLING_MULTIMODAL_LOCAL_V3,
        }:
            return ParsingPreset.MULTIMODAL_LOCAL_V2
        return ParsingPreset.TEXT_LOCAL_V1

    @property
    def uses_balanced_pdf_runtime(self) -> bool:
        return self in {
            self.DOCLING_TEXT_LOCAL_V2,
            self.DOCLING_MULTIMODAL_LOCAL_V3,
        }


@dataclass(frozen=True, slots=True)
class ParserProgress:
    """Content-safe, bounded progress for one PDF conversion segment."""

    stage: str
    total_pages: int
    completed_pages: int
    segment_number: int
    segment_count: int
    page_from: int
    page_to: int
    stage_pages: tuple[tuple[str, int], ...] = ()
    ocr_pages: int = 0
    ocr_regions: int = 0
    table_candidates: int = 0
    elapsed_ms: int = 0
    child_peak_rss_bytes: int | None = None

    def __post_init__(self) -> None:
        stage_names = tuple(name for name, _ in self.stage_pages)
        if (
            not self.stage
            or self.total_pages <= 0
            or not 0 <= self.completed_pages <= self.total_pages
            or self.segment_number <= 0
            or self.segment_count < self.segment_number
            or self.page_from <= 0
            or self.page_to < self.page_from
            or len(set(stage_names)) != len(stage_names)
            or any(not name or value < 0 for name, value in self.stage_pages)
            or self.ocr_pages < 0
            or self.ocr_regions < 0
            or self.table_candidates < 0
            or self.elapsed_ms < 0
            or (
                self.child_peak_rss_bytes is not None
                and self.child_peak_rss_bytes < 0
            )
        ):
            raise ValueError("parser progress is invalid")

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "pdf_parsing_progress_v1",
            "stage": self.stage,
            "total_pages": self.total_pages,
            "completed_pages": self.completed_pages,
            "segment_number": self.segment_number,
            "segment_count": self.segment_count,
            "page_from": self.page_from,
            "page_to": self.page_to,
            "stage_pages": dict(self.stage_pages),
            "ocr_pages": self.ocr_pages,
            "ocr_regions": self.ocr_regions,
            "table_candidates": self.table_candidates,
            "elapsed_ms": self.elapsed_ms,
        }
        if self.child_peak_rss_bytes is not None:
            payload["child_peak_rss_bytes"] = self.child_peak_rss_bytes
        return payload


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
