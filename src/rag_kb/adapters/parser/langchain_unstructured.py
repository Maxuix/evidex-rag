"""Local LangChain-Unstructured parsing and semantic chunk conversion."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import PurePath
from typing import Any

from rag_kb.document_processing import UNSTRUCTURED_CHUNKING_CONFIG
from rag_kb.domain import (
    ErrorCode,
    IndexChunkDraft,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
    ProcessedDocument,
)


_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def process_with_unstructured(
    source: ParserSource,
    limits: ParserLimits,
) -> ProcessedDocument:
    """Partition and chunk one bounded in-memory source using local Unstructured."""

    extension = PurePath(source.original_filename).suffix.lower()
    expected_media_type = _MEDIA_TYPES.get(extension)
    if expected_media_type is None:
        raise ParserExecutionError(ErrorCode.PARSER_NOT_CONFIGURED)
    if source.media_type != expected_media_type:
        raise ParserExecutionError(ErrorCode.FILE_MEDIA_TYPE_MISMATCH)

    try:
        from langchain_unstructured import UnstructuredLoader
    except ImportError as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_ISOLATION_FAILED,
            diagnostic={"check": "unstructured_dependency"},
        ) from error

    loader = UnstructuredLoader(
        file=BytesIO(source.content),
        metadata_filename=source.original_filename,
        content_type=source.media_type,
        partition_via_api=False,
        strategy="fast",
        include_page_breaks=True,
        chunking_strategy="by_title",
        max_characters=UNSTRUCTURED_CHUNKING_CONFIG["max_characters"],
        new_after_n_chars=UNSTRUCTURED_CHUNKING_CONFIG["new_after_n_chars"],
        overlap=UNSTRUCTURED_CHUNKING_CONFIG["overlap"],
        overlap_all=UNSTRUCTURED_CHUNKING_CONFIG["overlap_all"],
        combine_text_under_n_chars=UNSTRUCTURED_CHUNKING_CONFIG[
            "combine_text_under_n_chars"
        ],
        multipage_sections=UNSTRUCTURED_CHUNKING_CONFIG["multipage_sections"],
        include_orig_elements=True,
    )

    drafts: list[IndexChunkDraft] = []
    extracted_characters = 0
    try:
        documents = loader.lazy_load()
        for document in documents:
            text = _canonical_text(document.page_content)
            if not text:
                continue
            if len(drafts) >= limits.max_chunks:
                raise ParserExecutionError(
                    ErrorCode.PARSER_CHUNK_LIMIT_EXCEEDED,
                    diagnostic={
                        "limit_name": "max_chunks",
                        "limit": limits.max_chunks,
                    },
                )
            extracted_characters += len(text)
            if extracted_characters > limits.max_extracted_characters:
                raise ParserExecutionError(
                    ErrorCode.PARSER_RESOURCE_LIMIT,
                    diagnostic={
                        "limit_name": "max_extracted_characters",
                        "limit": limits.max_extracted_characters,
                    },
                )
            metadata = (
                document.metadata if isinstance(document.metadata, Mapping) else {}
            )
            original_elements = _original_elements(metadata, limits)
            source_location = _source_location(original_elements)
            hierarchy = _hierarchy(original_elements)
            processing_metadata = _processing_metadata(metadata, original_elements)
            _require_bounded_metadata(
                source_location,
                hierarchy,
                processing_metadata,
                limit=limits.max_metadata_bytes,
            )
            drafts.append(
                IndexChunkDraft(
                    ordinal=len(drafts),
                    text=text,
                    source_location=source_location,
                    hierarchy=hierarchy,
                    processing_metadata=processing_metadata,
                    content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
    except ParserExecutionError:
        raise
    except MemoryError:
        raise
    except BaseException as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "unstructured_loader"},
        ) from error

    if not drafts:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "non_empty_chunks"},
        )
    return ProcessedDocument(
        chunks=tuple(drafts),
        extracted_character_count=extracted_characters,
    )


def _canonical_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    )
    return normalized.strip()


def _original_elements(
    metadata: Mapping[str, Any],
    limits: ParserLimits,
) -> tuple[Any, ...]:
    encoded = metadata.get("orig_elements")
    if encoded is None:
        return ()
    if not isinstance(encoded, str):
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "orig_elements_type"},
        )
    encoded_size = len(encoded.encode("utf-8"))
    if encoded_size > limits.max_metadata_bytes:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_metadata_bytes",
                "limit": limits.max_metadata_bytes,
            },
        )
    try:
        from unstructured.staging.base import elements_from_base64_gzipped_json

        return tuple(elements_from_base64_gzipped_json(encoded))
    except BaseException as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "orig_elements_decode"},
        ) from error


def _source_location(elements: Iterable[Any]) -> dict[str, Any]:
    located: list[tuple[int, Mapping[str, Any]]] = []
    for element in elements:
        metadata = _element_metadata(element)
        page = metadata.get("page_number")
        if isinstance(page, int) and not isinstance(page, bool) and page > 0:
            located.append((page, metadata))
    if not located:
        return {}

    pages = sorted({page for page, _ in located})
    result: dict[str, Any] = {
        "page_start": pages[0],
        "page_end": pages[-1],
    }
    if len(pages) == 1:
        coordinates = _combined_coordinates(
            metadata.get("coordinates") for _, metadata in located
        )
        if coordinates is not None:
            result["coordinates"] = coordinates
    return result


def _combined_coordinates(values: Iterable[object]) -> dict[str, Any] | None:
    points: list[tuple[float, float]] = []
    system: str | None = None
    width: float | None = None
    height: float | None = None
    for value in values:
        if not isinstance(value, Mapping):
            continue
        candidate_system = value.get("system")
        candidate_width = _finite_number(value.get("layout_width"))
        candidate_height = _finite_number(value.get("layout_height"))
        candidate_points = value.get("points")
        if (
            not isinstance(candidate_system, str)
            or not _SAFE_VALUE.fullmatch(candidate_system)
            or candidate_width is None
            or candidate_height is None
            or not isinstance(candidate_points, (list, tuple))
        ):
            continue
        if system is None:
            system, width, height = (
                candidate_system,
                candidate_width,
                candidate_height,
            )
        elif (
            candidate_system != system
            or candidate_width != width
            or candidate_height != height
        ):
            return None
        for point in candidate_points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            x, y = _finite_number(point[0]), _finite_number(point[1])
            if x is not None and y is not None:
                points.append((x, y))
    if not points or system is None or width is None or height is None:
        return None
    minimum_x = min(point[0] for point in points)
    maximum_x = max(point[0] for point in points)
    minimum_y = min(point[1] for point in points)
    maximum_y = max(point[1] for point in points)
    return {
        "points": [
            [round(minimum_x, 4), round(minimum_y, 4)],
            [round(minimum_x, 4), round(maximum_y, 4)],
            [round(maximum_x, 4), round(maximum_y, 4)],
            [round(maximum_x, 4), round(minimum_y, 4)],
        ],
        "system": system,
        "layout_width": round(width, 4),
        "layout_height": round(height, 4),
    }


def _hierarchy(elements: Iterable[Any]) -> dict[str, Any]:
    titles: list[dict[str, Any]] = []
    for element in elements:
        if type(element).__name__ != "Title":
            continue
        title = _canonical_text(str(element))
        if not title:
            continue
        metadata = _element_metadata(element)
        depth = metadata.get("category_depth")
        titles.append(
            {
                "depth": (
                    depth
                    if isinstance(depth, int)
                    and not isinstance(depth, bool)
                    and 0 <= depth <= 32
                    else 0
                ),
                "text": title[:256],
            }
        )
        if len(titles) == 64:
            break
    return {"titles": titles} if titles else {}


def _processing_metadata(
    metadata: Mapping[str, Any],
    elements: Iterable[Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "integration": "langchain-unstructured",
        "category": _safe_metadata_value(metadata.get("category")),
    }
    languages = metadata.get("languages")
    if isinstance(languages, (list, tuple)):
        safe_languages = [
            item
            for item in languages
            if isinstance(item, str) and _SAFE_VALUE.fullmatch(item)
        ][:16]
        if safe_languages:
            result["languages"] = safe_languages
    element_types = sorted(
        {
            name
            for element in elements
            if (name := _safe_metadata_value(type(element).__name__)) is not None
        }
    )
    if element_types:
        result["element_types"] = element_types[:32]
    return {key: value for key, value in result.items() if value is not None}


def _element_metadata(element: Any) -> Mapping[str, Any]:
    metadata = getattr(element, "metadata", None)
    if metadata is None or not hasattr(metadata, "to_dict"):
        return {}
    value = metadata.to_dict()
    return value if isinstance(value, Mapping) else {}


def _safe_metadata_value(value: object) -> str | None:
    return value if isinstance(value, str) and _SAFE_VALUE.fullmatch(value) else None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _require_bounded_metadata(
    source_location: dict[str, Any],
    hierarchy: dict[str, Any],
    processing_metadata: dict[str, Any],
    *,
    limit: int,
) -> None:
    observed = len(
        json.dumps(
            {
                "source_location": source_location,
                "hierarchy": hierarchy,
                "processing_metadata": processing_metadata,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    if observed > limit:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={"limit_name": "max_metadata_bytes", "limit": limit},
        )
