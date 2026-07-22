"""Local PDF/DOCX multimodal partitioning with bounded temporary assets."""

from __future__ import annotations

import hashlib
import tempfile
from io import BytesIO
from pathlib import Path, PurePath
from typing import Any

from rag_kb.adapters.parser.docx_pictures import partition_docx_multimodal
from rag_kb.adapters.parser.langchain_unstructured import partition_with_unstructured
from rag_kb.adapters.parser.multimodal_elements import (
    bounded_image_asset,
    stable_element_key,
)
from rag_kb.document_processing import MULTIMODAL_PARSER_CONFIG, count_chunk_tokens
from rag_kb.domain import (
    ErrorCode,
    ParsedAssetDraft,
    ParsedDocument,
    ParsedElement,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
)


def partition_multimodal_with_unstructured(
    source: ParserSource,
    limits: ParserLimits,
    temp_root: Path | None = None,
) -> ParsedDocument:
    extension = PurePath(source.original_filename).suffix.lower()
    if extension == ".docx":
        return partition_docx_multimodal(
            source, limits, MULTIMODAL_PARSER_CONFIG["profile"]
        )
    if extension != ".pdf":
        parsed = partition_with_unstructured(source, limits)
        return _with_stable_keys(source, parsed)
    if source.media_type != "application/pdf":
        raise ParserExecutionError(ErrorCode.FILE_MEDIA_TYPE_MISMATCH)
    return _partition_pdf(source, limits, temp_root)


def _partition_pdf(
    source: ParserSource, limits: ParserLimits, temp_root: Path | None
) -> ParsedDocument:
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_NOT_CONFIGURED,
            diagnostic={"check": "unstructured_pdf_dependency"},
        ) from error
    checksum = hashlib.sha256(source.content).hexdigest()
    profile = MULTIMODAL_PARSER_CONFIG["profile"]
    elements: list[ParsedElement] = []
    assets: list[ParsedAssetDraft] = []
    extracted_characters = 0
    try:
        with tempfile.TemporaryDirectory(
            prefix="rag-kb-mm-", dir=temp_root
        ) as directory:
            raw_elements = partition_pdf(
                file=BytesIO(source.content),
                strategy="hi_res",
                infer_table_structure=True,
                include_page_breaks=True,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_output_dir=directory,
                extract_image_block_to_payload=False,
            )
            for raw in raw_elements:
                category = getattr(raw, "category", raw.__class__.__name__)
                text = str(raw).strip() if category != "PageBreak" else ""
                metadata = _metadata(raw)
                location = _source_location(metadata)
                hierarchy = _hierarchy(metadata)
                asset_key = None
                image_path = metadata.get("image_path")
                if category in {"Image", "Table"} and isinstance(image_path, str):
                    candidate = Path(image_path).resolve(strict=True)
                    if not candidate.is_relative_to(Path(directory).resolve(strict=True)):
                        raise ParserExecutionError(
                            ErrorCode.PARSER_OUTPUT_INVALID,
                            diagnostic={"check": "asset_temp_path"},
                        )
                    asset = bounded_image_asset(
                        candidate.read_bytes(),
                        kind="table_image" if category == "Table" else "pdf_image",
                        source_location=location,
                        limits=limits,
                    )
                    assets.append(asset)
                    asset_key = asset.asset_key
                table_html = metadata.get("text_as_html") if category == "Table" else None
                if table_html is not None and not isinstance(table_html, str):
                    table_html = None
                if table_html and len(table_html.encode("utf-8")) > limits.max_table_html_bytes:
                    raise ParserExecutionError(
                        ErrorCode.PARSER_RESOURCE_LIMIT,
                        diagnostic={"limit_name": "max_table_html_bytes", "limit": limits.max_table_html_bytes},
                    )
                extracted_characters += len(text)
                ordinal = len(elements)
                elements.append(
                    ParsedElement(
                        ordinal=ordinal,
                        text=text,
                        token_count=count_chunk_tokens(text) if text else 0,
                        category=category,
                        source_location=location,
                        hierarchy=hierarchy,
                        is_title=category == "Title",
                        is_table=category in {"Table", "TableChunk"},
                        element_key=stable_element_key(
                            checksum, profile, ordinal, category, location, text, asset_key
                        ),
                        asset_key=asset_key,
                        table_html=table_html,
                    )
                )
                _require_counts(elements, assets, extracted_characters, limits)
    except ParserExecutionError:
        raise
    except MemoryError:
        raise
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "unstructured_hi_res"},
        ) from error
    if not elements:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID, diagnostic={"check": "non_empty_elements"}
        )
    return ParsedDocument(tuple(elements), extracted_characters, tuple(assets))


def _with_stable_keys(source: ParserSource, parsed: ParsedDocument) -> ParsedDocument:
    checksum = hashlib.sha256(source.content).hexdigest()
    profile = MULTIMODAL_PARSER_CONFIG["profile"]
    elements = tuple(
        ParsedElement(
            ordinal=item.ordinal,
            text=item.text,
            token_count=item.token_count,
            category=item.category,
            source_location=item.source_location,
            hierarchy=item.hierarchy,
            is_title=item.is_title,
            is_table=item.is_table,
            element_key=stable_element_key(
                checksum,
                profile,
                item.ordinal,
                item.category,
                item.source_location,
                item.text,
                None,
            ),
        )
        for item in parsed.elements
    )
    return ParsedDocument(elements, parsed.extracted_character_count)


def _metadata(element: Any) -> dict[str, Any]:
    metadata = getattr(element, "metadata", None)
    value = metadata.to_dict() if metadata is not None and hasattr(metadata, "to_dict") else {}
    return value if isinstance(value, dict) else {}


def _source_location(metadata: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    if isinstance(metadata.get("page_number"), int):
        value["page_number"] = metadata["page_number"]
    coordinates = metadata.get("coordinates")
    if isinstance(coordinates, dict):
        points = coordinates.get("points")
        if isinstance(points, (list, tuple)):
            value["bbox_points"] = points
        system = coordinates.get("system")
        if isinstance(system, str):
            value["coordinate_system"] = system
    return value


def _hierarchy(metadata: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key in ("parent_id", "category_depth"):
        item = metadata.get(key)
        if isinstance(item, (str, int)) and not isinstance(item, bool):
            value[key] = item
    return value


def _require_counts(
    elements: list[ParsedElement],
    assets: list[ParsedAssetDraft],
    extracted_characters: int,
    limits: ParserLimits,
) -> None:
    checks = (
        ("max_units", len(elements), limits.max_units),
        ("max_assets", len(assets), limits.max_assets),
        ("max_total_asset_bytes", sum(len(item.content) for item in assets), limits.max_total_asset_bytes),
        ("max_extracted_characters", extracted_characters, limits.max_extracted_characters),
    )
    for name, observed, limit in checks:
        if observed > limit:
            raise ParserExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                diagnostic={"limit_name": name, "limit": limit},
            )
