"""Local PDF/DOCX multimodal partitioning with bounded temporary assets."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path, PurePath
from typing import Any

from pypdf import PdfReader

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
    scanned_pages = _scanned_page_numbers(source.content)
    if len(scanned_pages) > limits.max_assets:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={"limit_name": "max_assets", "limit": limits.max_assets},
        )
    elements: list[ParsedElement] = []
    assets: list[ParsedAssetDraft] = []
    extracted_characters = 0
    try:
        with tempfile.TemporaryDirectory(
            prefix="rag-kb-mm-", dir=temp_root
        ) as directory:
            temporary_root = Path(directory).resolve(strict=True)
            source_path = temporary_root / "source.pdf"
            source_path.write_bytes(source.content)
            page_assets: dict[int, ParsedAssetDraft] = {}
            for page_number in scanned_pages:
                page_content = _render_pdf_page(
                    source_path, temporary_root, page_number, limits
                )
                page_asset = bounded_image_asset(
                    page_content,
                    kind="page_image",
                    source_location={"page_number": page_number},
                    limits=limits,
                )
                page_assets[page_number] = replace(
                    page_asset,
                    processing_metadata={
                        **page_asset.processing_metadata,
                        "source": "pdftoppm",
                        "scanned_page": True,
                    },
                )
                assets.append(page_assets[page_number])
                _require_counts(elements, assets, extracted_characters, limits)
            raw_elements = partition_pdf(
                file=BytesIO(source.content),
                strategy="hi_res",
                infer_table_structure=True,
                include_page_breaks=True,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_output_dir=directory,
                extract_image_block_to_payload=False,
            )
            inserted_page_images: set[int] = set()
            scanned_page_cursor = 0

            def append_element(
                *,
                category: str,
                text: str,
                location: dict[str, Any],
                hierarchy: dict[str, Any],
                asset_key: str | None = None,
                table_html: str | None = None,
                is_title: bool = False,
                is_table: bool = False,
            ) -> None:
                ordinal = len(elements)
                elements.append(
                    ParsedElement(
                        ordinal=ordinal,
                        text=text,
                        token_count=count_chunk_tokens(text) if text else 0,
                        category=category,
                        source_location=location,
                        hierarchy=hierarchy,
                        is_title=is_title,
                        is_table=is_table,
                        element_key=stable_element_key(
                            checksum,
                            profile,
                            ordinal,
                            category,
                            location,
                            text,
                            asset_key,
                        ),
                        asset_key=asset_key,
                        table_html=table_html,
                    )
                )

            def append_page_image(page_number: int) -> None:
                if page_number in inserted_page_images:
                    return
                page_asset = page_assets[page_number]
                append_element(
                    category="PageImage",
                    text="",
                    location={"page_number": page_number},
                    hierarchy={},
                    asset_key=page_asset.asset_key,
                )
                inserted_page_images.add(page_number)

            def append_page_images_through(page_number: int) -> None:
                nonlocal scanned_page_cursor
                while (
                    scanned_page_cursor < len(scanned_pages)
                    and scanned_pages[scanned_page_cursor] <= page_number
                ):
                    append_page_image(scanned_pages[scanned_page_cursor])
                    scanned_page_cursor += 1

            for raw in raw_elements:
                category = getattr(raw, "category", raw.__class__.__name__)
                text = str(raw).strip() if category != "PageBreak" else ""
                metadata = _metadata(raw)
                location = _source_location(metadata)
                hierarchy = _hierarchy(metadata)
                page_number = location.get("page_number")
                if isinstance(page_number, int):
                    append_page_images_through(page_number)
                asset_key = None
                image_path = metadata.get("image_path")
                if category in {"Image", "Table"} and isinstance(image_path, str):
                    candidate = Path(image_path).resolve(strict=True)
                    if not candidate.is_relative_to(temporary_root):
                        raise ParserExecutionError(
                            ErrorCode.PARSER_OUTPUT_INVALID,
                            diagnostic={"check": "asset_temp_path"},
                        )
                    remaining_asset_bytes = limits.max_total_asset_bytes - sum(
                        len(item.content) for item in assets
                    )
                    if candidate.stat().st_size > remaining_asset_bytes:
                        raise ParserExecutionError(
                            ErrorCode.PARSER_RESOURCE_LIMIT,
                            diagnostic={
                                "limit_name": "max_total_asset_bytes",
                                "limit": limits.max_total_asset_bytes,
                            },
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
                normalized_category = (
                    "OCRText"
                    if isinstance(page_number, int)
                    and page_number in page_assets
                    and category
                    not in {"Image", "Table", "TableChunk", "PageBreak"}
                    else category
                )
                append_element(
                    category=normalized_category,
                    text=text,
                    location=location,
                    hierarchy=hierarchy,
                    asset_key=asset_key,
                    table_html=table_html,
                    is_title=normalized_category == "Title",
                    is_table=normalized_category in {"Table", "TableChunk"},
                )
                _require_counts(elements, assets, extracted_characters, limits)
            while scanned_page_cursor < len(scanned_pages):
                append_page_image(scanned_pages[scanned_page_cursor])
                scanned_page_cursor += 1
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


def _scanned_page_numbers(content: bytes) -> tuple[int, ...]:
    try:
        reader = PdfReader(BytesIO(content))
        return tuple(
            page_number
            for page_number, page in enumerate(reader.pages, start=1)
            if not (page.extract_text() or "").strip() and _page_has_image(page)
        )
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "pdf_text_layer"},
        ) from error


def _page_has_image(page: Any) -> bool:
    resources = page.get("/Resources")
    if resources is None:
        return False
    resources = resources.get_object()
    xobjects = resources.get("/XObject")
    if xobjects is None:
        return False
    xobjects = xobjects.get_object()
    return any(
        str(value.get_object().get("/Subtype")) == "/Image"
        for value in xobjects.values()
    )


def _render_pdf_page(
    source_path: Path,
    temporary_root: Path,
    page_number: int,
    limits: ParserLimits,
) -> bytes:
    output_prefix = temporary_root / f"page-image-{page_number}"
    try:
        completed = subprocess.run(
            (
                "pdftoppm",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-singlefile",
                "-png",
                str(source_path),
                str(output_prefix),
            ),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_NOT_CONFIGURED,
            diagnostic={"check": "pdftoppm"},
        ) from error
    if completed.returncode != 0:
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED,
            diagnostic={"check": "page_image_render"},
        )
    output_path = output_prefix.with_suffix(".png")
    try:
        resolved = output_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "page_image_output"},
        ) from error
    if not resolved.is_relative_to(temporary_root):
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "page_image_path"},
        )
    if resolved.stat().st_size > limits.max_total_asset_bytes:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_total_asset_bytes",
                "limit": limits.max_total_asset_bytes,
            },
        )
    return resolved.read_bytes()


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
