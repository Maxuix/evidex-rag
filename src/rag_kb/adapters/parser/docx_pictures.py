"""Bounded OOXML relationship extraction for DOCX text, tables, and pictures."""

from __future__ import annotations

import hashlib
import html
from io import BytesIO
from typing import Any

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

from rag_kb.document_processing import count_chunk_tokens
from rag_kb.domain import (
    ErrorCode,
    ParsedAssetDraft,
    ParsedDocument,
    ParsedElement,
    ParserExecutionError,
    ParserLimits,
    ParserSource,
)
from rag_kb.adapters.parser.multimodal_elements import (
    bounded_image_asset,
    stable_element_key,
)


def partition_docx_multimodal(
    source: ParserSource, limits: ParserLimits, profile: str
) -> ParsedDocument:
    try:
        document = Document(BytesIO(source.content))
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_CRASHED, diagnostic={"check": "docx_open"}
        ) from error

    checksum = hashlib.sha256(source.content).hexdigest()
    elements: list[ParsedElement] = []
    assets: list[ParsedAssetDraft] = []
    extracted_characters = 0
    body = document.element.body
    for block_ordinal, child in enumerate(body.iterchildren()):
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            category = _paragraph_category(paragraph)
            if text:
                extracted_characters += len(text)
                location = {"block_ordinal": block_ordinal}
                elements.append(
                    _element(
                        checksum,
                        profile,
                        len(elements),
                        category,
                        text,
                        location,
                        hierarchy={"style": paragraph.style.name if paragraph.style else None},
                    )
                )
            for picture_ordinal, blob in enumerate(_picture_blobs(paragraph)):
                location = {
                    "block_ordinal": block_ordinal,
                    "picture_ordinal": picture_ordinal,
                }
                asset = bounded_image_asset(
                    blob,
                    kind="docx_picture",
                    source_location=location,
                    limits=limits,
                )
                assets.append(asset)
                elements.append(
                    _element(
                        checksum,
                        profile,
                        len(elements),
                        "Image",
                        "",
                        location,
                        hierarchy={},
                        asset_key=asset.asset_key,
                    )
                )
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            rows = tuple(tuple(cell.text.strip() for cell in row.cells) for row in table.rows)
            text = "\n".join("\t".join(row) for row in rows).strip()
            table_html = _table_html(rows)
            if len(table_html.encode("utf-8")) > limits.max_table_html_bytes:
                raise ParserExecutionError(
                    ErrorCode.PARSER_RESOURCE_LIMIT,
                    diagnostic={
                        "limit_name": "max_table_html_bytes",
                        "limit": limits.max_table_html_bytes,
                    },
                )
            extracted_characters += len(text)
            elements.append(
                _element(
                    checksum,
                    profile,
                    len(elements),
                    "Table",
                    text,
                    {"block_ordinal": block_ordinal},
                    hierarchy={},
                    table_html=table_html,
                )
            )
        if len(elements) > limits.max_units:
            raise ParserExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                diagnostic={"limit_name": "max_units", "limit": limits.max_units},
            )
        if len(assets) > limits.max_assets:
            raise ParserExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                diagnostic={"limit_name": "max_assets", "limit": limits.max_assets},
            )
    _validate_totals(elements, assets, extracted_characters, limits)
    return ParsedDocument(tuple(elements), extracted_characters, tuple(assets))


def _paragraph_category(paragraph: Paragraph) -> str:
    style = paragraph.style.name.lower() if paragraph.style else ""
    if style.startswith("heading") or style == "title":
        return "Title"
    if "caption" in style:
        return "FigureCaption"
    return "NarrativeText"


def _picture_blobs(paragraph: Paragraph) -> tuple[bytes, ...]:
    blobs: list[bytes] = []
    for blip in paragraph._p.xpath(".//*[local-name()='blip']"):
        relationship_id = blip.get(qn("r:embed"))
        if not relationship_id:
            continue
        part = paragraph.part.related_parts.get(relationship_id)
        blob = getattr(part, "blob", None)
        if isinstance(blob, bytes):
            blobs.append(blob)
    return tuple(blobs)


def _element(
    checksum: str,
    profile: str,
    ordinal: int,
    category: str,
    text: str,
    source_location: dict[str, Any],
    hierarchy: dict[str, Any],
    *,
    asset_key: str | None = None,
    table_html: str | None = None,
) -> ParsedElement:
    return ParsedElement(
        ordinal=ordinal,
        text=text,
        token_count=count_chunk_tokens(text) if text else 0,
        category=category,
        source_location=source_location,
        hierarchy=hierarchy,
        is_title=category == "Title",
        is_table=category == "Table",
        element_key=stable_element_key(
            checksum, profile, ordinal, category, source_location, text, asset_key
        ),
        asset_key=asset_key,
        table_html=table_html,
    )


def _table_html(rows: tuple[tuple[str, ...], ...]) -> str:
    rendered = []
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 else "td"
        rendered.append(
            "<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in row) + "</tr>"
        )
    return "<table>" + "".join(rendered) + "</table>"


def _validate_totals(
    elements: list[ParsedElement],
    assets: list[ParsedAssetDraft],
    extracted_characters: int,
    limits: ParserLimits,
) -> None:
    if not elements:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID, diagnostic={"check": "non_empty_elements"}
        )
    if extracted_characters > limits.max_extracted_characters:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_extracted_characters",
                "limit": limits.max_extracted_characters,
            },
        )
    total = sum(len(asset.content) for asset in assets)
    if total > limits.max_total_asset_bytes:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_total_asset_bytes",
                "limit": limits.max_total_asset_bytes,
            },
        )
