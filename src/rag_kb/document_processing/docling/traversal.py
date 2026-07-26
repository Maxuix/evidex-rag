"""The single Docling item traversal shared by every downstream consumer."""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from typing import Any
import unicodedata

from docling_core.types.doc import DoclingDocument, DocItemLabel
from docling_core.types.doc.document import DocItem, PictureItem, TableItem

from rag_kb.domain import ErrorCode, ParserExecutionError, ParserLimits


_MAX_TITLE_CHARS = 256
_MAX_TITLE_DEPTH = 32
_MAX_TITLES = 64


class ItemKind(StrEnum):
    """The traversal's view of one Docling item, not a second document model."""

    TITLE = "title"
    SECTION_HEADER = "section_header"
    TEXT = "text"
    LIST_ITEM = "list_item"
    CODE = "code"
    FORMULA = "formula"
    TABLE = "table"
    PICTURE = "picture"
    CAPTION = "caption"
    FURNITURE = "furniture"
    UNKNOWN = "unknown"


_KIND_BY_LABEL = {
    DocItemLabel.TITLE: ItemKind.TITLE,
    DocItemLabel.SECTION_HEADER: ItemKind.SECTION_HEADER,
    DocItemLabel.TEXT: ItemKind.TEXT,
    DocItemLabel.PARAGRAPH: ItemKind.TEXT,
    DocItemLabel.REFERENCE: ItemKind.TEXT,
    DocItemLabel.FOOTNOTE: ItemKind.TEXT,
    DocItemLabel.HANDWRITTEN_TEXT: ItemKind.TEXT,
    DocItemLabel.LIST_ITEM: ItemKind.LIST_ITEM,
    DocItemLabel.CODE: ItemKind.CODE,
    DocItemLabel.FORMULA: ItemKind.FORMULA,
    DocItemLabel.TABLE: ItemKind.TABLE,
    DocItemLabel.DOCUMENT_INDEX: ItemKind.TABLE,
    DocItemLabel.PICTURE: ItemKind.PICTURE,
    DocItemLabel.CHART: ItemKind.PICTURE,
    DocItemLabel.CAPTION: ItemKind.CAPTION,
    DocItemLabel.PAGE_HEADER: ItemKind.FURNITURE,
    DocItemLabel.PAGE_FOOTER: ItemKind.FURNITURE,
}

#: Kinds whose text belongs in an assembled chunk. Captions stay out because
#: they are the author's description of a visual and are carried by relations;
#: pictures carry no text of their own.
TEXT_KINDS = frozenset(
    {
        ItemKind.TITLE,
        ItemKind.SECTION_HEADER,
        ItemKind.TEXT,
        ItemKind.LIST_ITEM,
        ItemKind.CODE,
        ItemKind.FORMULA,
        ItemKind.UNKNOWN,
    }
)


def classify_item(item: DocItem) -> ItemKind:
    """Map a Docling label onto the frozen traversal policy."""

    if isinstance(item, TableItem):
        return ItemKind.TABLE
    if isinstance(item, PictureItem):
        return ItemKind.PICTURE
    return _KIND_BY_LABEL.get(getattr(item, "label", None), ItemKind.UNKNOWN)


def iterate_body_items(document: DoclingDocument) -> Iterator[tuple[DocItem, int]]:
    """Yield body items in reading order, without groups and without furniture.

    ``iterate_items()`` already restricts itself to the body content layer and
    skips group nodes, so this stays the only place that decides what the
    project considers document content.
    """

    for item, level in document.iterate_items():
        if not isinstance(item, DocItem):
            continue
        if classify_item(item) is ItemKind.FURNITURE:
            continue
        yield item, level


def item_text(item: DocItem, document: DoclingDocument) -> str:
    """Return the canonical chunk text an item contributes, possibly empty."""

    if isinstance(item, TableItem):
        return table_text(item, document)
    if isinstance(item, PictureItem):
        return ""
    value = getattr(item, "text", None)
    return canonical_text(value) if isinstance(value, str) else ""


def table_text(table: TableItem, document: DoclingDocument) -> str:
    """Serialize a table through Docling instead of re-deriving cell layout."""

    try:
        return canonical_text(table.export_to_markdown(document))
    except ParserExecutionError:
        raise
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_table_serialization"},
        ) from error


def table_html(
    table: TableItem,
    document: DoclingDocument,
    limits: ParserLimits,
) -> str | None:
    """Return bounded structural HTML, or ``None`` when the table has no cells."""

    try:
        value = table.export_to_html(document, add_caption=False)
    except ParserExecutionError:
        raise
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_table_serialization"},
        ) from error
    value = value.strip()
    if not value:
        return None
    if len(value.encode("utf-8")) > limits.max_table_html_bytes:
        raise ParserExecutionError(
            ErrorCode.PARSER_RESOURCE_LIMIT,
            diagnostic={
                "limit_name": "max_table_html_bytes",
                "limit": limits.max_table_html_bytes,
            },
        )
    return value


def section_paths(
    document: DoclingDocument,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Build every item's heading path from traversal order and header levels.

    Docling already carries the heading structure, so the path is derived here
    once and shared by both chunking strategies instead of being recomputed as
    a per-element hierarchy.
    """

    paths: dict[str, tuple[dict[str, Any], ...]] = {}
    stack: list[tuple[int, dict[str, Any]]] = []
    for item, _level in iterate_body_items(document):
        kind = classify_item(item)
        if kind in {ItemKind.TITLE, ItemKind.SECTION_HEADER}:
            depth = 0 if kind is ItemKind.TITLE else _header_depth(item)
            while stack and stack[-1][0] >= depth:
                stack.pop()
            text = item_text(item, document)
            if text:
                stack.append((depth, {"depth": depth, "text": text[:_MAX_TITLE_CHARS]}))
        paths[item_ref(item)] = tuple(entry for _depth, entry in stack[-_MAX_TITLES:])
    return paths


def bounded_hierarchy(
    titles: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Return the persisted hierarchy shape, or an empty mapping."""

    return {"titles": [dict(entry) for entry in titles]} if titles else {}


def common_hierarchy(
    paths: tuple[tuple[dict[str, Any], ...], ...],
) -> dict[str, Any]:
    """Keep only the heading prefix every item in one chunk shares."""

    if not paths:
        return {}
    common = list(paths[0])
    for path in paths[1:]:
        length = 0
        for left, right in zip(common, path):
            if left != right:
                break
            length += 1
        common = common[:length]
    return bounded_hierarchy(tuple(common))


def _header_depth(item: DocItem) -> int:
    level = getattr(item, "level", None)
    if isinstance(level, bool) or not isinstance(level, int) or level < 1:
        return 1
    return min(level, _MAX_TITLE_DEPTH)


def item_ref(item: DocItem) -> str:
    """Return the Docling self reference used as the only item identity."""

    reference = getattr(item, "self_ref", "")
    if not isinstance(reference, str) or not reference:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "docling_item_self_ref"},
        )
    return reference


def parent_ref(item: DocItem) -> str | None:
    parent = getattr(item, "parent", None)
    reference = getattr(parent, "cref", None)
    return reference if isinstance(reference, str) and reference else None


def caption_refs(item: DocItem) -> tuple[str, ...]:
    captions = getattr(item, "captions", ())
    return tuple(
        reference.cref
        for reference in captions or ()
        if isinstance(getattr(reference, "cref", None), str) and reference.cref
    )


def canonical_text(value: str) -> str:
    return unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()
