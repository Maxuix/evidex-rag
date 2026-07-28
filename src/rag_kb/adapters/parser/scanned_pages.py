"""Bounded scanned-surface detection for sources Docling cannot self-report.

A converted ``DoclingDocument`` records no OCR provenance, so it cannot say
whether a page's text came from a text layer or from recognition. This probe
stops content-stream traversal at the first non-whitespace text fragment and
then inspects page resources only when deciding whether an image surface is
required as evidence.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import PurePath
from typing import Any

from pypdf import PdfReader

from rag_kb.domain import ErrorCode, ParserExecutionError, ParserSource


class _TextLayerFound(BaseException):
    """Stop pypdf content-stream traversal after the first visible text."""


def scanned_surfaces(source: ParserSource) -> frozenset[int]:
    """Return the 1-based pages that carry an image and no text layer."""

    if PurePath(source.original_filename).suffix.lower() != ".pdf":
        return frozenset()
    try:
        reader = PdfReader(BytesIO(source.content))
        return frozenset(
            number
            for number, page in enumerate(reader.pages, start=1)
            if not _has_text_layer(page) and _has_image(page)
        )
    except ParserExecutionError:
        raise
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "pdf_text_layer"},
        ) from error


def _has_text_layer(page: Any) -> bool:
    def stop_on_text(text: Any, *_: Any) -> None:
        if isinstance(text, str) and text.strip():
            raise _TextLayerFound

    try:
        extracted = page.extract_text(visitor_text=stop_on_text)
    except _TextLayerFound:
        return True
    return bool((extracted or "").strip())


def _has_image(page: Any) -> bool:
    resources = page.get("/Resources")
    if resources is None:
        return False
    xobjects = resources.get_object().get("/XObject")
    if xobjects is None:
        return False
    return any(
        str(value.get_object().get("/Subtype")) == "/Image"
        for value in xobjects.get_object().values()
    )
