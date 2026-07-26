"""Bounded scanned-surface detection for sources Docling cannot self-report.

A converted ``DoclingDocument`` records no OCR provenance, so it cannot say
whether a page's text came from a text layer or from recognition. This probe
reads only the PDF page structure — never its content — to decide which
surfaces need their rendered image as evidence.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import PurePath
from typing import Any

from pypdf import PdfReader

from rag_kb.domain import ErrorCode, ParserExecutionError, ParserSource


def scanned_surfaces(source: ParserSource) -> frozenset[int]:
    """Return the 1-based pages that carry an image and no text layer."""

    if PurePath(source.original_filename).suffix.lower() != ".pdf":
        return frozenset()
    try:
        reader = PdfReader(BytesIO(source.content))
        return frozenset(
            number
            for number, page in enumerate(reader.pages, start=1)
            if not (page.extract_text() or "").strip() and _has_image(page)
        )
    except ParserExecutionError:
        raise
    except Exception as error:
        raise ParserExecutionError(
            ErrorCode.PARSER_OUTPUT_INVALID,
            diagnostic={"check": "pdf_text_layer"},
        ) from error


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
