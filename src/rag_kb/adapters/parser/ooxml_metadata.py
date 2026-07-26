"""Bounded OOXML metadata Docling's public document model does not carry.

A converted spreadsheet exposes sheet ordinals but not sheet names, so a
citation could only say "sheet 2". This reads the workbook part directly — the
package structure only, never cell content — to recover the names.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import PurePath
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from rag_kb.domain import ErrorCode, ParserExecutionError, ParserSource


_WORKBOOK_PART = "xl/workbook.xml"
_SPREADSHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
#: The workbook part only lists sheets; a larger one is malformed, not richer.
_MAX_WORKBOOK_BYTES = 1_048_576
_MAX_SHEETS = 256
_MAX_LABEL_CHARS = 128


def worksheet_labels(source: ParserSource) -> dict[int, str]:
    """Map 1-based sheet ordinals to their names, in workbook order."""

    if PurePath(source.original_filename).suffix.lower() != ".xlsx":
        return {}
    try:
        with ZipFile(BytesIO(source.content)) as archive:
            entry = archive.getinfo(_WORKBOOK_PART)
            if entry.file_size > _MAX_WORKBOOK_BYTES:
                raise ParserExecutionError(
                    ErrorCode.PARSER_RESOURCE_LIMIT,
                    diagnostic={
                        "limit_name": "max_workbook_part_bytes",
                        "limit": _MAX_WORKBOOK_BYTES,
                    },
                )
            with archive.open(entry) as part:
                payload = part.read(_MAX_WORKBOOK_BYTES + 1)
        root = ElementTree.fromstring(payload)
    except ParserExecutionError:
        raise
    except (BadZipFile, KeyError, OSError, ValueError, ElementTree.ParseError) as error:
        # Admission already required the part to exist, so an unreadable
        # workbook is corrupt input rather than a missing capability.
        raise ParserExecutionError(
            ErrorCode.FILE_CONTENT_INVALID,
            diagnostic={"check": "ooxml_workbook_part"},
        ) from error

    labels: dict[int, str] = {}
    for ordinal, sheet in enumerate(
        root.iterfind(f"{_SPREADSHEET_NS}sheets/{_SPREADSHEET_NS}sheet"), start=1
    ):
        if ordinal > _MAX_SHEETS:
            raise ParserExecutionError(
                ErrorCode.PARSER_RESOURCE_LIMIT,
                diagnostic={"limit_name": "max_worksheets", "limit": _MAX_SHEETS},
            )
        name = (sheet.get("name") or "").strip()
        if name:
            labels[ordinal] = name[:_MAX_LABEL_CHARS]
    if not labels:
        raise ParserExecutionError(
            ErrorCode.FILE_CONTENT_INVALID,
            diagnostic={"check": "ooxml_worksheet_names"},
        )
    return labels
