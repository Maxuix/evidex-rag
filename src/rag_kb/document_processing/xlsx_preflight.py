"""Bound worksheet expansion before openpyxl creates cell objects."""

from __future__ import annotations

import posixpath
import re
from xml.parsers import expat
from zipfile import ZipFile

from rag_kb.document_processing.resource_preflight import (
    ResourcePreflightContentError, ResourcePreflightLimitError,
)

_CELL = re.compile(r"\$?([A-Za-z]{1,3})\$?([1-9][0-9]{0,6})\Z")
_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships}"


def _limit(name: str, maximum: int, observed: int) -> None:
    if observed > maximum:
        raise ResourcePreflightLimitError(name, maximum, observed=observed)


def _coordinates(value: str) -> tuple[int, int]:
    match = _CELL.fullmatch(value)
    if match is None:
        raise ResourcePreflightContentError("xlsx_cell_reference")
    column = 0
    for letter in match[1].upper():
        column = column * 26 + ord(letter) - ord("A") + 1
    row = int(match[2])
    if row > 1_048_576 or column > 16_384:
        raise ResourcePreflightContentError("xlsx_cell_reference")
    return row, column


def _scan(archive: ZipFile, name: str, start, end=lambda _: None) -> None:
    parser = expat.ParserCreate(namespace_separator="}")

    def reject_dtd(*_):
        raise ResourcePreflightContentError("xlsx_xml_dtd")

    parser.StartDoctypeDeclHandler = reject_dtd
    depth = 0

    def on_start(tag, attrs):
        nonlocal depth
        depth += 1
        _limit("max_xlsx_xml_depth", 128, depth)
        start(tag, attrs)

    def on_end(tag):
        nonlocal depth
        end(tag)
        depth -= 1

    parser.StartElementHandler = on_start
    parser.EndElementHandler = on_end
    try:
        with archive.open(name) as stream:
            while chunk := stream.read(64 * 1024):
                parser.Parse(chunk, False)
            parser.Parse(b"", True)
    except (expat.ExpatError, KeyError, ValueError) as error:
        if isinstance(error, (ResourcePreflightContentError, ResourcePreflightLimitError)):
            raise
        raise ResourcePreflightContentError("xlsx_xml") from error


def validate_xlsx_structure(
    archive: ZipFile, *, max_cells: int, max_columns: int,
    max_sheets: int, max_xml_bytes: int,
) -> None:
    """Check actual cell spans and merges; never trust worksheet dimension hints.

    Follow workbook relationships, so a worksheet at a nonstandard part name
    cannot bypass the check. SAX parsing retains only counters and coordinates.
    """
    entries = archive.infolist()
    if len({entry.filename for entry in entries}) != len(entries):
        raise ResourcePreflightContentError("xlsx_duplicate_part")
    names = {entry.filename for entry in entries}
    rels = "xl/_rels/workbook.xml.rels"
    # A hollow package cannot make openpyxl load any worksheets.
    if rels not in names:
        return
    sheets: list[str] = []

    def relationship(tag, attrs):
        if (
            tag != _REL_NS + "Relationship"
            or not attrs.get("Type", "").endswith("/worksheet")
        ):
            return
        if attrs.get("TargetMode") == "External":
            raise ResourcePreflightContentError("xlsx_external_worksheet")
        target = attrs.get("Target", "")
        name = posixpath.normpath(
            target.lstrip("/") if target.startswith("/")
            else posixpath.join("xl", target)
        )
        if not target or "\\" in target or name.startswith("../") or name not in names:
            raise ResourcePreflightContentError("xlsx_worksheet_target")
        sheets.append(name)
        _limit("max_xlsx_sheets", max_sheets, len(sheets))

    # Bound XML (including shared strings) before parsing or loading a workbook.
    xml_bytes = sum(
        entry.file_size for entry in entries
        if entry.filename.endswith((".xml", ".rels"))
    )
    _limit("max_xlsx_xml_bytes", max_xml_bytes, xml_bytes)
    _scan(archive, rels, relationship)
    xml_bytes += sum(
        archive.getinfo(name).file_size for name in set(sheets)
        if not name.endswith((".xml", ".rels"))
    )
    _limit("max_xlsx_xml_bytes", max_xml_bytes, xml_bytes)
    expanded = cell_count = merge_area = 0
    for name in sheets:
        row = column = 0
        current_cell = None
        bounds = None

        def include(first, last):
            nonlocal bounds
            r1, c1 = first
            r2, c2 = last
            if r2 < r1 or c2 < c1:
                raise ResourcePreflightContentError("xlsx_merge_reference")
            if bounds is None:
                bounds = (r1, c1, r2, c2)
            else:
                bounds = (min(bounds[0], r1), min(bounds[1], c1), max(bounds[2], r2), max(bounds[3], c2))
            width = bounds[3] - bounds[1] + 1
            area = width * (bounds[2] - bounds[0] + 1)
            _limit("max_xlsx_columns", max_columns, width)
            _limit("max_xlsx_cells", max_cells, expanded + area)

        def start(tag, attrs):
            nonlocal row, column, current_cell, cell_count, merge_area
            if tag == _SHEET_NS + "row":
                row = int(attrs.get("r", row + 1))
                column = 0
                if not 1 <= row <= 1_048_576:
                    raise ResourcePreflightContentError("xlsx_row_reference")
            elif tag == _SHEET_NS + "c":
                column += 1
                current_cell = _coordinates(attrs["r"]) if "r" in attrs else (row, column)
                column = current_cell[1]
                if not 1 <= current_cell[0] <= 1_048_576 or column > 16_384:
                    raise ResourcePreflightContentError("xlsx_cell_reference")
                cell_count += 1
                _limit("max_xlsx_cells", max_cells, cell_count)
            elif current_cell and tag in {_SHEET_NS + key for key in ("v", "f", "is")}:
                include(current_cell, current_cell)
            elif tag == _SHEET_NS + "mergeCell":
                parts = attrs.get("ref", "").split(":")
                if len(parts) not in (1, 2):
                    raise ResourcePreflightContentError("xlsx_merge_reference")
                first, last = _coordinates(parts[0]), _coordinates(parts[-1])
                include(first, last)
                merge_area += (last[0] - first[0] + 1) * (last[1] - first[1] + 1)
                _limit("max_xlsx_cells", max_cells, merge_area)

        def end(tag):
            nonlocal current_cell
            if tag == _SHEET_NS + "c":
                current_cell = None

        _scan(archive, name, start, end)
        if bounds is not None:
            expanded += (bounds[2] - bounds[0] + 1) * (bounds[3] - bounds[1] + 1)
