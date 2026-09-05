"""Bounded row packing shared by current structural and semantic profiles."""
from __future__ import annotations

import re

from docling_core.types.doc.document import TableItem

from rag_kb.document_processing.tokenization import count_chunk_tokens, split_by_tokens
from rag_kb.domain import ErrorCode, ParserExecutionError


def table_header_rows(table: TableItem) -> int:
    return max(
        (cell.end_row_offset_idx for cell in table.data.table_cells if cell.column_header),
        default=1,
    )


def table_chunks(
    text: str, *, maximum: int = 800, prefix: str = "", header_rows: int = 1,
) -> tuple[str, ...]:
    """Repeat the source header/context while keeping complete rows when possible."""
    prefix = prefix.strip()
    def attach(body: str) -> str:
        return f"{prefix}\n\n{body}" if prefix else body

    if count_chunk_tokens(attach(text)) <= maximum:
        return (attach(text),)
    lines = text.splitlines()
    separator = next((i for i, line in enumerate(lines)
                      if '-' in line and re.fullmatch(r"[|:\-\s]+", line)), None)
    header_end = min(len(lines), separator + max(1, header_rows)) if separator is not None else 0
    header = "\n".join(lines[:header_end])
    rows = lines[header_end:]
    def render(selected: list[str]) -> str:
        return attach("\n".join(([header] if header else []) + selected))

    budget = maximum - count_chunk_tokens(render([])) - 2
    if budget < 1:
        raise ParserExecutionError(ErrorCode.PARSER_RESOURCE_LIMIT,
                                   diagnostic={"limit_name": "table_header_tokens", "limit": maximum})
    parts: list[str] = []
    current: list[str] = []
    for row in rows:
        if current and count_chunk_tokens(render([*current, row])) > maximum:
            parts.append(render(current))
            current = []
        if count_chunk_tokens(render([row])) <= maximum:
            current.append(row)
            continue
        # A wide row retains the same source header on every bounded fragment.
        while True:
            pieces = split_by_tokens(row, max_tokens=budget, overlap_tokens=0)
            if all(count_chunk_tokens(render([piece])) <= maximum for piece in pieces):
                break
            budget -= 1
            if budget < 1:
                raise ParserExecutionError(ErrorCode.PARSER_RESOURCE_LIMIT,
                                           diagnostic={"limit_name": "table_header_tokens", "limit": maximum})
        parts.extend(render([piece]) for piece in pieces)
    if current:
        parts.append(render(current))
    return tuple(parts)
