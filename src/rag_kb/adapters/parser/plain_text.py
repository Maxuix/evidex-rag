"""Deterministic P1A plain-text parsing and chunk drafting."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from rag_kb.domain import (
    ErrorCode,
    IndexChunkDraft,
    ParsedBlock,
    ParsedDocument,
    ParserExecutionError,
    ParserSource,
    ProcessedDocument,
)


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")


def process_plain_text(
    source: ParserSource,
    *,
    max_characters: int,
    overlap_characters: int,
    max_chunks: int,
) -> ProcessedDocument:
    parsed = parse_plain_text(source)
    blocks = parsed.blocks
    drafts: list[IndexChunkDraft] = []
    for block in blocks:
        for start, end, text in _windows(
            block.text,
            max_characters=max_characters,
            overlap_characters=overlap_characters,
        ):
            if len(drafts) >= max_chunks:
                raise ParserExecutionError(
                    ErrorCode.PARSER_CHUNK_LIMIT_EXCEEDED,
                    diagnostic={"limit_name": "max_chunks", "limit": max_chunks},
                )
            drafts.append(
                IndexChunkDraft(
                    ordinal=len(drafts),
                    text=text,
                    start_character=block.start_character + start,
                    end_character=block.start_character + end,
                    heading_hierarchy=block.heading_hierarchy,
                    content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
    return ProcessedDocument(parsed=parsed, chunks=tuple(drafts))


def parse_plain_text(source: ParserSource) -> ParsedDocument:
    extension = _extension(source.original_filename)
    if extension not in {".txt", ".md"}:
        raise ParserExecutionError(ErrorCode.PARSER_NOT_CONFIGURED)
    expected_media_type = "text/plain" if extension == ".txt" else "text/markdown"
    if source.media_type != expected_media_type:
        raise ParserExecutionError(ErrorCode.FILE_MEDIA_TYPE_MISMATCH)
    try:
        decoded = source.content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ParserExecutionError(ErrorCode.FILE_INVALID_UTF8) from error

    canonical = unicodedata.normalize(
        "NFC", decoded.replace("\r\n", "\n").replace("\r", "\n")
    )
    blocks = _blocks(canonical, markdown=extension == ".md")
    return ParsedDocument(canonical_text=canonical, blocks=blocks)


class PlainTextTestParser:
    """The parser-neutral `.txt`/`.md` test adapter frozen for P1A."""

    def parse(self, source: ParserSource) -> ParsedDocument:
        return parse_plain_text(source)


def _extension(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot >= 0 else ""


def _blocks(text: str, *, markdown: bool) -> tuple[ParsedBlock, ...]:
    if not text:
        return ()
    headings: list[str] = []
    blocks: list[ParsedBlock] = []
    offset = 0
    paragraph_start: int | None = None
    paragraph_lines: list[str] = []
    paragraph_headings: tuple[str, ...] = ()

    def flush() -> None:
        nonlocal paragraph_start, paragraph_lines, paragraph_headings
        if paragraph_start is None:
            return
        raw = "\n".join(paragraph_lines)
        leading = len(raw) - len(raw.lstrip())
        value = raw.strip()
        if value:
            start = paragraph_start + leading
            blocks.append(
                ParsedBlock(
                    text=value,
                    start_character=start,
                    end_character=start + len(value),
                    heading_hierarchy=paragraph_headings,
                )
            )
        paragraph_start = None
        paragraph_lines = []
        paragraph_headings = ()

    for line_with_ending in text.splitlines(keepends=True):
        line = line_with_ending.removesuffix("\n")
        heading = _HEADING.match(line) if markdown else None
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            if len(headings) < level:
                headings.extend([""] * (level - len(headings)))
            headings[level - 1] = title
            del headings[level:]
            hierarchy = tuple(value for value in headings if value)
            blocks.append(
                ParsedBlock(
                    text=line.strip(),
                    start_character=offset + len(line) - len(line.lstrip()),
                    end_character=offset + len(line.rstrip()),
                    heading_hierarchy=hierarchy,
                )
            )
        elif not line.strip():
            flush()
        else:
            if paragraph_start is None:
                paragraph_start = offset
                paragraph_headings = tuple(value for value in headings if value)
            paragraph_lines.append(line)
        offset += len(line_with_ending)
    flush()
    return tuple(blocks)


def _windows(
    text: str,
    *,
    max_characters: int,
    overlap_characters: int,
):
    if not text:
        return
    start = 0
    while start < len(text):
        ceiling = min(start + max_characters, len(text))
        end = ceiling
        if ceiling < len(text):
            for boundary in ("\n\n", "\n"):
                candidate = text.rfind(boundary, start + 1, ceiling + 1)
                if candidate > start:
                    end = candidate + len(boundary)
                    break
        value = text[start:end]
        if value:
            yield start, end, value
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_characters)
