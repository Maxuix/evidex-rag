"""Application-facing isolated parser contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rag_kb.domain import (
    ParsedDocument,
    ParserSource,
    ParsingPreset,
    ProcessedDocument,
)

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


@runtime_checkable
class DocumentParser(Protocol):
    async def parse(
        self,
        source: ParserSource,
        *,
        preset: ParsingPreset,
    ) -> DoclingDocument: ...


@runtime_checkable
class DocumentProcessor(Protocol):
    async def process(self, source: ParserSource) -> ProcessedDocument: ...

    async def partition(self, source: ParserSource) -> ParsedDocument: ...

    async def partition_multimodal(self, source: ParserSource) -> ParsedDocument: ...
