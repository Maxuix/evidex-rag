"""Application-facing isolated parser contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_kb.domain import ParsedDocument, ParserSource, ProcessedDocument


@runtime_checkable
class DocumentProcessor(Protocol):
    async def process(self, source: ParserSource) -> ProcessedDocument: ...

    async def partition(self, source: ParserSource) -> ParsedDocument: ...

    async def partition_multimodal(self, source: ParserSource) -> ParsedDocument: ...
