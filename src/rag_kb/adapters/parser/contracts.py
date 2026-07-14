"""Application-facing isolated parser contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_kb.domain import ParserSource, ProcessedDocument


@runtime_checkable
class DocumentProcessor(Protocol):
    async def process(self, source: ParserSource) -> ProcessedDocument: ...
