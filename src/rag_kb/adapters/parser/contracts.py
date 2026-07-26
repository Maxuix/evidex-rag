"""The single application-facing parser contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rag_kb.domain import ParserSource, ParsingPreset

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
