"""In-process local Unstructured processor."""

from __future__ import annotations

import asyncio

from rag_kb.adapters.parser.langchain_unstructured import (
    partition_with_unstructured,
    process_with_unstructured,
)
from rag_kb.domain import (
    ParsedDocument,
    ParserLimits,
    ParserSource,
    ProcessedDocument,
)


class UnstructuredProcessor:
    """Run local Unstructured in the Worker process without blocking its event loop."""

    def __init__(self, limits: ParserLimits) -> None:
        self._limits = limits

    async def process(self, source: ParserSource) -> ProcessedDocument:
        return await asyncio.to_thread(
            process_with_unstructured,
            source,
            self._limits,
        )

    async def partition(self, source: ParserSource) -> ParsedDocument:
        return await asyncio.to_thread(
            partition_with_unstructured,
            source,
            self._limits,
        )
