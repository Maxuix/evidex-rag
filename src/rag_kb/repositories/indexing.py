"""Async relational persistence contract for one indexing execution."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rag_kb.domain import (
    IndexChunkWrite,
    IndexingCommand,
    IndexingPhase,
    IndexingTarget,
    VectorRecordWrite,
)


@runtime_checkable
class IndexingRepository(Protocol):
    async def prepare(self, command: IndexingCommand) -> IndexingTarget | None: ...

    async def set_phase(self, command: IndexingCommand, phase: IndexingPhase) -> bool: ...

    async def upsert_batch(
        self,
        command: IndexingCommand,
        chunks: tuple[IndexChunkWrite, ...],
        vectors: tuple[VectorRecordWrite, ...],
    ) -> bool: ...

    async def complete(self, command: IndexingCommand, *, expected_chunks: int) -> bool: ...

    async def count_chunks(self, command: IndexingCommand) -> int: ...

    async def fail(
        self,
        command: IndexingCommand,
        *,
        phase: IndexingPhase,
        error_code: str,
        error_detail: dict[str, Any],
    ) -> bool: ...
