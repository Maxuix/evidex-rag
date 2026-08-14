"""Persistence contract for the current-only entity graph projection."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain import (
    GraphChunkExtraction,
    GraphConfigSnapshot,
    GraphWorkItem,
)


@runtime_checkable
class GraphRepository(Protocol):
    async def get_config(self, kb_id: UUID) -> GraphConfigSnapshot | None: ...

    async def ensure_config(self, kb_id: UUID) -> GraphConfigSnapshot: ...

    async def configure(
        self,
        kb_id: UUID,
        *,
        chat_profile_revision_id: UUID | None,
        enabled: bool,
        extractor_version: str,
        force_rebuild: bool = False,
    ) -> GraphConfigSnapshot: ...

    async def retry(
        self,
        kb_id: UUID,
        *,
        extractor_version: str,
        force_rebuild: bool = False,
    ) -> GraphConfigSnapshot: ...

    async def invalidate_for_serving_change(self, kb_id: UUID) -> bool: ...

    async def invalidate_for_indexed_target(self, target_id: UUID) -> bool: ...

    async def next_work_item(self) -> GraphWorkItem | None: ...

    async def save_preflight_success(
        self, kb_id: UUID, *, build_id: UUID, extractor_version: str
    ) -> bool: ...

    async def save_chunk_extraction(
        self,
        *,
        kb_id: UUID,
        build_id: UUID,
        index_chunk_id: UUID,
        content_hash: str,
        extractor_version: str,
        extraction: GraphChunkExtraction,
    ) -> bool: ...

    async def mark_failed(
        self,
        kb_id: UUID,
        *,
        build_id: UUID,
        error_code: str,
    ) -> bool: ...

    async def finalize_if_complete(
        self,
        kb_id: UUID,
        *,
        build_id: UUID,
        observed_at: datetime,
    ) -> GraphConfigSnapshot | None: ...
