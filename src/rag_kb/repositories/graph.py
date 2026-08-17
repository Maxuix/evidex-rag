"""Persistence contract for Graphiti build orchestration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain import (
    GraphConfigSnapshot,
    GraphitiBuildSnapshot,
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

    async def get_graphiti_build(
        self, kb_id: UUID, *, build_id: UUID
    ) -> GraphitiBuildSnapshot | None: ...

    async def save_graphiti_episode(
        self,
        *,
        kb_id: UUID,
        build_id: UUID,
        index_chunk_id: UUID,
        content_hash: str,
        episode_uuid: str,
    ) -> bool: ...

    async def first_graphiti_episode_uuid(
        self, kb_id: UUID, *, build_id: UUID
    ) -> str | None: ...

    async def finalize_graphiti_if_complete(
        self, kb_id: UUID, *, build_id: UUID
    ) -> GraphConfigSnapshot | None: ...

    async def mark_failed(
        self,
        kb_id: UUID,
        *,
        build_id: UUID,
        error_code: str,
    ) -> bool: ...

    def take_retired_graphiti_builds(self) -> tuple[GraphitiBuildSnapshot, ...]: ...
