"""Persistence contract for Graphiti build orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from rag_kb.domain import (
    GraphChunkSource,
    GraphConfigSnapshot,
    GraphitiBuildSnapshot,
    GraphWorkItem,
)


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
        schema_profile_key: str | None = None,
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

    async def next_work_item(
        self,
        *,
        worker_id: str = "unknown",
        observed_at: datetime | None = None,
    ) -> GraphWorkItem | None: ...

    async def heartbeat_graph_work(
        self, work: GraphWorkItem, *, observed_at: datetime
    ) -> bool: ...

    async def release_graph_work(self, work: GraphWorkItem) -> bool: ...

    async def reconcile_graph_work_leases(
        self, *, observed_at: datetime, limit: int
    ) -> int: ...

    async def save_preflight_success(
        self,
        kb_id: UUID,
        *,
        build_id: UUID,
        extractor_version: str,
        lease_token: UUID | None = None,
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
        lease_token: UUID | None = None,
    ) -> bool: ...

    async def missing_graph_chunks(
        self,
        config: GraphConfigSnapshot,
        *,
        limit: int,
    ) -> tuple[GraphChunkSource, ...]: ...

    async def first_graphiti_episode_uuid(
        self, kb_id: UUID, *, build_id: UUID
    ) -> str | None: ...

    async def finalize_graphiti_if_complete(
        self,
        kb_id: UUID,
        *,
        build_id: UUID,
        lease_token: UUID | None = None,
    ) -> GraphConfigSnapshot | None: ...

    async def mark_failed(
        self,
        kb_id: UUID,
        *,
        build_id: UUID,
        error_code: str,
        lease_token: UUID | None = None,
    ) -> bool: ...

    def take_retired_graphiti_builds(self) -> tuple[GraphitiBuildSnapshot, ...]: ...
