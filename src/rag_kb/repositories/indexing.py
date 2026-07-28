"""Async relational persistence contract for one indexing execution."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from rag_kb.domain import (
    IndexChunkWrite,
    IndexChunkAssetRelationSnapshot,
    IndexChunkAssetRelationWrite,
    IndexCleanupResult,
    IndexChunkPlan,
    IndexArtifactManifest,
    IndexAssetWrite,
    IndexAssetSnapshot,
    IndexingCommand,
    IndexingLease,
    IndexingJobSnapshot,
    IndexingPhase,
    IndexingTarget,
    PromotionCommand,
    PromotionResult,
    PersistedVectorRepresentation,
    RetiredIndexTargetAssets,
    ReconciliationResult,
    VectorRecordWrite,
)


@runtime_checkable
class IndexingRepository(Protocol):
    async def get_asset(self, asset_id: UUID) -> IndexAssetSnapshot | None: ...

    async def list_retired_target_assets(
        self, *, data_before: datetime, limit: int
    ) -> tuple[RetiredIndexTargetAssets, ...]: ...

    async def list_relations(
        self,
        *,
        kb_id: UUID,
        index_revision_id: UUID,
        chunk_ids: tuple[UUID, ...] = (),
        asset_ids: tuple[UUID, ...] = (),
        limit: int = 500,
    ) -> tuple[IndexChunkAssetRelationSnapshot, ...]: ...

    async def oldest_claimable_at(
        self, *, observed_at: datetime, max_attempts: int
    ) -> datetime | None: ...

    async def get_job(self, job_id: UUID) -> IndexingJobSnapshot | None: ...

    async def retry_failed(
        self, job_id: UUID, *, observed_at: datetime
    ) -> IndexingJobSnapshot | None: ...

    async def cleanup_retired(
        self,
        *,
        target_ids: tuple[UUID, ...],
        data_before: datetime,
        tasks_before: datetime,
        limit: int,
    ) -> IndexCleanupResult: ...

    async def claim(
        self, *, worker_id: str, observed_at: datetime, max_attempts: int
    ) -> IndexingLease | None: ...

    async def heartbeat(
        self, lease: IndexingLease, *, observed_at: datetime
    ) -> bool: ...

    async def reschedule(
        self,
        lease: IndexingLease,
        *,
        observed_at: datetime,
        next_attempt_at: datetime,
        error_code: str,
        error_detail: dict[str, Any],
    ) -> bool: ...

    async def fail_owned(
        self,
        lease: IndexingLease,
        *,
        observed_at: datetime,
        error_code: str,
        error_detail: dict[str, Any],
    ) -> bool: ...

    async def release_terminal(self, lease: IndexingLease) -> bool: ...

    async def reconcile_stale(
        self,
        *,
        stale_before: datetime,
        observed_at: datetime,
        max_attempts: int,
        retry_at_by_attempt: tuple[datetime, ...],
        limit: int,
    ) -> ReconciliationResult: ...

    async def promote(self, command: PromotionCommand) -> PromotionResult | None: ...

    async def prepare(self, command: IndexingCommand) -> IndexingTarget | None: ...

    async def get_chunk_plan(
        self, command: IndexingCommand
    ) -> IndexChunkPlan | None: ...

    async def create_or_get_chunk_plan(
        self,
        command: IndexingCommand,
        proposed: IndexChunkPlan,
    ) -> IndexChunkPlan: ...

    async def get_artifact_manifest(
        self, command: IndexingCommand
    ) -> IndexArtifactManifest | None: ...

    async def create_or_get_artifact_manifest(
        self, command: IndexingCommand, proposed: IndexArtifactManifest
    ) -> IndexArtifactManifest: ...

    async def list_persisted_representations(
        self, command: IndexingCommand, *, limit: int
    ) -> tuple[PersistedVectorRepresentation, ...]: ...

    async def upsert_assets(
        self, command: IndexingCommand, assets: tuple[IndexAssetWrite, ...]
    ) -> bool: ...

    async def upsert_relations(
        self,
        command: IndexingCommand,
        relations: tuple[IndexChunkAssetRelationWrite, ...],
    ) -> bool: ...

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
