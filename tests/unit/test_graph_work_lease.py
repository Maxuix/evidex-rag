from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import unittest
from uuid import UUID, uuid4

from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    GraphChunkSource,
    GraphConfigSnapshot,
    GraphConfigStatus,
    GraphitiBuildSnapshot,
    GraphitiBuildStatus,
    GraphWorkItem,
    GraphWorkKind,
)
from rag_kb.graph.service import GraphExtractionWorker


WORKSPACE = UUID("01900000-0000-7000-8000-000000001a01")
KB_ID = UUID("01900000-0000-7000-8000-000000001a02")
BUILD_ID = UUID("01900000-0000-7000-8000-000000001a03")
CHUNK_ID = UUID("01900000-0000-7000-8000-000000001a04")
PROFILE_ID = UUID("01900000-0000-7000-8000-000000001a05")


class GraphWorkLeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_workers_serialize_same_build_episode_work(self) -> None:
        repository = _InMemoryLeaseRepository()
        graphiti = _DelayedGraphiti()
        factory = _factory(repository)
        first = GraphExtractionWorker(factory, graphiti, worker_id="worker-a")
        second = GraphExtractionWorker(factory, graphiti, worker_id="worker-b")

        claimed = await asyncio.gather(
            first.process_next_work_item(), second.process_next_work_item()
        )

        self.assertEqual(sorted(claimed), [False, True])
        self.assertEqual(graphiti.added, [CHUNK_ID])
        self.assertEqual(graphiti.max_active, 1)
        self.assertEqual(repository.released, 1)
        self.assertIsNone(repository.active_lease)

    async def test_token_mismatch_cannot_complete_or_release_lease(self) -> None:
        repository = _InMemoryLeaseRepository()
        work = await repository.next_work_item(worker_id="worker-a")
        assert work is not None
        wrong = GraphWorkItem(
            work.kind,
            work.config,
            work.chunk,
            lease_token=uuid4(),
            lease_owner="worker-a",
        )

        self.assertFalse(await repository.save_graphiti_episode(wrong))
        self.assertFalse(await repository.release_graph_work(wrong))
        self.assertIsNotNone(repository.active_lease)

    async def test_expired_lease_is_reclaimed(self) -> None:
        repository = _InMemoryLeaseRepository()
        work = await repository.next_work_item(worker_id="worker-a")
        assert work is not None
        repository.expires_at = datetime.now(UTC) - timedelta(seconds=1)

        reclaimed = await repository.next_work_item(worker_id="worker-b")

        assert reclaimed is not None
        self.assertNotEqual(reclaimed.lease_token, work.lease_token)
        self.assertEqual(reclaimed.lease_owner, "worker-b")


class _InMemoryLeaseRepository:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.active_lease: GraphWorkItem | None = None
        self.expires_at: datetime | None = None
        self.released = 0

    async def next_work_item(self, *, worker_id="unknown", observed_at=None):
        observed_at = observed_at or datetime.now(UTC)
        async with self._lock:
            if self.active_lease is not None and (
                self.expires_at is None or self.expires_at > observed_at
            ):
                return None
            token = uuid4()
            work = GraphWorkItem(
                GraphWorkKind.CHUNK,
                _config(),
                _chunk(),
                lease_token=token,
                lease_owner=worker_id,
            )
            self.active_lease = work
            self.expires_at = observed_at + timedelta(seconds=180)
            return work

    async def get_graphiti_build(self, kb_id, *, build_id):
        del kb_id, build_id
        return _build()

    async def heartbeat_graph_work(self, work, *, observed_at):
        if self._matches(work) and self.expires_at is not None and self.expires_at > observed_at:
            self.expires_at = observed_at + timedelta(seconds=180)
            return True
        return False

    async def save_graphiti_episode(self, work=None, *, lease_token=None, **kwargs):
        del kwargs
        if work is not None:
            return self._matches(work)
        return (
            self.active_lease is not None
            and lease_token == self.active_lease.lease_token
        )

    async def release_graph_work(self, work):
        if not self._matches(work):
            return False
        self.active_lease = None
        self.expires_at = None
        self.released += 1
        return True

    async def mark_failed(self, *args, **kwargs):
        del args, kwargs
        return True

    def take_retired_graphiti_builds(self):
        return ()

    def _matches(self, work: GraphWorkItem) -> bool:
        return (
            self.active_lease is not None
            and work.lease_token == self.active_lease.lease_token
            and work.lease_owner == self.active_lease.lease_owner
        )


class _DelayedGraphiti:
    def __init__(self) -> None:
        self.added: list[UUID] = []
        self.active = 0
        self.max_active = 0

    async def add_episode(self, build, chunk):
        del build
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.04)
            self.added.append(chunk.index_chunk_id)
            return "episode-1"
        finally:
            self.active -= 1


class _UnitOfWork:
    def __init__(self, repository) -> None:
        self.graph = repository
        self.workspace_id = WORKSPACE

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def commit(self):
        return None


def _factory(repository):
    @asynccontextmanager
    async def context():
        yield _UnitOfWork(repository)

    def factory(*, purpose, mode):
        del purpose, mode
        return context()

    return factory


def _config() -> GraphConfigSnapshot:
    return GraphConfigSnapshot(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        status=GraphConfigStatus.BUILDING,
        build_id=BUILD_ID,
        chat_profile_revision_id=PROFILE_ID,
        extractor_version=GRAPH_EXTRACTOR_VERSION,
        preflight_extractor_version=GRAPH_EXTRACTOR_VERSION,
        last_error_code=None,
        eligible_chunk_count=1,
    )


def _build() -> GraphitiBuildSnapshot:
    return GraphitiBuildSnapshot(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        build_id=BUILD_ID,
        group_id="lease-build",
        status=GraphitiBuildStatus.BUILDING,
        index_revision_id=uuid4(),
        serving_chunk_digest="a" * 64,
        expected_episode_count=1,
        chat_profile_revision_id=PROFILE_ID,
        embedding_profile_revision_id=PROFILE_ID,
        embedding_model="embedding-model",
        embedding_dimension=1024,
        extractor_version=GRAPH_EXTRACTOR_VERSION,
    )


def _chunk() -> GraphChunkSource:
    return GraphChunkSource(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        build_id=BUILD_ID,
        index_chunk_id=CHUNK_ID,
        index_revision_id=uuid4(),
        indexed_document_version_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        ordinal=0,
        modality="text",
        content="A grounded fact.",
        content_hash="b" * 64,
        source_location={},
        hierarchy={},
        source_metadata={},
    )


if __name__ == "__main__":
    unittest.main()
