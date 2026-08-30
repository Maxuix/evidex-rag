from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

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
from rag_kb.graph.service import GraphExtractionWorker, _graph_build_error_code
from rag_kb.document_processing.profiles import (
    SEMANTIC_CHUNKING_CONFIG,
    SEMANTIC_CHUNKING_CONFIG_V3,
    STRUCTURAL_CHUNKING_CONFIG_V4,
)
from rag_kb.repositories.sqlalchemy_graph import _graph_chunking_profile_compatible


WORKSPACE = UUID("01900000-0000-7000-8000-000000000901")
KB_ID = UUID("01900000-0000-7000-8000-000000000902")
BUILD_ID = UUID("01900000-0000-7000-8000-000000000903")
PROFILE_ID = UUID("01900000-0000-7000-8000-000000000904")
CHUNK_ID = UUID("01900000-0000-7000-8000-000000000905")
SECOND_CHUNK_ID = UUID("01900000-0000-7000-8000-00000000090a")
REVISION_ID = UUID("01900000-0000-7000-8000-000000000906")
TARGET_ID = UUID("01900000-0000-7000-8000-000000000907")
DOCUMENT_ID = UUID("01900000-0000-7000-8000-000000000908")
DOCUMENT_VERSION_ID = UUID("01900000-0000-7000-8000-000000000909")


class GraphitiBuildWorkerTests(unittest.IsolatedAsyncioTestCase):
    def test_failure_fingerprint_is_stable_and_does_not_depend_on_message(self) -> None:
        first = _graph_build_error_code(
            GraphWorkKind.CHUNK,
            RuntimeError("provider payload one"),
        )
        second = _graph_build_error_code(
            GraphWorkKind.CHUNK,
            RuntimeError("provider payload two"),
        )

        self.assertEqual(first, second)
        self.assertNotIn("provider", first)

    def test_graph_build_rejects_legacy_semantic_chunks_but_accepts_current_profiles(
        self,
    ) -> None:
        self.assertTrue(
            _graph_chunking_profile_compatible(SEMANTIC_CHUNKING_CONFIG)
        )
        self.assertTrue(
            _graph_chunking_profile_compatible(STRUCTURAL_CHUNKING_CONFIG_V4)
        )
        self.assertFalse(
            _graph_chunking_profile_compatible(SEMANTIC_CHUNKING_CONFIG_V3)
        )

    async def test_chunk_is_ingested_before_mapping_is_saved(self) -> None:
        repository = _GraphRepository(
            GraphWorkItem(GraphWorkKind.CHUNK, _config(), _chunk())
        )
        graphiti = _FakeGraphiti()

        await GraphExtractionWorker(
            _factory(repository),
            graphiti,
        ).process_next_work_item()

        self.assertEqual(graphiti.added, [(BUILD_ID, CHUNK_ID)])
        self.assertEqual(repository.mappings, [(CHUNK_ID, "episode-1")])
        self.assertEqual(repository.failed_codes, [])

    async def test_finalize_requires_known_episode_and_complete_probe(self) -> None:
        repository = _GraphRepository(
            GraphWorkItem(GraphWorkKind.FINALIZE, _config())
        )
        repository.first_episode_uuid = "episode-1"
        graphiti = _FakeGraphiti()

        await GraphExtractionWorker(
            _factory(repository),
            graphiti,
        ).process_next_work_item()

        self.assertEqual(graphiti.probes, [(BUILD_ID, "episode-1", True)])
        self.assertEqual(repository.finalized, [BUILD_ID])

    async def test_ineligible_episode_mapping_is_skipped_without_failing(self) -> None:
        repository = _GraphRepository(
            GraphWorkItem(GraphWorkKind.CHUNK, _config(), _chunk())
        )
        repository.accept_mapping = False
        graphiti = _FakeGraphiti()

        await GraphExtractionWorker(
            _factory(repository),
            graphiti,
        ).process_next_work_item()

        self.assertEqual(graphiti.added, [(BUILD_ID, CHUNK_ID)])
        self.assertEqual(repository.failed_codes, [])
        self.assertEqual(graphiti.deleted, [])

    async def test_rejected_bulk_mapping_rolls_back_the_whole_batch(self) -> None:
        repository = _GraphRepository(
            GraphWorkItem(GraphWorkKind.CHUNK, _config(), _chunk())
        )
        repository.batch_chunks = (_chunk(), _chunk(SECOND_CHUNK_ID))
        repository.mapping_acceptance = [True, False]
        graphiti = _BulkFakeGraphiti()

        await GraphExtractionWorker(
            _factory(repository),
            graphiti,
        ).process_next_work_item()

        self.assertEqual(graphiti.bulk_added, [(CHUNK_ID, SECOND_CHUNK_ID)])
        self.assertEqual(repository.mappings, [])
        self.assertEqual(repository.failed_codes, [])

    async def test_failed_probe_preserves_the_build_for_explicit_resume(self) -> None:
        repository = _GraphRepository(GraphWorkItem(GraphWorkKind.FINALIZE, _config()))
        graphiti = _FakeGraphiti(probe_success=False)

        await GraphExtractionWorker(
            _factory(repository),
            graphiti,
        ).process_next_work_item()

        self.assertRegex(
            repository.failed_codes[0],
            r"^graphiti_finalize_failed:[0-9a-f]{16}$",
        )
        self.assertEqual(graphiti.deleted, [])

    async def test_preflight_failure_persists_a_phase_specific_code(self) -> None:
        repository = _GraphRepository(
            GraphWorkItem(GraphWorkKind.PREFLIGHT, _config())
        )
        graphiti = _FakeGraphiti(probe_error=RuntimeError("provider detail"))

        await GraphExtractionWorker(
            _factory(repository),
            graphiti,
        ).process_next_work_item()

        self.assertRegex(
            repository.failed_codes[0],
            r"^graphiti_preflight_failed:[0-9a-f]{16}$",
        )

    async def test_episode_failure_persists_a_phase_specific_code(self) -> None:
        repository = _GraphRepository(
            GraphWorkItem(GraphWorkKind.CHUNK, _config(), _chunk())
        )
        graphiti = _FakeGraphiti(add_error=RuntimeError("provider detail"))

        await GraphExtractionWorker(
            _factory(repository),
            graphiti,
        ).process_next_work_item()

        self.assertRegex(
            repository.failed_codes[0],
            r"^graphiti_episode_extraction_failed:[0-9a-f]{16}$",
        )


class _GraphRepository:
    def __init__(self, work: GraphWorkItem) -> None:
        self.work = work
        self.mappings: list[tuple[UUID, str]] = []
        self.finalized: list[UUID] = []
        self.failed_codes: list[str] = []
        self.first_episode_uuid: str | None = None
        self.accept_mapping = True
        self.mapping_acceptance: list[bool] | None = None
        self.batch_chunks: tuple[GraphChunkSource, ...] = ()
        self._retired: list = []

    async def next_work_item(self):
        work, self.work = self.work, None
        return work

    async def get_graphiti_build(self, kb_id, *, build_id):
        del kb_id
        return _build(build_id)

    async def save_preflight_success(self, kb_id, *, build_id, extractor_version):
        del kb_id, build_id, extractor_version
        return True

    async def save_graphiti_episode(
        self,
        *,
        kb_id,
        build_id,
        index_chunk_id,
        content_hash,
        episode_uuid,
    ):
        del kb_id, build_id, content_hash
        accepted = (
            self.mapping_acceptance.pop(0)
            if self.mapping_acceptance is not None
            else self.accept_mapping
        )
        if not accepted:
            return False
        self.mappings.append((index_chunk_id, episode_uuid))
        return True

    async def missing_graph_chunks(self, config, *, limit):
        del config
        return self.batch_chunks[:limit]

    async def first_graphiti_episode_uuid(self, kb_id, *, build_id):
        del kb_id, build_id
        return self.first_episode_uuid

    async def finalize_graphiti_if_complete(self, kb_id, *, build_id):
        del kb_id
        self.finalized.append(build_id)
        return _config()

    async def mark_failed(self, kb_id, *, build_id, error_code):
        del kb_id
        self.failed_codes.append(error_code)
        return True

    def take_retired_graphiti_builds(self):
        retired = tuple(self._retired)
        self._retired = []
        return retired


class _FakeGraphiti:
    def __init__(
        self,
        *,
        probe_success: bool = True,
        probe_error: Exception | None = None,
        add_error: Exception | None = None,
    ) -> None:
        self.added: list[tuple[UUID, UUID]] = []
        self.probes: list[tuple[UUID, str | None, bool]] = []
        self.deleted: list[UUID] = []
        self.probe_success = probe_success
        self.probe_error = probe_error
        self.add_error = add_error

    async def probe(self, build, *, episode_uuid=None, require_complete=False):
        self.probes.append((build.build_id, episode_uuid, require_complete))
        if self.probe_error is not None:
            raise self.probe_error
        return self.probe_success

    async def add_episode(self, build, chunk):
        self.added.append((build.build_id, chunk.index_chunk_id))
        if self.add_error is not None:
            raise self.add_error
        return "episode-1"

    async def delete_graph(self, build):
        self.deleted.append(build.build_id)


class _BulkFakeGraphiti(_FakeGraphiti):
    def __init__(self) -> None:
        super().__init__()
        self.bulk_added: list[tuple[UUID, ...]] = []

    async def add_episodes_bulk(self, build, chunks):
        del build
        chunk_ids = tuple(chunk.index_chunk_id for chunk in chunks)
        self.bulk_added.append(chunk_ids)
        return tuple(f"episode-{index}" for index, _ in enumerate(chunks, start=1))


class _UnitOfWork:
    def __init__(self, repository: _GraphRepository) -> None:
        self.graph = repository
        self.workspace_id = WORKSPACE

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def commit(self) -> None:
        return None


def _factory(repository: _GraphRepository):
    @asynccontextmanager
    async def context() -> AsyncIterator[_UnitOfWork]:
        mappings_before = list(repository.mappings)
        try:
            yield _UnitOfWork(repository)
        except BaseException:
            repository.mappings[:] = mappings_before
            raise

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


def _build(build_id: UUID = BUILD_ID) -> GraphitiBuildSnapshot:
    return GraphitiBuildSnapshot(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        build_id=build_id,
        group_id=f"ws_{WORKSPACE}_kb_{KB_ID}_b_{build_id}",
        status=GraphitiBuildStatus.BUILDING,
        index_revision_id=REVISION_ID,
        serving_chunk_digest="a" * 64,
        expected_episode_count=1,
        chat_profile_revision_id=PROFILE_ID,
        embedding_profile_revision_id=PROFILE_ID,
        embedding_model="embedding-model",
        embedding_dimension=1024,
        extractor_version=GRAPH_EXTRACTOR_VERSION,
    )


def _chunk(index_chunk_id: UUID = CHUNK_ID) -> GraphChunkSource:
    return GraphChunkSource(
        workspace_id=WORKSPACE,
        knowledge_base_id=KB_ID,
        build_id=BUILD_ID,
        index_chunk_id=index_chunk_id,
        index_revision_id=REVISION_ID,
        indexed_document_version_id=TARGET_ID,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
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
