from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from rag_kb.adapters.graphiti.client import (
    GraphitiRuntime,
    graphiti_episode_uuid,
)
from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    GraphChunkSource,
    GraphitiBuildSnapshot,
    GraphitiBuildStatus,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST,
    SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY,
)
from rag_kb.graph.schema_profiles import get_graph_schema_registry


WORKSPACE_ID = UUID("01900000-0000-7000-8000-000000000a01")
KB_ID = UUID("01900000-0000-7000-8000-000000000a02")
BUILD_ID = UUID("01900000-0000-7000-8000-000000000a03")
PROFILE_ID = UUID("01900000-0000-7000-8000-000000000a04")
REVISION_ID = UUID("01900000-0000-7000-8000-000000000a05")
CHUNK_ID = UUID("01900000-0000-7000-8000-000000000a06")
TARGET_ID = UUID("01900000-0000-7000-8000-000000000a07")
DOCUMENT_ID = UUID("01900000-0000-7000-8000-000000000a08")
DOCUMENT_VERSION_ID = UUID("01900000-0000-7000-8000-000000000a09")
SOFTWARE_SCHEMA = get_graph_schema_registry().compile(
    SOFTWARE_GRAPH_SCHEMA_PROFILE_KEY,
    digest=SOFTWARE_GRAPH_SCHEMA_PROFILE_DIGEST,
    extractor_version=GRAPH_EXTRACTOR_VERSION,
)


class GraphitiEpisodeResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_probe_rejects_orphan_aliases_and_untyped_edges(self) -> None:
        for counts in ((1, 0, 1), (1, 0, 0, 1)):
            with self.subTest(counts=counts):
                driver = SimpleNamespace(
                    execute_query=AsyncMock(
                        side_effect=[([{"count": count}], [], []) for count in counts]
                    )
                )
                runtime = GraphitiRuntime(_unused_credentials)
                runtime._client = AsyncMock(return_value=(object(), driver))

                self.assertFalse(await runtime.probe(_build(), require_complete=True))

    async def test_replay_removes_uncommitted_episode_with_same_identity(self) -> None:
        driver = _Driver()
        graphiti = _Graphiti(driver)
        runtime = GraphitiRuntime(_unused_credentials)

        async def client(_build):
            return graphiti, driver

        runtime._client = client  # type: ignore[method-assign]
        modules = SimpleNamespace(
            EpisodeType=SimpleNamespace(text="text"),
            EpisodicNode=_Episode,
            NodeNotFoundError=_NodeNotFoundError,
            RELEVANT_SCHEMA_LIMIT=10,
        )

        with patch(
            "rag_kb.adapters.graphiti.client._graphiti_modules",
            return_value=modules,
        ):
            first = await runtime.add_episode(_build(), _chunk())
            replay = await runtime.add_episode(_build(), _chunk())

        self.assertEqual(first, replay)
        self.assertEqual(first, graphiti_episode_uuid(_build(), _chunk()))
        self.assertEqual(graphiti.removed, [first])
        self.assertEqual(len(graphiti.added), 2)
        self.assertEqual(graphiti.added[0]["uuid"], first)
        self.assertEqual(graphiti.added[1]["uuid"], first)
        self.assertEqual(graphiti.added[0]["previous_episode_uuids"], ["previous"])
        self.assertEqual(
            graphiti.added[0]["custom_extraction_instructions"],
            SOFTWARE_SCHEMA.extraction_instructions,
        )
        self.assertEqual(graphiti.added[0]["entity_types"], SOFTWARE_SCHEMA.entity_types)
        self.assertEqual(driver.episode_ids, {first})

    async def test_bulk_failure_does_not_replay_the_batch_serially(self) -> None:
        driver = _Driver()
        graphiti = _BulkGraphiti(driver, error=RuntimeError("provider unavailable"))
        runtime = GraphitiRuntime(_unused_credentials)

        async def client(_build):
            return graphiti, driver

        runtime._client = client  # type: ignore[method-assign]
        modules = SimpleNamespace(
            EpisodeType=SimpleNamespace(text="text"),
            EpisodicNode=_Episode,
            NodeNotFoundError=_NodeNotFoundError,
            RELEVANT_SCHEMA_LIMIT=10,
        )

        with patch(
            "rag_kb.adapters.graphiti.client._graphiti_modules",
            return_value=modules,
        ):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                await runtime.add_episodes_bulk(_build(), (_chunk(),))

        self.assertEqual(graphiti.bulk_calls, 1)
        self.assertEqual(graphiti.added, [])

    async def test_completion_marker_skips_provider_after_relational_gap(self) -> None:
        driver = _MarkerDriver()
        graphiti = _BulkGraphiti(driver)
        runtime = GraphitiRuntime(_unused_credentials)

        async def client(_build):
            return graphiti, driver

        runtime._client = client  # type: ignore[method-assign]
        modules = SimpleNamespace(
            EpisodeType=SimpleNamespace(text="text"),
            EpisodicNode=_Episode,
            NodeNotFoundError=_NodeNotFoundError,
            RELEVANT_SCHEMA_LIMIT=10,
        )

        with patch(
            "rag_kb.adapters.graphiti.client._graphiti_modules",
            return_value=modules,
        ):
            first = await runtime.add_episodes_bulk(_build(), (_chunk(),))
            replay = await runtime.add_episodes_bulk(_build(), (_chunk(),))

        self.assertEqual(first, replay)
        self.assertEqual(graphiti.bulk_calls, 1)
        self.assertEqual(graphiti.removed, [])
        self.assertEqual(driver.completed, set(first))

    def test_identity_is_scoped_to_build_chunk_and_content(self) -> None:
        baseline = graphiti_episode_uuid(_build(), _chunk())

        self.assertEqual(baseline, graphiti_episode_uuid(_build(), _chunk()))
        self.assertNotEqual(
            baseline,
            graphiti_episode_uuid(_build(), replace(_chunk(), content_hash="c" * 64)),
        )
        self.assertNotEqual(
            baseline,
            graphiti_episode_uuid(
                replace(
                    _build(),
                    build_id=UUID("01900000-0000-7000-8000-000000000a10"),
                ),
                _chunk(),
            ),
        )


class _NodeNotFoundError(RuntimeError):
    pass


class _Driver:
    def __init__(self) -> None:
        self.episode_ids: set[str] = set()


class _MarkerDriver(_Driver):
    def __init__(self) -> None:
        super().__init__()
        self.completed: set[str] = set()

    async def execute_query(self, query: str, **values):
        episode_uuids = tuple(values.get("episode_uuids", ()))
        if "RETURN episode.uuid AS uuid" in query:
            return (
                [
                    {"uuid": episode_uuid}
                    for episode_uuid in episode_uuids
                    if episode_uuid in self.completed
                ],
                [],
                [],
            )
        if "SET episode.rag_kb_ingestion_state" in query:
            self.completed.update(episode_uuids)
            return ([{"count": len(episode_uuids)}], [], [])
        return ([], [], [])


class _Episode:
    def __init__(self, **values) -> None:
        for name, value in values.items():
            setattr(self, name, value)

    @classmethod
    async def get_by_uuid(cls, driver: _Driver, episode_uuid: str):
        if episode_uuid not in driver.episode_ids:
            raise _NodeNotFoundError(episode_uuid)
        return cls(uuid=episode_uuid)

    async def save(self, driver: _Driver) -> None:
        driver.episode_ids.add(self.uuid)


class _Graphiti:
    def __init__(self, driver: _Driver) -> None:
        self.driver = driver
        self.removed: list[str] = []
        self.added: list[dict[str, object]] = []

    async def retrieve_episodes(self, *_args, **_kwargs):
        return [SimpleNamespace(uuid="previous")]

    async def remove_episode(self, episode_uuid: str) -> None:
        self.removed.append(episode_uuid)
        self.driver.episode_ids.remove(episode_uuid)

    async def add_episode(self, **values):
        self.added.append(values)
        return SimpleNamespace(episode=SimpleNamespace(uuid=values["uuid"]))


class _BulkGraphiti(_Graphiti):
    def __init__(self, driver: _Driver, *, error: Exception | None = None) -> None:
        super().__init__(driver)
        self.error = error
        self.bulk_calls = 0

    async def add_episode_bulk(self, episodes, **_values):
        self.bulk_calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            episodes=[SimpleNamespace(uuid=episode.uuid) for episode in episodes]
        )


async def _unused_credentials(_build):
    raise AssertionError("credentials must not be loaded")


def _build() -> GraphitiBuildSnapshot:
    return GraphitiBuildSnapshot(
        workspace_id=WORKSPACE_ID,
        knowledge_base_id=KB_ID,
        build_id=BUILD_ID,
        group_id=f"ws_{WORKSPACE_ID}_kb_{KB_ID}_b_{BUILD_ID}",
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


def _chunk() -> GraphChunkSource:
    return GraphChunkSource(
        workspace_id=WORKSPACE_ID,
        knowledge_base_id=KB_ID,
        build_id=BUILD_ID,
        index_chunk_id=CHUNK_ID,
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
