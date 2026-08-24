from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from rag_kb.adapters.graphiti.client import GraphitiRuntime
from rag_kb.domain import (
    GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST,
    GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
)


class GraphitiSchemaProfileAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_generic_build_passes_native_none_schema_arguments(self) -> None:
        class NodeNotFoundError(RuntimeError):
            pass

        class EpisodicNode:
            def __init__(self, **values) -> None:
                self.uuid = values["uuid"]

            @classmethod
            async def get_by_uuid(cls, _driver, _episode_uuid):
                raise NodeNotFoundError

            async def save(self, _driver) -> None:
                return None

        runtime = GraphitiRuntime.__new__(GraphitiRuntime)
        driver = SimpleNamespace(clone=lambda **_kwargs: driver)
        driver.execute_query = AsyncMock(return_value=([], [], []))
        graphiti = SimpleNamespace(
            add_episode=AsyncMock(
                side_effect=lambda **values: SimpleNamespace(
                    episode=SimpleNamespace(uuid=values["uuid"])
                )
            ),
            retrieve_episodes=AsyncMock(return_value=[]),
        )
        runtime._client = AsyncMock(return_value=(graphiti, driver))
        build = SimpleNamespace(
            group_id="generic-build",
            build_id=uuid4(),
            schema_profile_key=GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
            schema_profile_digest=GENERIC_GRAPH_SCHEMA_PROFILE_DIGEST,
            extractor_version="graphiti_v4",
        )
        chunk = SimpleNamespace(
            ordinal=0,
            content="A person lives in a city.",
            index_chunk_id="chunk-1",
            content_hash="a" * 64,
            reference_time=None,
        )

        with patch(
            "rag_kb.adapters.graphiti.client._graphiti_modules",
            return_value=SimpleNamespace(
                EpisodeType=SimpleNamespace(text="text"),
                EpisodicNode=EpisodicNode,
                NodeNotFoundError=NodeNotFoundError,
                RELEVANT_SCHEMA_LIMIT=10,
            ),
        ):
            await runtime.add_episode(build, chunk)

        values = graphiti.add_episode.await_args.kwargs
        self.assertIsNone(values["entity_types"])
        self.assertIsNone(values["edge_types"])
        self.assertIsNone(values["edge_type_map"])
        self.assertNotIn("AliasSurface", values["custom_extraction_instructions"])
        self.assertNotIn("repository", values["custom_extraction_instructions"].lower())

    async def test_profile_digest_mismatch_fails_before_external_client(self) -> None:
        runtime = GraphitiRuntime(_unused_credentials)
        runtime._client = AsyncMock(side_effect=AssertionError("client must not load"))
        build = SimpleNamespace(
            group_id="generic-build",
            schema_profile_key=GENERIC_GRAPH_SCHEMA_PROFILE_KEY,
            schema_profile_digest="0" * 64,
            extractor_version="graphiti_v4",
        )

        with self.assertRaisesRegex(RuntimeError, "graph_schema_profile_mismatch"):
            await runtime.probe(build)

        runtime._client.assert_not_awaited()


async def _unused_credentials(_build):
    raise AssertionError("credentials must not be loaded")


if __name__ == "__main__":
    unittest.main()
