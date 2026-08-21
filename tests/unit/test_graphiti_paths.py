from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from rag_kb.adapters.graphiti.client import (
    AliasSurfaceEntity,
    GRAPHITI_V2_EXTRACTION_INSTRUCTIONS,
    GraphitiRuntime,
    _GraphitiEntityResult,
    _rank_graphiti_paths,
)
from rag_kb.domain import GraphitiEdgeResult, GraphitiSearchQuery


class GraphitiPathResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_paths_resolves_a_named_node_before_edge_expansion(self) -> None:
        runtime = GraphitiRuntime.__new__(GraphitiRuntime)
        driver = SimpleNamespace(clone=lambda **_kwargs: driver)
        runtime._client = AsyncMock(return_value=(object(), driver))
        runtime._search_edges = AsyncMock(return_value=())
        runtime._search_entities = AsyncMock(
            return_value=(_GraphitiEntityResult("product", "WTC-7", 1),)
        )
        seed = _edge(1, "product", "WTC-7", "supplier", "梧桐芯片")
        outward = _edge(2, "supplier", "梧桐芯片", "group", "星澜集团")
        runtime._adjacent_edges = AsyncMock(
            side_effect=((seed,), (seed, outward))
        )

        paths = await runtime.search_paths(
            SimpleNamespace(group_id="graph-build"),
            GraphitiSearchQuery(
                workspace_id=uuid4(),
                knowledge_base_id=uuid4(),
                build_id=uuid4(),
                group_id="graph-build",
                query="WTC-7 最终属于哪个集团？",
                limit=8,
            ),
        )

        self.assertEqual(paths[0].hops, (seed, outward))
        self.assertTrue(paths[0].seed_entry)
        self.assertEqual(
            runtime._adjacent_edges.await_args_list[0].args[2],
            ("product",),
        )

    async def test_search_paths_hydrates_endpoints_missing_from_edge_projection(self) -> None:
        runtime = GraphitiRuntime.__new__(GraphitiRuntime)
        driver = SimpleNamespace(clone=lambda **_kwargs: driver)
        runtime._client = AsyncMock(return_value=(object(), driver))
        projected = GraphitiEdgeResult("edge-1", "fact", ("episode-1",), 1)
        hydrated = _edge(1, "source", "甲公司", "target", "乙公司")
        runtime._search_edges = AsyncMock(return_value=(projected,))
        runtime._search_entities = AsyncMock(return_value=())
        runtime._edges_by_uuid = AsyncMock(return_value=(hydrated,))
        runtime._adjacent_edges = AsyncMock(side_effect=((hydrated,), (hydrated,)))

        paths = await runtime.search_paths(
            SimpleNamespace(group_id="graph-build"),
            GraphitiSearchQuery(
                workspace_id=uuid4(),
                knowledge_base_id=uuid4(),
                build_id=uuid4(),
                group_id="graph-build",
                query="甲公司属于哪个主体？",
                limit=8,
            ),
        )

        self.assertEqual(paths[0].hops[0].endpoint_uuids, ("source", "target"))
        self.assertTrue(paths[0].seed_entry)

    async def test_episode_extraction_applies_the_v2_contract(self) -> None:
        runtime = GraphitiRuntime.__new__(GraphitiRuntime)
        graphiti = SimpleNamespace(
            add_episode=AsyncMock(
                return_value=SimpleNamespace(
                    episode=SimpleNamespace(uuid="episode-1")
                )
            )
        )
        runtime._client = AsyncMock(return_value=(graphiti, object()))
        build = SimpleNamespace(group_id="graph-build")
        chunk = SimpleNamespace(
            ordinal=3,
            content="甲公司曾用名乙公司。",
            index_chunk_id="chunk-1",
            reference_time=None,
        )

        with patch(
            "rag_kb.adapters.graphiti.client._graphiti_modules",
            return_value=SimpleNamespace(
                EpisodeType=SimpleNamespace(text="text")
            ),
        ):
            episode_uuid = await runtime.add_episode(build, chunk)

        self.assertEqual(episode_uuid, "episode-1")
        kwargs = graphiti.add_episode.await_args.kwargs
        self.assertEqual(
            kwargs["entity_types"],
            {"AliasSurface": AliasSurfaceEntity},
        )
        self.assertEqual(
            kwargs["custom_extraction_instructions"],
            GRAPHITI_V2_EXTRACTION_INSTRUCTIONS,
        )

    def test_query_grounded_seed_expands_only_through_the_opposite_endpoint(self) -> None:
        seed = _edge(1, "product", "WTC-7", "supplier", "梧桐芯片")
        outward = _edge(2, "supplier", "梧桐芯片", "group", "星澜集团")
        wrong_direction = _edge(3, "product", "WTC-7", "batch", "交付批次")

        paths = _rank_graphiti_paths(
            "WTC-7 最终属于哪个集团？",
            (seed,),
            (seed, outward, wrong_direction),
            limit=8,
        )

        self.assertEqual([path.hops for path in paths[:2]], [(seed, outward), (seed,)])
        self.assertTrue(all(path.seed_entry for path in paths))
        self.assertNotIn(wrong_direction, tuple(hop for path in paths for hop in path.hops))

    def test_ungrounded_ranked_fact_is_not_expanded_into_a_chain(self) -> None:
        seed = _edge(1, "a", "甲公司", "b", "乙公司")
        adjacent = _edge(2, "b", "乙公司", "c", "丙集团")

        paths = _rank_graphiti_paths(
            "请说明合作关系",
            (seed,),
            (seed, adjacent),
            limit=8,
        )

        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].hops, (seed,))
        self.assertFalse(paths[0].seed_entry)

    def test_ranked_adjacent_fact_outranks_arbitrary_adjacency_order(self) -> None:
        seed = _edge(1, "a", "甲公司", "bridge", "乙公司")
        relevant = _edge(2, "bridge", "乙公司", "answer", "丙公司")
        arbitrary_first = _edge(
            1,
            "bridge",
            "乙公司",
            "noise",
            "丁公司",
            edge_id="edge-noise",
        )

        paths = _rank_graphiti_paths(
            "甲公司关联的下游主体是谁？",
            (seed, relevant),
            (arbitrary_first, relevant),
            limit=8,
        )

        two_hop_paths = [path for path in paths if len(path.hops) == 2]
        self.assertEqual(two_hop_paths[0].hops, (seed, relevant))

    def test_question_may_omit_a_long_entity_legal_suffix(self) -> None:
        seed = _edge(1, "supplier", "梧桐芯片有限公司", "factory", "星澜智造有限公司")
        outward = _edge(2, "factory", "星澜智造有限公司", "group", "星澜集团")

        paths = _rank_graphiti_paths(
            "梧桐芯片提供的模组最终属于哪个集团？",
            (seed,),
            (seed, outward),
            limit=8,
        )

        self.assertEqual(paths[0].hops, (seed, outward))
        self.assertTrue(paths[0].seed_entry)

    def test_short_latin_prefix_does_not_ground_an_unrelated_entity(self) -> None:
        seed = _edge(
            1,
            "bank",
            "Bank of America Corporation",
            "project",
            "Atlas Project",
        )

        paths = _rank_graphiti_paths(
            "How does bankruptcy affect the market?",
            (seed,),
            (),
            limit=8,
        )

        self.assertFalse(paths[0].seed_entry)

    def test_self_loop_is_never_returned_as_a_path(self) -> None:
        self_loop = _edge(1, "alias", "星澜工厂", "alias", "星澜工厂")

        self.assertEqual(
            _rank_graphiti_paths(
                "星澜工厂属于哪个集团？",
                (self_loop,),
                (self_loop,),
                limit=8,
            ),
            (),
        )

    def test_extraction_contract_rejects_structure_and_alias_self_loops(self) -> None:
        self.assertIn("section labels", GRAPHITI_V2_EXTRACTION_INSTRUCTIONS)
        self.assertIn("distinct nodes", GRAPHITI_V2_EXTRACTION_INSTRUCTIONS)
        self.assertIn("do not infer", GRAPHITI_V2_EXTRACTION_INSTRUCTIONS.lower())


def _edge(
    rank: int,
    source_uuid: str,
    source_name: str,
    target_uuid: str,
    target_name: str,
    *,
    edge_id: str | None = None,
) -> GraphitiEdgeResult:
    return GraphitiEdgeResult(
        edge_uuid=edge_id or f"edge-{rank}",
        fact=f"{source_name} relates to {target_name}",
        episode_uuids=(f"episode-{rank}",),
        rank=rank,
        source_entity_uuid=source_uuid,
        source_entity_name=source_name,
        target_entity_uuid=target_uuid,
        target_entity_name=target_name,
    )


if __name__ == "__main__":
    unittest.main()
