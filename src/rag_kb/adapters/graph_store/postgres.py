"""Bounded PostgreSQL entity graph lookup and traversal."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any
from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_kb.domain import (
    GRAPH_MAX_INTERMEDIATE_DEGREE,
    GRAPH_MAX_NEIGHBORS_PER_ENTRY,
    GraphChunkEvidence,
    GraphConfigSnapshot,
    GraphConfigStatus,
    GraphEntityCandidate,
    GraphEntityLookupQuery,
    GraphPathCandidate,
    GraphPathHop,
    GraphTraversalQuery,
    GraphTraversalResult,
)


class PgGraphStore:
    """Serve only the current, ready graph projection from PostgreSQL."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get_config(
        self, workspace_id: UUID, knowledge_base_id: UUID
    ) -> GraphConfigSnapshot | None:
        async with self._sessions() as session:
            async with session.begin():
                row = (
                    await session.execute(
                        self._config_statement(),
                        {
                            "workspace_id": workspace_id,
                            "kb_id": knowledge_base_id,
                        },
                    )
                ).mappings().one_or_none()
        if row is None:
            return None
        return _config_snapshot(row)

    async def find_entity_candidates(
        self, query: GraphEntityLookupQuery
    ) -> tuple[GraphEntityCandidate, ...]:
        statement = self._candidate_statement(query.prefix).bindparams(
            bindparam("surface_value")
        )
        async with self._sessions() as session:
            async with session.begin():
                rows = (
                    await session.execute(
                        statement,
                        {
                            "workspace_id": query.workspace_id,
                            "kb_id": query.knowledge_base_id,
                            "build_id": query.build_id,
                            "surface_value": query.normalized_surface,
                            "limit": query.limit,
                        },
                    )
                ).mappings().all()
        return tuple(
            GraphEntityCandidate(
                entity_key=row["entity_key"],
                entity_type=row["entity_type"],
                surface=row["surface"],
                normalized_surface=row["normalized_surface"],
                index_chunk_id=row["index_chunk_id"],
                indexed_document_version_id=row["indexed_document_version_id"],
                index_revision_id=row["index_revision_id"],
            )
            for row in rows
        )

    async def traverse(
        self, query: GraphTraversalQuery
    ) -> GraphTraversalResult | None:
        entry_keys = tuple(dict.fromkeys(query.entry_entity_keys))
        one_hop_statement = self._one_hop_statement().bindparams(
            bindparam("entry_keys", expanding=True),
            bindparam("seed_chunk_ids", expanding=True),
        )
        two_hop_statement = self._two_hop_statement().bindparams(
            bindparam("entry_keys", expanding=True),
            bindparam("seed_chunk_ids", expanding=True),
        )
        async with self._sessions() as session:
            async with session.begin():
                active_revision_id = await session.scalar(
                    text(
                        "SELECT active_index_revision_id "
                        "FROM knowledge_base "
                        "WHERE workspace_id = :workspace_id "
                        "AND id = :kb_id AND deleted_at IS NULL"
                    ),
                    {
                        "workspace_id": query.workspace_id,
                        "kb_id": query.knowledge_base_id,
                    },
                )
                if active_revision_id is None:
                    return None
                if active_revision_id != query.index_revision_id:
                    return GraphTraversalResult(
                        resolved_active_revision_id=active_revision_id
                    )
                parameters = {
                    "workspace_id": query.workspace_id,
                    "kb_id": query.knowledge_base_id,
                    "build_id": query.build_id,
                    "index_revision_id": query.index_revision_id,
                    "entry_keys": list(entry_keys),
                    "seed_chunk_ids": list(query.seed_chunk_ids),
                }
                one_rows = (
                    await session.execute(one_hop_statement, parameters)
                ).mappings().all()
                two_rows = (
                    await session.execute(two_hop_statement, parameters)
                ).mappings().all() if query.max_hops == 2 else ()

        seed_chunks = set(query.seed_chunk_ids)
        one_edges = _aggregate_edges(
            (_edge_from_row(row) for row in one_rows),
            seed_chunks,
        )
        two_edges = _aggregate_edge_pairs(
            (_edge_pair_from_row(row) for row in two_rows),
            seed_chunks,
        )
        paths, chunks, rejected = _build_paths(
            entry_keys,
            one_edges,
            two_edges,
            seed_chunks,
            max_paths=query.max_paths,
        )
        return GraphTraversalResult(
            resolved_active_revision_id=query.index_revision_id,
            paths=paths,
            chunks=chunks,
            rejected_path_count=rejected,
        )

    @staticmethod
    def _config_statement():
        return text(
            """
            WITH eligible AS (
              SELECT chunk.id AS index_chunk_id, chunk.content_hash
              FROM index_chunk chunk
              JOIN indexed_document_version target
                ON target.id = chunk.indexed_document_version_id
               AND target.workspace_id = chunk.workspace_id
               AND target.kb_id = chunk.kb_id
               AND target.build_status = 'ready'
               AND target.serving_status = 'serving'
              JOIN index_revision revision
                ON revision.id = target.index_revision_id
               AND revision.workspace_id = target.workspace_id
               AND revision.kb_id = target.kb_id
               AND revision.status = 'active'
              JOIN knowledge_base kb
                ON kb.id = chunk.kb_id
               AND kb.workspace_id = chunk.workspace_id
               AND kb.active_index_revision_id = target.index_revision_id
               AND kb.deleted_at IS NULL
              JOIN document doc
                ON doc.id = target.document_id
               AND doc.workspace_id = target.workspace_id
               AND doc.kb_id = target.kb_id
               AND doc.deleted_at IS NULL
              JOIN document_version version
                ON version.id = target.document_version_id
               AND version.workspace_id = target.workspace_id
               AND version.kb_id = target.kb_id
               AND version.document_id = target.document_id
               AND version.source_status = 'available'
              WHERE chunk.workspace_id = :workspace_id
                AND chunk.kb_id = :kb_id
                AND chunk.modality IN ('text', 'table')
            ), completed AS (
              SELECT graph.result_status
              FROM index_graph_chunk graph
              JOIN eligible ON eligible.index_chunk_id = graph.index_chunk_id
                AND eligible.content_hash = graph.content_hash
              JOIN knowledge_base_graph_config config
                ON config.workspace_id = graph.workspace_id
               AND config.kb_id = graph.kb_id
               AND config.build_id = graph.build_id
               AND config.extractor_version = graph.extractor_version
              WHERE graph.workspace_id = :workspace_id
                AND graph.kb_id = :kb_id
                AND graph.build_id = config.build_id
            )
            SELECT config.kb_id, config.workspace_id, config.status,
                   config.build_id, config.chat_profile_revision_id,
                   config.extractor_version, config.preflight_extractor_version,
                   config.last_error_code,
                   (SELECT count(*) FROM eligible) AS eligible_chunk_count,
                   (SELECT count(*) FROM completed) AS processed_chunk_count,
                   (SELECT count(*) FROM completed WHERE result_status = 'extracted')
                     AS extracted_chunk_count,
                   (SELECT count(*) FROM completed WHERE result_status = 'empty')
                     AS empty_chunk_count,
                   (SELECT count(*) FROM completed WHERE result_status = 'skipped_protocol')
                     AS protocol_skipped_count,
                   (SELECT count(*) FROM completed WHERE result_status = 'skipped_resource')
                     AS resource_skipped_count
            FROM knowledge_base_graph_config config
            WHERE config.workspace_id = :workspace_id
              AND config.kb_id = :kb_id
            """
        )

    @staticmethod
    def _candidate_statement(prefix: bool):
        operator = "LIKE" if prefix else "="
        value = ":surface_value || '%'" if prefix else ":surface_value"
        return text(
            f"""
            SELECT DISTINCT ON (mention.entity_key)
                   mention.entity_key, mention.entity_type,
                   mention.surface, mention.normalized_surface,
                   chunk.id AS index_chunk_id,
                   target.id AS indexed_document_version_id,
                   target.index_revision_id
            FROM graph_entity_mention mention
            JOIN index_graph_chunk graph
              ON graph.workspace_id = mention.workspace_id
             AND graph.kb_id = mention.kb_id
             AND graph.build_id = mention.build_id
             AND graph.index_chunk_id = mention.index_chunk_id
             AND graph.result_status = 'extracted'
            JOIN knowledge_base_graph_config config
              ON config.workspace_id = mention.workspace_id
             AND config.kb_id = mention.kb_id
             AND config.build_id = mention.build_id
             AND config.extractor_version = graph.extractor_version
             AND config.status = 'ready'
            JOIN index_chunk chunk
              ON chunk.workspace_id = graph.workspace_id
             AND chunk.kb_id = graph.kb_id
             AND chunk.id = graph.index_chunk_id
             AND chunk.excluded_at IS NULL
            JOIN indexed_document_version target
              ON target.workspace_id = chunk.workspace_id
             AND target.kb_id = chunk.kb_id
             AND target.id = chunk.indexed_document_version_id
             AND target.index_revision_id = (
                   SELECT active_index_revision_id
                   FROM knowledge_base
                   WHERE workspace_id = :workspace_id
                     AND id = :kb_id AND deleted_at IS NULL
                 )
             AND target.build_status = 'ready'
             AND target.serving_status = 'serving'
            JOIN document_version version
              ON version.workspace_id = target.workspace_id
             AND version.kb_id = target.kb_id
             AND version.id = target.document_version_id
             AND version.document_id = target.document_id
             AND version.source_status = 'available'
            JOIN document doc
              ON doc.workspace_id = target.workspace_id
             AND doc.kb_id = target.kb_id
             AND doc.id = target.document_id
             AND doc.deleted_at IS NULL
            JOIN knowledge_base kb
              ON kb.workspace_id = target.workspace_id
             AND kb.id = target.kb_id
             AND kb.deleted_at IS NULL
             AND kb.active_index_revision_id = target.index_revision_id
            WHERE mention.workspace_id = :workspace_id
              AND mention.kb_id = :kb_id
              AND mention.build_id = :build_id
              AND mention.normalized_surface {operator} {value}
            ORDER BY mention.entity_key, mention.index_chunk_id, mention.ordinal
            LIMIT :limit
            """
        )

    @staticmethod
    def _one_hop_statement():
        return text(_visible_edge_cte() + """
            SELECT edge.*
            FROM visible_edges edge
            WHERE edge.subject_entity_key IN :entry_keys
               OR edge.object_entity_key IN :entry_keys
               OR edge.source_chunk_id IN :seed_chunk_ids
            ORDER BY edge.relation_id, edge.source_chunk_id
            """)

    @staticmethod
    def _two_hop_statement():
        return text(_visible_edge_cte() + """
            SELECT
              first_edge.relation_id AS first_relation_id,
              first_edge.source_workspace_id AS first_source_workspace_id,
              first_edge.source_kb_id AS first_source_kb_id,
              first_edge.subject_entity_key AS first_subject_entity_key,
              first_edge.object_entity_key AS first_object_entity_key,
              first_edge.predicate AS first_predicate,
              first_edge.normalized_predicate AS first_normalized_predicate,
              first_edge.source_chunk_id AS first_source_chunk_id,
              first_edge.source_index_revision_id AS first_source_index_revision_id,
              first_edge.source_location AS first_source_location,
              first_edge.source_hierarchy AS first_source_hierarchy,
              first_edge.source_metadata AS first_source_metadata,
              first_edge.source_indexed_document_version_id AS first_source_indexed_document_version_id,
              first_edge.source_document_id AS first_source_document_id,
              first_edge.source_document_version_id AS first_source_document_version_id,
              first_edge.source_ordinal AS first_source_ordinal,
              first_edge.source_text AS first_source_text,
              first_edge.source_modality AS first_source_modality,
              first_edge.source_evidence_group_key AS first_source_evidence_group_key,
              first_edge.source_document_display_name AS first_source_document_display_name,
              first_edge.source_document_original_filename AS first_source_document_original_filename,
              second_edge.relation_id AS second_relation_id,
              second_edge.source_workspace_id AS second_source_workspace_id,
              second_edge.source_kb_id AS second_source_kb_id,
              second_edge.subject_entity_key AS second_subject_entity_key,
              second_edge.object_entity_key AS second_object_entity_key,
              second_edge.predicate AS second_predicate,
              second_edge.normalized_predicate AS second_normalized_predicate,
              second_edge.source_chunk_id AS second_source_chunk_id,
              second_edge.source_index_revision_id AS second_source_index_revision_id,
              second_edge.source_location AS second_source_location,
              second_edge.source_hierarchy AS second_source_hierarchy,
              second_edge.source_metadata AS second_source_metadata,
              second_edge.source_indexed_document_version_id AS second_source_indexed_document_version_id,
              second_edge.source_document_id AS second_source_document_id,
              second_edge.source_document_version_id AS second_source_document_version_id,
              second_edge.source_ordinal AS second_source_ordinal,
              second_edge.source_text AS second_source_text,
              second_edge.source_modality AS second_source_modality,
              second_edge.source_evidence_group_key AS second_source_evidence_group_key,
              second_edge.source_document_display_name AS second_source_document_display_name,
              second_edge.source_document_original_filename AS second_source_document_original_filename
            FROM visible_edges first_edge
            JOIN visible_edges second_edge
              ON second_edge.relation_id <> first_edge.relation_id
             AND (
                  (second_edge.subject_entity_key = first_edge.object_entity_key
                   OR second_edge.object_entity_key = first_edge.object_entity_key
                   OR second_edge.subject_entity_key = first_edge.subject_entity_key
                   OR second_edge.object_entity_key = first_edge.subject_entity_key)
             )
            WHERE first_edge.subject_entity_key IN :entry_keys
               OR first_edge.object_entity_key IN :entry_keys
               OR first_edge.source_chunk_id IN :seed_chunk_ids
            ORDER BY first_edge.relation_id, second_edge.relation_id
            """)


@dataclass(frozen=True, slots=True)
class _GraphEdge:
    relation_id: UUID
    subject_entity_key: str
    object_entity_key: str
    predicate: str
    normalized_predicate: str
    source_chunk: GraphChunkEvidence
    support_count: int = 1


def _config_snapshot(row: Any) -> GraphConfigSnapshot:
    return GraphConfigSnapshot(
        workspace_id=row["workspace_id"],
        knowledge_base_id=row["kb_id"],
        status=GraphConfigStatus(row["status"]),
        build_id=row["build_id"],
        chat_profile_revision_id=row["chat_profile_revision_id"],
        extractor_version=row["extractor_version"],
        preflight_extractor_version=row["preflight_extractor_version"],
        last_error_code=row["last_error_code"],
        eligible_chunk_count=int(row["eligible_chunk_count"] or 0),
        processed_chunk_count=int(row["processed_chunk_count"] or 0),
        extracted_chunk_count=int(row["extracted_chunk_count"] or 0),
        empty_chunk_count=int(row["empty_chunk_count"] or 0),
        protocol_skipped_count=int(row["protocol_skipped_count"] or 0),
        resource_skipped_count=int(row["resource_skipped_count"] or 0),
    )


def _visible_edge_cte() -> str:
    return """
            WITH visible_edges AS (
              SELECT rel.id AS relation_id,
                     rel.workspace_id AS source_workspace_id,
                     rel.kb_id AS source_kb_id,
                     rel.subject_entity_key, rel.object_entity_key,
                     rel.predicate, rel.normalized_predicate,
                     chunk.id AS source_chunk_id,
                     target.index_revision_id AS source_index_revision_id,
                     target.id AS source_indexed_document_version_id,
                     target.document_id AS source_document_id,
                     target.document_version_id AS source_document_version_id,
                     chunk.ordinal AS source_ordinal,
                     chunk.content AS source_text,
                     chunk.source_location,
                     chunk.hierarchy AS source_hierarchy,
                     chunk.source_metadata,
                     chunk.modality AS source_modality,
                     chunk.evidence_group_key AS source_evidence_group_key,
                     doc.display_name AS source_document_display_name,
                     version.original_filename AS source_document_original_filename
              FROM graph_relation_assertion rel
              JOIN index_graph_chunk graph
                ON graph.workspace_id = rel.workspace_id
               AND graph.kb_id = rel.kb_id
               AND graph.build_id = rel.build_id
               AND graph.index_chunk_id = rel.index_chunk_id
               AND graph.result_status = 'extracted'
              JOIN index_chunk chunk
                ON chunk.workspace_id = graph.workspace_id
               AND chunk.kb_id = graph.kb_id
               AND chunk.id = graph.index_chunk_id
               AND chunk.modality IN ('text', 'table')
               AND chunk.excluded_at IS NULL
              JOIN indexed_document_version target
                ON target.workspace_id = chunk.workspace_id
               AND target.kb_id = chunk.kb_id
               AND target.id = chunk.indexed_document_version_id
               AND target.index_revision_id = :index_revision_id
               AND target.build_status = 'ready'
               AND target.serving_status = 'serving'
              JOIN index_revision revision
                ON revision.workspace_id = target.workspace_id
               AND revision.kb_id = target.kb_id
               AND revision.id = target.index_revision_id
               AND revision.status = 'active'
              JOIN knowledge_base kb
                ON kb.workspace_id = target.workspace_id
               AND kb.id = target.kb_id
               AND kb.active_index_revision_id = target.index_revision_id
               AND kb.deleted_at IS NULL
              JOIN knowledge_base_graph_config config
                ON config.workspace_id = rel.workspace_id
               AND config.kb_id = rel.kb_id
               AND config.build_id = rel.build_id
               AND config.extractor_version = graph.extractor_version
               AND config.status = 'ready'
              JOIN document doc
                ON doc.workspace_id = target.workspace_id
               AND doc.kb_id = target.kb_id
               AND doc.id = target.document_id
               AND doc.deleted_at IS NULL
              JOIN document_version version
                ON version.workspace_id = target.workspace_id
               AND version.kb_id = target.kb_id
               AND version.id = target.document_version_id
               AND version.document_id = target.document_id
               AND version.source_status = 'available'
              WHERE rel.workspace_id = :workspace_id
                AND rel.kb_id = :kb_id
                AND rel.build_id = :build_id
                AND rel.subject_entity_key <> rel.object_entity_key
            )
            """


def _chunk_from_values(values: dict[str, Any], prefix: str = "") -> GraphChunkEvidence:
    def get(name: str) -> Any:
        return values[f"{prefix}{name}"]

    return GraphChunkEvidence(
        workspace_id=get("source_workspace_id"),
        knowledge_base_id=get("source_kb_id"),
        index_revision_id=get("source_index_revision_id"),
        index_chunk_id=get("source_chunk_id"),
        indexed_document_version_id=get("source_indexed_document_version_id"),
        document_id=get("source_document_id"),
        document_version_id=get("source_document_version_id"),
        ordinal=get("source_ordinal"),
        text=get("source_text"),
        source_location=get("source_location"),
        hierarchy=get("source_hierarchy"),
        source_metadata=get("source_metadata"),
        modality=get("source_modality"),
        evidence_group_key=get("source_evidence_group_key"),
        document_display_name=get("source_document_display_name"),
        document_original_filename=get("source_document_original_filename"),
    )


def _edge_from_row(row: Any, prefix: str = "") -> _GraphEdge:
    values = dict(row)
    chunk = _chunk_from_values(values, prefix)
    return _GraphEdge(
        relation_id=values[f"{prefix}relation_id"],
        subject_entity_key=values[f"{prefix}subject_entity_key"],
        object_entity_key=values[f"{prefix}object_entity_key"],
        predicate=values[f"{prefix}predicate"],
        normalized_predicate=values[f"{prefix}normalized_predicate"],
        source_chunk=chunk,
    )


def _edge_pair_from_row(row: Any) -> tuple[_GraphEdge, _GraphEdge]:
    return _edge_from_row(row, "first_"), _edge_from_row(row, "second_")


def _aggregate_edges(
    values: Any,
    seed_chunks: set[UUID],
) -> tuple[_GraphEdge, ...]:
    grouped: dict[tuple[str, str, str], list[_GraphEdge]] = {}
    for edge in values:
        grouped.setdefault(
            (edge.subject_entity_key, edge.object_entity_key, edge.normalized_predicate),
            [],
        ).append(edge)
    return tuple(_representative(key, values, seed_chunks) for key, values in sorted(grouped.items()))


def _aggregate_edge_pairs(
    values: Any,
    seed_chunks: set[UUID],
) -> tuple[tuple[_GraphEdge, _GraphEdge], ...]:
    materialized = tuple(values)
    grouped: dict[tuple[str, str, str], list[_GraphEdge]] = {}
    for first, second in materialized:
        for edge in (first, second):
            grouped.setdefault(
                (edge.subject_entity_key, edge.object_entity_key, edge.normalized_predicate),
                [],
            ).append(edge)
    representatives = {
        key: _representative(key, values, seed_chunks)
        for key, values in grouped.items()
    }
    return tuple(
        (
            representatives[
                (first.subject_entity_key, first.object_entity_key, first.normalized_predicate)
            ],
            representatives[
                (second.subject_entity_key, second.object_entity_key, second.normalized_predicate)
            ],
        )
        for first, second in materialized
    )


def _representative(
    key: tuple[str, str, str],
    values: list[_GraphEdge],
    seed_chunks: set[UUID],
) -> _GraphEdge:
    representative = min(values, key=lambda edge: _edge_sort_key(edge, seed_chunks))
    return _GraphEdge(
        relation_id=representative.relation_id,
        subject_entity_key=key[0],
        object_entity_key=key[1],
        predicate=representative.predicate,
        normalized_predicate=key[2],
        source_chunk=representative.source_chunk,
        support_count=len(values),
    )


def _edge_sort_key(edge: _GraphEdge, seed_chunks: set[UUID]) -> tuple[int, int, int]:
    return (
        0 if edge.source_chunk.index_chunk_id in seed_chunks else 1,
        edge.source_chunk.index_chunk_id.int,
        edge.relation_id.int,
    )


def _build_paths(
    entry_keys: tuple[str, ...],
    one_edges: tuple[_GraphEdge, ...],
    two_edges: tuple[tuple[_GraphEdge, _GraphEdge], ...],
    seed_chunks: set[UUID],
    *,
    max_paths: int,
) -> tuple[tuple[GraphPathCandidate, ...], tuple[GraphChunkEvidence, ...], int]:
    adjacency: dict[str, list[_GraphEdge]] = {}
    for edge in one_edges:
        adjacency.setdefault(edge.subject_entity_key, []).append(edge)
        adjacency.setdefault(edge.object_entity_key, []).append(edge)

    seed_entry_keys: set[str] = set()
    anchor_by_entry: dict[str, UUID] = {}
    for edge in one_edges:
        if edge.source_chunk.index_chunk_id in seed_chunks:
            for key in (edge.subject_entity_key, edge.object_entity_key):
                seed_entry_keys.add(key)
                anchor_by_entry.setdefault(key, edge.source_chunk.index_chunk_id)

    candidates: list[GraphPathCandidate] = []
    seen: set[str] = set()
    entries = tuple(dict.fromkeys(entry_keys + tuple(sorted(seed_entry_keys))))
    for entry in entries:
        neighbors = _bounded_neighbors(adjacency.get(entry, ()), entry, seed_chunks)
        for edge in neighbors:
            path = _path_for_one(entry, edge, seed_entry_keys, anchor_by_entry)
            if path.path_id not in seen:
                candidates.append(path)
                seen.add(path.path_id)

    bounded_first_edges = {
        entry: {
            edge.relation_id
            for edge in _bounded_neighbors(
                adjacency.get(entry, ()), entry, seed_chunks
            )
        }
        for entry in entries
    }
    two_hop_neighbors: dict[str, set[str]] = {}
    for first, second in two_edges:
        intermediate = _shared_intermediate(first, second, entries)
        if intermediate is None:
            continue
        entry, final = intermediate
        if first.relation_id not in bounded_first_edges.get(entry, set()):
            continue
        seen_neighbors = two_hop_neighbors.setdefault(entry, set())
        if (
            final not in seen_neighbors
            and len(seen_neighbors) >= GRAPH_MAX_NEIGHBORS_PER_ENTRY
        ):
            continue
        seen_neighbors.add(final)
        if len(
            {
                entry,
                first.subject_entity_key,
                first.object_entity_key,
                second.subject_entity_key,
                second.object_entity_key,
            }
        ) < 3:
            continue
        intermediate_key = _other_endpoint(first, entry)
        if intermediate_key is None:
            continue
        degree = _distinct_degree(adjacency, intermediate_key)
        if degree > GRAPH_MAX_INTERMEDIATE_DEGREE:
            continue
        if first.source_chunk.index_chunk_id in seed_chunks:
            seed_entry_keys.add(entry)
            anchor_by_entry.setdefault(entry, first.source_chunk.index_chunk_id)
        path = _path_for_two(
            entry,
            first,
            second,
            seed_entry_keys,
            anchor_by_entry,
        )
        if path.path_id not in seen:
            candidates.append(path)
            seen.add(path.path_id)

    candidates.sort(
        key=lambda path: (
            path.hop_count,
            0 if path.seed_entry else 1,
            -sum(hop.support_count for hop in path.hops),
            path.path_id,
        )
    )
    rejected = max(0, len(candidates) - max_paths)
    selected = tuple(
        GraphPathCandidate(
            path_id=path.path_id,
            entry_entity_key=path.entry_entity_key,
            hops=path.hops,
            anchor_chunk_id=path.anchor_chunk_id,
            rank=rank,
            seed_entry=path.seed_entry,
        )
        for rank, path in enumerate(candidates[:max_paths], start=1)
    )
    edge_by_chunk = {
        edge.source_chunk.index_chunk_id: edge.source_chunk
        for edge in one_edges
    }
    for first, second in two_edges:
        edge_by_chunk[first.source_chunk.index_chunk_id] = first.source_chunk
        edge_by_chunk[second.source_chunk.index_chunk_id] = second.source_chunk
    chunks = tuple(
        edge_by_chunk[chunk_id]
        for chunk_id in sorted(
            {
                chunk_id
                for path in selected
                for chunk_id in path.source_chunk_ids
            },
            key=lambda value: value.int,
        )
        if chunk_id in edge_by_chunk
    )
    return selected, chunks, rejected


def _bounded_neighbors(
    edges: list[_GraphEdge] | tuple[_GraphEdge, ...],
    entry: str,
    seed_chunks: set[UUID],
) -> tuple[_GraphEdge, ...]:
    selected: list[_GraphEdge] = []
    neighbor_keys: set[str] = set()
    for edge in sorted(edges, key=lambda item: _edge_sort_key(item, seed_chunks)):
        neighbor = _other_endpoint(edge, entry)
        if neighbor is None:
            continue
        if neighbor not in neighbor_keys and len(neighbor_keys) >= 8:
            continue
        neighbor_keys.add(neighbor)
        selected.append(edge)
    return tuple(selected)


def _path_for_one(
    entry: str,
    edge: _GraphEdge,
    seed_entry_keys: set[str],
    anchor_by_entry: dict[str, UUID],
) -> GraphPathCandidate:
    anchor = anchor_by_entry.get(entry, edge.source_chunk.index_chunk_id)
    return _path_candidate(entry, (edge,), anchor, entry in seed_entry_keys)


def _path_for_two(
    entry: str,
    first: _GraphEdge,
    second: _GraphEdge,
    seed_entry_keys: set[str],
    anchor_by_entry: dict[str, UUID],
) -> GraphPathCandidate:
    anchor = anchor_by_entry.get(entry, first.source_chunk.index_chunk_id)
    return _path_candidate(entry, (first, second), anchor, entry in seed_entry_keys)


def _path_candidate(
    entry: str,
    edges: tuple[_GraphEdge, ...],
    anchor: UUID,
    seed_entry: bool,
) -> GraphPathCandidate:
    path_material = ":".join(
        [entry]
        + [
            f"{edge.relation_id}:{edge.subject_entity_key}:{edge.object_entity_key}"
            for edge in edges
        ]
    )
    path_id = hashlib.sha256(path_material.encode("utf-8")).hexdigest()[:32]
    return GraphPathCandidate(
        path_id=path_id,
        entry_entity_key=entry,
        hops=tuple(
            GraphPathHop(
                subject_entity_key=edge.subject_entity_key,
                object_entity_key=edge.object_entity_key,
                predicate=edge.predicate,
                normalized_predicate=edge.normalized_predicate,
                relation_id=edge.relation_id,
                source_chunk_id=edge.source_chunk.index_chunk_id,
                source_index_revision_id=edge.source_chunk.index_revision_id,
                source_location=edge.source_chunk.source_location,
                support_count=edge.support_count,
            )
            for edge in edges
        ),
        anchor_chunk_id=anchor,
        rank=1,
        seed_entry=seed_entry,
    )


def _shared_intermediate(
    first: _GraphEdge,
    second: _GraphEdge,
    entry_keys: tuple[str, ...],
) -> tuple[str, str] | None:
    for entry in entry_keys:
        first_other = _other_endpoint(first, entry)
        if first_other is None:
            continue
        if first_other in {
            second.subject_entity_key,
            second.object_entity_key,
        }:
            final = (
                second.object_entity_key
                if second.subject_entity_key == first_other
                else second.subject_entity_key
            )
            if final != entry:
                return entry, final
    return None


def _other_endpoint(edge: _GraphEdge, entity_key: str) -> str | None:
    if edge.subject_entity_key == entity_key:
        return edge.object_entity_key
    if edge.object_entity_key == entity_key:
        return edge.subject_entity_key
    return None


def _distinct_degree(adjacency: dict[str, list[_GraphEdge]], key: str) -> int:
    neighbors = {
        _other_endpoint(edge, key)
        for edge in adjacency.get(key, ())
        if _other_endpoint(edge, key) is not None
    }
    return len(neighbors)
