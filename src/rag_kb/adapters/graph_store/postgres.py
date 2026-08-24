"""PostgreSQL metadata and evidence mapping for Graphiti retrieval."""

from __future__ import annotations

from typing import Any
import unicodedata
from uuid import UUID, NAMESPACE_URL, uuid4, uuid5

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_kb.domain import (
    GraphChunkEvidence,
    GraphConfigSnapshot,
    GraphConfigStatus,
    GraphitiBuildSnapshot,
    GraphitiBuildStatus,
    GraphitiEdgeResult,
    GraphitiPathResult,
    GraphPathCandidate,
    GraphPathHop,
    GraphTraversalResult,
)
from rag_kb.domain.graph import GRAPH_MAX_PATHS


class PgGraphStore:
    """Resolve active Graphiti builds and hydrate edges to original chunks."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get_config(
        self,
        workspace_id: UUID,
        knowledge_base_id: UUID,
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
        return _config_snapshot(row) if row is not None else None

    async def get_active_graphiti_build(
        self,
        workspace_id: UUID,
        knowledge_base_id: UUID,
    ) -> GraphitiBuildSnapshot | None:
        async with self._sessions() as session:
            async with session.begin():
                row = (
                    await session.execute(
                        text(
                            """
                            SELECT build.*
                            FROM knowledge_base_graph_config config
                            JOIN graphiti_graph_build build
                              ON build.workspace_id = config.workspace_id
                             AND build.kb_id = config.kb_id
                             AND build.build_id = config.active_build_id
                             AND build.status = 'ready'
                            WHERE config.workspace_id = :workspace_id
                              AND config.kb_id = :kb_id
                              AND config.status <> 'disabled'
                            """
                        ),
                        {"workspace_id": workspace_id, "kb_id": knowledge_base_id},
                    )
                ).mappings().one_or_none()
        if row is None:
            return None
        return GraphitiBuildSnapshot(
            workspace_id=row["workspace_id"],
            knowledge_base_id=row["kb_id"],
            build_id=row["build_id"],
            group_id=row["group_id"],
            status=GraphitiBuildStatus(row["status"]),
            index_revision_id=row["index_revision_id"],
            serving_chunk_digest=row["serving_chunk_digest"],
            expected_episode_count=row["expected_episode_count"],
            chat_profile_revision_id=row["chat_profile_revision_id"],
            embedding_profile_revision_id=row["embedding_profile_revision_id"],
            embedding_model=row["embedding_model"],
            embedding_dimension=row["embedding_dimension"],
            extractor_version=row["extractor_version"],
            superseded_by=row["superseded_by"],
            schema_profile_key=row["schema_profile_key"],
            schema_profile_digest=row["schema_profile_digest"],
        )

    async def first_graphiti_episode_uuid(
        self,
        workspace_id: UUID,
        knowledge_base_id: UUID,
        build_id: UUID,
    ) -> str | None:
        async with self._sessions() as session:
            async with session.begin():
                return await session.scalar(
                    text(
                        """
                        SELECT episode_uuid
                        FROM graphiti_episode_chunk
                        WHERE workspace_id = :workspace_id AND kb_id = :kb_id
                          AND build_id = :build_id
                        ORDER BY created_at, id
                        LIMIT 1
                        """
                    ),
                    {
                        "workspace_id": workspace_id,
                        "kb_id": knowledge_base_id,
                        "build_id": build_id,
                    },
                )

    async def hydrate_graphiti_edges(
        self,
        *,
        workspace_id: UUID,
        knowledge_base_id: UUID,
        build_id: UUID,
        index_revision_id: UUID,
        edges: tuple[GraphitiEdgeResult, ...],
    ) -> GraphTraversalResult | None:
        """Compatibility hydration for evaluator edge probes.

        Production retrieval uses ``hydrate_graphiti_paths``. Edges without
        real endpoints are intentionally not converted into fabricated paths.
        """

        paths = tuple(
            GraphitiPathResult(
                path_id=str(uuid5(NAMESPACE_URL, f"graphiti:{edge.edge_uuid}")),
                entry_entity_uuid=edge.source_entity_uuid,
                hops=(edge,),
                rank=edge.rank,
                seed_entry=False,
            )
            for edge in edges
            if edge.source_entity_uuid
            and edge.target_entity_uuid
            and edge.source_entity_uuid != edge.target_entity_uuid
        )
        return await self.hydrate_graphiti_paths(
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            build_id=build_id,
            index_revision_id=index_revision_id,
            paths=paths,
        )

    async def hydrate_graphiti_paths(
        self,
        *,
        workspace_id: UUID,
        knowledge_base_id: UUID,
        build_id: UUID,
        index_revision_id: UUID,
        paths: tuple[GraphitiPathResult, ...],
    ) -> GraphTraversalResult | None:
        episode_ids = tuple(
            dict.fromkeys(
                episode_uuid
                for path in paths
                for hop in path.hops
                for episode_uuid in hop.episode_uuids
            )
        )
        if not episode_ids:
            return GraphTraversalResult(resolved_active_revision_id=index_revision_id)
        async with self._sessions() as session:
            async with session.begin():
                active_revision_id = await session.scalar(
                    text(
                        "SELECT active_index_revision_id FROM knowledge_base "
                        "WHERE workspace_id = :workspace_id AND id = :kb_id "
                        "AND deleted_at IS NULL"
                    ),
                    {"workspace_id": workspace_id, "kb_id": knowledge_base_id},
                )
                if active_revision_id is None:
                    return None
                if active_revision_id != index_revision_id:
                    return GraphTraversalResult(
                        resolved_active_revision_id=active_revision_id
                    )
                rows = (
                    await session.execute(
                        text(
                            """
                            SELECT mapping.episode_uuid,
                                   chunk.workspace_id AS source_workspace_id,
                                   chunk.kb_id AS source_kb_id,
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
                            FROM graphiti_episode_chunk mapping
                            JOIN index_chunk chunk
                              ON chunk.workspace_id = mapping.workspace_id
                             AND chunk.kb_id = mapping.kb_id
                             AND chunk.id = mapping.index_chunk_id
                             AND chunk.content_hash = mapping.content_hash
                             AND chunk.excluded_at IS NULL
                            JOIN indexed_document_version target
                              ON target.workspace_id = chunk.workspace_id
                             AND target.kb_id = chunk.kb_id
                             AND target.id = chunk.indexed_document_version_id
                             AND target.index_revision_id = :index_revision_id
                             AND target.build_status = 'ready'
                             AND target.serving_status = 'serving'
                            JOIN document doc
                              ON doc.workspace_id = target.workspace_id
                             AND doc.kb_id = target.kb_id
                             AND doc.id = target.document_id
                             AND doc.deleted_at IS NULL
                            JOIN document_version version
                              ON version.workspace_id = target.workspace_id
                             AND version.kb_id = target.kb_id
                             AND version.id = target.document_version_id
                             AND version.source_status = 'available'
                            WHERE mapping.workspace_id = :workspace_id
                              AND mapping.kb_id = :kb_id
                              AND mapping.build_id = :build_id
                              AND mapping.index_revision_id = :index_revision_id
                              AND mapping.episode_uuid IN :episode_uuids
                            """
                        ).bindparams(bindparam("episode_uuids", expanding=True)),
                        {
                            "workspace_id": workspace_id,
                            "kb_id": knowledge_base_id,
                            "build_id": build_id,
                            "index_revision_id": index_revision_id,
                            "episode_uuids": list(episode_ids),
                        },
                    )
                ).mappings().all()
        chunks_by_episode = {
            row["episode_uuid"]: _chunk_from_values(dict(row)) for row in rows
        }
        hydrated_paths, chunks, rejected = _bounded_graphiti_paths(
            paths,
            chunks_by_episode,
        )
        return GraphTraversalResult(
            resolved_active_revision_id=index_revision_id,
            paths=hydrated_paths,
            chunks=chunks,
            rejected_path_count=rejected,
            mapped_episode_ids=tuple(
                dict.fromkeys(str(row["episode_uuid"]) for row in rows)
            ),
            mapped_episode_chunks=tuple(
                dict.fromkeys(
                    (
                        str(row["episode_uuid"]),
                        row["source_chunk_id"],
                    )
                    for row in rows
                )
            ),
        )

    async def schedule_graphiti_rebuild(
        self,
        workspace_id: UUID,
        knowledge_base_id: UUID,
        failed_build_id: UUID,
    ) -> bool:
        new_build_id = uuid4()
        async with self._sessions() as session:
            async with session.begin():
                row = (
                    await session.execute(
                        text(
                            """
                            SELECT config.build_id, config.chat_profile_revision_id,
                                   build.index_revision_id,
                                   build.serving_chunk_digest,
                                   build.expected_episode_count,
                                   build.embedding_profile_revision_id,
                                   build.embedding_model,
                                   build.embedding_dimension,
                                   build.extractor_version,
                                   build.schema_profile_key,
                                   build.schema_profile_digest
                            FROM knowledge_base_graph_config config
                            JOIN graphiti_graph_build build
                              ON build.workspace_id = config.workspace_id
                             AND build.kb_id = config.kb_id
                             AND build.build_id = config.active_build_id
                            WHERE config.workspace_id = :workspace_id
                              AND config.kb_id = :kb_id
                              AND config.active_build_id = :failed_build_id
                              AND config.build_id = :failed_build_id
                            FOR UPDATE OF config
                            """
                        ),
                        {
                            "workspace_id": workspace_id,
                            "kb_id": knowledge_base_id,
                            "failed_build_id": failed_build_id,
                        },
                    )
                ).mappings().one_or_none()
                if row is None:
                    return False
                group_id = f"ws_{workspace_id}_kb_{knowledge_base_id}_b_{new_build_id}"
                await session.execute(
                    text(
                        """
                        INSERT INTO graphiti_graph_build (
                            build_id, workspace_id, kb_id, group_id, status,
                            index_revision_id, serving_chunk_digest,
                            expected_episode_count, chat_profile_revision_id,
                            embedding_profile_revision_id, embedding_model,
                            embedding_dimension, extractor_version,
                            schema_profile_key, schema_profile_digest
                        ) VALUES (
                            :build_id, :workspace_id, :kb_id, :group_id, 'building',
                            :index_revision_id, :serving_chunk_digest,
                            :expected_episode_count, :chat_profile_revision_id,
                            :embedding_profile_revision_id, :embedding_model,
                            :embedding_dimension, :extractor_version,
                            :schema_profile_key, :schema_profile_digest
                        )
                        """
                    ),
                    {
                        **dict(row),
                        "build_id": new_build_id,
                        "workspace_id": workspace_id,
                        "kb_id": knowledge_base_id,
                        "group_id": group_id,
                    },
                )
                await session.execute(
                    text(
                        """
                        UPDATE knowledge_base_graph_config
                        SET build_id = :build_id, status = 'building',
                            preflight_extractor_version = NULL,
                            last_error_code = 'graph_runtime_probe_failed',
                            updated_at = now()
                        WHERE workspace_id = :workspace_id AND kb_id = :kb_id
                          AND active_build_id = :failed_build_id
                          AND build_id = :failed_build_id
                        """
                    ),
                    {
                        "build_id": new_build_id,
                        "workspace_id": workspace_id,
                        "kb_id": knowledge_base_id,
                        "failed_build_id": failed_build_id,
                    },
                )
        return True

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
                AND chunk.excluded_at IS NULL
            ), completed AS (
              SELECT mapping.index_chunk_id
              FROM graphiti_episode_chunk mapping
              JOIN eligible
                ON eligible.index_chunk_id = mapping.index_chunk_id
               AND eligible.content_hash = mapping.content_hash
              JOIN knowledge_base_graph_config config
                ON config.workspace_id = mapping.workspace_id
               AND config.kb_id = mapping.kb_id
               AND config.build_id = mapping.build_id
              WHERE mapping.workspace_id = :workspace_id
                AND mapping.kb_id = :kb_id
            )
            SELECT config.kb_id, config.workspace_id, config.status,
                   config.build_id, config.active_build_id,
                   config.chat_profile_revision_id,
                   config.extractor_version, config.preflight_extractor_version,
                   config.last_error_code,
                   config.schema_profile_key, config.schema_profile_digest,
                   build.group_id, build.index_revision_id,
                   build.embedding_profile_revision_id, build.embedding_model,
                   build.embedding_dimension,
                   active_build.schema_profile_key AS active_build_schema_profile_key,
                   active_build.schema_profile_digest AS active_build_schema_profile_digest,
                   (SELECT count(*) FROM eligible) AS eligible_chunk_count,
                   (SELECT count(*) FROM completed) AS processed_chunk_count
            FROM knowledge_base_graph_config config
            LEFT JOIN graphiti_graph_build build
              ON build.workspace_id = config.workspace_id
             AND build.kb_id = config.kb_id
             AND build.build_id = config.build_id
            LEFT JOIN graphiti_graph_build active_build
              ON active_build.workspace_id = config.workspace_id
             AND active_build.kb_id = config.kb_id
             AND active_build.build_id = config.active_build_id
            WHERE config.workspace_id = :workspace_id
              AND config.kb_id = :kb_id
            """
        )


def _bounded_graphiti_paths(
    raw_paths: tuple[GraphitiPathResult, ...],
    chunks_by_episode: dict[str, GraphChunkEvidence],
) -> tuple[tuple[GraphPathCandidate, ...], tuple[GraphChunkEvidence, ...], int]:
    paths: list[GraphPathCandidate] = []
    chunks: dict[UUID, GraphChunkEvidence] = {}
    rejected = 0
    for raw_path in raw_paths:
        if len(paths) >= GRAPH_MAX_PATHS:
            rejected += 1
            continue
        hydrated_hops: list[GraphPathHop] = []
        path_chunks: dict[UUID, GraphChunkEvidence] = {}
        complete = True
        for edge in raw_path.hops:
            mapped = tuple(
                chunks_by_episode[episode_uuid]
                for episode_uuid in edge.episode_uuids
                if episode_uuid in chunks_by_episode
            )
            if not mapped:
                complete = False
                break
            chunk = max(
                mapped,
                key=lambda candidate: _graph_support_key(edge, candidate),
            )
            path_chunks.setdefault(chunk.index_chunk_id, chunk)
            hydrated_hops.append(
                GraphPathHop(
                    subject_entity_key=edge.source_entity_uuid,
                    object_entity_key=edge.target_entity_uuid,
                    predicate=edge.fact or edge.relation_type or "graphiti_fact",
                    normalized_predicate=_normalized_graph_text(
                        edge.relation_type or edge.fact or "graphiti_fact"
                    ),
                    relation_id=uuid5(NAMESPACE_URL, f"graphiti:{edge.edge_uuid}"),
                    source_chunk_id=chunk.index_chunk_id,
                    source_index_revision_id=chunk.index_revision_id,
                    source_location=chunk.source_location,
                    support_count=len({item.index_chunk_id for item in mapped}),
                )
            )
        if not complete or len(hydrated_hops) != len(raw_path.hops):
            rejected += 1
            continue
        chunks.update(path_chunks)
        paths.append(
            GraphPathCandidate(
                path_id=raw_path.path_id,
                entry_entity_key=raw_path.entry_entity_uuid,
                hops=tuple(hydrated_hops),
                anchor_chunk_id=hydrated_hops[0].source_chunk_id,
                rank=raw_path.rank,
                seed_entry=raw_path.seed_entry,
            )
        )
    return tuple(paths), tuple(chunks.values()), rejected


def _graph_support_key(
    edge: GraphitiEdgeResult,
    chunk: GraphChunkEvidence,
) -> tuple[int, int, int, int]:
    """Prefer the mapped episode that most directly supports an edge fact."""

    content = _normalized_graph_text(chunk.text)
    fact = _normalized_graph_text(edge.fact)
    source = _normalized_graph_text(edge.source_entity_name)
    target = _normalized_graph_text(edge.target_entity_name)
    fact_characters = set(fact)
    overlap = len(fact_characters & set(content))
    return (
        int(bool(source) and source in content)
        + int(bool(target) and target in content),
        int(bool(fact) and fact in content),
        overlap,
        -chunk.ordinal,
    )


def _normalized_graph_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _config_snapshot(row: Any) -> GraphConfigSnapshot:
    processed = int(row["processed_chunk_count"] or 0)
    return GraphConfigSnapshot(
        workspace_id=row["workspace_id"],
        knowledge_base_id=row["kb_id"],
        status=GraphConfigStatus(row["status"]),
        build_id=row["build_id"],
        active_build_id=row["active_build_id"],
        chat_profile_revision_id=row["chat_profile_revision_id"],
        extractor_version=row["extractor_version"],
        preflight_extractor_version=row["preflight_extractor_version"],
        last_error_code=row["last_error_code"],
        eligible_chunk_count=int(row["eligible_chunk_count"] or 0),
        processed_chunk_count=processed,
        extracted_chunk_count=processed,
        group_id=row["group_id"],
        index_revision_id=row["index_revision_id"],
        embedding_profile_revision_id=row["embedding_profile_revision_id"],
        embedding_model=row["embedding_model"],
        embedding_dimension=row["embedding_dimension"],
        schema_profile_key=row["schema_profile_key"],
        schema_profile_digest=row["schema_profile_digest"],
        active_build_schema_profile_key=row["active_build_schema_profile_key"],
        active_build_schema_profile_digest=row["active_build_schema_profile_digest"],
    )


def _chunk_from_values(values: dict[str, Any]) -> GraphChunkEvidence:
    return GraphChunkEvidence(
        workspace_id=values["source_workspace_id"],
        knowledge_base_id=values["source_kb_id"],
        index_revision_id=values["source_index_revision_id"],
        index_chunk_id=values["source_chunk_id"],
        indexed_document_version_id=values["source_indexed_document_version_id"],
        document_id=values["source_document_id"],
        document_version_id=values["source_document_version_id"],
        ordinal=values["source_ordinal"],
        text=values["source_text"],
        source_location=values["source_location"],
        hierarchy=values["source_hierarchy"],
        source_metadata=values["source_metadata"],
        modality=values["source_modality"],
        evidence_group_key=values["source_evidence_group_key"],
        document_display_name=values["source_document_display_name"],
        document_original_filename=values["source_document_original_filename"],
    )
