from __future__ import annotations

import os
import unittest
from uuid import UUID, uuid4

import asyncpg

from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.domain import (
    GRAPH_EXTRACTOR_VERSION,
    GraphConfigStatus,
    GraphWorkKind,
)
from rag_kb.uow import execute_in_transaction
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")
WORKSPACE_ID = UUID("01900000-0000-7000-8000-000000001a01")


@unittest.skipUnless(
    MIGRATION_DSN and RUNTIME_SQLALCHEMY_DSN,
    "database integration DSNs are not configured",
)
class GraphBuildResumeDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute("TRUNCATE TABLE workspace CASCADE")
            self.fixture = await _seed_graph_fixture(connection)
        finally:
            await connection.close()
        self.database = create_database_resources(
            RUNTIME_SQLALCHEMY_DSN,
            pool_size=2,
            max_overflow=0,
            process=DatabaseProcess.API,
        )
        self.factory = SqlAlchemyUnitOfWorkFactory(
            self.database.sessions,
            WORKSPACE_ID,
        )

    async def asyncTearDown(self) -> None:
        await self.database.close()

    async def test_failed_retry_reuses_frozen_build_and_existing_mapping(self) -> None:
        async def build_and_fail(uow):
            configured = await uow.graph.configure(
                self.fixture.kb_id,
                chat_profile_revision_id=self.fixture.chat_profile_revision_id,
                enabled=True,
                extractor_version=GRAPH_EXTRACTOR_VERSION,
            )
            await uow.graph.save_preflight_success(
                self.fixture.kb_id,
                build_id=configured.build_id,
                extractor_version=GRAPH_EXTRACTOR_VERSION,
            )
            saved = await uow.graph.save_graphiti_episode(
                kb_id=self.fixture.kb_id,
                build_id=configured.build_id,
                index_chunk_id=self.fixture.chunk_id,
                content_hash=self.fixture.content_hash,
                episode_uuid="episode-1",
            )
            await uow.graph.mark_failed(
                self.fixture.kb_id,
                build_id=configured.build_id,
                error_code="graphiti_episode_extraction_failed:test",
            )
            return configured, saved, uow.graph.take_retired_graphiti_builds()

        failed, saved, retired = await execute_in_transaction(
            self.factory,
            build_and_fail,
        )
        self.assertTrue(saved)
        self.assertEqual(retired, ())

        async def resume(uow):
            snapshot = await uow.graph.retry(
                self.fixture.kb_id,
                extractor_version=GRAPH_EXTRACTOR_VERSION,
            )
            work = await uow.graph.next_work_item()
            return snapshot, work, uow.graph.take_retired_graphiti_builds()

        resumed, work, retired = await execute_in_transaction(self.factory, resume)

        self.assertEqual(resumed.status, GraphConfigStatus.BUILDING)
        self.assertEqual(resumed.build_id, failed.build_id)
        self.assertEqual(resumed.group_id, failed.group_id)
        self.assertEqual(resumed.processed_chunk_count, 1)
        self.assertEqual(
            resumed.preflight_extractor_version,
            GRAPH_EXTRACTOR_VERSION,
        )
        self.assertEqual(retired, ())
        self.assertIsNotNone(work)
        assert work is not None
        self.assertEqual(work.kind, GraphWorkKind.FINALIZE)

        await execute_in_transaction(
            self.factory,
            lambda uow: uow.graph.mark_failed(
                self.fixture.kb_id,
                build_id=resumed.build_id,
                error_code="graphiti_finalize_failed:test",
            ),
        )
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute(
                "UPDATE index_chunk SET content = $1, content_hash = $2 WHERE id = $3",
                "changed grounded fact",
                "c" * 64,
                self.fixture.chunk_id,
            )
        finally:
            await connection.close()

        async def rotate(uow):
            snapshot = await uow.graph.retry(
                self.fixture.kb_id,
                extractor_version=GRAPH_EXTRACTOR_VERSION,
            )
            return snapshot, uow.graph.take_retired_graphiti_builds()

        rotated, retired = await execute_in_transaction(self.factory, rotate)

        self.assertNotEqual(rotated.build_id, resumed.build_id)
        self.assertEqual(rotated.status, GraphConfigStatus.BUILDING)
        self.assertEqual(rotated.processed_chunk_count, 0)
        self.assertIsNone(rotated.preflight_extractor_version)
        self.assertEqual(tuple(item.build_id for item in retired), (resumed.build_id,))

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            previous = await connection.fetchrow(
                "SELECT status, superseded_by FROM graphiti_graph_build WHERE build_id = $1",
                resumed.build_id,
            )
            mapping_count = await connection.fetchval(
                "SELECT count(*) FROM graphiti_episode_chunk WHERE build_id = $1",
                resumed.build_id,
            )
        finally:
            await connection.close()
        assert previous is not None
        self.assertEqual(previous["status"], "superseded")
        self.assertEqual(previous["superseded_by"], rotated.build_id)
        self.assertEqual(mapping_count, 1)


class _Fixture:
    def __init__(
        self,
        *,
        kb_id: UUID,
        chat_profile_revision_id: UUID,
        chunk_id: UUID,
        content_hash: str,
    ) -> None:
        self.kb_id = kb_id
        self.chat_profile_revision_id = chat_profile_revision_id
        self.chunk_id = chunk_id
        self.content_hash = content_hash


async def _seed_graph_fixture(connection: asyncpg.Connection) -> _Fixture:
    async with connection.transaction():
        await connection.execute(
            "INSERT INTO workspace (id, name) VALUES ($1, 'graph-resume')",
            WORKSPACE_ID,
        )
        provider_id = await connection.fetchval(
            "INSERT INTO model_provider (workspace_id, name) VALUES ($1, 'provider') RETURNING id",
            WORKSPACE_ID,
        )
        provider_revision_id = await connection.fetchval(
            """
            INSERT INTO model_provider_revision (
                workspace_id, provider_id, revision, protocol, base_url,
                secret_reference, timeout_seconds, max_retries,
                max_concurrency, configuration_fingerprint
            ) VALUES (
                $1, $2, 1, 'openai_compatible', 'https://provider.invalid/v1',
                'TEST_SECRET', 30, 0, 1, 'sha256:provider'
            ) RETURNING id
            """,
            WORKSPACE_ID,
            provider_id,
        )
        chat_profile_revision_id = await _profile_revision(
            connection,
            provider_id=provider_id,
            provider_revision_id=provider_revision_id,
            name="chat",
            kind="chat",
            model="chat-model",
        )
        embedding_profile_revision_id = await _profile_revision(
            connection,
            provider_id=provider_id,
            provider_revision_id=provider_revision_id,
            name="embedding",
            kind="text_embedding",
            model="embedding-model",
        )
        embedding_space_id = await connection.fetchval(
            """
            INSERT INTO embedding_space (
                workspace_id, provider_identity, endpoint_identity,
                requested_model, resolved_model, model_version, dimension,
                distance_metric, vector_data_type, normalization,
                configuration_fingerprint, compatibility_fingerprint,
                model_profile_revision_id
            ) VALUES (
                $1, 'provider', 'endpoint', 'embedding-model',
                'embedding-model', 'v1', 1024, 'cosine', 'float32', 'l2',
                'sha256:embedding-config', $2, $3
            ) RETURNING id
            """,
            WORKSPACE_ID,
            f"sha256:{uuid4().hex}",
            embedding_profile_revision_id,
        )
        kb_id = await connection.fetchval(
            "INSERT INTO knowledge_base (workspace_id, name) VALUES ($1, 'kb') RETURNING id",
            WORKSPACE_ID,
        )
        revision_id = await connection.fetchval(
            """
            INSERT INTO index_revision (
                workspace_id, kb_id, embedding_space_id, status,
                source_snapshot_seq, parser_config, chunking_config
            ) VALUES ($1, $2, $3, 'active', 0, '{}', '{}') RETURNING id
            """,
            WORKSPACE_ID,
            kb_id,
            embedding_space_id,
        )
        await connection.execute(
            "UPDATE knowledge_base SET active_index_revision_id = $1, provisioned_at = now() WHERE id = $2",
            revision_id,
            kb_id,
        )
        document_id = await connection.fetchval(
            "INSERT INTO document (workspace_id, kb_id, display_name) "
            "VALUES ($1, $2, 'document') RETURNING id",
            WORKSPACE_ID,
            kb_id,
        )
        document_version_id = await connection.fetchval(
            """
            INSERT INTO document_version (
                workspace_id, kb_id, document_id, version_number,
                source_status, checksum_sha256, storage_uri,
                original_filename, media_type, size_bytes
            ) VALUES (
                $1, $2, $3, 1, 'available', $4,
                'file:///graph-resume.txt', 'graph-resume.txt', 'text/plain', 1
            ) RETURNING id
            """,
            WORKSPACE_ID,
            kb_id,
            document_id,
            "a" * 64,
        )
        await connection.execute(
            "UPDATE document SET current_version_id = $1 WHERE id = $2",
            document_version_id,
            document_id,
        )
        indexed_id = await connection.fetchval(
            """
            INSERT INTO indexed_document_version (
                workspace_id, kb_id, document_id, document_version_id,
                index_revision_id, source_change_seq, build_status, serving_status
            ) VALUES ($1, $2, $3, $4, $5, 1, 'ready', 'serving') RETURNING id
            """,
            WORKSPACE_ID,
            kb_id,
            document_id,
            document_version_id,
            revision_id,
        )
        chunk_id = uuid4()
        content_hash = "b" * 64
        await connection.execute(
            """
            INSERT INTO index_chunk (
                id, workspace_id, kb_id, indexed_document_version_id,
                ordinal, content, content_hash, token_count,
                source_location, hierarchy, source_metadata, unit_key, modality
            ) VALUES ($1, $2, $3, $4, 0, 'grounded fact', $5, 2,
                      '{}', '{}', '{}', $6, 'text')
            """,
            chunk_id,
            WORKSPACE_ID,
            kb_id,
            indexed_id,
            content_hash,
            f"graph:{chunk_id}",
        )
    return _Fixture(
        kb_id=kb_id,
        chat_profile_revision_id=chat_profile_revision_id,
        chunk_id=chunk_id,
        content_hash=content_hash,
    )


async def _profile_revision(
    connection: asyncpg.Connection,
    *,
    provider_id: UUID,
    provider_revision_id: UUID,
    name: str,
    kind: str,
    model: str,
) -> UUID:
    profile_id = await connection.fetchval(
        "INSERT INTO model_profile (workspace_id, provider_id, name, kind) VALUES ($1, $2, $3, $4) RETURNING id",
        WORKSPACE_ID,
        provider_id,
        name,
        kind,
    )
    return await connection.fetchval(
        """
        INSERT INTO model_profile_revision (
            workspace_id, profile_id, provider_revision_id, revision, model,
            configuration, configuration_fingerprint, capability_fingerprint,
            compatibility_fingerprint, validation_status, validated_at
        ) VALUES (
            $1, $2, $3, 1, $4, '{}', $5, $6, $7, 'valid', now()
        ) RETURNING id
        """,
        WORKSPACE_ID,
        profile_id,
        provider_revision_id,
        model,
        f"sha256:{name}-config",
        f"sha256:{name}-capability",
        f"sha256:{name}-compatibility",
    )


if __name__ == "__main__":
    unittest.main()
