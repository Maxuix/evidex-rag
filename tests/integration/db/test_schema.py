from __future__ import annotations

import asyncio
import os
import unittest
from uuid import UUID

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine

from rag_kb.db.compatibility import (
    EXPECTED_APPLICATION_TABLES,
    validate_database_compatibility,
)
from rag_kb.db.readiness import validate_runtime_readiness


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")


@unittest.skipUnless(
    MIGRATION_DSN and RUNTIME_DSN and RUNTIME_SQLALCHEMY_DSN,
    "database integration DSNs are not configured",
)
class DatabaseSchemaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute("TRUNCATE TABLE workspace CASCADE")
        finally:
            await connection.close()

    async def create_foundation(
        self, connection: asyncpg.Connection, *, suffix: str = "default"
    ) -> tuple[UUID, UUID, UUID]:
        workspace_id = await connection.fetchval(
            "INSERT INTO workspace (name) VALUES ($1) RETURNING id",
            f"workspace-{suffix}",
        )
        embedding_space_id = await connection.fetchval(
            """
            INSERT INTO embedding_space (
                workspace_id, provider_identity, endpoint_identity,
                requested_model, resolved_model, model_version, dimension,
                distance_metric, vector_data_type, normalization,
                configuration_fingerprint, compatibility_fingerprint
            ) VALUES (
                $1, 'alibaba-cloud-model-studio-qwen',
                'alibaba-model-studio-beijing-embedding',
                'qwen3.7-text-embedding', 'qwen3.7-text-embedding',
                'qwen3.7-text-embedding', 1024,
                'cosine', 'float32', 'l2', $2, $3
            ) RETURNING id
            """,
            workspace_id,
            f"configuration-{suffix}",
            f"compatibility-{suffix}",
        )
        kb_id = await connection.fetchval(
            """
            INSERT INTO knowledge_base (workspace_id, name)
            VALUES ($1, $2) RETURNING id
            """,
            workspace_id,
            f"kb-{suffix}",
        )
        return workspace_id, embedding_space_id, kb_id

    async def create_revision(
        self,
        connection: asyncpg.Connection,
        workspace_id: UUID,
        embedding_space_id: UUID,
        kb_id: UUID,
        *,
        status: str = "building",
    ) -> UUID:
        return await connection.fetchval(
            """
            INSERT INTO index_revision (
                workspace_id, kb_id, embedding_space_id, status,
                source_snapshot_seq, parser_config, chunking_config
            ) VALUES ($1, $2, $3, $4, 0, '{}'::jsonb, '{}'::jsonb)
            RETURNING id
            """,
            workspace_id,
            kb_id,
            embedding_space_id,
            status,
        )

    async def create_document_versions(
        self,
        connection: asyncpg.Connection,
        workspace_id: UUID,
        kb_id: UUID,
        *,
        count: int = 2,
    ) -> tuple[UUID, list[UUID]]:
        document_id = await connection.fetchval(
            """
            INSERT INTO document (workspace_id, kb_id, display_name)
            VALUES ($1, $2, 'test document') RETURNING id
            """,
            workspace_id,
            kb_id,
        )
        versions: list[UUID] = []
        for version_number in range(1, count + 1):
            version_id = await connection.fetchval(
                """
                INSERT INTO document_version (
                    workspace_id, kb_id, document_id, version_number,
                    source_status, checksum_sha256, storage_uri,
                    original_filename, media_type, size_bytes
                ) VALUES (
                    $1, $2, $3, $4, 'available', $5, $6,
                    'document.txt', 'text/plain', 10
                ) RETURNING id
                """,
                workspace_id,
                kb_id,
                document_id,
                version_number,
                f"{version_number:064d}",
                f"file:///sources/{version_number}.txt",
            )
            versions.append(version_id)
        await connection.execute(
            "UPDATE document SET current_version_id = $1 WHERE id = $2",
            versions[-1],
            document_id,
        )
        return document_id, versions

    async def test_runtime_role_is_dml_only_and_compatibility_is_read_only(self) -> None:
        engine = create_async_engine(RUNTIME_SQLALCHEMY_DSN)
        try:
            async with engine.connect() as connection:
                report = await validate_database_compatibility(connection)
            readiness = await validate_runtime_readiness(engine)
        finally:
            await engine.dispose()

        self.assertEqual(
            report.application_table_count, len(EXPECTED_APPLICATION_TABLES)
        )
        self.assertEqual(report.vector_type, "vector(1024)")
        self.assertEqual(report.cross_modal_vector_type, "vector(768)")
        self.assertEqual(readiness.database, "ready")
        self.assertEqual(readiness.queue, "ready")
        self.assertEqual(readiness.queue_backend, "postgresql")

        runtime = await asyncpg.connect(RUNTIME_DSN)
        try:
            workspace_id = await runtime.fetchval(
                "INSERT INTO workspace (name) VALUES ('runtime-dml') RETURNING id"
            )
            await runtime.execute("DELETE FROM workspace WHERE id = $1", workspace_id)
            with self.assertRaises(asyncpg.InsufficientPrivilegeError):
                await runtime.execute("CREATE TABLE runtime_must_not_create (id int)")
            with self.assertRaises(asyncpg.InsufficientPrivilegeError):
                await runtime.execute("CREATE TEMP TABLE runtime_temp_forbidden (id int)")
            with self.assertRaises(asyncpg.InsufficientPrivilegeError):
                await runtime.execute(
                    "UPDATE alembic_version SET version_num = 'runtime-mutation'"
                )
            revision = await runtime.fetchval("SELECT version_num FROM alembic_version")
            self.assertEqual(revision, "0011_chat_final_llm_context")
        finally:
            await runtime.close()

    async def test_same_kb_selector_and_deferred_active_rule(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, embedding_space_id, kb_one = await self.create_foundation(
                connection, suffix="selector-one"
            )
            kb_two = await connection.fetchval(
                """
                INSERT INTO knowledge_base (workspace_id, name)
                VALUES ($1, 'kb-selector-two') RETURNING id
                """,
                workspace_id,
            )
            revision_two = await self.create_revision(
                connection,
                workspace_id,
                embedding_space_id,
                kb_two,
                status="active",
            )

            transaction = connection.transaction()
            await transaction.start()
            await connection.execute(
                "UPDATE knowledge_base SET active_index_revision_id = $1 WHERE id = $2",
                revision_two,
                kb_one,
            )
            with self.assertRaises(asyncpg.ForeignKeyViolationError):
                await transaction.commit()

            transaction = connection.transaction()
            await transaction.start()
            await connection.execute(
                "UPDATE knowledge_base SET provisioned_at = now() WHERE id = $1",
                kb_one,
            )
            with self.assertRaises(asyncpg.CheckViolationError):
                await transaction.commit()

            transaction = connection.transaction()
            await transaction.start()
            revision_one = await self.create_revision(
                connection,
                workspace_id,
                embedding_space_id,
                kb_one,
                status="active",
            )
            await connection.execute(
                """
                UPDATE knowledge_base
                SET active_index_revision_id = $1, provisioned_at = now()
                WHERE id = $2
                """,
                revision_one,
                kb_one,
            )
            await transaction.commit()
        finally:
            await connection.close()

    async def test_concurrent_active_revision_constraint(self) -> None:
        setup = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                setup, suffix="active-race"
            )
        finally:
            await setup.close()

        async def insert_active() -> UUID:
            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                async with connection.transaction():
                    return await self.create_revision(
                        connection,
                        workspace_id,
                        embedding_space_id,
                        kb_id,
                        status="active",
                    )
            finally:
                await connection.close()

        results = await asyncio.gather(
            insert_active(), insert_active(), return_exceptions=True
        )
        successes = [result for result in results if isinstance(result, UUID)]
        failures = [result for result in results if isinstance(result, Exception)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], asyncpg.UniqueViolationError)

    async def test_serving_constraints_are_database_enforced(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                connection, suffix="serving"
            )
            revision_id = await self.create_revision(
                connection, workspace_id, embedding_space_id, kb_id
            )
            document_id, versions = await self.create_document_versions(
                connection, workspace_id, kb_id
            )

            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    """
                    INSERT INTO indexed_document_version (
                        workspace_id, kb_id, document_id, document_version_id,
                        index_revision_id, source_change_seq,
                        build_status, serving_status
                    ) VALUES ($1, $2, $3, $4, $5, 1, 'queued', 'serving')
                    """,
                    workspace_id,
                    kb_id,
                    document_id,
                    versions[0],
                    revision_id,
                )

            indexed_version_id = await connection.fetchval(
                """
                INSERT INTO indexed_document_version (
                    workspace_id, kb_id, document_id, document_version_id,
                    index_revision_id, source_change_seq,
                    build_status, serving_status
                ) VALUES ($1, $2, $3, $4, $5, 1, 'ready', 'serving')
                RETURNING id
                """,
                workspace_id,
                kb_id,
                document_id,
                versions[0],
                revision_id,
            )
            with self.assertRaises(asyncpg.UniqueViolationError):
                await connection.execute(
                    """
                    INSERT INTO indexed_document_version (
                        workspace_id, kb_id, document_id, document_version_id,
                        index_revision_id, source_change_seq,
                        build_status, serving_status
                    ) VALUES ($1, $2, $3, $4, $5, 2, 'ready', 'serving')
                    """,
                    workspace_id,
                    kb_id,
                    document_id,
                    versions[1],
                    revision_id,
                )

            chunk_id = await connection.fetchval(
                """
                INSERT INTO index_chunk (
                    workspace_id, kb_id, indexed_document_version_id, ordinal,
                    content, content_hash, token_count, source_location,
                    unit_key, modality
                ) VALUES (
                    $1, $2, $3, 0, 'test', $4, 1, '{}'::jsonb,
                    'schema-test', 'text'
                )
                RETURNING id
                """,
                workspace_id,
                kb_id,
                indexed_version_id,
                "0" * 64,
            )
            embedding = "[" + ",".join(["1", *("0" for _ in range(1023))]) + "]"
            await connection.execute(
                """
                INSERT INTO vector_record_1024 (
                    workspace_id, kb_id, index_chunk_id,
                    embedding_space_id, embedding
                ) VALUES ($1, $2, $3, $4, $5::vector)
                """,
                workspace_id,
                kb_id,
                chunk_id,
                embedding_space_id,
                embedding,
            )
            distance = await connection.fetchval(
                """
                SELECT embedding <=> $1::vector
                FROM vector_record_1024 WHERE index_chunk_id = $2
                """,
                embedding,
                chunk_id,
            )
            self.assertAlmostEqual(distance, 0.0)
        finally:
            await connection.close()

    async def test_concurrent_source_change_allocation_has_no_gaps(self) -> None:
        setup = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, _, kb_id = await self.create_foundation(
                setup, suffix="source-sequence"
            )
            document_id, versions = await self.create_document_versions(
                setup, workspace_id, kb_id
            )
        finally:
            await setup.close()

        async def allocate(version_id: UUID) -> int:
            connection = await asyncpg.connect(MIGRATION_DSN)
            try:
                async with connection.transaction():
                    sequence = await connection.fetchval(
                        """
                        UPDATE knowledge_base
                        SET source_change_seq = source_change_seq + 1
                        WHERE id = $1
                        RETURNING source_change_seq
                        """,
                        kb_id,
                    )
                    await connection.execute(
                        """
                        INSERT INTO source_change (
                            workspace_id, kb_id, source_change_seq,
                            document_id, document_version_id, change_kind
                        ) VALUES ($1, $2, $3, $4, $5, 'upsert')
                        """,
                        workspace_id,
                        kb_id,
                        sequence,
                        document_id,
                        version_id,
                    )
                    return sequence
            finally:
                await connection.close()

        allocated = await asyncio.gather(*(allocate(version) for version in versions))
        self.assertEqual(sorted(allocated), [1, 2])

        check = await asyncpg.connect(MIGRATION_DSN)
        try:
            ledger = await check.fetch(
                """
                SELECT source_change_seq FROM source_change
                WHERE kb_id = $1 ORDER BY source_change_seq
                """,
                kb_id,
            )
        finally:
            await check.close()
        self.assertEqual([row["source_change_seq"] for row in ledger], [1, 2])

    async def test_composite_relation_rejects_cross_target_edges_and_duplicates(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            workspace_id, embedding_space_id, kb_id = await self.create_foundation(
                connection, suffix="composite-relation"
            )
            revision_id = await self.create_revision(
                connection, workspace_id, embedding_space_id, kb_id, status="active"
            )
            document_id, versions = await self.create_document_versions(
                connection, workspace_id, kb_id, count=2
            )

            async def create_target(version_id: UUID, sequence: int) -> UUID:
                return await connection.fetchval(
                    """
                    INSERT INTO indexed_document_version (
                        workspace_id, kb_id, document_id, document_version_id,
                        index_revision_id, source_change_seq,
                        build_status, serving_status
                    ) VALUES ($1, $2, $3, $4, $5, $6, 'processing', 'candidate')
                    RETURNING id
                    """,
                    workspace_id,
                    kb_id,
                    document_id,
                    version_id,
                    revision_id,
                    sequence,
                )

            first_target = await create_target(versions[0], 1)
            second_target = await create_target(versions[1], 2)
            asset_id = await connection.fetchval(
                """
                INSERT INTO index_asset (
                    workspace_id, kb_id, document_id, document_version_id,
                    indexed_document_version_id, asset_key, kind, storage_uri,
                    media_type, checksum_sha256, source_location
                ) VALUES (
                    $1, $2, $3, $4, $5, 'asset-1', 'image', 'local://asset-1',
                    'image/png', $6, '{}'::jsonb
                ) RETURNING id
                """,
                workspace_id,
                kb_id,
                document_id,
                versions[0],
                first_target,
                "a" * 64,
            )

            async def create_chunk(target_id: UUID, ordinal: int, unit_key: str) -> UUID:
                return await connection.fetchval(
                    """
                    INSERT INTO index_chunk (
                        workspace_id, kb_id, indexed_document_version_id,
                        ordinal, unit_key, modality, content, content_hash,
                        token_count, source_location
                    ) VALUES ($1, $2, $3, $4, $5, 'text', 'body', $6, 1, '{}'::jsonb)
                    RETURNING id
                    """,
                    workspace_id,
                    kb_id,
                    target_id,
                    ordinal,
                    unit_key,
                    f"{ordinal:064d}",
                )

            text_chunk = await create_chunk(first_target, 0, "text-1")
            visual_unit = await create_chunk(first_target, 1, "visual-1")
            other_chunk = await create_chunk(second_target, 0, "text-2")
            statement = """
                INSERT INTO index_chunk_asset_relation (
                    workspace_id, kb_id, indexed_document_version_id,
                    chunk_id, visual_unit_id, asset_id, relation_type,
                    confidence_micros, ordinal, provenance, evidence_group_key
                ) VALUES ($1, $2, $3, $4, $5, $6, 'caption_of',
                    1000000, 0, 'author_caption_v2', 'figure-1')
            """
            await connection.execute(
                statement,
                workspace_id,
                kb_id,
                first_target,
                text_chunk,
                visual_unit,
                asset_id,
            )
            with self.assertRaises(asyncpg.UniqueViolationError):
                await connection.execute(
                    statement,
                    workspace_id,
                    kb_id,
                    first_target,
                    text_chunk,
                    visual_unit,
                    asset_id,
                )
            with self.assertRaises(asyncpg.ForeignKeyViolationError):
                await connection.execute(
                    statement,
                    workspace_id,
                    kb_id,
                    first_target,
                    other_chunk,
                    visual_unit,
                    asset_id,
                )
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    """
                    UPDATE index_chunk
                    SET embedding_text = 'body', embedding_text_hash = NULL
                    WHERE id = $1
                    """,
                    text_chunk,
                )
        finally:
            await connection.close()
