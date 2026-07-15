from __future__ import annotations

import os
import unittest
from dataclasses import dataclass, replace
from uuid import UUID, uuid4

import asyncpg
from sqlalchemy import event

from rag_kb.adapters import FixedPgVectorSpace, PgVectorStore
from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.domain import (
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    ResourceNotFoundError,
    RetrievalExecutionError,
    RetrievalRequest,
)
from rag_kb.retrieval import RetrievalService


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")
WORKSPACE = UUID("01900000-0000-7000-8000-000000001001")
OTHER_WORKSPACE = UUID("01900000-0000-7000-8000-000000001002")
EXPECTED_FINGERPRINT = "sha256:retrieval-compatible"


@unittest.skipUnless(
    MIGRATION_DSN and RUNTIME_SQLALCHEMY_DSN,
    "database integration DSNs are not configured",
)
class ExactRetrievalDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute("TRUNCATE TABLE workspace CASCADE")
        finally:
            await connection.close()
        self.database = create_database_resources(
            RUNTIME_SQLALCHEMY_DSN,
            pool_size=3,
            max_overflow=0,
            process=DatabaseProcess.API,
        )
        self.definition = _embedding_space()
        self.policy = SingleWorkspaceAccessPolicy(WORKSPACE)
        self.context = AuthContext("principal", "client", WORKSPACE)
        self.provider = _Provider(self.definition, _axis_vector(0))
        self.vector_store = PgVectorStore(
            self.database.sessions,
            FixedPgVectorSpace(self.definition),
        )
        self.service = RetrievalService(
            self.policy,
            self.provider,
            self.vector_store,
        )

    async def asyncTearDown(self) -> None:
        await self.database.close()

    async def test_exact_search_is_one_statement_with_stable_top_k(self) -> None:
        foundation = await self._foundation()
        first = UUID("01900000-0000-7000-8000-000000001111")
        second = UUID("01900000-0000-7000-8000-000000001112")
        third = UUID("01900000-0000-7000-8000-000000001113")
        await self._target(foundation, chunk_id=second, vector=_axis_vector(0))
        await self._target(foundation, chunk_id=first, vector=_axis_vector(0))
        await self._target(foundation, chunk_id=third, vector=_axis_vector(1))
        statements: list[str] = []

        def capture_statement(*args) -> None:
            statements.append(args[2])

        event.listen(
            self.database.engine.sync_engine,
            "before_cursor_execute",
            capture_statement,
        )
        try:
            pack = await self.service.retrieve(
                self.context,
                RetrievalRequest(
                    foundation.kb_id,
                    "query",
                    top_k=2,
                    include_debug=True,
                ),
            )
        finally:
            event.remove(
                self.database.engine.sync_engine,
                "before_cursor_execute",
                capture_statement,
            )

        self.assertEqual(len(statements), 1)
        self.assertIn("LEFT OUTER JOIN LATERAL", statements[0])
        self.assertIn("<=>", statements[0])
        self.assertEqual(pack.index_revision_id, foundation.revision_id)
        self.assertEqual(
            tuple(item.index_chunk_id for item in pack.evidence),
            (first, second),
        )
        self.assertEqual(tuple(item.rank for item in pack.evidence), (1, 2))
        self.assertTrue(all(item.score == 1.0 for item in pack.evidence))
        self.assertIsNotNone(pack.debug)
        assert pack.debug is not None
        self.assertEqual(pack.debug.result_count, 2)

    async def test_empty_result_keeps_revision_and_missing_scope_is_not_found(self) -> None:
        foundation = await self._foundation()

        empty = await self.service.retrieve(
            self.context,
            RetrievalRequest(foundation.kb_id, "no matches", top_k=10),
        )

        self.assertEqual(empty.index_revision_id, foundation.revision_id)
        self.assertEqual(empty.evidence, ())
        with self.assertRaises(ResourceNotFoundError):
            await self.service.retrieve(
                self.context,
                RetrievalRequest(uuid4(), "missing knowledge base"),
            )

        other = await self._foundation(
            workspace_id=OTHER_WORKSPACE,
            compatibility_fingerprint="sha256:other-workspace-space",
        )
        with self.assertRaises(ResourceNotFoundError):
            await self.service.retrieve(
                self.context,
                RetrievalRequest(other.kb_id, "cross workspace"),
            )

    async def test_only_current_available_ready_serving_active_content_is_returned(self) -> None:
        foundation = await self._foundation()
        valid = UUID("01900000-0000-7000-8000-000000001201")
        await self._target(foundation, chunk_id=valid, vector=_axis_vector(0))
        await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001202"),
            vector=_axis_vector(0),
            serving_status="candidate",
        )
        await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001203"),
            vector=_axis_vector(0),
            build_status="failed",
            serving_status="candidate",
        )
        await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001204"),
            vector=_axis_vector(0),
            serving_status="retired",
        )
        await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001205"),
            vector=_axis_vector(0),
            source_status="unavailable",
        )
        await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001206"),
            vector=_axis_vector(0),
            deleted=True,
        )
        await self._target(
            foundation,
            chunk_id=UUID("01900000-0000-7000-8000-000000001207"),
            vector=_axis_vector(0),
            current=False,
        )
        retired_revision = await self._revision(
            foundation,
            status="retired",
        )
        await self._target(
            replace(foundation, revision_id=retired_revision),
            chunk_id=UUID("01900000-0000-7000-8000-000000001208"),
            vector=_axis_vector(0),
        )

        pack = await self.service.retrieve(
            self.context,
            RetrievalRequest(foundation.kb_id, "query", top_k=100),
        )

        self.assertEqual(
            tuple(item.index_chunk_id for item in pack.evidence),
            (valid,),
        )

    async def test_active_embedding_space_mismatch_fails_closed(self) -> None:
        foundation = await self._foundation(
            compatibility_fingerprint="sha256:different-space"
        )

        with self.assertRaises(RetrievalExecutionError) as failure:
            await self.service.retrieve(
                self.context,
                RetrievalRequest(foundation.kb_id, "query"),
            )

        self.assertEqual(failure.exception.code, ErrorCode.EMBEDDING_SPACE_MISMATCH)

    async def _foundation(
        self,
        *,
        workspace_id: UUID = WORKSPACE,
        compatibility_fingerprint: str = EXPECTED_FINGERPRINT,
    ) -> "_Foundation":
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                await connection.execute(
                    "INSERT INTO workspace (id, name) VALUES ($1, $2)",
                    workspace_id,
                    f"workspace-{workspace_id}",
                )
                embedding_space_id = await connection.fetchval(
                    """
                    INSERT INTO embedding_space (
                        workspace_id, provider_identity, endpoint_identity,
                        requested_model, resolved_model, model_version, dimension,
                        distance_metric, vector_data_type, normalization,
                        configuration_fingerprint, compatibility_fingerprint
                    ) VALUES (
                        $1, 'test', 'test-endpoint', 'test-embedding',
                        'test-embedding', 'v1', 1024, 'cosine', 'float32', 'l2',
                        'sha256:configuration', $2
                    ) RETURNING id
                    """,
                    workspace_id,
                    compatibility_fingerprint,
                )
                kb_id = await connection.fetchval(
                    """
                    INSERT INTO knowledge_base (workspace_id, name)
                    VALUES ($1, $2) RETURNING id
                    """,
                    workspace_id,
                    f"kb-{uuid4()}",
                )
                revision_id = await connection.fetchval(
                    """
                    INSERT INTO index_revision (
                        workspace_id, kb_id, embedding_space_id, status,
                        source_snapshot_seq, parser_config, chunking_config
                    ) VALUES ($1, $2, $3, 'active', 0, '{}', '{}')
                    RETURNING id
                    """,
                    workspace_id,
                    kb_id,
                    embedding_space_id,
                )
                await connection.execute(
                    """
                    UPDATE knowledge_base
                       SET active_index_revision_id = $1, provisioned_at = now()
                     WHERE id = $2
                    """,
                    revision_id,
                    kb_id,
                )
        finally:
            await connection.close()
        return _Foundation(
            workspace_id,
            kb_id,
            revision_id,
            embedding_space_id,
        )

    async def _revision(
        self,
        foundation: "_Foundation",
        *,
        status: str,
    ) -> UUID:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            return await connection.fetchval(
                """
                INSERT INTO index_revision (
                    workspace_id, kb_id, embedding_space_id, status,
                    source_snapshot_seq, parser_config, chunking_config
                ) VALUES ($1, $2, $3, $4, 0, '{}', '{}') RETURNING id
                """,
                foundation.workspace_id,
                foundation.kb_id,
                foundation.embedding_space_id,
                status,
            )
        finally:
            await connection.close()

    async def _target(
        self,
        foundation: "_Foundation",
        *,
        chunk_id: UUID,
        vector: tuple[float, ...],
        build_status: str = "ready",
        serving_status: str = "serving",
        source_status: str = "available",
        deleted: bool = False,
        current: bool = True,
    ) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                document_id = await connection.fetchval(
                    """
                    INSERT INTO document (
                        workspace_id, kb_id, display_name, deleted_at
                    ) VALUES ($1, $2, $3, CASE WHEN $4 THEN now() ELSE NULL END)
                    RETURNING id
                    """,
                    foundation.workspace_id,
                    foundation.kb_id,
                    f"document-{chunk_id}",
                    deleted,
                )
                version_id = await self._document_version(
                    connection,
                    foundation,
                    document_id,
                    version_number=1,
                    source_status=source_status,
                )
                current_version_id = version_id
                if not current:
                    current_version_id = await self._document_version(
                        connection,
                        foundation,
                        document_id,
                        version_number=2,
                        source_status="available",
                    )
                await connection.execute(
                    "UPDATE document SET current_version_id = $1 WHERE id = $2",
                    current_version_id,
                    document_id,
                )
                indexed_id = await connection.fetchval(
                    """
                    INSERT INTO indexed_document_version (
                        workspace_id, kb_id, document_id, document_version_id,
                        index_revision_id, source_change_seq, build_status,
                        serving_status
                    ) VALUES ($1, $2, $3, $4, $5, 1, $6, $7) RETURNING id
                    """,
                    foundation.workspace_id,
                    foundation.kb_id,
                    document_id,
                    version_id,
                    foundation.revision_id,
                    build_status,
                    serving_status,
                )
                await connection.execute(
                    """
                    INSERT INTO index_chunk (
                        id, workspace_id, kb_id, indexed_document_version_id,
                        ordinal, content, content_hash, token_count,
                        source_location, hierarchy, source_metadata
                    ) VALUES (
                        $1, $2, $3, $4, 0, $5, $6, 2,
                        '{"line_start": 1, "line_end": 1}', '{}',
                        '{"filename": "fixture.txt"}'
                    )
                    """,
                    chunk_id,
                    foundation.workspace_id,
                    foundation.kb_id,
                    indexed_id,
                    f"evidence-{chunk_id}",
                    chunk_id.hex.ljust(64, "0")[:64],
                )
                await connection.execute(
                    """
                    INSERT INTO vector_record_1024 (
                        workspace_id, kb_id, index_chunk_id,
                        embedding_space_id, embedding
                    ) VALUES ($1, $2, $3, $4, $5::vector)
                    """,
                    foundation.workspace_id,
                    foundation.kb_id,
                    chunk_id,
                    foundation.embedding_space_id,
                    _vector_literal(vector),
                )
        finally:
            await connection.close()

    async def _document_version(
        self,
        connection: asyncpg.Connection,
        foundation: "_Foundation",
        document_id: UUID,
        *,
        version_number: int,
        source_status: str,
    ) -> UUID:
        return await connection.fetchval(
            """
            INSERT INTO document_version (
                workspace_id, kb_id, document_id, version_number, source_status,
                checksum_sha256, storage_uri, original_filename, media_type,
                size_bytes
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, 'fixture.txt', 'text/plain', 1
            ) RETURNING id
            """,
            foundation.workspace_id,
            foundation.kb_id,
            document_id,
            version_number,
            source_status,
            f"{version_number:064d}",
            f"file:///fixture/{uuid4()}.txt",
        )


@dataclass(frozen=True, slots=True)
class _Foundation:
    workspace_id: UUID
    kb_id: UUID
    revision_id: UUID
    embedding_space_id: UUID


class _Provider:
    max_batch_size = 10

    def __init__(
        self,
        embedding_space: EmbeddingSpaceDefinition,
        vector: tuple[float, ...],
    ) -> None:
        self.embedding_space = embedding_space
        self.vector = vector

    async def embed(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        if len(texts) != 1:
            raise AssertionError("retrieval embeds exactly one query")
        return EmbeddingBatch(
            model=self.embedding_space.resolved_model,
            vectors=(self.vector,),
        )


def _embedding_space() -> EmbeddingSpaceDefinition:
    return EmbeddingSpaceDefinition(
        provider_identity="test",
        endpoint_identity="test-endpoint",
        requested_model="test-embedding",
        resolved_model="test-embedding",
        model_version="v1",
        deployment_revision=None,
        dimension=1024,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization="l2",
        configuration_fingerprint="sha256:configuration",
        tokenizer_fingerprint=None,
        compatibility_fingerprint=EXPECTED_FINGERPRINT,
    )


def _axis_vector(index: int) -> tuple[float, ...]:
    values = [0.0] * 1024
    values[index] = 1.0
    return tuple(values)


def _vector_literal(vector: tuple[float, ...]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"
