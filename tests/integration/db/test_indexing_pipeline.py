from __future__ import annotations

import io
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg

from rag_kb.adapters import FixedPgVectorSpace, IsolatedPlainTextProcessor, LocalFileStore
from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.domain import (
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexProfileDefinition,
    IndexingCommand,
    IndexingExecutionError,
    IndexingPhase,
    ParserLimits,
)
from rag_kb.indexing import IndexingPipeline
from rag_kb.services import SourceFileService
from rag_kb.services.content import DocumentService, KnowledgeBaseService
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")
WORKSPACE = UUID("01900000-0000-7000-8000-000000000401")


@unittest.skipUnless(
    MIGRATION_DSN and RUNTIME_SQLALCHEMY_DSN,
    "database integration DSNs are not configured",
)
class IndexingPipelineDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute("TRUNCATE TABLE workspace CASCADE")
        finally:
            await connection.close()
        self.database = create_database_resources(
            RUNTIME_SQLALCHEMY_DSN,
            pool_size=4,
            max_overflow=0,
            process=DatabaseProcess.WORKER,
        )
        self.factory = SqlAlchemyUnitOfWorkFactory(self.database.sessions, WORKSPACE)
        self.policy = SingleWorkspaceAccessPolicy(WORKSPACE)
        self.context = AuthContext("principal", "client", WORKSPACE)
        self.knowledge_bases = KnowledgeBaseService(
            self.factory,
            self.policy,
            embedding_space=_embedding(),
            index_profile=_profile(),
        )
        self.documents = DocumentService(self.factory, self.policy)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "staging").mkdir()
        (self.root / "final").mkdir()
        self.store = LocalFileStore(self.root / "staging", self.root / "final")
        self.processor = IsolatedPlainTextProcessor(ParserLimits())

    async def asyncTearDown(self) -> None:
        await self.database.close()
        self.temporary.cleanup()

    async def test_txt_markdown_and_completed_replay_produce_one_non_serving_set(self) -> None:
        kb = await self._create_kb()
        uploaded = (
            await self._upload(kb.id, "guide.txt", "text/plain", b"first\n\nsecond"),
            await self._upload(
                kb.id,
                "handbook.md",
                "text/markdown",
                b"# Handbook\r\n\r\nrestart-safe evidence",
            ),
        )
        provider = _Provider()
        pipeline = self._pipeline(provider)
        results = []
        for item in uploaded:
            results.append(await pipeline.execute(_command(item)))
        replay = await pipeline.execute(_command(uploaded[0]))

        self.assertTrue(all(result.status == "ready" for result in results))
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.chunk_count, results[0].chunk_count)
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            state = await connection.fetchrow(
                """
                SELECT
                    (SELECT count(*) FROM indexed_document_version
                      WHERE build_status = 'ready' AND serving_status = 'candidate') AS ready_candidates,
                    (SELECT count(*) FROM indexed_document_version
                      WHERE serving_status = 'serving') AS serving,
                    (SELECT count(*) FROM indexing_job WHERE status = 'completed') AS completed_jobs,
                    (SELECT count(*) FROM index_chunk) AS chunks,
                    (SELECT count(*) FROM vector_record_1024) AS vectors,
                    (SELECT bool_and(vector_dims(embedding) = 1024)
                       FROM vector_record_1024) AS dimensions_valid
                """
            )
        finally:
            await connection.close()
        self.assertEqual(state["ready_candidates"], 2)
        self.assertEqual(state["serving"], 0)
        self.assertEqual(state["completed_jobs"], 2)
        self.assertEqual(state["chunks"], state["vectors"])
        self.assertTrue(state["dimensions_valid"])

    async def test_partial_provider_failure_is_durable_and_replay_converges(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(
            kb.id, "large.txt", "text/plain", b"a" * 2501
        )
        provider = _Provider(max_batch_size=1, fail_call=2)
        pipeline = self._pipeline(provider)

        with self.assertRaises(IndexingExecutionError) as failure:
            await pipeline.execute(_command(uploaded))
        self.assertEqual(failure.exception.code, ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE)
        failed = await self._target_state(uploaded.indexed_document_version_id)
        self.assertEqual(
            tuple(failed),
            ("failed", "candidate", "failed", "embedding", "EMBEDDING_PROVIDER_UNAVAILABLE", 1, 1),
        )

        provider.fail_call = None
        provider.calls = 0
        completed = await pipeline.execute(_command(uploaded))
        self.assertEqual((completed.status, completed.chunk_count), ("ready", 2))
        replayed = await self._target_state(uploaded.indexed_document_version_id)
        self.assertEqual(
            tuple(replayed),
            ("ready", "candidate", "completed", "completed", None, 2, 2),
        )

    async def test_embedding_mismatch_fails_before_any_derived_write(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        provider = _Provider(
            embedding_space=replace(
                _embedding(), configuration_fingerprint="sha256:changed"
            )
        )

        with self.assertRaises(IndexingExecutionError) as failure:
            await self._pipeline(provider).execute(_command(uploaded))
        self.assertEqual(failure.exception.code, ErrorCode.EMBEDDING_SPACE_MISMATCH)
        state = await self._target_state(uploaded.indexed_document_version_id)
        self.assertEqual(
            tuple(state),
            ("failed", "candidate", "failed", "embedding", "EMBEDDING_SPACE_MISMATCH", 0, 0),
        )

    async def test_conflicting_stable_key_records_persistence_failure(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute(
                """
                INSERT INTO index_chunk(
                    id, workspace_id, kb_id, indexed_document_version_id, ordinal,
                    content, content_hash, token_count, source_location, hierarchy,
                    source_metadata
                ) VALUES ($1, $2, $3, $4, 0, 'conflict', $5, 8, '{}', '{}', '{}')
                """,
                uuid4(),
                WORKSPACE,
                kb.id,
                uploaded.indexed_document_version_id,
                "0" * 64,
            )
        finally:
            await connection.close()

        with self.assertRaises(IndexingExecutionError) as failure:
            await self._pipeline(_Provider()).execute(_command(uploaded))
        self.assertEqual(failure.exception.code, ErrorCode.INDEX_PERSISTENCE_FAILED)
        state = await self._target_state(uploaded.indexed_document_version_id)
        self.assertEqual(
            tuple(state),
            ("failed", "candidate", "failed", "persisting", "INDEX_PERSISTENCE_FAILED", 1, 0),
        )

    async def test_missing_source_records_source_phase_without_partial_content(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        identity = self.store.parse_uri(uploaded.document.current_version.storage_uri)
        await self.store.delete(identity)

        with self.assertRaises(IndexingExecutionError) as failure:
            await self._pipeline(_Provider()).execute(_command(uploaded))
        self.assertEqual(failure.exception.code, ErrorCode.SOURCE_FILE_MISSING)
        state = await self._target_state(uploaded.indexed_document_version_id)
        self.assertEqual(
            tuple(state),
            ("failed", "candidate", "failed", "source_read", "SOURCE_FILE_MISSING", 0, 0),
        )

    def _pipeline(self, provider) -> IndexingPipeline:
        return IndexingPipeline(
            self.factory,
            self.store,
            self.processor,
            provider,
            FixedPgVectorSpace(_embedding()),
        )

    async def _create_kb(self):
        return await self.knowledge_bases.create(
            self.context,
            uuid4(),
            name=f"kb-{uuid4()}",
            retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
        )

    async def _upload(self, kb_id, filename, media_type, content):
        return await SourceFileService(self.documents, self.store).store_and_activate(
            self.context,
            uuid4(),
            kb_id=kb_id,
            document_id=None,
            display_name=filename,
            original_filename=filename,
            media_type=media_type,
            source=io.BytesIO(content),
        )

    async def _target_state(self, target_id):
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            return await connection.fetchrow(
                """
                SELECT idv.build_status::text,
                       idv.serving_status::text,
                       job.status::text,
                       job.phase,
                       job.error_code,
                       (SELECT count(*) FROM index_chunk c
                         WHERE c.indexed_document_version_id = idv.id),
                       (SELECT count(*) FROM vector_record_1024 v
                         JOIN index_chunk c ON c.id = v.index_chunk_id
                        WHERE c.indexed_document_version_id = idv.id)
                  FROM indexed_document_version idv
                  JOIN indexing_job job ON job.indexed_document_version_id = idv.id
                 WHERE idv.id = $1
                """,
                target_id,
            )
        finally:
            await connection.close()


class _Provider:
    def __init__(
        self,
        *,
        embedding_space=None,
        max_batch_size=10,
        fail_call=None,
    ) -> None:
        self.embedding_space = embedding_space or _embedding()
        self.max_batch_size = max_batch_size
        self.fail_call = fail_call
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        if self.calls == self.fail_call:
            raise IndexingExecutionError(
                ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                phase=IndexingPhase.EMBEDDING,
                diagnostic={"retry_exhausted": True},
            )
        return EmbeddingBatch(
            model=self.embedding_space.resolved_model,
            vectors=tuple(_vector() for _ in texts),
        )


def _command(uploaded):
    return IndexingCommand(
        uploaded.job_id,
        uploaded.indexed_document_version_id,
    )


def _embedding():
    return EmbeddingSpaceDefinition(
        provider_identity="alibaba-cloud-model-studio-qwen",
        endpoint_identity="alibaba-model-studio-beijing-embedding",
        requested_model="text-embedding-v4",
        resolved_model="text-embedding-v4",
        model_version="text-embedding-v4 (Qwen3-Embedding series)",
        deployment_revision=None,
        dimension=1024,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization="l2",
        configuration_fingerprint="sha256:c135eb852aefd97be80fd82dd168f7cf1ccff0c99eb4fcfdbbf3e337cedeee66",
        tokenizer_fingerprint=None,
        compatibility_fingerprint="sha256:7bd706a3642d7ee17a5a0112a3e0d7e50abaa26bf19c5511d240f222ae4d1153",
    )


def _profile():
    return IndexProfileDefinition(
        parser_config={
            "profile": "plain_text_test_v1",
            "encoding": "utf-8",
            "bom": "optional",
            "line_endings": "lf",
        },
        chunking_config={
            "profile": "paragraph_window_v1",
            "boundary_order": ["paragraph", "line", "codepoint"],
            "max_characters": 2000,
            "overlap_characters": 200,
        },
    )


def _vector():
    return (1.0,) + (0.0,) * 1023


if __name__ == "__main__":
    unittest.main()
