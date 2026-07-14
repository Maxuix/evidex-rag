from __future__ import annotations

import asyncio
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
    PromotionCommand,
    PromotionReason,
)
from rag_kb.indexing import CandidatePromotionService, IndexingPipeline
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

    async def test_txt_markdown_and_completed_replay_produce_serving_sets(self) -> None:
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
        not_ready = await CandidatePromotionService(self.factory).promote(
            _promotion_command(uploaded[0])
        )
        self.assertEqual(
            (not_ready.status, not_ready.reason),
            ("not_ready", PromotionReason.NOT_READY),
        )
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
        self.assertEqual(state["ready_candidates"], 0)
        self.assertEqual(state["serving"], 2)
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
            ("ready", "serving", "completed", "completed", None, 2, 2),
        )

    async def test_new_version_switches_atomically_after_ready(self) -> None:
        kb = await self._create_kb()
        first = await self._upload(kb.id, "guide.txt", "text/plain", b"version one")
        pipeline = self._pipeline(_Provider())
        await pipeline.execute(_command(first))

        second = await self._upload(
            kb.id,
            "guide.txt",
            "text/plain",
            b"version two",
            document_id=first.document.id,
        )
        self.assertEqual(
            await self._serving_states(first.document.id),
            (
                (first.indexed_document_version_id, "serving"),
                (second.indexed_document_version_id, "candidate"),
            ),
        )

        promoted = await pipeline.execute(_command(second))

        self.assertEqual(promoted.serving_status, "serving")
        self.assertEqual(
            await self._serving_states(first.document.id),
            (
                (first.indexed_document_version_id, "retired"),
                (second.indexed_document_version_id, "serving"),
            ),
        )

    async def test_reverse_completion_cannot_restore_superseded_version(self) -> None:
        kb = await self._create_kb()
        first = await self._upload(kb.id, "guide.txt", "text/plain", b"version one")
        second = await self._upload(
            kb.id,
            "guide.txt",
            "text/plain",
            b"version two",
            document_id=first.document.id,
        )
        pipeline = self._pipeline(_Provider())

        current = await pipeline.execute(_command(second))
        stale = await pipeline.execute(_command(first))
        stale_replay = await pipeline.execute(_command(first))

        self.assertEqual(
            (
                current.serving_status,
                stale.serving_status,
                stale_replay.serving_status,
            ),
            ("serving", "retired", "retired"),
        )
        self.assertTrue(stale_replay.replayed)
        self.assertEqual(
            await self._serving_states(first.document.id),
            (
                (first.indexed_document_version_id, "retired"),
                (second.indexed_document_version_id, "serving"),
            ),
        )

    async def test_late_worker_after_delete_cannot_restore_serving_state(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        await self.documents.delete(self.context, uuid4(), uploaded.document.id)

        result = await self._pipeline(_Provider()).execute(_command(uploaded))

        self.assertEqual(
            (result.status, result.serving_status),
            ("cancelled", "retired"),
        )
        self.assertEqual(
            await self._serving_states(uploaded.document.id),
            ((uploaded.indexed_document_version_id, "retired"),),
        )

    async def test_concurrent_promotions_converge_on_current_version(self) -> None:
        kb = await self._create_kb()
        first = await self._upload(kb.id, "guide.txt", "text/plain", b"version one")
        second = await self._upload(
            kb.id,
            "guide.txt",
            "text/plain",
            b"version two",
            document_id=first.document.id,
        )
        await self._mark_ready(first, second)
        promotion = CandidatePromotionService(self.factory)

        first_result, second_result = await asyncio.gather(
            promotion.promote(_promotion_command(first)),
            promotion.promote(_promotion_command(second)),
        )

        self.assertEqual(first_result.reason, PromotionReason.SUPERSEDED)
        self.assertEqual(second_result.status, "serving")
        self.assertEqual(
            await self._serving_states(first.document.id),
            (
                (first.indexed_document_version_id, "retired"),
                (second.indexed_document_version_id, "serving"),
            ),
        )

    async def test_promotion_delete_race_always_ends_retired(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        await self._mark_ready(uploaded)
        promotion = CandidatePromotionService(self.factory)

        await asyncio.gather(
            promotion.promote(_promotion_command(uploaded)),
            self.documents.delete(self.context, uuid4(), uploaded.document.id),
        )

        self.assertEqual(
            await self._serving_states(uploaded.document.id),
            ((uploaded.indexed_document_version_id, "retired"),),
        )

    async def test_later_source_change_retires_otherwise_eligible_candidate(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        await self._mark_ready(uploaded)
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                await connection.execute(
                    "UPDATE knowledge_base SET source_change_seq = 2 WHERE id = $1",
                    kb.id,
                )
                await connection.execute(
                    """
                    INSERT INTO source_change(
                        workspace_id, kb_id, source_change_seq, document_id,
                        document_version_id, change_kind
                    ) VALUES ($1, $2, 2, $3, $4, 'upsert')
                    """,
                    WORKSPACE,
                    kb.id,
                    uploaded.document.id,
                    uploaded.document_version_id,
                )
        finally:
            await connection.close()

        result = await CandidatePromotionService(self.factory).promote(
            _promotion_command(uploaded)
        )

        self.assertEqual(result.reason, PromotionReason.LATER_SOURCE_CHANGE)
        self.assertEqual(result.status, "retired")

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

    async def _upload(
        self,
        kb_id,
        filename,
        media_type,
        content,
        *,
        document_id=None,
    ):
        return await SourceFileService(self.documents, self.store).store_and_activate(
            self.context,
            uuid4(),
            kb_id=kb_id,
            document_id=document_id,
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

    async def _serving_states(self, document_id):
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            rows = await connection.fetch(
                """
                SELECT id, serving_status::text
                  FROM indexed_document_version
                 WHERE document_id = $1
                 ORDER BY source_change_seq
                """,
                document_id,
            )
            return tuple((row["id"], row["serving_status"]) for row in rows)
        finally:
            await connection.close()

    async def _mark_ready(self, *uploaded):
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                for item in uploaded:
                    await connection.execute(
                        """
                        UPDATE indexed_document_version
                           SET build_status = 'ready'
                         WHERE id = $1
                        """,
                        item.indexed_document_version_id,
                    )
                    await connection.execute(
                        """
                        UPDATE indexing_job
                           SET status = 'completed', phase = 'completed'
                         WHERE id = $1
                        """,
                        item.job_id,
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


def _promotion_command(uploaded):
    return PromotionCommand(
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
