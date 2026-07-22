from __future__ import annotations

import io
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg

from rag_kb.adapters import LocalFileStore
from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.document_processing import index_profile
from rag_kb.domain import (
    AdmissionLimits,
    DocumentSource,
    EmbeddingSpaceDefinition,
    FileAdmissionError,
    IdempotencyKeyReusedError,
    IndexProfileDefinition,
)
from rag_kb.services import FileAdmissionService, FileReconciliationService, SourceFileService
from rag_kb.services.content import DocumentService, KnowledgeBaseService
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")
WORKSPACE = UUID("01900000-0000-7000-8000-000000000301")


@unittest.skipUnless(
    MIGRATION_DSN and RUNTIME_SQLALCHEMY_DSN,
    "database integration DSNs are not configured",
)
class FileConsistencyTests(unittest.IsolatedAsyncioTestCase):
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

    async def asyncTearDown(self) -> None:
        await self.database.close()
        self.temporary.cleanup()

    def reconciler(self, *, orphan_grace_seconds: float = 300) -> FileReconciliationService:
        return FileReconciliationService(
            self.factory,
            self.documents,
            LocalFileStore(self.root / "staging", self.root / "final"),
            batch_size=100,
            orphan_grace_seconds=orphan_grace_seconds,
            cleanup_max_attempts=3,
            cleanup_base_delay_seconds=0.01,
        )

    async def test_committed_file_survives_restart_and_missing_file_is_compensated(self) -> None:
        kb = await self._create_kb()
        result = await SourceFileService(self.documents, self.store).store_and_activate(
            self.context,
            uuid4(),
            kb_id=kb.id,
            document_id=None,
            display_name="guide",
            original_filename="guide.md",
            media_type="text/markdown",
            source=io.BytesIO(b"restart-safe source"),
        )
        current = result.document.current_version
        self.assertIsNotNone(current)
        identity = self.store.parse_uri(current.storage_uri)
        restarted = LocalFileStore(self.root / "staging", self.root / "final")
        self.assertEqual(await restarted.read_final(identity), b"restart-safe source")

        await restarted.delete(identity)
        reconciled = await self.reconciler().run_once(self.context)
        self.assertEqual(reconciled.missing_compensated, 1)

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            state = await connection.fetchrow(
                """
                SELECT dv.source_status::text AS source_status,
                       idv.serving_status::text AS serving_status,
                       job.status::text AS job_status,
                       kb.source_change_seq,
                       (SELECT count(*) FROM source_change WHERE kb_id = kb.id) AS changes
                FROM document_version dv
                JOIN document d ON d.id = dv.document_id
                JOIN knowledge_base kb ON kb.id = dv.kb_id
                JOIN indexed_document_version idv ON idv.document_version_id = dv.id
                JOIN indexing_job job ON job.indexed_document_version_id = idv.id
                WHERE dv.id = $1
                """,
                result.document_version_id,
            )
        finally:
            await connection.close()
        self.assertEqual(
            tuple(state),
            ("unavailable", "retired", "cancelled", 2, 2),
        )

    async def test_admitted_upload_is_idempotent_and_never_serves_partial_content(self) -> None:
        kb = await self._create_kb()
        admission = FileAdmissionService(AdmissionLimits())
        source_files = SourceFileService(self.documents, self.store)
        key = uuid4()

        source = io.BytesIO(b"# Guide\r\nrestart-safe")
        admitted = admission.validate(
            source,
            original_filename="guide.md",
            media_type="text/markdown; charset=utf-8",
        )
        first = await source_files.store_and_activate(
            self.context,
            key,
            kb_id=kb.id,
            document_id=None,
            display_name="Guide",
            original_filename=admitted.original_filename,
            media_type=admitted.media_type,
            source=source,
        )

        replay_source = io.BytesIO(b"# Guide\r\nrestart-safe")
        replay_admitted = admission.validate(
            replay_source,
            original_filename="guide.md",
            media_type="text/markdown",
        )
        replay = await source_files.store_and_activate(
            self.context,
            key,
            kb_id=kb.id,
            document_id=None,
            display_name="Guide",
            original_filename=replay_admitted.original_filename,
            media_type=replay_admitted.media_type,
            source=replay_source,
        )
        self.assertEqual(first.document_version_id, replay.document_version_id)
        self.assertEqual(first.job_id, replay.job_id)

        conflict_source = io.BytesIO(b"different")
        conflict_admitted = admission.validate(
            conflict_source,
            original_filename="guide.md",
            media_type="text/markdown",
        )
        with self.assertRaises(IdempotencyKeyReusedError):
            await source_files.store_and_activate(
                self.context,
                key,
                kb_id=kb.id,
                document_id=None,
                display_name="Guide",
                original_filename=conflict_admitted.original_filename,
                media_type=conflict_admitted.media_type,
                source=conflict_source,
            )

        for bad in (b"\xff", b"1\n" * 200_001):
            with self.assertRaises(FileAdmissionError):
                admission.validate(
                    io.BytesIO(bad),
                    original_filename="bad.txt",
                    media_type="text/plain",
                )

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            state = await connection.fetchrow(
                """
                SELECT
                    (SELECT count(*) FROM document_version) AS versions,
                    (SELECT count(*) FROM source_change) AS changes,
                    (SELECT count(*) FROM indexed_document_version) AS targets,
                    (SELECT count(*) FROM indexing_job) AS jobs,
                    (SELECT serving_status::text FROM indexed_document_version) AS serving,
                    (SELECT status::text FROM indexing_job) AS job_status
                """
            )
        finally:
            await connection.close()
        self.assertEqual(tuple(state), (1, 1, 1, 1, "candidate", "queued"))
        stored = await self.store.list_files()
        self.assertEqual(len(stored), 1)

    async def test_restart_recovers_staging_and_finalized_pending_mutations(self) -> None:
        kb = await self._create_kb()
        for ordinal, finalize_first in ((1, False), (2, True)):
            key = uuid4()
            staged = await self.store.stage(
                WORKSPACE,
                f"pending-{ordinal}",
                io.BytesIO(f"pending-{ordinal}".encode()),
            )
            reserved = await self.documents.reserve_version(
                self.context,
                key,
                kb_id=kb.id,
                document_id=None,
                display_name=f"pending-{ordinal}",
                source=DocumentSource(
                    checksum_sha256=staged.digest.checksum_sha256,
                    storage_uri=staged.identity.storage_uri,
                    original_filename=f"pending-{ordinal}.txt",
                    media_type="text/plain",
                    size_bytes=staged.digest.size_bytes,
                ),
            )
            self.assertIsNotNone(reserved.document_version_id)
            if finalize_first:
                await self.store.finalize(staged.identity, staged.digest)

        reconciled = await self.reconciler().run_once(self.context)
        self.assertEqual(reconciled.pending_activated, 2)

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            state = await connection.fetchrow(
                """
                SELECT
                    (SELECT count(*) FROM document_version WHERE source_status = 'available') AS available,
                    (SELECT count(*) FROM content_mutation WHERE status = 'completed') AS completed,
                    (SELECT count(*) FROM source_change) AS changes,
                    (SELECT count(*) FROM indexing_job) AS jobs
                """
            )
        finally:
            await connection.close()
        self.assertEqual(tuple(state), (2, 3, 2, 2))

    async def test_delete_cleanup_and_orphan_sweep_are_idempotent(self) -> None:
        kb = await self._create_kb()
        result = await SourceFileService(self.documents, self.store).store_and_activate(
            self.context,
            uuid4(),
            kb_id=kb.id,
            document_id=None,
            display_name="delete-me",
            original_filename="delete-me.txt",
            media_type="text/plain",
            source=io.BytesIO(b"delete me"),
        )
        await self.documents.delete(self.context, uuid4(), result.document.id)

        orphan = await self.store.stage(
            WORKSPACE, "orphan", io.BytesIO(b"orphan")
        )
        orphan_file = next(
            item
            for item in await self.store.list_files()
            if item.identity == orphan.identity
        )
        os.utime(
            self.root / orphan_file.location.value / orphan_file.opaque_name,
            (1, 1),
        )

        now = datetime.now(UTC) + timedelta(seconds=1)
        first = await self.reconciler(orphan_grace_seconds=0).run_once(
            self.context, now=now
        )
        second = await self.reconciler(orphan_grace_seconds=0).run_once(
            self.context, now=now
        )
        self.assertEqual(first.cleanup_completed, 1)
        self.assertEqual(first.orphans_removed, 1)
        self.assertEqual(second.cleanup_completed, 0)

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            cleanup = await connection.fetchrow(
                "SELECT status, attempt_count FROM source_file_cleanup"
            )
        finally:
            await connection.close()
        self.assertEqual(tuple(cleanup), ("completed", 0))

    async def test_cleanup_failure_is_bounded_and_content_safe(self) -> None:
        kb = await self._create_kb()
        key = uuid4()
        reserved = await self.documents.reserve_version(
            self.context,
            key,
            kb_id=kb.id,
            document_id=None,
            display_name="invalid-storage",
            source=DocumentSource(
                checksum_sha256="0" * 64,
                storage_uri="file:///caller-controlled/path",
                original_filename="secret-name.txt",
                media_type="text/plain",
                size_bytes=0,
            ),
        )
        await self.documents.activate_reserved_version(
            self.context,
            key,
            document_id=reserved.document.id,
        )
        await self.documents.delete(self.context, uuid4(), reserved.document.id)

        start = datetime.now(UTC)
        reconciler = self.reconciler()
        for offset in (0, 1, 2):
            result = await reconciler.run_once(
                self.context,
                now=start + timedelta(seconds=offset),
            )
            self.assertEqual(result.cleanup_failed, 1)

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            cleanup = await connection.fetchrow(
                """
                SELECT status, attempt_count, last_error_code
                FROM source_file_cleanup
                """
            )
        finally:
            await connection.close()
        self.assertEqual(
            tuple(cleanup),
            ("failed", 3, "FILE_STORAGE_IDENTITY_INVALID"),
        )

    async def _create_kb(self):
        return await self.knowledge_bases.create(
            self.context,
            uuid4(),
            name="file-consistency-kb",
            retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
        )


def _embedding() -> EmbeddingSpaceDefinition:
    return EmbeddingSpaceDefinition(
        provider_identity="alibaba-cloud-model-studio-qwen",
        endpoint_identity="alibaba-model-studio-beijing-embedding",
        requested_model="qwen3.7-text-embedding",
        resolved_model="qwen3.7-text-embedding",
        model_version="qwen3.7-text-embedding",
        deployment_revision=None,
        dimension=1024,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization="l2",
        configuration_fingerprint="sha256:" + "1" * 64,
        tokenizer_fingerprint=None,
        compatibility_fingerprint="sha256:" + "2" * 64,
    )


def _profile() -> IndexProfileDefinition:
    return index_profile()
