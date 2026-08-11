from __future__ import annotations

import asyncio
import os
import unittest
from uuid import UUID, uuid4

import asyncpg

from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.document_processing.profiles import index_profile
from rag_kb.domain import (
    ChunkingPreset,
    DocumentSource,
    DuplicateDocumentError,
    EmbeddingSpaceDefinition,
    IdempotencyKeyReusedError,
    IndexProfileDefinition,
    ResourceNameConflictError,
    ResourceNotFoundError,
)
from rag_kb.services.content import DocumentService, KnowledgeBaseService
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")
WORKSPACE = UUID("01900000-0000-7000-8000-000000000201")
OTHER_WORKSPACE = UUID("01900000-0000-7000-8000-000000000202")


@unittest.skipUnless(
    MIGRATION_DSN and RUNTIME_SQLALCHEMY_DSN,
    "database integration DSNs are not configured",
)
class ContentLifecycleTests(unittest.IsolatedAsyncioTestCase):
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
            process=DatabaseProcess.API,
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

    async def asyncTearDown(self) -> None:
        await self.database.close()

    async def test_concurrent_create_replays_one_knowledge_base_and_rejects_hash_change(self) -> None:
        key = uuid4()
        first, second = await asyncio.gather(
            self.knowledge_bases.create(
                self.context,
                key,
                name="engineering",
                retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
            ),
            self.knowledge_bases.create(
                self.context,
                key,
                name="engineering",
                retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
            ),
        )
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.active_index_revision_id, second.active_index_revision_id)

        with self.assertRaises(IdempotencyKeyReusedError):
            await self.knowledge_bases.create(
                self.context,
                key,
                name="different",
                retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
            )
        with self.assertRaises(IdempotencyKeyReusedError):
            await self.knowledge_bases.create(
                self.context,
                key,
                name="engineering",
                chunking_preset=ChunkingPreset.SEMANTIC_BALANCED_V1,
                retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
            )

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            counts = await connection.fetchrow(
                """
                SELECT
                    (SELECT count(*) FROM knowledge_base) AS knowledge_bases,
                    (SELECT count(*) FROM index_revision) AS revisions,
                    (SELECT count(*) FROM embedding_space) AS spaces,
                    (SELECT count(*) FROM content_mutation) AS mutations
                """
            )
        finally:
            await connection.close()
        self.assertEqual(tuple(counts), (1, 1, 1, 1))

    async def test_semantic_create_registers_required_analysis_on_text_space(
        self,
    ) -> None:
        knowledge_base = await self.knowledge_bases.create(
            self.context,
            uuid4(),
            name="semantic-roles",
            chunking_preset=ChunkingPreset.SEMANTIC_BALANCED_V1,
            retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
        )

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            rows = await connection.fetch(
                """
                SELECT role, embedding_space_id, required
                FROM index_revision_embedding_space
                WHERE index_revision_id = $1
                ORDER BY role
                """,
                knowledge_base.active_index_revision_id,
            )
        finally:
            await connection.close()

        self.assertEqual(
            [(row["role"], row["required"]) for row in rows],
            [("semantic_analysis", True), ("text_retrieval", True)],
        )
        self.assertEqual({row["embedding_space_id"] for row in rows}, {
            knowledge_base.embedding_space_id
        })

    async def test_version_activation_is_atomic_immutable_and_soft_delete_is_idempotent(self) -> None:
        kb = await self._create_kb()
        create_key = uuid4()
        reserved = await self.documents.reserve_version(
            self.context,
            create_key,
            kb_id=kb.id,
            document_id=None,
            display_name="guide.md",
            source=_source("1"),
        )
        self.assertIsNone(reserved.document.current_version)
        self.assertIsNotNone(reserved.document_version_id)

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            before = await connection.fetchrow(
                """
                SELECT kb.source_change_seq,
                       (SELECT count(*) FROM source_change) AS changes,
                       (SELECT count(*) FROM indexing_job) AS jobs
                FROM knowledge_base kb WHERE kb.id = $1
                """,
                kb.id,
            )
        finally:
            await connection.close()
        self.assertEqual(tuple(before), (0, 0, 0))

        activated = await self.documents.activate_reserved_version(
            self.context,
            create_key,
            document_id=reserved.document.id,
        )
        self.assertEqual(activated.source_change_seq, 1)
        self.assertEqual(activated.job_status, "queued")
        self.assertEqual(activated.document.current_version.source_status, "available")

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    "UPDATE document_version SET checksum_sha256 = $1 WHERE id = $2",
                    "f" * 64,
                    activated.document_version_id,
                )
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute(
                    "UPDATE source_change SET source_change_seq = 9 WHERE id = $1",
                    activated.source_change_id,
                )
        finally:
            await connection.close()

        delete_key = uuid4()
        deleted = await self.documents.delete(
            self.context, delete_key, activated.document.id
        )
        replay = await self.documents.delete(
            self.context, delete_key, activated.document.id
        )
        self.assertEqual(deleted.document.id, replay.document.id)
        self.assertIsNotNone(deleted.document.deleted_at)

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            state = await connection.fetchrow(
                """
                SELECT kb.source_change_seq,
                       (SELECT count(*) FROM source_change) AS changes,
                       (SELECT source_status::text FROM document_version WHERE id = $2) AS source_status,
                       (SELECT serving_status::text FROM indexed_document_version WHERE id = $3) AS serving_status,
                       (SELECT status::text FROM indexing_job WHERE id = $4) AS job_status
                       ,(SELECT count(*) FROM source_file_cleanup WHERE document_version_id = $2) AS cleanup_count
                FROM knowledge_base kb WHERE kb.id = $1
                """,
                kb.id,
                activated.document_version_id,
                activated.indexed_document_version_id,
                activated.job_id,
            )
        finally:
            await connection.close()
        self.assertEqual(
            tuple(state),
            (2, 2, "deleted", "retired", "cancelled", 1),
        )

    async def test_concurrent_version_activation_allocates_gapless_source_changes(self) -> None:
        kb = await self._create_kb()
        reservations = []
        for ordinal in (1, 2):
            key = uuid4()
            reserved = await self.documents.reserve_version(
                self.context,
                key,
                kb_id=kb.id,
                document_id=None,
                display_name=f"document-{ordinal}.txt",
                source=_source(str(ordinal)),
            )
            reservations.append((key, reserved))

        activated = await asyncio.gather(
            *(
                self.documents.activate_reserved_version(
                    self.context, key, document_id=reserved.document.id
                )
                for key, reserved in reservations
            )
        )
        self.assertEqual(sorted(item.source_change_seq for item in activated), [1, 2])

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            sequences = await connection.fetch(
                "SELECT source_change_seq FROM source_change ORDER BY source_change_seq"
            )
        finally:
            await connection.close()
        self.assertEqual([row["source_change_seq"] for row in sequences], [1, 2])

    async def test_knowledge_base_delete_retires_content_and_allows_name_reuse(
        self,
    ) -> None:
        kb = await self._create_kb()
        upload_key = uuid4()
        reserved = await self.documents.reserve_version(
            self.context,
            upload_key,
            kb_id=kb.id,
            document_id=None,
            display_name="guide.txt",
            source=_source("8"),
        )
        activated = await self.documents.activate_reserved_version(
            self.context,
            upload_key,
            document_id=reserved.document.id,
        )

        delete_key = uuid4()
        deleted = await self.knowledge_bases.delete(
            self.context,
            delete_key,
            kb.id,
        )
        replay = await self.knowledge_bases.delete(
            self.context,
            delete_key,
            kb.id,
        )
        self.assertEqual(replay.id, deleted.id)
        self.assertIsNotNone(deleted.deleted_at)
        with self.assertRaises(ResourceNotFoundError):
            await self.knowledge_bases.get(self.context, kb.id)
        with self.assertRaises(ResourceNotFoundError):
            await self.documents.get(self.context, activated.document.id)

        recreated = await self.knowledge_bases.create(
            self.context,
            uuid4(),
            name="lifecycle-kb",
            retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
        )
        self.assertNotEqual(recreated.id, kb.id)

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            state = await connection.fetchrow(
                """
                SELECT
                    (SELECT deleted_at IS NOT NULL FROM knowledge_base WHERE id = $1)
                        AS kb_deleted,
                    (SELECT deleted_at IS NOT NULL FROM document WHERE id = $2)
                        AS document_deleted,
                    (SELECT source_status::text FROM document_version WHERE id = $3)
                        AS source_status,
                    (SELECT serving_status::text FROM indexed_document_version WHERE id = $4)
                        AS serving_status,
                    (SELECT status::text FROM indexing_job WHERE id = $5)
                        AS job_status,
                    (SELECT status::text FROM index_revision WHERE id = $6)
                        AS revision_status,
                    (SELECT count(*) FROM source_file_cleanup
                     WHERE document_version_id = $3) AS cleanup_count
                """,
                kb.id,
                activated.document.id,
                activated.document_version_id,
                activated.indexed_document_version_id,
                activated.job_id,
                kb.active_index_revision_id,
            )
        finally:
            await connection.close()
        self.assertEqual(
            tuple(state),
            (True, True, "deleted", "retired", "cancelled", "retired", 1),
        )

    async def test_document_detail_reports_active_revision_manifest_counts(self) -> None:
        kb = await self._create_kb()
        key = uuid4()
        reserved = await self.documents.reserve_version(
            self.context,
            key,
            kb_id=kb.id,
            document_id=None,
            display_name="scan.pdf",
            source=_source("7"),
        )
        activated = await self.documents.activate_reserved_version(
            self.context,
            key,
            document_id=reserved.document.id,
        )
        assert activated.indexed_document_version_id is not None
        assert activated.document.current_version is not None

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute(
                """
                INSERT INTO index_artifact_manifest (
                    indexed_document_version_id,
                    source_checksum_sha256,
                    profile_fingerprint,
                    element_sequence_hash,
                    asset_manifest_hash,
                    unit_plan,
                    representation_matrix,
                    relation_plan,
                    unit_count,
                    asset_count,
                    representation_count,
                    relation_count,
                    relation_manifest_hash,
                    manifest_hash
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    '[]'::jsonb, '[]'::jsonb, '[]'::jsonb,
                    2, 1, 3, 0, $6, $7
                )
                """,
                activated.indexed_document_version_id,
                activated.document.current_version.checksum_sha256,
                "1" * 64,
                "2" * 64,
                "3" * 64,
                "4" * 64,
                "5" * 64,
            )
        finally:
            await connection.close()

        detail = await self.documents.get_detail(
            self.context, activated.document.id
        )

        self.assertEqual(detail.document.id, activated.document.id)
        assert detail.index is not None
        self.assertEqual(
            (
                detail.index.build_status,
                detail.index.serving_status,
                detail.index.unit_count,
                detail.index.asset_count,
                detail.index.representation_count,
            ),
            ("queued", "candidate", 2, 1, 3),
        )

    async def test_workspace_bound_repository_hides_foreign_identifiers(self) -> None:
        kb = await self._create_kb()
        other_factory = SqlAlchemyUnitOfWorkFactory(
            self.database.sessions, OTHER_WORKSPACE
        )
        other_service = KnowledgeBaseService(
            other_factory,
            SingleWorkspaceAccessPolicy(OTHER_WORKSPACE),
            embedding_space=_embedding(),
            index_profile=_profile(),
        )
        with self.assertRaises(ResourceNotFoundError):
            await other_service.get(
                AuthContext("other-principal", "other-client", OTHER_WORKSPACE),
                kb.id,
            )

    async def test_update_and_keyset_pagination_persist_typed_configuration(self) -> None:
        created = []
        for name in ("charlie", "alpha", "bravo"):
            created.append(
                await self.knowledge_bases.create(
                    self.context,
                    uuid4(),
                    name=name,
                    retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
                )
            )
        update_key = uuid4()
        updated = await self.knowledge_bases.update(
            self.context,
            update_key,
            created[0].id,
            name="delta",
            retrieval_defaults={"strategy": "exact_vector", "top_k": 7},
            answer_policy_defaults={
                "answer_style": "summary",
                "insufficiency_policy": "partial_answer",
            },
        )
        replay = await self.knowledge_bases.update(
            self.context,
            update_key,
            created[0].id,
            name="delta",
            retrieval_defaults={"strategy": "exact_vector", "top_k": 7},
            answer_policy_defaults={
                "answer_style": "summary",
                "insufficiency_policy": "partial_answer",
            },
        )
        self.assertEqual(replay.id, updated.id)
        self.assertEqual(replay.retrieval_defaults["top_k"], 7)
        self.assertEqual(replay.answer_policy_defaults["answer_style"], "summary")
        self.assertEqual(
            replay.answer_policy_defaults["insufficiency_policy"], "partial_answer"
        )

        first = await self.knowledge_bases.list(
            self.context, limit=2, sort="name", after=None
        )
        second = await self.knowledge_bases.list(
            self.context, limit=2, sort="name", after=first.next_values
        )
        self.assertEqual(
            [item.name for item in (*first.items, *second.items)],
            ["alpha", "bravo", "delta"],
        )

        with self.assertRaises(ResourceNameConflictError):
            await self.knowledge_bases.update(
                self.context,
                uuid4(),
                created[1].id,
                name="bravo",
                retrieval_defaults=None,
            )

    async def test_new_document_same_checksum_is_rejected_but_new_version_is_allowed(
        self,
    ) -> None:
        kb = await self._create_kb()
        source = _source("9")
        first_key = uuid4()
        first = await self.documents.reserve_version(
            self.context,
            first_key,
            kb_id=kb.id,
            document_id=None,
            display_name="first.txt",
            source=source,
        )

        replay = await self.documents.reserve_version(
            self.context,
            first_key,
            kb_id=kb.id,
            document_id=None,
            display_name="first.txt",
            source=source,
        )
        self.assertEqual(replay.document.id, first.document.id)
        await self.documents.activate_reserved_version(
            self.context,
            first_key,
            document_id=first.document.id,
        )

        with self.assertRaises(DuplicateDocumentError) as raised:
            await self.documents.reserve_version(
                self.context,
                uuid4(),
                kb_id=kb.id,
                document_id=None,
                display_name="second.txt",
                source=source,
            )
        self.assertEqual(raised.exception.existing_document_id, first.document.id)

        new_version = await self.documents.reserve_version(
            self.context,
            uuid4(),
            kb_id=kb.id,
            document_id=first.document.id,
            display_name="first-v2.txt",
            source=source,
        )
        self.assertEqual(new_version.document.id, first.document.id)

    async def _create_kb(self):
        return await self.knowledge_bases.create(
            self.context,
            uuid4(),
            name="lifecycle-kb",
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


def _source(ordinal: str) -> DocumentSource:
    return DocumentSource(
        checksum_sha256=ordinal.zfill(64),
        storage_uri=f"file:///sources/{ordinal}.txt",
        original_filename=f"{ordinal}.txt",
        media_type="text/plain",
        size_bytes=10,
    )
