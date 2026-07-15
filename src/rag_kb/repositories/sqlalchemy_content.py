"""SQLAlchemy persistence for knowledge-base and document lifecycle state."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_kb.db.models import (
    ContentMutation as ContentMutationRow,
    Document as DocumentRow,
    DocumentSourceStatus,
    DocumentVersion as DocumentVersionRow,
    EmbeddingSpace as EmbeddingSpaceRow,
    IndexedDocumentVersion as IndexedDocumentVersionRow,
    IndexBuildStatus,
    IndexingJob as IndexingJobRow,
    IndexRevision as IndexRevisionRow,
    IndexRevisionStatus,
    IndexServingStatus,
    JobStatus,
    KnowledgeBase as KnowledgeBaseRow,
    SourceChange as SourceChangeRow,
    SourceChangeKind,
    SourceFileCleanup as SourceFileCleanupRow,
    Workspace as WorkspaceRow,
)
from rag_kb.domain import (
    ContentMutation,
    Document,
    DocumentMutationResult,
    DocumentSource,
    DocumentVersion,
    EmbeddingSpaceDefinition,
    IdempotencyScope,
    IndexProfileDefinition,
    KnowledgeBase,
    Page,
    PendingFileMutation,
    ResourceNameConflictError,
    ResourceStateConflictError,
    SourceFileCleanup,
    SourceFileReference,
)


class SqlAlchemyKnowledgeBaseRepository:
    def __init__(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        ensure_active: Callable[[], None],
    ) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def create(
        self,
        *,
        name: str,
        retrieval_defaults: dict[str, Any],
        answer_policy_defaults: dict[str, Any],
        embedding_space: EmbeddingSpaceDefinition,
        index_profile: IndexProfileDefinition,
    ) -> KnowledgeBase:
        self._ensure_active()
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": f"content-foundation:{self._workspace_id}"},
        )
        workspace = await self._session.get(WorkspaceRow, self._workspace_id)
        if workspace is None:
            workspace = WorkspaceRow(
                id=self._workspace_id,
                name=f"development-workspace-{self._workspace_id}",
            )
            self._session.add(workspace)
            await self._session.flush()

        embedding = await self._session.scalar(
            select(EmbeddingSpaceRow).where(
                EmbeddingSpaceRow.compatibility_fingerprint
                == embedding_space.compatibility_fingerprint
            )
        )
        if embedding is None:
            embedding = EmbeddingSpaceRow(
                workspace_id=self._workspace_id,
                provider_identity=embedding_space.provider_identity,
                endpoint_identity=embedding_space.endpoint_identity,
                requested_model=embedding_space.requested_model,
                resolved_model=embedding_space.resolved_model,
                model_version=embedding_space.model_version,
                deployment_revision=embedding_space.deployment_revision,
                dimension=embedding_space.dimension,
                distance_metric=embedding_space.distance_metric,
                vector_data_type=embedding_space.vector_data_type,
                normalization=embedding_space.normalization,
                configuration_fingerprint=embedding_space.configuration_fingerprint,
                tokenizer_fingerprint=embedding_space.tokenizer_fingerprint,
                compatibility_fingerprint=embedding_space.compatibility_fingerprint,
            )
            self._session.add(embedding)
            await self._session.flush()
        elif not _embedding_matches(embedding, self._workspace_id, embedding_space):
            raise ResourceStateConflictError(
                "the configured embedding space conflicts with persisted state"
            )

        duplicate = await self._session.scalar(
            select(KnowledgeBaseRow.id).where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.name == name,
            )
        )
        if duplicate is not None:
            raise ResourceNameConflictError("knowledge-base name already exists")

        kb = KnowledgeBaseRow(
            workspace_id=self._workspace_id,
            name=name,
            retrieval_defaults=dict(retrieval_defaults),
            answer_policy_defaults=dict(answer_policy_defaults),
        )
        self._session.add(kb)
        await self._session.flush()
        revision = IndexRevisionRow(
            workspace_id=self._workspace_id,
            kb_id=kb.id,
            embedding_space_id=embedding.id,
            status=IndexRevisionStatus.ACTIVE,
            source_snapshot_seq=0,
            parser_config=dict(index_profile.parser_config),
            chunking_config=dict(index_profile.chunking_config),
        )
        self._session.add(revision)
        await self._session.flush()
        now = datetime.now(UTC)
        kb.active_index_revision_id = revision.id
        kb.provisioned_at = now
        kb.updated_at = now
        await self._session.flush()
        return _knowledge_base(kb, embedding.id)

    async def get(self, kb_id: UUID) -> KnowledgeBase | None:
        self._ensure_active()
        row = (
            await self._session.execute(
                select(KnowledgeBaseRow, IndexRevisionRow.embedding_space_id)
                .join(
                    IndexRevisionRow,
                    IndexRevisionRow.id == KnowledgeBaseRow.active_index_revision_id,
                )
                .where(
                    KnowledgeBaseRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.id == kb_id,
                )
            )
        ).one_or_none()
        return _knowledge_base(row[0], row[1]) if row is not None else None

    async def list(
        self,
        *,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[KnowledgeBase]:
        self._ensure_active()
        descending = sort.startswith("-")
        field = sort.removeprefix("-")
        column = {
            "created_at": KnowledgeBaseRow.created_at,
            "name": KnowledgeBaseRow.name,
        }[field]
        statement = (
            select(KnowledgeBaseRow, IndexRevisionRow.embedding_space_id)
            .join(
                IndexRevisionRow,
                IndexRevisionRow.id == KnowledgeBaseRow.active_index_revision_id,
            )
            .where(KnowledgeBaseRow.workspace_id == self._workspace_id)
        )
        statement = _with_after(statement, column, KnowledgeBaseRow.id, after, descending)
        ordering = column.desc() if descending else column.asc()
        id_ordering = KnowledgeBaseRow.id.desc() if descending else KnowledgeBaseRow.id.asc()
        rows = (await self._session.execute(statement.order_by(ordering, id_ordering).limit(limit + 1))).all()
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = tuple(_knowledge_base(row[0], row[1]) for row in rows)
        next_values = _cursor_values(items[-1], field) if has_more and items else None
        return Page(items=items, next_values=next_values)

    async def update(
        self,
        kb_id: UUID,
        *,
        name: str | None,
        retrieval_defaults: dict[str, Any] | None,
        answer_policy_defaults: dict[str, Any] | None,
    ) -> KnowledgeBase | None:
        self._ensure_active()
        kb = await self._session.scalar(
            select(KnowledgeBaseRow).where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == kb_id,
            ).with_for_update()
        )
        if kb is None:
            return None
        if name is not None and name != kb.name:
            duplicate = await self._session.scalar(
                select(KnowledgeBaseRow.id).where(
                    KnowledgeBaseRow.workspace_id == self._workspace_id,
                    KnowledgeBaseRow.name == name,
                    KnowledgeBaseRow.id != kb_id,
                )
            )
            if duplicate is not None:
                raise ResourceNameConflictError("knowledge-base name already exists")
            kb.name = name
        if retrieval_defaults is not None:
            kb.retrieval_defaults = dict(retrieval_defaults)
        if answer_policy_defaults is not None:
            kb.answer_policy_defaults = dict(answer_policy_defaults)
        kb.updated_at = datetime.now(UTC)
        await self._session.flush()
        assert kb.active_index_revision_id is not None
        embedding_id = await self._session.scalar(
            select(IndexRevisionRow.embedding_space_id).where(
                IndexRevisionRow.id == kb.active_index_revision_id,
                IndexRevisionRow.kb_id == kb.id,
            )
        )
        assert embedding_id is not None
        return _knowledge_base(kb, embedding_id)


class SqlAlchemyDocumentRepository:
    def __init__(self, session: AsyncSession, workspace_id: UUID, ensure_active: Callable[[], None]) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def get(self, document_id: UUID) -> Document | None:
        self._ensure_active()
        row = (
            await self._session.execute(
                select(DocumentRow, DocumentVersionRow)
                .outerjoin(
                    DocumentVersionRow,
                    DocumentVersionRow.id == DocumentRow.current_version_id,
                )
                .where(
                    DocumentRow.workspace_id == self._workspace_id,
                    DocumentRow.id == document_id,
                )
            )
        ).one_or_none()
        return _document(row[0], row[1]) if row is not None else None

    async def list(
        self,
        *,
        kb_id: UUID,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[Document]:
        self._ensure_active()
        kb_exists = await self._session.scalar(
            select(KnowledgeBaseRow.id).where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == kb_id,
            )
        )
        if kb_exists is None:
            return Page(items=())
        descending = sort.startswith("-")
        field = sort.removeprefix("-")
        column = {"created_at": DocumentRow.created_at, "display_name": DocumentRow.display_name}[field]
        statement = (
            select(DocumentRow, DocumentVersionRow)
            .outerjoin(DocumentVersionRow, DocumentVersionRow.id == DocumentRow.current_version_id)
            .where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.kb_id == kb_id,
                DocumentRow.deleted_at.is_(None),
            )
        )
        statement = _with_after(statement, column, DocumentRow.id, after, descending)
        ordering = column.desc() if descending else column.asc()
        id_ordering = DocumentRow.id.desc() if descending else DocumentRow.id.asc()
        rows = (await self._session.execute(statement.order_by(ordering, id_ordering).limit(limit + 1))).all()
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = tuple(_document(row[0], row[1]) for row in rows)
        next_values = _cursor_values(items[-1], field) if has_more and items else None
        return Page(items=items, next_values=next_values)

    async def reserve_version(
        self,
        *,
        kb_id: UUID,
        document_id: UUID | None,
        display_name: str,
        source: DocumentSource,
    ) -> DocumentMutationResult:
        self._ensure_active()
        kb_exists = await self._session.scalar(
            select(KnowledgeBaseRow.id).where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == kb_id,
                KnowledgeBaseRow.provisioned_at.is_not(None),
            )
        )
        if kb_exists is None:
            raise ResourceStateConflictError("knowledge base is unavailable")
        if document_id is None:
            document = DocumentRow(
                workspace_id=self._workspace_id,
                kb_id=kb_id,
                display_name=display_name,
            )
            self._session.add(document)
            await self._session.flush()
            version_number = 1
        else:
            document = await self._session.scalar(
                select(DocumentRow).where(
                    DocumentRow.workspace_id == self._workspace_id,
                    DocumentRow.kb_id == kb_id,
                    DocumentRow.id == document_id,
                ).with_for_update()
            )
            if document is None:
                raise ResourceStateConflictError("document is unavailable")
            maximum = await self._session.scalar(
                select(func.max(DocumentVersionRow.version_number)).where(
                    DocumentVersionRow.workspace_id == self._workspace_id,
                    DocumentVersionRow.document_id == document.id,
                )
            )
            version_number = (maximum or 0) + 1
            document.display_name = display_name
            document.updated_at = datetime.now(UTC)
        version = DocumentVersionRow(
            workspace_id=self._workspace_id,
            kb_id=kb_id,
            document_id=document.id,
            version_number=version_number,
            source_status=DocumentSourceStatus.UNAVAILABLE,
            checksum_sha256=source.checksum_sha256,
            storage_uri=source.storage_uri,
            original_filename=source.original_filename,
            media_type=source.media_type,
            size_bytes=source.size_bytes,
        )
        self._session.add(version)
        await self._session.flush()
        return DocumentMutationResult(
            document=_document(document, None),
            document_version_id=version.id,
        )

    async def activate_version(
        self,
        *,
        document_id: UUID,
        document_version_id: UUID,
    ) -> DocumentMutationResult:
        self._ensure_active()
        document = await self._session.scalar(
            select(DocumentRow).where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.id == document_id,
            ).with_for_update()
        )
        version = await self._session.scalar(
            select(DocumentVersionRow).where(
                DocumentVersionRow.workspace_id == self._workspace_id,
                DocumentVersionRow.document_id == document_id,
                DocumentVersionRow.id == document_version_id,
            ).with_for_update()
        )
        if document is None or version is None:
            raise ResourceStateConflictError("reserved document version is unavailable")
        if version.source_status is not DocumentSourceStatus.UNAVAILABLE:
            raise ResourceStateConflictError("reserved document version is not provisional")

        allocation = await self._session.execute(
            update(KnowledgeBaseRow)
            .where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == document.kb_id,
                KnowledgeBaseRow.provisioned_at.is_not(None),
            )
            .values(
                source_change_seq=KnowledgeBaseRow.source_change_seq + 1,
                updated_at=func.now(),
            )
            .returning(
                KnowledgeBaseRow.source_change_seq,
                KnowledgeBaseRow.active_index_revision_id,
            )
        )
        allocated = allocation.one_or_none()
        if allocated is None or allocated.active_index_revision_id is None:
            raise ResourceStateConflictError("knowledge base has no active revision")
        version.source_status = DocumentSourceStatus.AVAILABLE
        document.current_version_id = version.id
        document.deleted_at = None
        document.updated_at = datetime.now(UTC)
        change = SourceChangeRow(
            workspace_id=self._workspace_id,
            kb_id=document.kb_id,
            source_change_seq=allocated.source_change_seq,
            document_id=document.id,
            document_version_id=version.id,
            change_kind=SourceChangeKind.UPSERT,
        )
        self._session.add(change)
        target = IndexedDocumentVersionRow(
            workspace_id=self._workspace_id,
            kb_id=document.kb_id,
            document_id=document.id,
            document_version_id=version.id,
            index_revision_id=allocated.active_index_revision_id,
            source_change_seq=allocated.source_change_seq,
            build_status=IndexBuildStatus.QUEUED,
            serving_status=IndexServingStatus.CANDIDATE,
        )
        self._session.add(target)
        await self._session.flush()
        job = IndexingJobRow(
            workspace_id=self._workspace_id,
            kb_id=document.kb_id,
            indexed_document_version_id=target.id,
            status=JobStatus.QUEUED,
            phase="queued",
        )
        self._session.add(job)
        await self._session.flush()
        return DocumentMutationResult(
            document=_document(document, version),
            document_version_id=version.id,
            source_change_id=change.id,
            source_change_seq=allocated.source_change_seq,
            indexed_document_version_id=target.id,
            index_revision_id=allocated.active_index_revision_id,
            job_id=job.id,
            job_status=job.status.value,
        )

    async def soft_delete(self, document_id: UUID) -> DocumentMutationResult | None:
        self._ensure_active()
        document = await self._session.scalar(
            select(DocumentRow).where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.id == document_id,
            ).with_for_update()
        )
        if document is None:
            return None
        current = None
        if document.current_version_id is not None:
            current = await self._session.get(DocumentVersionRow, document.current_version_id)
        await self._schedule_file_cleanup(document.id)
        if document.deleted_at is not None:
            return DocumentMutationResult(document=_document(document, current))
        allocation = await self._session.execute(
            update(KnowledgeBaseRow)
            .where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == document.kb_id,
            )
            .values(
                source_change_seq=KnowledgeBaseRow.source_change_seq + 1,
                updated_at=func.now(),
            )
            .returning(
                KnowledgeBaseRow.source_change_seq,
                KnowledgeBaseRow.active_index_revision_id,
            )
        )
        allocated = allocation.one()
        now = datetime.now(UTC)
        document.deleted_at = now
        document.updated_at = now
        await self._session.execute(
            update(DocumentVersionRow)
            .where(
                DocumentVersionRow.workspace_id == self._workspace_id,
                DocumentVersionRow.document_id == document.id,
            )
            .values(source_status=DocumentSourceStatus.DELETED)
        )
        change = SourceChangeRow(
            workspace_id=self._workspace_id,
            kb_id=document.kb_id,
            source_change_seq=allocated.source_change_seq,
            document_id=document.id,
            document_version_id=None,
            change_kind=SourceChangeKind.DELETE,
        )
        self._session.add(change)
        if allocated.active_index_revision_id is not None:
            targets = select(IndexedDocumentVersionRow.id).where(
                IndexedDocumentVersionRow.workspace_id == self._workspace_id,
                IndexedDocumentVersionRow.document_id == document.id,
                IndexedDocumentVersionRow.index_revision_id == allocated.active_index_revision_id,
            )
            await self._session.execute(
                update(IndexedDocumentVersionRow)
                .where(IndexedDocumentVersionRow.id.in_(targets))
                .values(serving_status=IndexServingStatus.RETIRED, updated_at=func.now())
            )
            await self._session.execute(
                update(IndexingJobRow)
                .where(
                    IndexingJobRow.indexed_document_version_id.in_(targets),
                    IndexingJobRow.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
                )
                .values(
                    status=JobStatus.CANCELLED,
                    phase="cancelled",
                    claimed_by=None,
                    claimed_at=None,
                    heartbeat_at=None,
                    next_attempt_at=None,
                    updated_at=func.now(),
                )
            )
        await self._session.flush()
        if current is not None:
            current.source_status = DocumentSourceStatus.DELETED
        return DocumentMutationResult(
            document=_document(document, current),
            source_change_id=change.id,
            source_change_seq=allocated.source_change_seq,
            index_revision_id=allocated.active_index_revision_id,
        )

    async def _schedule_file_cleanup(self, document_id: UUID) -> None:
        versions = (
            await self._session.execute(
                select(DocumentVersionRow.id, DocumentVersionRow.storage_uri).where(
                    DocumentVersionRow.workspace_id == self._workspace_id,
                    DocumentVersionRow.document_id == document_id,
                )
            )
        ).all()
        if not versions:
            return
        await self._session.execute(
            pg_insert(SourceFileCleanupRow)
            .values(
                [
                    {
                        "workspace_id": self._workspace_id,
                        "document_version_id": version.id,
                        "storage_uri": version.storage_uri,
                        "reason": "document_deleted",
                    }
                    for version in versions
                ]
            )
            .on_conflict_do_nothing(index_elements=["document_version_id"])
        )


class SqlAlchemyContentMutationRepository:
    def __init__(self, session: AsyncSession, workspace_id: UUID, ensure_active: Callable[[], None]) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def lock(self, scope: IdempotencyScope) -> None:
        self._ensure_active()
        key = f"{scope.principal_id}\x1f{scope.client_id}\x1f{scope.endpoint}\x1f{scope.idempotency_key}"
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": key},
        )

    async def get(self, scope: IdempotencyScope) -> ContentMutation | None:
        self._ensure_active()
        row = await self._session.scalar(
            select(ContentMutationRow).where(
                ContentMutationRow.workspace_id == self._workspace_id,
                ContentMutationRow.principal_id == scope.principal_id,
                ContentMutationRow.client_id == scope.client_id,
                ContentMutationRow.endpoint == scope.endpoint,
                ContentMutationRow.idempotency_key == scope.idempotency_key,
            )
        )
        return _mutation(row) if row is not None else None

    async def add(
        self,
        *,
        scope: IdempotencyScope,
        request_hash: str,
        operation: str,
        status: str,
        result: DocumentMutationResult | KnowledgeBase,
    ) -> ContentMutation:
        self._ensure_active()
        values = _result_ids(result)
        row = ContentMutationRow(
            workspace_id=self._workspace_id,
            principal_id=scope.principal_id,
            client_id=scope.client_id,
            endpoint=scope.endpoint,
            idempotency_key=scope.idempotency_key,
            request_hash=request_hash,
            operation=operation,
            status=status,
            **values,
        )
        self._session.add(row)
        await self._session.flush()
        return _mutation(row)

    async def complete(
        self, scope: IdempotencyScope, result: DocumentMutationResult
    ) -> ContentMutation:
        self._ensure_active()
        row = await self._session.scalar(
            select(ContentMutationRow).where(
                ContentMutationRow.workspace_id == self._workspace_id,
                ContentMutationRow.principal_id == scope.principal_id,
                ContentMutationRow.client_id == scope.client_id,
                ContentMutationRow.endpoint == scope.endpoint,
                ContentMutationRow.idempotency_key == scope.idempotency_key,
            ).with_for_update()
        )
        if row is None:
            raise ResourceStateConflictError("content mutation reservation is missing")
        for name, value in _result_ids(result).items():
            setattr(row, name, value)
        row.status = "completed"
        row.updated_at = datetime.now(UTC)
        await self._session.flush()
        return _mutation(row)

    async def add_indexing_retry(
        self,
        *,
        scope: IdempotencyScope,
        request_hash: str,
        kb_id: UUID,
        document_id: UUID,
        document_version_id: UUID,
        indexed_document_version_id: UUID,
        index_revision_id: UUID,
        job_id: UUID,
    ) -> ContentMutation:
        self._ensure_active()
        row = ContentMutationRow(
            workspace_id=self._workspace_id,
            principal_id=scope.principal_id,
            client_id=scope.client_id,
            endpoint=scope.endpoint,
            idempotency_key=scope.idempotency_key,
            request_hash=request_hash,
            operation="indexing_job.retry",
            status="completed",
            kb_id=kb_id,
            document_id=document_id,
            document_version_id=document_version_id,
            source_change_id=None,
            indexed_document_version_id=indexed_document_version_id,
            index_revision_id=index_revision_id,
            job_id=job_id,
        )
        self._session.add(row)
        await self._session.flush()
        return _mutation(row)


class SqlAlchemyFileConsistencyRepository:
    def __init__(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        ensure_active: Callable[[], None],
    ) -> None:
        self._session = session
        self._workspace_id = workspace_id
        self._ensure_active = ensure_active

    async def delete_expired_cleanup_records(
        self,
        *,
        before: datetime,
        limit: int,
    ) -> int:
        self._ensure_active()
        ids = tuple(
            (
                await self._session.execute(
                    select(SourceFileCleanupRow.id)
                    .where(
                        SourceFileCleanupRow.workspace_id == self._workspace_id,
                        SourceFileCleanupRow.status.in_(("completed", "failed")),
                        SourceFileCleanupRow.updated_at <= before,
                    )
                    .order_by(SourceFileCleanupRow.updated_at, SourceFileCleanupRow.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).scalars()
        )
        if not ids:
            return 0
        result = await self._session.execute(
            delete(SourceFileCleanupRow).where(
                SourceFileCleanupRow.workspace_id == self._workspace_id,
                SourceFileCleanupRow.id.in_(ids),
            )
        )
        return int(result.rowcount or 0)

    async def list_references(self) -> tuple[SourceFileReference, ...]:
        self._ensure_active()
        rows = (
            await self._session.execute(
                select(DocumentVersionRow).where(
                    DocumentVersionRow.workspace_id == self._workspace_id
                )
            )
        ).scalars()
        return tuple(
            SourceFileReference(
                document_id=row.document_id,
                document_version_id=row.id,
                source_status=row.source_status.value,
                storage_uri=row.storage_uri,
                checksum_sha256=row.checksum_sha256,
                size_bytes=row.size_bytes,
            )
            for row in rows
        )

    async def list_pending_mutations(
        self, *, limit: int
    ) -> tuple[PendingFileMutation, ...]:
        self._ensure_active()
        rows = (
            await self._session.execute(
                select(ContentMutationRow, DocumentVersionRow)
                .join(
                    DocumentVersionRow,
                    DocumentVersionRow.id == ContentMutationRow.document_version_id,
                )
                .where(
                    ContentMutationRow.workspace_id == self._workspace_id,
                    ContentMutationRow.status == "pending",
                    ContentMutationRow.operation == "document.version.reserve",
                    DocumentVersionRow.source_status == DocumentSourceStatus.UNAVAILABLE,
                )
                .order_by(ContentMutationRow.created_at, ContentMutationRow.id)
                .limit(limit)
            )
        ).all()
        return tuple(
            PendingFileMutation(
                scope=IdempotencyScope(
                    mutation.principal_id,
                    mutation.client_id,
                    mutation.endpoint,
                    mutation.idempotency_key,
                ),
                document_id=mutation.document_id,
                document_version_id=version.id,
                storage_uri=version.storage_uri,
                checksum_sha256=version.checksum_sha256,
                size_bytes=version.size_bytes,
            )
            for mutation, version in rows
            if mutation.document_id is not None
        )

    async def list_due_cleanup(
        self, *, now: datetime, limit: int
    ) -> tuple[SourceFileCleanup, ...]:
        self._ensure_active()
        rows = (
            await self._session.execute(
                select(SourceFileCleanupRow)
                .where(
                    SourceFileCleanupRow.workspace_id == self._workspace_id,
                    SourceFileCleanupRow.status == "pending",
                    SourceFileCleanupRow.next_attempt_at <= now,
                )
                .order_by(SourceFileCleanupRow.next_attempt_at, SourceFileCleanupRow.id)
                .limit(limit)
            )
        ).scalars()
        return tuple(_file_cleanup(row) for row in rows)

    async def schedule_cleanup(
        self,
        *,
        document_version_id: UUID,
        storage_uri: str,
        reason: str,
    ) -> None:
        self._ensure_active()
        await self._session.execute(
            pg_insert(SourceFileCleanupRow)
            .values(
                workspace_id=self._workspace_id,
                document_version_id=document_version_id,
                storage_uri=storage_uri,
                reason=reason,
            )
            .on_conflict_do_nothing(index_elements=["document_version_id"])
        )

    async def complete_cleanup(self, cleanup_id: UUID, *, now: datetime) -> bool:
        self._ensure_active()
        result = await self._session.execute(
            update(SourceFileCleanupRow)
            .where(
                SourceFileCleanupRow.workspace_id == self._workspace_id,
                SourceFileCleanupRow.id == cleanup_id,
                SourceFileCleanupRow.status == "pending",
            )
            .values(
                status="completed",
                completed_at=now,
                last_error_code=None,
                updated_at=now,
            )
        )
        return bool(result.rowcount)

    async def fail_cleanup(
        self,
        cleanup_id: UUID,
        *,
        expected_attempt_count: int,
        error_code: str,
        next_attempt_at: datetime,
        terminal: bool,
    ) -> bool:
        self._ensure_active()
        result = await self._session.execute(
            update(SourceFileCleanupRow)
            .where(
                SourceFileCleanupRow.workspace_id == self._workspace_id,
                SourceFileCleanupRow.id == cleanup_id,
                SourceFileCleanupRow.status == "pending",
                SourceFileCleanupRow.attempt_count == expected_attempt_count,
            )
            .values(
                status="failed" if terminal else "pending",
                attempt_count=SourceFileCleanupRow.attempt_count + 1,
                last_error_code=error_code,
                next_attempt_at=next_attempt_at,
                updated_at=func.now(),
            )
        )
        return bool(result.rowcount)

    async def compensate_missing_file(self, document_version_id: UUID) -> bool:
        self._ensure_active()
        version = await self._session.scalar(
            select(DocumentVersionRow)
            .where(
                DocumentVersionRow.workspace_id == self._workspace_id,
                DocumentVersionRow.id == document_version_id,
            )
            .with_for_update()
        )
        if version is None or version.source_status is not DocumentSourceStatus.AVAILABLE:
            return False
        document = await self._session.scalar(
            select(DocumentRow)
            .where(
                DocumentRow.workspace_id == self._workspace_id,
                DocumentRow.id == version.document_id,
            )
            .with_for_update()
        )
        if document is None:
            return False
        allocation = await self._session.execute(
            update(KnowledgeBaseRow)
            .where(
                KnowledgeBaseRow.workspace_id == self._workspace_id,
                KnowledgeBaseRow.id == version.kb_id,
            )
            .values(
                source_change_seq=KnowledgeBaseRow.source_change_seq + 1,
                updated_at=func.now(),
            )
            .returning(KnowledgeBaseRow.source_change_seq)
        )
        source_change_seq = allocation.scalar_one()
        version.source_status = DocumentSourceStatus.UNAVAILABLE
        self._session.add(
            SourceChangeRow(
                workspace_id=self._workspace_id,
                kb_id=version.kb_id,
                source_change_seq=source_change_seq,
                document_id=version.document_id,
                document_version_id=None,
                change_kind=SourceChangeKind.DELETE,
            )
        )
        targets = select(IndexedDocumentVersionRow.id).where(
            IndexedDocumentVersionRow.workspace_id == self._workspace_id,
            IndexedDocumentVersionRow.document_version_id == version.id,
        )
        await self._session.execute(
            update(IndexedDocumentVersionRow)
            .where(IndexedDocumentVersionRow.id.in_(targets))
            .values(serving_status=IndexServingStatus.RETIRED, updated_at=func.now())
        )
        await self._session.execute(
            update(IndexingJobRow)
            .where(
                IndexingJobRow.indexed_document_version_id.in_(targets),
                IndexingJobRow.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
            )
            .values(
                status=JobStatus.CANCELLED,
                phase="source_missing",
                claimed_by=None,
                claimed_at=None,
                heartbeat_at=None,
                next_attempt_at=None,
                updated_at=func.now(),
            )
        )
        await self._session.flush()
        return True


def _embedding_matches(row: EmbeddingSpaceRow, workspace_id: UUID, value: EmbeddingSpaceDefinition) -> bool:
    return (
        row.workspace_id == workspace_id
        and row.provider_identity == value.provider_identity
        and row.endpoint_identity == value.endpoint_identity
        and row.requested_model == value.requested_model
        and row.resolved_model == value.resolved_model
        and row.model_version == value.model_version
        and row.deployment_revision == value.deployment_revision
        and row.dimension == value.dimension
        and row.distance_metric == value.distance_metric
        and row.vector_data_type == value.vector_data_type
        and row.normalization == value.normalization
        and row.configuration_fingerprint == value.configuration_fingerprint
        and row.tokenizer_fingerprint == value.tokenizer_fingerprint
    )


def _knowledge_base(row: KnowledgeBaseRow, embedding_space_id: UUID) -> KnowledgeBase:
    assert row.active_index_revision_id is not None
    assert row.provisioned_at is not None
    return KnowledgeBase(
        id=row.id,
        workspace_id=row.workspace_id,
        name=row.name,
        source_change_seq=row.source_change_seq,
        active_index_revision_id=row.active_index_revision_id,
        embedding_space_id=embedding_space_id,
        retrieval_defaults=dict(row.retrieval_defaults),
        answer_policy_defaults=dict(row.answer_policy_defaults),
        provisioned_at=row.provisioned_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _version(row: DocumentVersionRow) -> DocumentVersion:
    return DocumentVersion(
        id=row.id,
        document_id=row.document_id,
        version_number=row.version_number,
        source_status=row.source_status.value,
        checksum_sha256=row.checksum_sha256,
        storage_uri=row.storage_uri,
        original_filename=row.original_filename,
        media_type=row.media_type,
        size_bytes=row.size_bytes,
        created_at=row.created_at,
    )


def _document(row: DocumentRow, version: DocumentVersionRow | None) -> Document:
    return Document(
        id=row.id,
        workspace_id=row.workspace_id,
        kb_id=row.kb_id,
        display_name=row.display_name,
        current_version=_version(version) if version is not None else None,
        deleted_at=row.deleted_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _mutation(row: ContentMutationRow) -> ContentMutation:
    return ContentMutation(
        scope=IdempotencyScope(row.principal_id, row.client_id, row.endpoint, row.idempotency_key),
        request_hash=row.request_hash,
        operation=row.operation,
        status=row.status,
        kb_id=row.kb_id,
        document_id=row.document_id,
        document_version_id=row.document_version_id,
        source_change_id=row.source_change_id,
        indexed_document_version_id=row.indexed_document_version_id,
        index_revision_id=row.index_revision_id,
        job_id=row.job_id,
    )


def _file_cleanup(row: SourceFileCleanupRow) -> SourceFileCleanup:
    return SourceFileCleanup(
        id=row.id,
        document_version_id=row.document_version_id,
        storage_uri=row.storage_uri,
        reason=row.reason,
        status=row.status,
        attempt_count=row.attempt_count,
        next_attempt_at=row.next_attempt_at,
    )


def _result_ids(result: DocumentMutationResult | KnowledgeBase) -> dict[str, UUID | None]:
    if isinstance(result, KnowledgeBase):
        return {
            "kb_id": result.id,
            "document_id": None,
            "document_version_id": None,
            "source_change_id": None,
            "indexed_document_version_id": None,
            "index_revision_id": result.active_index_revision_id,
            "job_id": None,
        }
    return {
        "kb_id": result.document.kb_id,
        "document_id": result.document.id,
        "document_version_id": result.document_version_id,
        "source_change_id": result.source_change_id,
        "indexed_document_version_id": result.indexed_document_version_id,
        "index_revision_id": result.index_revision_id,
        "job_id": result.job_id,
    }


def _with_after(statement, column, id_column, after: tuple[str, ...] | None, descending: bool):
    if after is None:
        return statement
    if len(after) != 2:
        raise ValueError("cursor must contain sort value and id")
    raw_value, raw_id = after
    value: Any = datetime.fromisoformat(raw_value) if getattr(column.type, "python_type", None) is datetime else raw_value
    item_id = UUID(raw_id)
    compare = column < value if descending else column > value
    compare_id = id_column < item_id if descending else id_column > item_id
    return statement.where(or_(compare, and_(column == value, compare_id)))


def _cursor_values(item: KnowledgeBase | Document, field: str) -> tuple[str, str]:
    value = getattr(item, field)
    rendered = value.isoformat() if isinstance(value, datetime) else str(value)
    return rendered, str(item.id)
