"""Local source-file orchestration and restart-safe reconciliation."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from io import BytesIO
import logging
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID

from rag_kb.auth import AuthContext
from rag_kb.domain import (
    DocumentMutationResult,
    DocumentSource,
    FileLocation,
    FileReconciliationResult,
    FileStoreError,
    InvalidStorageIdentityError,
    ResourceStateConflictError,
    SourceFileDigest,
    SourceFileIdentity,
    StagedSourceFile,
)
from rag_kb.document_processing.markdown_bundle import MARKDOWN_BUNDLE_MEDIA_TYPE
from rag_kb.ports.files import SourceFileStore
from rag_kb.services.content import (
    CREATE_DOCUMENT_ENDPOINT,
    CREATE_VERSION_ENDPOINT,
    DocumentService,
)
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, execute_in_transaction

if TYPE_CHECKING:
    from rag_kb.services.markdown_media import MarkdownMediaNormalizer


LOGGER = logging.getLogger("rag_kb.files.reconciliation")


class SourceFileService:
    """Stage bytes, reserve immutable metadata, finalize, then activate."""

    def __init__(
        self,
        documents: DocumentService,
        file_store: SourceFileStore,
        markdown_media: MarkdownMediaNormalizer | None = None,
    ) -> None:
        self._documents = documents
        self._file_store = file_store
        self._markdown_media = markdown_media

    async def store_and_activate(
        self,
        context: AuthContext,
        idempotency_key: UUID,
        *,
        kb_id: UUID,
        document_id: UUID | None,
        display_name: str,
        original_filename: str,
        media_type: str,
        source: BinaryIO,
        normalize_markdown_media: bool = False,
    ) -> DocumentMutationResult:
        endpoint = (
            CREATE_DOCUMENT_ENDPOINT
            if document_id is None
            else CREATE_VERSION_ENDPOINT
        )
        key_material = hashlib.sha256(
            "\x1f".join(
                (
                    str(context.workspace_id),
                    context.principal_id,
                    context.client_id,
                    endpoint,
                    str(idempotency_key),
                    str(kb_id),
                    str(document_id) if document_id is not None else "new",
                )
            ).encode("utf-8")
        ).hexdigest()
        stored_media_type = media_type
        if normalize_markdown_media:
            if self._markdown_media is None:
                raise RuntimeError("Markdown media normalizer is not configured")
            source.seek(0)
            raw_content = source.read()
            if not isinstance(raw_content, bytes):
                raise TypeError("source file must yield bytes")
            raw_checksum = hashlib.sha256(raw_content).hexdigest()
            identity = SourceFileIdentity(
                workspace_id=context.workspace_id,
                key=hashlib.sha256(
                    f"markdown-media-v2\x1f{key_material}\x1f{raw_checksum}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )
            existing = await self._file_store.inspect(
                identity,
                FileLocation.FINAL,
            )
            if existing is None:
                existing = await self._file_store.inspect(
                    identity,
                    FileLocation.STAGING,
                )
            if existing is None:
                normalized = await self._markdown_media.normalize(
                    raw_content,
                    original_filename=original_filename,
                    media_type=media_type,
                )
                staged = await self._file_store.stage_at(
                    identity,
                    BytesIO(normalized),
                )
            else:
                staged = StagedSourceFile(identity=identity, digest=existing)
            stored_media_type = MARKDOWN_BUNDLE_MEDIA_TYPE
        else:
            staged = await self._file_store.stage(
                context.workspace_id,
                key_material,
                source,
            )
        try:
            reserved = await self._documents.reserve_version(
                context,
                idempotency_key,
                kb_id=kb_id,
                document_id=document_id,
                display_name=display_name,
                source=DocumentSource(
                    checksum_sha256=staged.digest.checksum_sha256,
                    storage_uri=staged.identity.storage_uri,
                    original_filename=original_filename,
                    media_type=stored_media_type,
                    size_bytes=staged.digest.size_bytes,
                ),
            )
        except Exception:
            if not normalize_markdown_media:
                await self._file_store.discard_staged(staged.identity)
            raise
        await self._file_store.finalize(staged.identity, staged.digest)
        return await self._documents.activate_reserved_version(
            context,
            idempotency_key,
            document_id=reserved.document.id,
        )


class FileReconciliationService:
    """One bounded janitor pass; safe to repeat after any process failure."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        documents: DocumentService,
        file_store: SourceFileStore,
        *,
        batch_size: int,
        orphan_grace_seconds: float,
        cleanup_max_attempts: int,
        cleanup_base_delay_seconds: float,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._documents = documents
        self._file_store = file_store
        self._batch_size = batch_size
        self._orphan_grace = timedelta(seconds=orphan_grace_seconds)
        self._cleanup_max_attempts = cleanup_max_attempts
        self._cleanup_base_delay_seconds = cleanup_base_delay_seconds

    async def run_once(
        self,
        context: AuthContext,
        *,
        now: datetime | None = None,
    ) -> FileReconciliationResult:
        observed_at = now or datetime.now(UTC)

        async def load(uow: UnitOfWork):
            if uow.workspace_id != context.workspace_id:
                raise RuntimeError("reconciliation workspace does not match identity")
            return (
                await uow.file_consistency.list_references(),
                await uow.file_consistency.list_pending_mutations(
                    limit=self._batch_size
                ),
                await uow.file_consistency.list_due_cleanup(
                    now=observed_at,
                    limit=self._batch_size,
                ),
            )

        references, pending, cleanup = await execute_in_transaction(
            self._unit_of_work,
            load,
        )

        activated = 0
        pending_waiting = 0
        pending_failed = 0
        pending_conflicted = 0
        staging_references = {item.storage_uri for item in pending}
        final_references = {
            item.storage_uri
            for item in references
            if item.source_status == "available"
        }

        async def fail_pending(
            mutation: PendingFileMutation,
            failure_code: str,
        ) -> bool:
            async def fail(uow: UnitOfWork) -> bool:
                return await uow.file_consistency.fail_pending_file_mutation(
                    scope=mutation.scope,
                    document_version_id=mutation.document_version_id,
                    failure_code=failure_code,
                    failed_at=observed_at,
                )

            return await execute_in_transaction(
                self._unit_of_work,
                fail,
            )

        for mutation in pending:
            expected = SourceFileDigest(
                mutation.checksum_sha256,
                mutation.size_bytes,
            )
            mutation_context = AuthContext(
                mutation.scope.principal_id,
                mutation.scope.client_id,
                context.workspace_id,
            )
            try:
                identity = self._checked_identity(
                    mutation.storage_uri, context.workspace_id
                )
                final = await self._file_store.inspect(identity, FileLocation.FINAL)
                if final is not None:
                    if final != expected:
                        changed = await fail_pending(mutation, "SOURCE_FILE_INTEGRITY")
                        pending_failed += int(changed)
                        if changed:
                            staging_references.discard(mutation.storage_uri)
                            final_references.discard(mutation.storage_uri)
                        continue
                else:
                    staged = await self._file_store.inspect(
                        identity, FileLocation.STAGING
                    )
                    if staged is None:
                        # A rename can happen between the two inspections.
                        final = await self._file_store.inspect(
                            identity, FileLocation.FINAL
                        )
                        if final is None:
                            if observed_at < mutation.reserved_at + self._orphan_grace:
                                pending_waiting += 1
                                continue
                            changed = await fail_pending(
                                mutation, "SOURCE_FILE_MISSING"
                            )
                            pending_failed += int(changed)
                            if changed:
                                staging_references.discard(mutation.storage_uri)
                                final_references.discard(mutation.storage_uri)
                            continue
                    if final is None:
                        assert staged is not None
                        if staged != expected:
                            changed = await fail_pending(
                                mutation, "SOURCE_FILE_INTEGRITY"
                            )
                            pending_failed += int(changed)
                            if changed:
                                staging_references.discard(mutation.storage_uri)
                                final_references.discard(mutation.storage_uri)
                            continue
                        await self._file_store.finalize(identity, expected)
                        final = await self._file_store.inspect(
                            identity, FileLocation.FINAL
                        )
                        if final != expected:
                            changed = await fail_pending(
                                mutation, "SOURCE_FILE_INTEGRITY"
                            )
                            pending_failed += int(changed)
                            if changed:
                                staging_references.discard(mutation.storage_uri)
                                final_references.discard(mutation.storage_uri)
                            continue
                await self._documents.activate_reserved_version(
                    mutation_context,
                    mutation.scope.idempotency_key,
                    document_id=mutation.document_id,
                )
                activated += 1
                staging_references.discard(mutation.storage_uri)
                final_references.add(mutation.storage_uri)
            except ResourceStateConflictError:
                pending_conflicted += 1
                changed = await fail_pending(
                    mutation, "RESERVED_VERSION_STATE_CONFLICT"
                )
                pending_failed += int(changed)
                if changed:
                    staging_references.discard(mutation.storage_uri)
                    final_references.discard(mutation.storage_uri)
            except InvalidStorageIdentityError:
                changed = await fail_pending(
                    mutation, "FILE_STORAGE_IDENTITY_INVALID"
                )
                pending_failed += int(changed)
                if changed:
                    staging_references.discard(mutation.storage_uri)
                    final_references.discard(mutation.storage_uri)
            except (FileStoreError, OSError):
                LOGGER.debug(
                    "pending_file_mutation_deferred reason_code=%s document_id=%s "
                    "document_version_id=%s",
                    "FILE_STORAGE_TEMPORARILY_UNAVAILABLE",
                    mutation.document_id,
                    mutation.document_version_id,
                )
                continue

        missing_compensated = 0
        for reference in references:
            if reference.source_status != "available":
                continue
            expected = SourceFileDigest(
                reference.checksum_sha256,
                reference.size_bytes,
            )
            needs_compensation = False
            try:
                identity = self._checked_identity(
                    reference.storage_uri, context.workspace_id
                )
                actual = await self._file_store.inspect(identity, FileLocation.FINAL)
                needs_compensation = actual != expected
            except (FileStoreError, OSError):
                needs_compensation = True
            if not needs_compensation:
                continue

            async def compensate(uow: UnitOfWork) -> bool:
                return await uow.file_consistency.compensate_missing_file(
                    reference.document_version_id
                )

            changed = await execute_in_transaction(
                self._unit_of_work,
                compensate,
            )
            missing_compensated += int(changed)
            if changed:
                try:
                    identity = self._checked_identity(
                        reference.storage_uri, context.workspace_id
                    )
                    actual = await self._file_store.inspect(
                        identity, FileLocation.FINAL
                    )
                    if actual is not None:
                        await self._schedule_cleanup(
                            reference.document_version_id,
                            reference.storage_uri,
                            "integrity_mismatch",
                        )
                except (FileStoreError, OSError):
                    pass

        cleanup_completed = 0
        cleanup_failed = 0
        for task in cleanup:
            try:
                identity = self._checked_identity(task.storage_uri, context.workspace_id)
                await self._file_store.delete(identity)
            except (FileStoreError, OSError) as error:
                attempt = task.attempt_count + 1
                terminal = attempt >= self._cleanup_max_attempts
                delay = self._cleanup_base_delay_seconds * (2 ** min(attempt - 1, 10))
                error_code = (
                    "FILE_STORAGE_IDENTITY_INVALID"
                    if isinstance(error, InvalidStorageIdentityError)
                    else "FILE_DELETE_FAILED"
                )

                async def fail(uow: UnitOfWork) -> bool:
                    return await uow.file_consistency.fail_cleanup(
                        task.id,
                        expected_attempt_count=task.attempt_count,
                        error_code=error_code,
                        next_attempt_at=observed_at + timedelta(seconds=delay),
                        terminal=terminal,
                    )

                changed = await execute_in_transaction(
                    self._unit_of_work,
                    fail,
                )
                cleanup_failed += int(changed)
                continue

            async def complete(uow: UnitOfWork) -> bool:
                return await uow.file_consistency.complete_cleanup(
                    task.id, now=observed_at
                )

            changed = await execute_in_transaction(
                self._unit_of_work,
                complete,
            )
            cleanup_completed += int(changed)

        cutoff = observed_at - self._orphan_grace
        orphans_removed = 0
        for stored in await self._file_store.list_files():
            if stored.modified_at > cutoff:
                continue
            if stored.identity is not None:
                if stored.identity.workspace_id != context.workspace_id:
                    continue
                uri = stored.identity.storage_uri
            else:
                first_segment = stored.opaque_name.split("/", 1)[0]
                if first_segment != str(context.workspace_id):
                    continue
                uri = None
            referenced = (
                uri in staging_references
                if stored.location is FileLocation.STAGING
                else uri in final_references
            )
            if referenced:
                continue
            try:
                await self._file_store.delete_stored(stored)
                orphans_removed += 1
            except OSError:
                continue

        return FileReconciliationResult(
            pending_activated=activated,
            pending_waiting=pending_waiting,
            pending_failed=pending_failed,
            pending_conflicted=pending_conflicted,
            missing_compensated=missing_compensated,
            cleanup_completed=cleanup_completed,
            cleanup_failed=cleanup_failed,
            orphans_removed=orphans_removed,
        )

    async def _schedule_cleanup(
        self,
        document_version_id: UUID,
        storage_uri: str,
        reason: str,
    ) -> None:
        async def schedule(uow: UnitOfWork) -> None:
            await uow.file_consistency.schedule_cleanup(
                document_version_id=document_version_id,
                storage_uri=storage_uri,
                reason=reason,
            )

        await execute_in_transaction(
            self._unit_of_work,
            schedule,
        )

    def _checked_identity(
        self, storage_uri: str, workspace_id: UUID
    ) -> SourceFileIdentity:
        identity = self._file_store.parse_uri(storage_uri)
        if identity.workspace_id != workspace_id:
            raise InvalidStorageIdentityError("source storage workspace differs")
        return identity
