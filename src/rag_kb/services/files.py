"""Local source-file orchestration and restart-safe reconciliation."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import BinaryIO
from uuid import UUID

from rag_kb.adapters import SourceFileStore
from rag_kb.auth import AuthContext
from rag_kb.domain import (
    DocumentMutationResult,
    DocumentSource,
    FileLocation,
    FileReconciliationResult,
    FileStoreError,
    InvalidStorageIdentityError,
    SourceFileDigest,
    SourceFileIdentity,
)
from rag_kb.services.content import (
    CREATE_DOCUMENT_ENDPOINT,
    CREATE_VERSION_ENDPOINT,
    DocumentService,
)
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


class SourceFileService:
    """Stage bytes, reserve immutable metadata, finalize, then activate."""

    def __init__(self, documents: DocumentService, file_store: SourceFileStore) -> None:
        self._documents = documents
        self._file_store = file_store

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
        staged = await self._file_store.stage(
            context.workspace_id,
            key_material,
            source,
        )
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
                media_type=media_type,
                size_bytes=staged.digest.size_bytes,
            ),
        )
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
            purpose=UnitOfWorkPurpose.RECONCILIATION,
        )

        activated = 0
        for mutation in pending:
            if (
                mutation.scope.principal_id != context.principal_id
                or mutation.scope.client_id != context.client_id
            ):
                continue
            expected = SourceFileDigest(
                mutation.checksum_sha256,
                mutation.size_bytes,
            )
            try:
                identity = self._checked_identity(
                    mutation.storage_uri, context.workspace_id
                )
                final = await self._file_store.inspect(identity, FileLocation.FINAL)
                if final is not None:
                    if final != expected:
                        await self._schedule_cleanup(
                            mutation.document_version_id,
                            mutation.storage_uri,
                            "integrity_mismatch",
                        )
                        continue
                else:
                    staged = await self._file_store.inspect(
                        identity, FileLocation.STAGING
                    )
                    if staged is None:
                        continue
                    if staged != expected:
                        await self._schedule_cleanup(
                            mutation.document_version_id,
                            mutation.storage_uri,
                            "integrity_mismatch",
                        )
                        continue
                    await self._file_store.finalize(identity, expected)
                await self._documents.activate_reserved_version(
                    context,
                    mutation.scope.idempotency_key,
                    document_id=mutation.document_id,
                )
                activated += 1
            except (FileStoreError, OSError):
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
                purpose=UnitOfWorkPurpose.RECONCILIATION,
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
                    purpose=UnitOfWorkPurpose.RECONCILIATION,
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
                purpose=UnitOfWorkPurpose.RECONCILIATION,
            )
            cleanup_completed += int(changed)

        staging_references = {item.storage_uri for item in pending}
        final_references = {item.storage_uri for item in references}
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
            purpose=UnitOfWorkPurpose.RECONCILIATION,
        )

    def _checked_identity(
        self, storage_uri: str, workspace_id: UUID
    ) -> SourceFileIdentity:
        identity = self._file_store.parse_uri(storage_uri)
        if identity.workspace_id != workspace_id:
            raise InvalidStorageIdentityError("source storage workspace differs")
        return identity
