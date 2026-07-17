"""One retry-safe indexing execution with no external I/O inside DB transactions."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from typing import TypeVar

from rag_kb.adapters import (
    DocumentProcessor,
    EmbeddingModelAdapter,
    FixedPgVectorSpace,
    SourceFileStore,
)
from rag_kb.domain import (
    ErrorCode,
    FileStoreError,
    IndexChunkWrite,
    IndexingCancelled,
    IndexingCommand,
    IndexingExecutionError,
    IndexingPhase,
    IndexingResult,
    IndexingTarget,
    InvalidStorageIdentityError,
    ParserExecutionError,
    ParserSource,
    PromotionCommand,
    SourceFileMissingError,
    VectorRecordWrite,
    stable_chunk_id,
    stable_vector_id,
    validate_embedding_vector,
)
from rag_kb.document_processing import (
    UNSTRUCTURED_CHUNKING_CONFIG,
    UNSTRUCTURED_PARSER_CONFIG,
)
from rag_kb.indexing.promotion import CandidatePromotionService
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


ResultT = TypeVar("ResultT")


class IndexingPipeline:
    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        file_store: SourceFileStore,
        document_processor: DocumentProcessor,
        embedding_provider: EmbeddingModelAdapter,
        vector_space: FixedPgVectorSpace,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._file_store = file_store
        self._document_processor = document_processor
        self._embedding_provider = embedding_provider
        self._vector_space = vector_space
        self._promotion = CandidatePromotionService(unit_of_work)

    async def execute(self, command: IndexingCommand) -> IndexingResult:
        phase = IndexingPhase.SOURCE_READ
        try:
            target = await self._prepare(command)
            if target is None:
                raise IndexingExecutionError(
                    ErrorCode.INDEX_TARGET_INVALID,
                    phase=phase,
                    diagnostic={"check": "job_target_mapping"},
            )
            if target.already_complete:
                promotion = await self._promotion.promote(
                    _promotion_command(command)
                )
                return IndexingResult(
                    command.job_id,
                    command.indexed_document_version_id,
                    "ready",
                    await self._stored_chunk_count(command),
                    replayed=True,
                    serving_status=promotion.status.value,
                )
            self._require_revision_profile(target)
            self._vector_space.require_compatible(
                target.embedding_space,
                self._embedding_provider.embedding_space,
            )
            content = await self._read_source(target)

            phase = IndexingPhase.PARSING
            await self._set_phase(command, phase)
            try:
                processed = await self._document_processor.process(
                    ParserSource(
                        original_filename=target.original_filename,
                        media_type=target.media_type,
                        content=content,
                    )
                )
            except ParserExecutionError as error:
                raise IndexingExecutionError(
                    error.code,
                    phase=phase,
                    diagnostic=error.diagnostic,
                ) from error
            except Exception as error:
                raise IndexingExecutionError(
                    ErrorCode.PARSER_CRASHED,
                    phase=phase,
                    diagnostic={"check": "processor_contract"},
                ) from error

            phase = IndexingPhase.EMBEDDING
            await self._set_phase(command, phase)
            batch_size = self._embedding_provider.max_batch_size
            for offset in range(0, len(processed.chunks), batch_size):
                drafts = processed.chunks[offset : offset + batch_size]
                try:
                    embedded = await self._embedding_provider.embed_documents(
                        tuple(draft.text for draft in drafts)
                    )
                except IndexingExecutionError:
                    raise
                except Exception as error:
                    raise IndexingExecutionError(
                        ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                        phase=phase,
                        diagnostic={"check": "provider_contract"},
                    ) from error
                if len(embedded.vectors) != len(drafts):
                    raise IndexingExecutionError(
                        ErrorCode.EMBEDDING_RESPONSE_INVALID,
                        phase=phase,
                        diagnostic={
                            "check": "batch_count",
                            "expected": len(drafts),
                            "observed": len(embedded.vectors),
                        },
                    )
                for vector in embedded.vectors:
                    validate_embedding_vector(vector, target.embedding_space)
                chunks, vectors = self._writes(target, drafts, embedded.vectors)
                phase = IndexingPhase.PERSISTING
                await self._upsert(command, chunks, vectors)
                phase = IndexingPhase.EMBEDDING

            phase = IndexingPhase.VALIDATING
            await self._set_phase(command, phase)
            await self._complete(command, expected_chunks=len(processed.chunks))
            promotion = await self._promotion.promote(_promotion_command(command))
            return IndexingResult(
                command.job_id,
                command.indexed_document_version_id,
                "ready",
                len(processed.chunks),
                serving_status=promotion.status.value,
            )
        except IndexingCancelled:
            return IndexingResult(
                command.job_id,
                command.indexed_document_version_id,
                "cancelled",
                0,
                serving_status="retired",
            )
        except IndexingExecutionError as error:
            await self._record_failure(command, error)
            raise
        except Exception as error:
            failure = IndexingExecutionError(
                ErrorCode.INDEX_PERSISTENCE_FAILED,
                phase=phase,
                diagnostic={"operation": phase.value},
            )
            await self._record_failure(command, failure)
            raise failure from error

    async def _prepare(self, command: IndexingCommand) -> IndexingTarget | None:
        return await self._transaction(lambda uow: uow.indexing.prepare(command))

    async def _set_phase(self, command: IndexingCommand, phase: IndexingPhase) -> None:
        changed = await self._transaction(
            lambda uow: uow.indexing.set_phase(command, phase)
        )
        if not changed:
            raise IndexingCancelled

    async def _upsert(
        self,
        command: IndexingCommand,
        chunks: tuple[IndexChunkWrite, ...],
        vectors: tuple[VectorRecordWrite, ...],
    ) -> None:
        try:
            changed = await self._transaction(
                lambda uow: uow.indexing.upsert_batch(command, chunks, vectors)
            )
        except IndexingExecutionError:
            raise
        except Exception as error:
            raise IndexingExecutionError(
                ErrorCode.INDEX_PERSISTENCE_FAILED,
                phase=IndexingPhase.PERSISTING,
                diagnostic={"operation": "upsert_batch"},
            ) from error
        if not changed:
            raise IndexingCancelled

    async def _complete(
        self,
        command: IndexingCommand,
        *,
        expected_chunks: int,
    ) -> None:
        changed = await self._transaction(
            lambda uow: uow.indexing.complete(
                command, expected_chunks=expected_chunks
            )
        )
        if not changed:
            raise IndexingCancelled

    async def _record_failure(
        self,
        command: IndexingCommand,
        error: IndexingExecutionError,
    ) -> None:
        async def persist(uow: UnitOfWork) -> bool:
            return await uow.indexing.fail(
                command,
                phase=error.phase,
                error_code=error.code.value,
                error_detail=_safe_diagnostic(error.diagnostic),
            )

        await self._transaction(persist)

    async def _stored_chunk_count(self, command: IndexingCommand) -> int:
        return await self._transaction(lambda uow: uow.indexing.count_chunks(command))

    async def _transaction(
        self,
        operation: Callable[[UnitOfWork], Awaitable[ResultT]],
    ) -> ResultT:
        return await execute_in_transaction(
            self._unit_of_work,
            operation,
            purpose=UnitOfWorkPurpose.INDEXING,
        )

    async def _read_source(self, target: IndexingTarget) -> bytes:
        try:
            identity = self._file_store.parse_uri(target.storage_uri)
            if identity.workspace_id != target.workspace_id:
                raise InvalidStorageIdentityError("source workspace differs")
            content = await self._file_store.read_final(identity)
        except SourceFileMissingError as error:
            raise IndexingExecutionError(
                ErrorCode.SOURCE_FILE_MISSING,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "final_file"},
            ) from error
        except (FileStoreError, OSError) as error:
            raise IndexingExecutionError(
                ErrorCode.SOURCE_FILE_INTEGRITY,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "storage_identity"},
            ) from error
        if (
            len(content) != target.size_bytes
            or hashlib.sha256(content).hexdigest() != target.checksum_sha256
        ):
            raise IndexingExecutionError(
                ErrorCode.SOURCE_FILE_INTEGRITY,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "checksum_and_size"},
            )
        return content

    @staticmethod
    def _require_revision_profile(target: IndexingTarget) -> None:
        if (
            target.parser_config != UNSTRUCTURED_PARSER_CONFIG
            or target.chunking_config != UNSTRUCTURED_CHUNKING_CONFIG
        ):
            raise IndexingExecutionError(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                phase=IndexingPhase.SOURCE_READ,
                diagnostic={"check": "parser_chunking_profile"},
            )

    @staticmethod
    def _writes(target, drafts, embeddings):
        chunks: list[IndexChunkWrite] = []
        vectors: list[VectorRecordWrite] = []
        for draft, embedding in zip(drafts, embeddings, strict=True):
            chunk_id = stable_chunk_id(target.indexed_document_version_id, draft.ordinal)
            chunks.append(
                IndexChunkWrite(
                    id=chunk_id,
                    ordinal=draft.ordinal,
                    content=draft.text,
                    content_hash=draft.content_sha256,
                    token_count=len(draft.text),
                    source_location=dict(draft.source_location),
                    hierarchy=dict(draft.hierarchy),
                    source_metadata={
                        "document_id": str(target.document_id),
                        "document_version_id": str(target.document_version_id),
                        "original_filename": target.original_filename,
                        "media_type": target.media_type,
                        "checksum_sha256": target.checksum_sha256,
                        "processing": dict(draft.processing_metadata),
                    },
                )
            )
            vectors.append(
                VectorRecordWrite(
                    id=stable_vector_id(target.embedding_space_id, chunk_id),
                    index_chunk_id=chunk_id,
                    embedding_space_id=target.embedding_space_id,
                    embedding=embedding,
                )
            )
        return tuple(chunks), tuple(vectors)


def _safe_diagnostic(value: dict) -> dict:
    allowed = {
        "check",
        "expected",
        "observed",
        "expected_chunks",
        "observed_chunks",
        "observed_vectors",
        "fields",
        "http_status",
        "retryable",
        "retry_exhausted",
        "operation",
        "limit_name",
        "limit",
        "exit_kind",
    }
    return {key: item for key, item in value.items() if key in allowed}


def _promotion_command(command: IndexingCommand) -> PromotionCommand:
    return PromotionCommand(
        job_id=command.job_id,
        indexed_document_version_id=command.indexed_document_version_id,
    )
