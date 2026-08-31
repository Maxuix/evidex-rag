"""Application services for database-only content lifecycle use cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.document_processing.profiles import index_profile, profile_for_preset
from rag_kb.domain import (
    ChunkingPreset,
    Document,
    DocumentChunkInspection,
    DocumentDetail,
    DocumentMutationResult,
    DocumentSource,
    EmbeddingSpaceDefinition,
    IdempotencyKeyReusedError,
    IdempotencyScope,
    IndexProfileDefinition,
    KnowledgeBase,
    ModelKind,
    ModelValidationStatus,
    Page,
    ParsingPreset,
    ResourceNotFoundError,
    ResourceStateConflictError,
    canonical_request_hash,
)
from rag_kb.uow import execute_in_transaction

if TYPE_CHECKING:
    from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWork, SqlAlchemyUnitOfWorkFactory


CREATE_KB_ENDPOINT = "POST /api/v1/knowledge-bases"
DELETE_KB_ENDPOINT = "DELETE /api/v1/knowledge-bases/{kb_id}"
PATCH_KB_ENDPOINT = "PATCH /api/v1/knowledge-bases/{kb_id}"
DELETE_DOCUMENT_ENDPOINT = "DELETE /api/v1/documents/{document_id}"
CREATE_DOCUMENT_ENDPOINT = "POST /api/v1/knowledge-bases/{kb_id}/documents"
CREATE_VERSION_ENDPOINT = "POST /api/v1/documents/{document_id}/versions"


@dataclass(frozen=True, slots=True)
class ContentServices:
    knowledge_bases: "KnowledgeBaseService"
    documents: "DocumentService"


def build_content_services(
    unit_of_work: SqlAlchemyUnitOfWorkFactory,
    access_policy: SingleWorkspaceAccessPolicy,
    embedding: Any,
    multimodal_embedding: Any | None = None,
) -> ContentServices:
    """Construct content services without exposing domain configuration to the API."""

    definition = (
        embedding_space_definition(embedding) if embedding is not None else None
    )
    multimodal_definition = (
        embedding_space_definition(multimodal_embedding)
        if multimodal_embedding is not None
        else None
    )
    profile = index_profile()
    return ContentServices(
        knowledge_bases=KnowledgeBaseService(
            unit_of_work,
            access_policy,
            embedding_space=definition,
            cross_modal_embedding_space=multimodal_definition,
            index_profile=profile,
        ),
        documents=DocumentService(unit_of_work, access_policy),
    )


def embedding_space_definition(embedding: Any) -> EmbeddingSpaceDefinition:
    """Map immutable provider settings to the persisted compatibility fact."""

    return EmbeddingSpaceDefinition(
        provider_identity=embedding.provider_identity,
        endpoint_identity=embedding.logical_endpoint_identity,
        requested_model=embedding.model,
        resolved_model=embedding.resolved_model,
        model_version=embedding.model_version,
        deployment_revision=None,
        dimension=embedding.dimension,
        distance_metric=embedding.metric,
        vector_data_type=embedding.vector_data_type,
        normalization=embedding.normalization,
        configuration_fingerprint=embedding.configuration_fingerprint,
        tokenizer_fingerprint=None,
        compatibility_fingerprint=embedding.compatibility_fingerprint,
    )


def unconfigured_embedding_space_definition() -> EmbeddingSpaceDefinition:
    """Return a non-persistable runtime shape for the fail-closed adapter."""

    fingerprint = "sha256:" + "0" * 64
    return EmbeddingSpaceDefinition(
        provider_identity="unconfigured",
        endpoint_identity="unconfigured",
        requested_model="unconfigured",
        resolved_model="unconfigured",
        model_version="unconfigured",
        deployment_revision=None,
        dimension=1024,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization="l2",
        configuration_fingerprint=fingerprint,
        tokenizer_fingerprint=None,
        compatibility_fingerprint=fingerprint,
    )


async def _selected_embedding_space(
    uow: SqlAlchemyUnitOfWork,
    kind: ModelKind,
    fallback: EmbeddingSpaceDefinition | None,
    revision_id: UUID | None = None,
    require_unified: bool = False,
) -> EmbeddingSpaceDefinition:
    repository = getattr(uow, "model_settings", None)
    if repository is None:
        if fallback is None:
            raise ResourceStateConflictError("an embedding model must be selected")
        return fallback
    selection = await repository.get_selection()
    revision_id = revision_id or (
        selection.text_embedding_profile_revision_id
        if kind is ModelKind.TEXT_EMBEDDING
        else selection.multimodal_embedding_profile_revision_id
    )
    if revision_id is None:
        if fallback is None:
            raise ResourceStateConflictError("an embedding model must be selected")
        return fallback
    bundle = await repository.get_profile_revision(revision_id)
    if bundle is None or bundle.profile.kind is not kind:
        raise ResourceStateConflictError("the selected embedding model is invalid")
    if (
        not bundle.profile.enabled
        or not bundle.provider.enabled
        or bundle.current_revision.validation_status
        is not ModelValidationStatus.VALID
    ):
        raise ResourceStateConflictError("the selected embedding model is unavailable")
    revision = bundle.current_revision
    snapshot = revision.validation_snapshot
    if revision.compatibility_fingerprint is None or snapshot is None:
        raise ResourceStateConflictError("embedding compatibility is missing")
    if require_unified and (
        bundle.profile.kind is not ModelKind.MULTIMODAL_EMBEDDING
        or not snapshot.shared_text_image_space_confirmed
        or {"text_document", "text_query", "image"}
        - {value.value for value in snapshot.input_capabilities}
    ):
        raise ResourceStateConflictError(
            "the selected multimodal model is not eligible for a unified space"
        )
    return EmbeddingSpaceDefinition(
        provider_identity=bundle.provider.name,
        endpoint_identity=bundle.provider_revision.configuration_fingerprint,
        requested_model=revision.model,
        resolved_model=revision.model,
        model_version=revision.model,
        deployment_revision=None,
        dimension=snapshot.selected_dimension,
        distance_metric=snapshot.distance_metric,
        vector_data_type=snapshot.vector_data_type,
        normalization=snapshot.normalization,
        configuration_fingerprint=revision.configuration_fingerprint,
        tokenizer_fingerprint=None,
        compatibility_fingerprint=revision.compatibility_fingerprint,
        model_profile_revision_id=revision.id,
        dimension_request_mode=snapshot.dimension_request_mode.value,
    )


class KnowledgeBaseService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWorkFactory,
        access_policy: SingleWorkspaceAccessPolicy,
        *,
        embedding_space: EmbeddingSpaceDefinition | None,
        cross_modal_embedding_space: EmbeddingSpaceDefinition | None = None,
        index_profile: IndexProfileDefinition,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._access_policy = access_policy
        self._embedding_space = embedding_space
        self._cross_modal_embedding_space = cross_modal_embedding_space
        expected = profile_for_preset(ChunkingPreset.STRUCTURAL_BALANCED_V2)
        if index_profile != expected:
            raise ValueError("default index profile must match the preset registry")

    async def create(
        self,
        context: AuthContext,
        idempotency_key: UUID,
        *,
        name: str,
        parsing_preset: ParsingPreset | str = ParsingPreset.TEXT_LOCAL_V1,
        chunking_preset: ChunkingPreset | str = ChunkingPreset.STRUCTURAL_BALANCED_V2,
        retrieval_defaults: dict[str, Any],
        embedding_selection: dict[str, Any] | None = None,
    ) -> KnowledgeBase:
        self._authorize(context)
        resolved_preset = ChunkingPreset(chunking_preset)
        resolved_parsing = ParsingPreset(parsing_preset)
        resolved_profile = profile_for_preset(resolved_preset, resolved_parsing)
        scope = IdempotencyScope(
            context.principal_id,
            context.client_id,
            CREATE_KB_ENDPOINT,
            idempotency_key,
        )
        request_hash = canonical_request_hash(
            {
                "name": name,
                "parsing": {"preset": resolved_parsing.value},
                "chunking": {"preset": resolved_preset.value},
                "retrieval_defaults": retrieval_defaults,
                "embedding": (
                    embedding_selection
                    if embedding_selection is not None
                    else {"strategy": "default_for_parsing"}
                ),
            }
        )
        async def persist(uow: SqlAlchemyUnitOfWork) -> KnowledgeBase:
            _require_scope(uow, context)
            await uow.content_mutations.lock(scope)
            prior = await uow.content_mutations.get(scope)
            if prior is not None:
                if prior.request_hash != request_hash:
                    raise IdempotencyKeyReusedError(
                        "idempotency key was already used with a different request"
                    )
                assert prior.kb_id is not None
                existing = await uow.knowledge_bases.get(prior.kb_id)
                if existing is None:
                    raise ResourceNotFoundError("idempotent knowledge base is unavailable")
                return existing
            strategy = (
                embedding_selection.get("strategy")
                if embedding_selection is not None
                else (
                    "text_only"
                    if resolved_parsing is ParsingPreset.TEXT_LOCAL_V1
                    else "dual_space"
                )
            )
            if resolved_parsing is ParsingPreset.TEXT_LOCAL_V1 and strategy != "text_only":
                raise ResourceStateConflictError(
                    "text parsing requires text-only embedding"
                )
            if resolved_parsing is ParsingPreset.MULTIMODAL_LOCAL_V2 and strategy == "text_only":
                raise ResourceStateConflictError(
                    "multimodal parsing requires dual or unified embedding"
                )
            if strategy == "unified_multimodal":
                unified = await _selected_embedding_space(
                    uow,
                    ModelKind.MULTIMODAL_EMBEDDING,
                    self._cross_modal_embedding_space,
                    revision_id=(embedding_selection or {}).get(
                        "profile_revision_id"
                    ),
                    require_unified=True,
                )
                embedding_space = unified
                cross_modal_embedding_space = unified
            else:
                embedding_space = await _selected_embedding_space(
                    uow,
                    ModelKind.TEXT_EMBEDDING,
                    self._embedding_space,
                    revision_id=(embedding_selection or {}).get(
                        "text_profile_revision_id"
                    ),
                )
                cross_modal_embedding_space = None
            if strategy == "dual_space":
                cross_modal_embedding_space = await _selected_embedding_space(
                    uow,
                    ModelKind.MULTIMODAL_EMBEDDING,
                    self._cross_modal_embedding_space,
                    revision_id=(embedding_selection or {}).get(
                        "multimodal_profile_revision_id"
                    ),
                )
            created = await uow.knowledge_bases.create(
                name=name,
                retrieval_defaults=retrieval_defaults,
                embedding_space=embedding_space,
                cross_modal_embedding_space=cross_modal_embedding_space,
                index_profile=resolved_profile,
            )
            await uow.content_mutations.add(
                scope=scope,
                request_hash=request_hash,
                operation="knowledge_base.create",
                status="completed",
                result=created,
            )
            return created

        return await execute_in_transaction(self._unit_of_work, persist)

    async def get(self, context: AuthContext, kb_id: UUID) -> KnowledgeBase:
        self._authorize(context)

        async def load(uow: SqlAlchemyUnitOfWork) -> KnowledgeBase:
            _require_scope(uow, context)
            result = await uow.knowledge_bases.get(kb_id)
            if result is None:
                raise ResourceNotFoundError("knowledge base was not found")
            return result

        return await execute_in_transaction(self._unit_of_work, load)

    async def list(
        self,
        context: AuthContext,
        *,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[KnowledgeBase]:
        self._authorize(context)

        async def load(uow: SqlAlchemyUnitOfWork) -> Page[KnowledgeBase]:
            _require_scope(uow, context)
            return await uow.knowledge_bases.list(limit=limit, sort=sort, after=after)

        return await execute_in_transaction(self._unit_of_work, load)

    async def update(
        self,
        context: AuthContext,
        idempotency_key: UUID,
        kb_id: UUID,
        *,
        name: str | None,
        retrieval_defaults: dict[str, Any] | None,
    ) -> KnowledgeBase:
        self._authorize(context)
        scope = IdempotencyScope(
            context.principal_id,
            context.client_id,
            PATCH_KB_ENDPOINT,
            idempotency_key,
        )
        request_hash = canonical_request_hash(
            {
                "kb_id": str(kb_id),
                "name": name,
                "retrieval_defaults": retrieval_defaults,
            }
        )

        async def persist(uow: SqlAlchemyUnitOfWork) -> KnowledgeBase:
            _require_scope(uow, context)
            await uow.content_mutations.lock(scope)
            prior = await uow.content_mutations.get(scope)
            if prior is not None:
                _require_same_hash(prior.request_hash, request_hash)
                assert prior.kb_id is not None
                replay = await uow.knowledge_bases.get(prior.kb_id)
                if replay is None:
                    raise ResourceNotFoundError("knowledge base was not found")
                return replay
            updated = await uow.knowledge_bases.update(
                kb_id,
                name=name,
                retrieval_defaults=retrieval_defaults,
            )
            if updated is None:
                raise ResourceNotFoundError("knowledge base was not found")
            await uow.content_mutations.add(
                scope=scope,
                request_hash=request_hash,
                operation="knowledge_base.update",
                status="completed",
                result=updated,
            )
            return updated

        return await execute_in_transaction(self._unit_of_work, persist)

    async def delete(
        self,
        context: AuthContext,
        idempotency_key: UUID,
        kb_id: UUID,
    ) -> KnowledgeBase:
        self._authorize(context)
        scope = IdempotencyScope(
            context.principal_id,
            context.client_id,
            DELETE_KB_ENDPOINT,
            idempotency_key,
        )
        request_hash = canonical_request_hash({"kb_id": str(kb_id)})

        async def persist(uow: SqlAlchemyUnitOfWork) -> KnowledgeBase:
            _require_scope(uow, context)
            await uow.content_mutations.lock(scope)
            prior = await uow.content_mutations.get(scope)
            if prior is not None:
                _require_same_hash(prior.request_hash, request_hash)
                assert prior.kb_id is not None
                replay = await uow.knowledge_bases.get(
                    prior.kb_id, include_deleted=True
                )
                if replay is None:
                    raise ResourceNotFoundError("knowledge base was not found")
                return replay
            deleted = await uow.knowledge_bases.soft_delete(kb_id)
            if deleted is None:
                raise ResourceNotFoundError("knowledge base was not found")
            await uow.content_mutations.add(
                scope=scope,
                request_hash=request_hash,
                operation="knowledge_base.delete",
                status="completed",
                result=deleted,
            )
            return deleted

        return await execute_in_transaction(self._unit_of_work, persist)

    def _authorize(self, context: AuthContext) -> None:
        self._access_policy.metadata_filter(context)


class DocumentService:
    """Relational lifecycle; file orchestration is intentionally a later layer."""

    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWorkFactory,
        access_policy: SingleWorkspaceAccessPolicy,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._access_policy = access_policy

    async def get(self, context: AuthContext, document_id: UUID) -> Document:
        self._authorize(context)

        async def load(uow: SqlAlchemyUnitOfWork) -> Document:
            _require_scope(uow, context)
            document = await uow.documents.get(document_id)
            if document is None:
                raise ResourceNotFoundError("document was not found")
            return document

        return await execute_in_transaction(self._unit_of_work, load)

    async def get_detail(
        self, context: AuthContext, document_id: UUID
    ) -> DocumentDetail:
        self._authorize(context)

        async def load(uow: SqlAlchemyUnitOfWork) -> DocumentDetail:
            _require_scope(uow, context)
            detail = await uow.documents.get_detail(document_id)
            if detail is None:
                raise ResourceNotFoundError("document was not found")
            return detail

        return await execute_in_transaction(
            self._unit_of_work, load
        )

    async def inspect_chunks(
        self,
        context: AuthContext,
        document_id: UUID,
        *,
        limit: int,
        after: tuple[str, ...] | None,
    ) -> DocumentChunkInspection:
        self._authorize(context)

        async def load(uow: SqlAlchemyUnitOfWork) -> DocumentChunkInspection:
            _require_scope(uow, context)
            inspection = await uow.documents.inspect_chunks(
                document_id, limit=limit, after=after
            )
            if inspection is None:
                raise ResourceNotFoundError("document was not found")
            return inspection

        return await execute_in_transaction(
            self._unit_of_work, load
        )

    async def list(
        self,
        context: AuthContext,
        *,
        kb_id: UUID,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[Document]:
        self._authorize(context)

        async def load(uow: SqlAlchemyUnitOfWork) -> Page[Document]:
            _require_scope(uow, context)
            if await uow.knowledge_bases.get(kb_id) is None:
                raise ResourceNotFoundError("knowledge base was not found")
            return await uow.documents.list(kb_id=kb_id, limit=limit, sort=sort, after=after)

        return await execute_in_transaction(self._unit_of_work, load)

    async def reserve_version(
        self,
        context: AuthContext,
        idempotency_key: UUID,
        *,
        kb_id: UUID,
        document_id: UUID | None,
        display_name: str,
        source: DocumentSource,
    ) -> DocumentMutationResult:
        """Reserve an unavailable immutable version before later file finalization."""

        self._authorize(context)
        endpoint = CREATE_DOCUMENT_ENDPOINT if document_id is None else CREATE_VERSION_ENDPOINT
        scope = IdempotencyScope(context.principal_id, context.client_id, endpoint, idempotency_key)
        request_hash = canonical_request_hash(
            {
                "kb_id": str(kb_id),
                "document_id": str(document_id) if document_id else None,
                "display_name": display_name,
                "checksum_sha256": source.checksum_sha256,
                "storage_uri": source.storage_uri,
                "original_filename": source.original_filename,
                "media_type": source.media_type,
                "size_bytes": source.size_bytes,
            }
        )

        async def persist(uow: SqlAlchemyUnitOfWork) -> DocumentMutationResult:
            _require_scope(uow, context)
            await uow.content_mutations.lock(scope)
            prior = await uow.content_mutations.get(scope)
            if prior is not None:
                _require_same_hash(prior.request_hash, request_hash)
                _require_replayable_mutation(prior)
                return await _document_result_from_mutation(uow, prior)
            reserved = await uow.documents.reserve_version(
                kb_id=kb_id,
                document_id=document_id,
                display_name=display_name,
                source=source,
            )
            await uow.content_mutations.add(
                scope=scope,
                request_hash=request_hash,
                operation="document.version.reserve",
                status="pending",
                result=reserved,
            )
            return reserved

        return await execute_in_transaction(self._unit_of_work, persist)

    async def activate_reserved_version(
        self,
        context: AuthContext,
        idempotency_key: UUID,
        *,
        document_id: UUID,
    ) -> DocumentMutationResult:
        """Publish a finalized source and create its index target atomically."""

        self._authorize(context)

        async def persist(uow: SqlAlchemyUnitOfWork) -> DocumentMutationResult:
            _require_scope(uow, context)
            selected_scope = None
            mutation = None
            for endpoint in (CREATE_DOCUMENT_ENDPOINT, CREATE_VERSION_ENDPOINT):
                candidate = IdempotencyScope(context.principal_id, context.client_id, endpoint, idempotency_key)
                await uow.content_mutations.lock(candidate)
                found = await uow.content_mutations.get(candidate)
                if found is not None:
                    selected_scope, mutation = candidate, found
                    break
            if mutation is None or selected_scope is None:
                raise ResourceNotFoundError("document version reservation was not found")
            if mutation.document_id != document_id or mutation.document_version_id is None:
                raise IdempotencyKeyReusedError("idempotency key targets another document")
            _require_replayable_mutation(mutation)
            if mutation.status == "completed":
                return await _document_result_from_mutation(uow, mutation)
            activated = await uow.documents.activate_version(
                document_id=document_id, document_version_id=mutation.document_version_id
            )
            await uow.content_mutations.complete(selected_scope, activated)
            return activated

        return await execute_in_transaction(self._unit_of_work, persist)

    async def delete(
        self,
        context: AuthContext,
        idempotency_key: UUID,
        document_id: UUID,
    ) -> DocumentMutationResult:
        self._authorize(context)
        scope = IdempotencyScope(context.principal_id, context.client_id, DELETE_DOCUMENT_ENDPOINT, idempotency_key)
        request_hash = canonical_request_hash({"document_id": str(document_id)})

        async def persist(uow: SqlAlchemyUnitOfWork) -> DocumentMutationResult:
            _require_scope(uow, context)
            await uow.content_mutations.lock(scope)
            prior = await uow.content_mutations.get(scope)
            if prior is not None:
                _require_same_hash(prior.request_hash, request_hash)
                return await _document_result_from_mutation(uow, prior)
            deleted = await uow.documents.soft_delete(document_id)
            if deleted is None:
                raise ResourceNotFoundError("document was not found")
            await uow.content_mutations.add(
                scope=scope,
                request_hash=request_hash,
                operation="document.delete",
                status="completed",
                result=deleted,
            )
            return deleted

        return await execute_in_transaction(self._unit_of_work, persist)

    async def exclude_chunk(
        self,
        context: AuthContext,
        *,
        document_id: UUID,
        chunk_id: UUID,
    ) -> datetime:
        self._authorize(context)

        async def persist(uow: SqlAlchemyUnitOfWork) -> datetime:
            _require_scope(uow, context)
            excluded_at = await uow.documents.exclude_chunk(
                document_id=document_id,
                chunk_id=chunk_id,
            )
            if excluded_at is None:
                raise ResourceNotFoundError("document chunk was not found")
            return excluded_at

        return await execute_in_transaction(self._unit_of_work, persist)

    def _authorize(self, context: AuthContext) -> None:
        self._access_policy.metadata_filter(context)


async def _document_result_from_mutation(
    uow: SqlAlchemyUnitOfWork, mutation
) -> DocumentMutationResult:
    _require_replayable_mutation(mutation)
    assert mutation.document_id is not None
    document = await uow.documents.get(mutation.document_id)
    if document is None:
        raise ResourceNotFoundError("idempotent document is unavailable")
    return DocumentMutationResult(
        document=document,
        document_version_id=mutation.document_version_id,
        source_change_id=mutation.source_change_id,
        indexed_document_version_id=mutation.indexed_document_version_id,
        index_revision_id=mutation.index_revision_id,
        job_id=mutation.job_id,
        job_status="queued" if mutation.job_id is not None else None,
    )


def _require_same_hash(actual: str, expected: str) -> None:
    if actual != expected:
        raise IdempotencyKeyReusedError("idempotency key was already used with a different request")


def _require_replayable_mutation(mutation) -> None:
    if mutation.status == "failed":
        raise ResourceStateConflictError(
            mutation.failure_code or "content mutation reached a terminal failure"
        )


def _require_scope(uow: SqlAlchemyUnitOfWork, context: AuthContext) -> None:
    if uow.workspace_id != context.workspace_id:
        raise RuntimeError("Unit of Work workspace does not match authorized identity")
