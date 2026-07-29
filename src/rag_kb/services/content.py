"""Application services for database-only content lifecycle use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from rag_kb.auth import AccessPolicy, AuthContext
from rag_kb.document_processing import index_profile, profile_for_preset
from rag_kb.domain import (
    AnswerPolicyDefaults,
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
    Page,
    ParsingPreset,
    ResourceNotFoundError,
    ResourceStateConflictError,
    canonical_request_hash,
    validate_p1_answer_policy_defaults,
)
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


CREATE_KB_ENDPOINT = "POST /api/v1/knowledge-bases"
PATCH_KB_ENDPOINT = "PATCH /api/v1/knowledge-bases/{kb_id}"
DELETE_DOCUMENT_ENDPOINT = "DELETE /api/v1/documents/{document_id}"
CREATE_DOCUMENT_ENDPOINT = "POST /api/v1/knowledge-bases/{kb_id}/documents"
CREATE_VERSION_ENDPOINT = "POST /api/v1/documents/{document_id}/versions"


@dataclass(frozen=True, slots=True)
class ContentServices:
    knowledge_bases: "KnowledgeBaseService"
    documents: "DocumentService"


def build_content_services(
    unit_of_work: UnitOfWorkFactory,
    access_policy: AccessPolicy,
    embedding: Any,
    multimodal_embedding: Any | None = None,
) -> ContentServices:
    """Construct content services without exposing domain configuration to the API."""

    definition = embedding_space_definition(embedding)
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


class KnowledgeBaseService:
    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        access_policy: AccessPolicy,
        *,
        embedding_space: EmbeddingSpaceDefinition,
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
        answer_policy_defaults: dict[str, Any] | None = None,
    ) -> KnowledgeBase:
        self._authorize(context)
        resolved_answer_defaults = (
            AnswerPolicyDefaults().as_dict()
            if answer_policy_defaults is None
            else validate_p1_answer_policy_defaults(answer_policy_defaults).as_dict()
        )
        resolved_preset = ChunkingPreset(chunking_preset)
        resolved_parsing = ParsingPreset(parsing_preset)
        resolved_profile = profile_for_preset(resolved_preset, resolved_parsing)
        if (
            resolved_parsing is ParsingPreset.MULTIMODAL_LOCAL_V2
            and self._cross_modal_embedding_space is None
        ):
            raise ResourceStateConflictError(
                "multimodal parsing requires a configured cross-modal embedding space"
            )
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
                "answer_policy_defaults": resolved_answer_defaults,
            }
        )
        async def persist(uow: UnitOfWork) -> KnowledgeBase:
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
            created = await uow.knowledge_bases.create(
                name=name,
                retrieval_defaults=retrieval_defaults,
                answer_policy_defaults=resolved_answer_defaults,
                embedding_space=self._embedding_space,
                cross_modal_embedding_space=(
                    self._cross_modal_embedding_space
                    if resolved_parsing is ParsingPreset.MULTIMODAL_LOCAL_V2
                    else None
                ),
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

        async def load(uow: UnitOfWork) -> KnowledgeBase:
            _require_scope(uow, context)
            result = await uow.knowledge_bases.get(kb_id)
            if result is None:
                raise ResourceNotFoundError("knowledge base was not found")
            return result

        return await execute_in_transaction(self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST)

    async def list(
        self,
        context: AuthContext,
        *,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[KnowledgeBase]:
        self._authorize(context)

        async def load(uow: UnitOfWork) -> Page[KnowledgeBase]:
            _require_scope(uow, context)
            return await uow.knowledge_bases.list(limit=limit, sort=sort, after=after)

        return await execute_in_transaction(self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST)

    async def update(
        self,
        context: AuthContext,
        idempotency_key: UUID,
        kb_id: UUID,
        *,
        name: str | None,
        retrieval_defaults: dict[str, Any] | None,
        answer_policy_defaults: dict[str, Any] | None = None,
    ) -> KnowledgeBase:
        self._authorize(context)
        if answer_policy_defaults is not None:
            answer_policy_defaults = validate_p1_answer_policy_defaults(
                answer_policy_defaults
            ).as_dict()
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
                "answer_policy_defaults": answer_policy_defaults,
            }
        )

        async def persist(uow: UnitOfWork) -> KnowledgeBase:
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
                answer_policy_defaults=answer_policy_defaults,
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

    def _authorize(self, context: AuthContext) -> None:
        self._access_policy.metadata_filter(context)


class DocumentService:
    """Relational lifecycle; file orchestration is intentionally a later layer."""

    def __init__(self, unit_of_work: UnitOfWorkFactory, access_policy: AccessPolicy) -> None:
        self._unit_of_work = unit_of_work
        self._access_policy = access_policy

    async def get(self, context: AuthContext, document_id: UUID) -> Document:
        self._authorize(context)

        async def load(uow: UnitOfWork) -> Document:
            _require_scope(uow, context)
            document = await uow.documents.get(document_id)
            if document is None:
                raise ResourceNotFoundError("document was not found")
            return document

        return await execute_in_transaction(self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST)

    async def get_detail(
        self, context: AuthContext, document_id: UUID
    ) -> DocumentDetail:
        self._authorize(context)

        async def load(uow: UnitOfWork) -> DocumentDetail:
            _require_scope(uow, context)
            detail = await uow.documents.get_detail(document_id)
            if detail is None:
                raise ResourceNotFoundError("document was not found")
            return detail

        return await execute_in_transaction(
            self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST
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

        async def load(uow: UnitOfWork) -> DocumentChunkInspection:
            _require_scope(uow, context)
            inspection = await uow.documents.inspect_chunks(
                document_id, limit=limit, after=after
            )
            if inspection is None:
                raise ResourceNotFoundError("document was not found")
            return inspection

        return await execute_in_transaction(
            self._unit_of_work, load, purpose=UnitOfWorkPurpose.READ_SNAPSHOT
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

        async def load(uow: UnitOfWork) -> Page[Document]:
            _require_scope(uow, context)
            if await uow.knowledge_bases.get(kb_id) is None:
                raise ResourceNotFoundError("knowledge base was not found")
            return await uow.documents.list(kb_id=kb_id, limit=limit, sort=sort, after=after)

        return await execute_in_transaction(self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST)

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

        async def persist(uow: UnitOfWork) -> DocumentMutationResult:
            _require_scope(uow, context)
            await uow.content_mutations.lock(scope)
            prior = await uow.content_mutations.get(scope)
            if prior is not None:
                _require_same_hash(prior.request_hash, request_hash)
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

        async def persist(uow: UnitOfWork) -> DocumentMutationResult:
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

        async def persist(uow: UnitOfWork) -> DocumentMutationResult:
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

    def _authorize(self, context: AuthContext) -> None:
        self._access_policy.metadata_filter(context)


async def _document_result_from_mutation(uow: UnitOfWork, mutation) -> DocumentMutationResult:
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


def _require_scope(uow: UnitOfWork, context: AuthContext) -> None:
    if uow.workspace_id != context.workspace_id:
        raise RuntimeError("Unit of Work workspace does not match authorized identity")
