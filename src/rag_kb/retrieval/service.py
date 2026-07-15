"""Authorized retrieval planning and evidence normalization."""

from __future__ import annotations

import math

from rag_kb.adapters.model_api import EmbeddingProvider
from rag_kb.adapters.vector_store import VectorStore
from rag_kb.auth import AccessPolicy, AuthContext
from rag_kb.domain import (
    ErrorCode,
    Evidence,
    EvidencePack,
    IndexingExecutionError,
    ResourceNotFoundError,
    RetrievalDebug,
    RetrievalExecutionError,
    RetrievalQueryPlan,
    RetrievalRequest,
    RetrievalStrategy,
    VectorSearchResult,
)


class RetrievalService:
    """Create mandatory plans and return only normalized, locatable evidence."""

    def __init__(
        self,
        access_policy: AccessPolicy,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._access_policy = access_policy
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def retrieve(
        self,
        context: AuthContext,
        request: RetrievalRequest,
    ) -> EvidencePack:
        metadata_filter = self._access_policy.metadata_filter(context)
        self._require_enabled(request)
        if request.include_debug:
            self._access_policy.authorize_retrieval_debug(context)

        plan = RetrievalQueryPlan(
            workspace_id=metadata_filter.workspace_id,
            knowledge_base_id=request.knowledge_base_id,
            strategy=request.strategy,
            top_k=request.top_k,
            rerank=request.rerank,
        )
        query_embedding = await self._embed_query(request.query)
        result = await self._vector_store.search(plan, query_embedding)
        if result is None:
            raise ResourceNotFoundError("knowledge base or active revision was not found")
        evidence = self._normalize(plan, result)
        debug = (
            RetrievalDebug(
                query_plan=plan,
                resolved_active_revision_id=result.resolved_active_revision_id,
                result_count=len(evidence),
            )
            if request.include_debug
            else None
        )
        return EvidencePack(
            knowledge_base_id=plan.knowledge_base_id,
            index_revision_id=result.resolved_active_revision_id,
            strategy=plan.strategy,
            evidence=evidence,
            debug=debug,
        )

    @staticmethod
    def _require_enabled(request: RetrievalRequest) -> None:
        if request.strategy is not RetrievalStrategy.EXACT_VECTOR:
            raise RetrievalExecutionError(
                ErrorCode.CAPABILITY_NOT_ENABLED,
                diagnostic={"capability": request.strategy.value},
            )
        if request.rerank:
            raise RetrievalExecutionError(
                ErrorCode.CAPABILITY_NOT_ENABLED,
                diagnostic={"capability": "rerank"},
            )

    async def _embed_query(self, query: str) -> tuple[float, ...]:
        try:
            batch = await self._embedding_provider.embed((query,))
        except IndexingExecutionError as error:
            raise RetrievalExecutionError(
                error.code,
                diagnostic=error.diagnostic,
            ) from error
        except Exception as error:
            raise RetrievalExecutionError(
                ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                diagnostic={"check": "provider_contract"},
            ) from error
        if len(batch.vectors) != 1:
            raise RetrievalExecutionError(
                ErrorCode.EMBEDDING_RESPONSE_INVALID,
                diagnostic={"check": "query_cardinality"},
            )
        vector = batch.vectors[0]
        definition = self._embedding_provider.embedding_space
        if (
            batch.model != definition.resolved_model
            or definition.distance_metric != "cosine"
            or definition.vector_data_type != "float32"
            or definition.normalization != "l2"
        ):
            raise RetrievalExecutionError(
                ErrorCode.EMBEDDING_SPACE_MISMATCH,
                diagnostic={"check": "query_embedding_space"},
            )
        if len(vector) != definition.dimension or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or abs(float(value)) > 3.4028235e38
            for value in vector
        ):
            raise RetrievalExecutionError(
                ErrorCode.EMBEDDING_RESPONSE_INVALID,
                diagnostic={"check": "query_vector_compatibility"},
            )
        normalized = tuple(float(value) for value in vector)
        norm = math.sqrt(sum(value * value for value in normalized))
        if abs(norm - 1.0) > 0.001:
            raise RetrievalExecutionError(
                ErrorCode.EMBEDDING_RESPONSE_INVALID,
                diagnostic={"check": "query_vector_normalization"},
            )
        return normalized

    @staticmethod
    def _normalize(
        plan: RetrievalQueryPlan,
        result: VectorSearchResult,
    ) -> tuple[Evidence, ...]:
        if len(result.hits) > plan.top_k:
            raise RetrievalExecutionError(
                ErrorCode.INTERNAL_SERVER_ERROR,
                diagnostic={"check": "result_limit"},
            )
        chunk_ids = [hit.index_chunk_id for hit in result.hits]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RetrievalExecutionError(
                ErrorCode.INTERNAL_SERVER_ERROR,
                diagnostic={"check": "duplicate_chunk"},
            )
        for hit in result.hits:
            if (
                hit.workspace_id != plan.workspace_id
                or hit.knowledge_base_id != plan.knowledge_base_id
                or hit.index_revision_id != result.resolved_active_revision_id
                or hit.build_status != plan.build_status
                or hit.serving_status != plan.serving_status
                or not hit.is_current_serving_version
            ):
                raise RetrievalExecutionError(
                    ErrorCode.INTERNAL_SERVER_ERROR,
                    diagnostic={"check": "mandatory_scope"},
                )
        ordered = sorted(
            result.hits,
            key=lambda hit: (hit.cosine_distance, hit.index_chunk_id.int),
        )
        return tuple(
            Evidence(
                rank=rank,
                index_chunk_id=hit.index_chunk_id,
                indexed_document_version_id=hit.indexed_document_version_id,
                document_id=hit.document_id,
                document_version_id=hit.document_version_id,
                index_revision_id=hit.index_revision_id,
                ordinal=hit.ordinal,
                text=hit.text,
                source_location=hit.source_location,
                hierarchy=hit.hierarchy,
                source_metadata=hit.source_metadata,
                score=1.0 - hit.cosine_distance,
            )
            for rank, hit in enumerate(ordered, start=1)
        )
