"""Authorized retrieval planning and evidence normalization."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
import math
from typing import Any, Literal, Protocol
from uuid import UUID

from rag_kb.auth import AccessPolicy, AuthContext
from rag_kb.domain import (
    AdjacentChunkAnchor,
    AdjacentChunkQuery,
    AdjacentChunkResult,
    ErrorCode,
    EmbeddingSpaceDefinition,
    ChunkAssetRelationType,
    Evidence,
    EvidenceAsset,
    evidence_group_identity,
    EvidencePack,
    EvidenceScoreKind,
    IndexChunkAssetRelationSnapshot,
    IndexingExecutionError,
    LexicalSearchResult,
    ResourceNotFoundError,
    RetrievalDebug,
    RetrievalExecutionError,
    RetrievalQueryPlan,
    RetrievalRequest,
    RetrievalStrategy,
    RelatedVisualEvidence,
    RerankDocument,
    RerankMode,
    VectorSearchHit,
    VectorSearchResult,
    validate_embedding_vector,
)
from rag_kb.document_processing.lexical import (
    LEXICAL_ANALYZER_VERSION,
    LEXICAL_QUERY_VERSION,
)
from rag_kb.ports.model_api import (
    EmbeddingModelAdapter,
    MultimodalEmbeddingAdapter,
    RerankerAdapterError,
    TextRerankerAdapter,
)
from rag_kb.ports.retrieval import LexicalStore, VectorStore
from rag_kb.retrieval.reranker import (
    RerankedHit,
    order_model_scored_evidence,
    rerank_hits,
    score_hits,
)
from rag_kb.retrieval.fusion import (
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_lanes,
)
from rag_kb.retrieval.eligibility import EvidenceEligibilityPolicy
from rag_kb.retrieval.profile import (
    EXACT_PROFILE_VERSION,
    HYBRID_PROFILE_VERSION,
    RetrievalExecutionProfile,
)


class CompositeEvidenceHydrator(Protocol):
    async def hydrate(
        self,
        context: AuthContext,
        *,
        kb_id: UUID,
        index_revision_id: UUID,
        chunk_ids: tuple[UUID, ...],
        asset_ids: tuple[UUID, ...],
    ) -> tuple[IndexChunkAssetRelationSnapshot, ...]: ...


RetrievalMode = Literal["vector", "hybrid"]
RetrievalCapabilityStrategy = Literal["exact_vector", "hybrid"]
RetrievalCapabilityProfile = Literal["exact_vector_v2", "hybrid_fts_rrf_v2"]


@dataclass(frozen=True, slots=True)
class RetrievalCapability:
    """A process-wide request-mode capability, independent of KB readiness."""

    mode: RetrievalMode
    strategy: RetrievalCapabilityStrategy
    profile_version: RetrievalCapabilityProfile
    enabled: bool


@dataclass(frozen=True, slots=True)
class RetrievalCapabilitiesSnapshot:
    """Stable capability snapshot exposed by the retrieval composition root."""

    default_mode: Literal["vector"]
    modes: tuple[RetrievalCapability, ...]


class RetrievalService:
    """Create mandatory plans and return only normalized, locatable evidence."""

    def __init__(
        self,
        access_policy: AccessPolicy,
        embedding_provider: EmbeddingModelAdapter,
        vector_store: VectorStore,
        *,
        candidate_multiplier: int = 4,
        max_candidate_count: int = 40,
        vector_weight: float = 0.65,
        lexical_weight: float = 0.35,
        mmr_lambda: float = 0.75,
        multimodal_embedding_provider: MultimodalEmbeddingAdapter | None = None,
        cross_modal_candidate_count: int = 20,
        cross_modal_min_cosine_similarity: float = 0.25,
        text_min_cosine_similarity: float = 0.35,
        rrf_k: int = 60,
        cross_modal_weight_micros: int = 1_000_000,
        lexical_store: LexicalStore | None = None,
        hybrid_enabled: bool = False,
        lexical_analyzer_version: str = LEXICAL_ANALYZER_VERSION,
        lexical_query_version: str = LEXICAL_QUERY_VERSION,
        lexical_candidate_count: int = 40,
        dense_weight_micros: int = 1_000_000,
        lexical_weight_micros: int = 1_000_000,
        min_rerank_score: float = 0.45,
        relation_hydrator: CompositeEvidenceHydrator | None = None,
        deadline_seconds: float = 240.0,
        embedding_model_resolver: (
            Callable[[EmbeddingSpaceDefinition], Awaitable[EmbeddingModelAdapter]]
            | None
        ) = None,
        multimodal_embedding_model_resolver: (
            Callable[[EmbeddingSpaceDefinition], Awaitable[MultimodalEmbeddingAdapter]]
            | None
        ) = None,
        text_reranker: TextRerankerAdapter | None = None,
    ) -> None:
        if candidate_multiplier < 2:
            raise ValueError("candidate_multiplier must be at least two")
        if max_candidate_count < 10:
            raise ValueError("max_candidate_count must be at least ten")
        if not math.isclose(vector_weight + lexical_weight, 1.0, abs_tol=1e-9):
            raise ValueError("rerank weights must sum to one")
        if not 0.0 < mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between zero and one")
        if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
            raise ValueError("retrieval deadline must be positive")
        self._access_policy = access_policy
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._candidate_multiplier = candidate_multiplier
        self._max_candidate_count = max_candidate_count
        self._vector_weight = vector_weight
        self._lexical_weight = lexical_weight
        self._mmr_lambda = mmr_lambda
        self._multimodal_embedding_provider = multimodal_embedding_provider
        self._cross_modal_candidate_count = cross_modal_candidate_count
        self._cross_modal_min_cosine_similarity = cross_modal_min_cosine_similarity
        self._text_min_cosine_similarity = text_min_cosine_similarity
        self._rrf_k = rrf_k
        self._cross_modal_weight_micros = cross_modal_weight_micros
        self._lexical_store = lexical_store
        self._hybrid_enabled = hybrid_enabled
        self._lexical_analyzer_version = lexical_analyzer_version
        self._lexical_query_version = lexical_query_version
        self._lexical_candidate_count = lexical_candidate_count
        self._dense_weight_micros = dense_weight_micros
        self._lexical_weight_micros = lexical_weight_micros
        self._eligibility = EvidenceEligibilityPolicy(
            text_min_cosine_similarity,
            min_rerank_score,
            cross_modal_min_cosine_similarity,
        )
        self._relation_hydrator = relation_hydrator
        self._deadline_seconds = deadline_seconds
        self._embedding_model_resolver = embedding_model_resolver
        self._multimodal_embedding_model_resolver = (
            multimodal_embedding_model_resolver
        )
        self._text_reranker = text_reranker
        common_profile = {
            "top_k": 10,
            "rerank_mode": RerankMode.CLASSIC,
            "dense_candidate_count": max_candidate_count,
            "lexical_candidate_count": lexical_candidate_count,
            "cross_modal_candidate_count": cross_modal_candidate_count,
            "rrf_k": rrf_k,
            "dense_weight_micros": dense_weight_micros,
            "lexical_weight_micros": lexical_weight_micros,
            "cross_modal_weight_micros": cross_modal_weight_micros,
            "min_cosine_similarity": text_min_cosine_similarity,
            "min_rerank_score": min_rerank_score,
            "cross_modal_min_cosine_similarity": (
                cross_modal_min_cosine_similarity
            ),
            "rerank_vector_weight": vector_weight,
            "rerank_lexical_weight": lexical_weight,
            "mmr_lambda": mmr_lambda,
        }
        self._exact_profile = RetrievalExecutionProfile(
            profile_version=EXACT_PROFILE_VERSION,
            strategy=RetrievalStrategy.EXACT_VECTOR,
            lexical_analyzer_version=None,
            lexical_query_version=None,
            **common_profile,
        )
        self._hybrid_profile = RetrievalExecutionProfile(
            profile_version=HYBRID_PROFILE_VERSION,
            strategy=RetrievalStrategy.HYBRID,
            lexical_analyzer_version=lexical_analyzer_version,
            lexical_query_version=lexical_query_version,
            **common_profile,
        )

    def hybrid_request_enabled(self) -> bool:
        """Return whether this process can accept a hybrid request mode.

        This is deliberately a pure capability predicate.  It does not inspect
        a knowledge base, serving target, lexical manifest, unit of work, or
        any external provider.
        """

        return self._hybrid_enabled and self._lexical_store is not None

    def capabilities_snapshot(self) -> RetrievalCapabilitiesSnapshot:
        """Return the deterministic process-wide request capability snapshot."""

        return RetrievalCapabilitiesSnapshot(
            default_mode="vector",
            modes=(
                RetrievalCapability(
                    mode="vector",
                    strategy="exact_vector",
                    profile_version="exact_vector_v2",
                    enabled=True,
                ),
                RetrievalCapability(
                    mode="hybrid",
                    strategy="hybrid",
                    profile_version="hybrid_fts_rrf_v2",
                    enabled=self.hybrid_request_enabled(),
                ),
            ),
        )

    def execution_profile(
        self,
        *,
        strategy: RetrievalStrategy,
        top_k: int,
        rerank_mode: RerankMode,
    ) -> RetrievalExecutionProfile:
        base = (
            self._hybrid_profile
            if strategy is RetrievalStrategy.HYBRID
            else self._exact_profile
        )
        return replace(
            base,
            top_k=top_k,
            rerank_mode=rerank_mode,
            dense_candidate_count=max(
                top_k,
                min(
                    top_k * self._candidate_multiplier,
                    self._max_candidate_count,
                ),
            ),
            lexical_candidate_count=max(
                top_k, base.lexical_candidate_count
            ),
            cross_modal_candidate_count=max(
                top_k, base.cross_modal_candidate_count
            ),
        )

    async def retrieve(
        self,
        context: AuthContext,
        request: RetrievalRequest,
    ) -> EvidencePack:
        deadline = asyncio.timeout(self._deadline_seconds)
        try:
            async with deadline:
                return await self._retrieve(context, request)
        except TimeoutError as error:
            if not deadline.expired():
                raise
            raise RetrievalExecutionError(
                ErrorCode.RETRIEVAL_DEADLINE_EXCEEDED,
                diagnostic={"check": "absolute_deadline"},
            ) from error

    async def retrieve_adjacent_evidence(
        self,
        context: AuthContext,
        *,
        knowledge_base_id: UUID,
        index_revision_id: UUID,
        anchors: tuple[Evidence, ...],
    ) -> tuple[Evidence, ...]:
        deadline = asyncio.timeout(self._deadline_seconds)
        try:
            async with deadline:
                return await self._retrieve_adjacent_evidence(
                    context,
                    knowledge_base_id=knowledge_base_id,
                    index_revision_id=index_revision_id,
                    anchors=anchors,
                )
        except TimeoutError as error:
            if not deadline.expired():
                raise
            raise RetrievalExecutionError(
                ErrorCode.RETRIEVAL_DEADLINE_EXCEEDED,
                diagnostic={"check": "adjacency_absolute_deadline"},
            ) from error

    async def _retrieve_adjacent_evidence(
        self,
        context: AuthContext,
        *,
        knowledge_base_id: UUID,
        index_revision_id: UUID,
        anchors: tuple[Evidence, ...],
    ) -> tuple[Evidence, ...]:
        metadata_filter = self._access_policy.metadata_filter(context)
        if (
            not 1 <= len(anchors) <= 2
            or len({item.index_chunk_id for item in anchors}) != len(anchors)
            or any(
                item.index_revision_id != index_revision_id
                or item.modality not in {"text", "table"}
                or item.score_kind is EvidenceScoreKind.ADJACENCY
                for item in anchors
            )
        ):
            raise RetrievalExecutionError(
                ErrorCode.INTERNAL_SERVER_ERROR,
                diagnostic={"check": "adjacency_anchor_scope"},
            )
        query = AdjacentChunkQuery(
            workspace_id=metadata_filter.workspace_id,
            knowledge_base_id=knowledge_base_id,
            index_revision_id=index_revision_id,
            anchors=tuple(
                AdjacentChunkAnchor(
                    index_chunk_id=item.index_chunk_id,
                    indexed_document_version_id=(
                        item.indexed_document_version_id
                    ),
                    ordinal=item.ordinal,
                )
                for item in anchors
            ),
        )
        result = await self._vector_store.adjacent_chunks(query)
        if result is None:
            raise ResourceNotFoundError(
                "knowledge base or active revision was not found"
            )
        if result.resolved_active_revision_id != index_revision_id:
            raise RetrievalExecutionError(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                diagnostic={"check": "adjacency_frozen_revision"},
            )
        if result.validated_anchor_count != len(anchors):
            raise RetrievalExecutionError(
                ErrorCode.INTERNAL_SERVER_ERROR,
                diagnostic={"check": "adjacency_anchor_scope"},
            )
        self._validate_adjacent_scope(query, result)
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
                score=0.0,
                score_kind=EvidenceScoreKind.ADJACENCY,
                modality=hit.modality,
                asset=self._asset(hit),
                evidence_group_key=hit.evidence_group_key,
                matched_representations=("adjacency",),
                document_display_name=hit.document_display_name,
                document_original_filename=hit.document_original_filename,
                adjacency_anchor_index_chunk_id=(
                    hit.anchor_index_chunk_id
                ),
                adjacency_offset=hit.offset,
            )
            for rank, hit in enumerate(result.hits, start=1)
        )

    async def _retrieve(
        self,
        context: AuthContext,
        request: RetrievalRequest,
    ) -> EvidencePack:
        metadata_filter = self._access_policy.metadata_filter(context)
        self._require_enabled(request)
        if request.include_debug:
            self._access_policy.authorize_retrieval_debug(context)
        profile = self.execution_profile(
            strategy=request.strategy,
            top_k=request.top_k,
            rerank_mode=request.rerank_mode,
        )
        if request.strategy is RetrievalStrategy.HYBRID:
            return await self._retrieve_hybrid(context, request, profile)

        plan = RetrievalQueryPlan(
            workspace_id=metadata_filter.workspace_id,
            knowledge_base_id=request.knowledge_base_id,
            strategy=request.strategy,
            top_k=request.top_k,
            candidate_count=(
                profile.dense_candidate_count
                if request.rerank
                else None
            ),
            rerank_mode=request.rerank_mode,
        )
        embedding_provider, cross_provider = await self._embedding_providers(plan)
        multimodal = cross_provider is not None
        relations: tuple[IndexChunkAssetRelationSnapshot, ...] = ()
        cross_result: VectorSearchResult | None = None
        if multimodal:
            assert cross_provider is not None
            if _providers_share_space(embedding_provider, cross_provider):
                query_embedding = await self._embed_query(
                    request.query, embedding_provider
                )
                cross_embedding = query_embedding
            else:
                query_embedding, cross_embedding = await _gather_cancel_on_error(
                    self._embed_query(request.query, embedding_provider),
                    self._embed_multimodal_query(request.query, cross_provider),
                )
            cross_plan = RetrievalQueryPlan(
                workspace_id=plan.workspace_id,
                knowledge_base_id=plan.knowledge_base_id,
                strategy=plan.strategy,
                top_k=plan.top_k,
                candidate_count=profile.cross_modal_candidate_count,
                rerank_mode=(
                    plan.rerank_mode
                    if plan.rerank
                    else RerankMode.CLASSIC
                ),
            )
            result, cross_result = await _gather_cancel_on_error(
                self._search_text(plan, query_embedding, embedding_provider),
                self._vector_store.search_space(
                    cross_plan,
                    cross_embedding,
                    space_role="cross_modal_retrieval",
                    representation_kinds=("native_image", "table_image"),
                    expected_space=cross_provider.embedding_space,
                ),
            )
            if result is None:
                raise ResourceNotFoundError(
                    "knowledge base or active revision was not found"
                )
            if cross_result is None:
                raise RetrievalExecutionError(
                    ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                    diagnostic={"check": "cross_modal_snapshot_missing"},
                )
            self._validate_multimodal_results(plan, result, cross_plan, cross_result)
            if self._relation_hydrator is not None:
                all_hits = result.hits + cross_result.hits
                relations = await self._relation_hydrator.hydrate(
                    context,
                    kb_id=plan.knowledge_base_id,
                    index_revision_id=result.resolved_active_revision_id,
                    chunk_ids=tuple(dict.fromkeys(hit.index_chunk_id for hit in all_hits)),
                    asset_ids=tuple(
                        dict.fromkeys(
                            hit.index_asset_id
                            for hit in all_hits
                            if hit.index_asset_id is not None
                        )
                    ),
                )
                self._validate_relations(plan, result, relations)
            candidates = self._multimodal_evidence(
                plan,
                result,
                cross_result,
                request.query,
                relations,
                profile,
            )
            evidence, model_candidate_count, model_window_count = (
                await self._finish_reranking(
                    request.query,
                    candidates,
                    plan,
                )
            )
        else:
            query_embedding = await self._embed_query(
                request.query, embedding_provider
            )
            result = await self._search_text(
                plan, query_embedding, embedding_provider
            )
            if result is None:
                raise ResourceNotFoundError(
                    "knowledge base or active revision was not found"
                )
            self._validate_scope(plan, result)
            evidence, model_candidate_count, model_window_count = (
                await self._normalize(
                    plan,
                    result,
                    query=request.query,
                    profile=profile,
                )
            )
        debug = (
            RetrievalDebug(
                query_plan=plan,
                resolved_active_revision_id=result.resolved_active_revision_id,
                result_count=len(evidence),
                text_candidate_count=len(result.hits) if multimodal else None,
                cross_modal_candidate_count=(
                    len(cross_result.hits)
                    if multimodal and cross_result is not None
                    else None
                ),
                hydrated_relation_count=len(relations) if multimodal else None,
                evidence_group_count=(
                    len(
                        {
                            evidence_group_identity(
                                item.indexed_document_version_id,
                                item.evidence_group_key
                                or str(item.index_chunk_id),
                            )
                            for item in evidence
                        }
                    )
                    if multimodal
                    else None
                ),
                model_rerank_candidate_count=model_candidate_count,
                model_rerank_window_count=model_window_count,
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

    async def _retrieve_hybrid(
        self,
        context: AuthContext,
        request: RetrievalRequest,
        profile: RetrievalExecutionProfile,
    ) -> EvidencePack:
        lexical_store = self._lexical_store
        assert lexical_store is not None
        metadata_filter = self._access_policy.metadata_filter(context)
        dense_count = profile.dense_candidate_count
        plan = RetrievalQueryPlan(
            workspace_id=metadata_filter.workspace_id,
            knowledge_base_id=request.knowledge_base_id,
            strategy=RetrievalStrategy.HYBRID,
            top_k=request.top_k,
            candidate_count=dense_count,
            rerank_mode=request.rerank_mode,
        )
        embedding_provider, cross_provider = await self._embedding_providers(plan)
        multimodal = cross_provider is not None
        cross_result: VectorSearchResult | None = None
        if multimodal:
            assert cross_provider is not None
            if _providers_share_space(embedding_provider, cross_provider):
                query_embedding = await self._embed_query(
                    request.query, embedding_provider
                )
                cross_embedding = query_embedding
            else:
                query_embedding, cross_embedding = await _gather_cancel_on_error(
                    self._embed_query(request.query, embedding_provider),
                    self._embed_multimodal_query(request.query, cross_provider),
                )
            cross_plan = replace(
                plan,
                candidate_count=profile.cross_modal_candidate_count,
            )
            dense_result, lexical_result, cross_result = (
                await _gather_cancel_on_error(
                    self._search_text(plan, query_embedding, embedding_provider),
                    lexical_store.search(
                        plan,
                        request.query,
                        query_embedding,
                        analyzer_version=(
                            profile.lexical_analyzer_version or ""
                        ),
                        query_version=profile.lexical_query_version or "",
                        candidate_count=profile.lexical_candidate_count,
                    ),
                    self._vector_store.search_space(
                        cross_plan,
                        cross_embedding,
                        space_role="cross_modal_retrieval",
                        representation_kinds=(
                            "native_image",
                            "table_image",
                        ),
                        expected_space=(
                            cross_provider.embedding_space
                        ),
                    ),
                )
            )
        else:
            query_embedding = await self._embed_query(
                request.query, embedding_provider
            )
            dense_result, lexical_result = await _gather_cancel_on_error(
                self._search_text(plan, query_embedding, embedding_provider),
                lexical_store.search(
                    plan,
                    request.query,
                    query_embedding,
                    analyzer_version=profile.lexical_analyzer_version or "",
                    query_version=profile.lexical_query_version or "",
                    candidate_count=profile.lexical_candidate_count,
                ),
            )
        if dense_result is None or lexical_result is None:
            raise ResourceNotFoundError(
                "knowledge base or active revision was not found"
            )
        self._validate_scope(plan, dense_result)
        self._validate_lexical_result(plan, dense_result, lexical_result)
        if cross_result is not None:
            self._validate_multimodal_results(
                plan, dense_result, plan, cross_result
            )

        all_hits = dense_result.hits + lexical_result.hits
        if cross_result is not None:
            all_hits += cross_result.hits
        relations: tuple[IndexChunkAssetRelationSnapshot, ...] = ()
        if self._relation_hydrator is not None and all_hits:
            relations = await self._relation_hydrator.hydrate(
                context,
                kb_id=plan.knowledge_base_id,
                index_revision_id=dense_result.resolved_active_revision_id,
                chunk_ids=tuple(
                    dict.fromkeys(hit.index_chunk_id for hit in all_hits)
                ),
                asset_ids=tuple(
                    dict.fromkeys(
                        hit.index_asset_id
                        for hit in all_hits
                        if hit.index_asset_id is not None
                    )
                ),
            )
            self._validate_relations(plan, dense_result, relations)
        candidates = self._hybrid_evidence(
            plan,
            dense_result,
            lexical_result,
            cross_result,
            request.query,
            relations,
            profile,
        )
        evidence, model_candidate_count, model_window_count = (
            await self._finish_reranking(
                request.query,
                candidates,
                plan,
            )
        )
        debug = (
            RetrievalDebug(
                query_plan=plan,
                resolved_active_revision_id=dense_result.resolved_active_revision_id,
                result_count=len(evidence),
                text_candidate_count=len(dense_result.hits),
                lexical_candidate_count=len(lexical_result.hits),
                cross_modal_candidate_count=(
                    len(cross_result.hits) if cross_result is not None else None
                ),
                lexical_analyzer_version=lexical_result.analyzer_version,
                lexical_manifest_target_count=(
                    lexical_result.manifest_target_count
                ),
                hydrated_relation_count=len(relations),
                evidence_group_count=len(
                    {
                        evidence_group_identity(
                            item.indexed_document_version_id,
                            item.evidence_group_key
                            or str(item.index_chunk_id),
                        )
                        for item in evidence
                    }
                ),
                model_rerank_candidate_count=model_candidate_count,
                model_rerank_window_count=model_window_count,
            )
            if request.include_debug
            else None
        )
        return EvidencePack(
            knowledge_base_id=plan.knowledge_base_id,
            index_revision_id=dense_result.resolved_active_revision_id,
            strategy=RetrievalStrategy.HYBRID,
            evidence=evidence,
            debug=debug,
        )

    def _multimodal_evidence(
        self,
        plan: RetrievalQueryPlan,
        text_result: VectorSearchResult,
        cross_result: VectorSearchResult,
        query: str,
        relations: tuple[IndexChunkAssetRelationSnapshot, ...],
        profile: RetrievalExecutionProfile,
    ) -> tuple[Evidence, ...]:
        output_limit = self._candidate_evidence_limit(plan)
        text_hits = tuple(
            hit
            for hit in text_result.hits
            if 1.0 - hit.cosine_distance >= profile.min_cosine_similarity
        )
        cross_hits = tuple(
            hit
            for hit in cross_result.hits
            if 1.0 - hit.cosine_distance
            >= profile.cross_modal_min_cosine_similarity
        )
        reranked = rerank_hits(
            query,
            text_hits,
            top_k=min(len(text_hits), output_limit) or 1,
            vector_weight=profile.rerank_vector_weight,
            lexical_weight=profile.rerank_lexical_weight,
            mmr_lambda=profile.mmr_lambda,
        )
        text_order = tuple(item.hit for item in reranked)
        metrics = {item.hit.index_chunk_id: item for item in reranked}
        strong_relations = tuple(
            relation
            for relation in relations
            if ChunkAssetRelationType(relation.relation_type).is_strong
        )
        group_keys: dict[UUID, list[str]] = {}
        for relation in strong_relations:
            for chunk_id in (relation.chunk_id, relation.visual_unit_id):
                keys = group_keys.setdefault(chunk_id, [])
                if relation.evidence_group_key not in keys:
                    keys.append(relation.evidence_group_key)
        group_keys_by_chunk = {
            chunk_id: tuple(keys) for chunk_id, keys in group_keys.items()
        }
        fused = reciprocal_rank_fusion(
            text_order,
            tuple(sorted(cross_hits, key=lambda hit: (hit.cosine_distance, hit.index_chunk_id.int))),
            rrf_k=profile.rrf_k,
            cross_modal_weight_micros=profile.cross_modal_weight_micros,
            top_k=max(plan.top_k, len(text_order) + len(cross_hits)),
            group_keys_by_chunk=group_keys_by_chunk,
        )
        cross_ranks = {
            hit.index_chunk_id: rank
            for rank, hit in enumerate(
                sorted(
                    cross_hits,
                    key=lambda hit: (hit.cosine_distance, hit.index_chunk_id.int),
                ),
                start=1,
            )
        }
        by_chunk: dict[UUID, Evidence] = {}
        for item in fused:
            hit = item.hit
            group_relations = tuple(
                relation
                for relation in strong_relations
                if (
                    relation.indexed_document_version_id
                    == hit.indexed_document_version_id
                    and relation.evidence_group_key == item.group_key
                )
            )
            parent = min(
                (
                    relation
                    for relation in group_relations
                    if hit.index_chunk_id == relation.visual_unit_id
                    or hit.index_asset_id == relation.asset_id
                ),
                key=_relation_priority,
                default=None,
            )
            base_chunk_id = parent.chunk_id if parent is not None else hit.index_chunk_id
            group_relations = _preferred_asset_relations(
                relation
                for relation in group_relations
                if relation.chunk_id == base_chunk_id
            )
            rerank = metrics.get(base_chunk_id) or metrics.get(hit.index_chunk_id)
            related_visuals = tuple(
                RelatedVisualEvidence(
                    visual_unit_id=relation.visual_unit_id,
                    asset=self._relation_asset(relation),
                    relation_type=relation.relation_type,
                    relation_confidence_micros=relation.confidence_micros,
                    relation_provenance=relation.provenance,
                    evidence_group_key=relation.evidence_group_key,
                    figure_label=relation.figure_label,
                    parent_chunk_id=relation.chunk_id,
                    modality=relation.visual_modality,
                    source_location=relation.visual_source_location,
                    text_space_rank=item.text_rank,
                    cross_modal_rank=cross_ranks.get(relation.visual_unit_id),
                )
                for relation in group_relations
            )
            value = Evidence(
                rank=1,
                index_chunk_id=base_chunk_id,
                indexed_document_version_id=hit.indexed_document_version_id,
                document_id=parent.document_id if parent is not None else hit.document_id,
                document_version_id=(
                    parent.document_version_id if parent is not None else hit.document_version_id
                ),
                index_revision_id=hit.index_revision_id,
                ordinal=parent.chunk_ordinal if parent is not None else hit.ordinal,
                text=parent.chunk_content if parent is not None else hit.text,
                source_location=(
                    parent.chunk_source_location if parent is not None else hit.source_location
                ),
                hierarchy=parent.chunk_hierarchy if parent is not None else hit.hierarchy,
                source_metadata=(
                    parent.chunk_source_metadata if parent is not None else hit.source_metadata
                ),
                score=item.score,
                score_kind=EvidenceScoreKind.RECIPROCAL_RANK_FUSION,
                vector_similarity=(
                    rerank.vector_similarity
                    if rerank is not None
                    else 1.0 - hit.cosine_distance
                ),
                lexical_score=rerank.lexical_score if rerank is not None else 0.0,
                lexical_coverage=rerank.lexical_coverage if rerank is not None else 0.0,
                modality=parent.chunk_modality if parent is not None else hit.modality,
                asset=self._asset(hit),
                evidence_group_key=item.group_key,
                matched_representations=item.matched_representations,
                text_space_rank=item.text_rank,
                cross_modal_rank=item.cross_modal_rank,
                fusion_score=item.score,
                related_visuals=related_visuals,
                document_display_name=hit.document_display_name,
                document_original_filename=hit.document_original_filename,
            )
            existing = by_chunk.get(base_chunk_id)
            if existing is None:
                by_chunk[base_chunk_id] = value
            else:
                visuals = {visual.asset.id: visual for visual in existing.related_visuals}
                visuals.update({visual.asset.id: visual for visual in related_visuals})
                by_chunk[base_chunk_id] = replace(
                    existing,
                    matched_representations=tuple(
                        sorted(
                            set(existing.matched_representations)
                            | set(value.matched_representations)
                        )
                    ),
                    text_space_rank=_minimum_rank(
                        existing.text_space_rank, value.text_space_rank
                    ),
                    cross_modal_rank=_minimum_rank(
                        existing.cross_modal_rank, value.cross_modal_rank
                    ),
                    related_visuals=tuple(visuals.values()),
                )
            if len(by_chunk) >= output_limit:
                break
        return tuple(
            replace(value, rank=rank)
            for rank, value in enumerate(by_chunk.values(), start=1)
        )

    def _hybrid_evidence(
        self,
        plan: RetrievalQueryPlan,
        dense_result: VectorSearchResult,
        lexical_result: LexicalSearchResult,
        cross_result: VectorSearchResult | None,
        query: str,
        relations: tuple[IndexChunkAssetRelationSnapshot, ...],
        profile: RetrievalExecutionProfile,
    ) -> tuple[Evidence, ...]:
        output_limit = self._candidate_evidence_limit(plan)
        merged: dict[UUID, VectorSearchHit] = {}
        for hit in dense_result.hits:
            merged[hit.index_chunk_id] = hit
        for hit in lexical_result.hits:
            current = merged.get(hit.index_chunk_id)
            if current is None:
                merged[hit.index_chunk_id] = hit
            else:
                merged[hit.index_chunk_id] = replace(
                    current,
                    lexical_rank=hit.lexical_rank,
                    lexical_score=hit.lexical_score,
                )
        scored = score_hits(
            query,
            merged.values(),
            vector_weight=profile.rerank_vector_weight,
            lexical_weight=profile.rerank_lexical_weight,
        )
        metrics = {item.hit.index_chunk_id: item for item in scored}
        eligibility = EvidenceEligibilityPolicy(
            profile.min_cosine_similarity,
            profile.min_rerank_score,
            profile.cross_modal_min_cosine_similarity,
        )
        eligible_ids = {
            item.hit.index_chunk_id
            for item in scored
            if eligibility.usable_text_candidate(item)
        }
        dense_hits = tuple(
            metrics[hit.index_chunk_id].hit
            for hit in dense_result.hits
            if hit.index_chunk_id in eligible_ids
        )
        lexical_hits = tuple(
            metrics[hit.index_chunk_id].hit
            for hit in lexical_result.hits
            if hit.index_chunk_id in eligible_ids
        )
        cross_hits = tuple(
            sorted(
                (
                    hit
                    for hit in (cross_result.hits if cross_result else ())
                    if 1.0 - hit.cosine_distance
                    >= profile.cross_modal_min_cosine_similarity
                ),
                key=lambda hit: (hit.cosine_distance, hit.index_chunk_id.int),
            )
        )
        strong_relations = tuple(
            relation
            for relation in relations
            if ChunkAssetRelationType(relation.relation_type).is_strong
        )
        group_keys: dict[UUID, list[str]] = {}
        for relation in strong_relations:
            for chunk_id in (relation.chunk_id, relation.visual_unit_id):
                keys = group_keys.setdefault(chunk_id, [])
                if relation.evidence_group_key not in keys:
                    keys.append(relation.evidence_group_key)
        lanes = [
            ("dense_text", dense_hits, profile.dense_weight_micros),
            ("lexical", lexical_hits, profile.lexical_weight_micros),
        ]
        if cross_result is not None:
            lanes.append(
                (
                    "cross_modal",
                    cross_hits,
                    profile.cross_modal_weight_micros,
                )
            )
        fused = reciprocal_rank_fusion_lanes(
            tuple(lanes),
            rrf_k=profile.rrf_k,
            top_k=max(
                plan.top_k,
                len(dense_hits) + len(lexical_hits) + len(cross_hits),
            ),
            group_keys_by_chunk={
                chunk_id: tuple(keys) for chunk_id, keys in group_keys.items()
            },
        )
        cross_ranks = {
            hit.index_chunk_id: rank
            for rank, hit in enumerate(cross_hits, start=1)
        }
        by_chunk: dict[UUID, Evidence] = {}
        for item in fused:
            hit = item.hit
            group_relations = tuple(
                relation
                for relation in strong_relations
                if relation.indexed_document_version_id
                == hit.indexed_document_version_id
                and relation.evidence_group_key == item.group_key
            )
            parent = min(
                (
                    relation
                    for relation in group_relations
                    if hit.index_chunk_id == relation.visual_unit_id
                    or hit.index_asset_id == relation.asset_id
                ),
                key=_relation_priority,
                default=None,
            )
            base_chunk_id = (
                parent.chunk_id if parent is not None else hit.index_chunk_id
            )
            group_relations = _preferred_asset_relations(
                relation
                for relation in group_relations
                if relation.chunk_id == base_chunk_id
            )
            rerank = metrics.get(base_chunk_id) or metrics.get(
                hit.index_chunk_id
            )
            related_visuals = tuple(
                RelatedVisualEvidence(
                    visual_unit_id=relation.visual_unit_id,
                    asset=self._relation_asset(relation),
                    relation_type=relation.relation_type,
                    relation_confidence_micros=relation.confidence_micros,
                    relation_provenance=relation.provenance,
                    evidence_group_key=relation.evidence_group_key,
                    figure_label=relation.figure_label,
                    parent_chunk_id=relation.chunk_id,
                    modality=relation.visual_modality,
                    source_location=relation.visual_source_location,
                    text_space_rank=item.text_rank,
                    lexical_rank=item.lexical_rank,
                    cross_modal_rank=cross_ranks.get(
                        relation.visual_unit_id
                    ),
                )
                for relation in group_relations
            )
            value = Evidence(
                rank=1,
                index_chunk_id=base_chunk_id,
                indexed_document_version_id=hit.indexed_document_version_id,
                document_id=(
                    parent.document_id if parent is not None else hit.document_id
                ),
                document_version_id=(
                    parent.document_version_id
                    if parent is not None
                    else hit.document_version_id
                ),
                index_revision_id=hit.index_revision_id,
                ordinal=parent.chunk_ordinal if parent is not None else hit.ordinal,
                text=parent.chunk_content if parent is not None else hit.text,
                source_location=(
                    parent.chunk_source_location
                    if parent is not None
                    else hit.source_location
                ),
                hierarchy=(
                    parent.chunk_hierarchy if parent is not None else hit.hierarchy
                ),
                source_metadata=(
                    parent.chunk_source_metadata
                    if parent is not None
                    else hit.source_metadata
                ),
                score=item.score,
                score_kind=EvidenceScoreKind.RECIPROCAL_RANK_FUSION,
                vector_similarity=(
                    rerank.vector_similarity
                    if rerank is not None
                    else 1.0 - hit.cosine_distance
                ),
                lexical_score=(
                    rerank.lexical_score if rerank is not None else 0.0
                ),
                lexical_coverage=(
                    rerank.lexical_coverage if rerank is not None else 0.0
                ),
                modality=parent.chunk_modality if parent is not None else hit.modality,
                asset=self._asset(hit),
                evidence_group_key=item.group_key,
                matched_representations=item.matched_representations,
                text_space_rank=item.text_rank,
                lexical_rank=item.lexical_rank,
                cross_modal_rank=item.cross_modal_rank,
                fusion_score=item.score,
                related_visuals=related_visuals,
                document_display_name=hit.document_display_name,
                document_original_filename=hit.document_original_filename,
            )
            existing = by_chunk.get(base_chunk_id)
            if existing is None:
                by_chunk[base_chunk_id] = value
            else:
                visuals = {
                    visual.asset.id: visual for visual in existing.related_visuals
                }
                visuals.update(
                    {visual.asset.id: visual for visual in related_visuals}
                )
                by_chunk[base_chunk_id] = replace(
                    existing,
                    matched_representations=tuple(
                        sorted(
                            set(existing.matched_representations)
                            | set(value.matched_representations)
                        )
                    ),
                    text_space_rank=_minimum_rank(
                        existing.text_space_rank, value.text_space_rank
                    ),
                    lexical_rank=_minimum_rank(
                        existing.lexical_rank, value.lexical_rank
                    ),
                    cross_modal_rank=_minimum_rank(
                        existing.cross_modal_rank, value.cross_modal_rank
                    ),
                    related_visuals=tuple(visuals.values()),
                )
            if len(by_chunk) >= output_limit:
                break
        return tuple(
            replace(value, rank=rank)
            for rank, value in enumerate(by_chunk.values(), start=1)
        )

    def _require_enabled(self, request: RetrievalRequest) -> None:
        if request.strategy is RetrievalStrategy.HYBRID:
            if not self.hybrid_request_enabled():
                raise RetrievalExecutionError(
                    ErrorCode.CAPABILITY_NOT_ENABLED,
                    diagnostic={"capability": request.strategy.value},
                )
            if request.rerank_mode is RerankMode.NONE:
                raise RetrievalExecutionError(
                    ErrorCode.CAPABILITY_NOT_ENABLED,
                    diagnostic={"capability": "hybrid_reranking"},
                )
            return
        if request.strategy is not RetrievalStrategy.EXACT_VECTOR:
            raise RetrievalExecutionError(
                ErrorCode.CAPABILITY_NOT_ENABLED,
                diagnostic={"capability": request.strategy.value},
            )

    async def _text_embedding_provider(
        self,
        plan: RetrievalQueryPlan,
    ) -> EmbeddingModelAdapter:
        if self._embedding_model_resolver is None or not hasattr(
            self._vector_store, "resolve_space"
        ):
            return self._embedding_provider
        space = await self._vector_store.resolve_space(plan, "text_retrieval")
        if space is None:
            raise ResourceNotFoundError(
                "knowledge base or active revision was not found"
            )
        if space.model_profile_revision_id is None:
            return self._embedding_provider
        if self._multimodal_embedding_model_resolver is not None:
            cross_space = await self._vector_store.resolve_space(
                plan, "cross_modal_retrieval"
            )
            if (
                cross_space is not None
                and cross_space.compatibility_fingerprint
                == space.compatibility_fingerprint
            ):
                return await self._multimodal_embedding_model_resolver(space)
        return await self._embedding_model_resolver(space)

    async def _embedding_providers(
        self,
        plan: RetrievalQueryPlan,
    ) -> tuple[EmbeddingModelAdapter, MultimodalEmbeddingAdapter | None]:
        if not hasattr(self._vector_store, "resolve_spaces"):
            text_provider = await self._text_embedding_provider(plan)
            multimodal = bool(
                hasattr(self._vector_store, "has_space_role")
                and await self._vector_store.has_space_role(
                    plan, "cross_modal_retrieval"
                )
            )
            if not multimodal:
                return text_provider, None
            cross_provider = await self._cross_modal_embedding_provider(plan)
            if cross_provider is None:
                raise RetrievalExecutionError(
                    ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                    diagnostic={"check": "cross_modal_provider_required"},
                )
            return text_provider, cross_provider

        spaces = await self._vector_store.resolve_spaces(plan)
        text_space = spaces.get("text_retrieval")
        if text_space is None:
            raise ResourceNotFoundError(
                "knowledge base or active revision was not found"
            )
        cross_space = spaces.get("cross_modal_retrieval")
        unified = (
            cross_space is not None
            and cross_space.compatibility_fingerprint
            == text_space.compatibility_fingerprint
        )
        if text_space.model_profile_revision_id is None:
            text_provider = self._embedding_provider
        elif unified and self._multimodal_embedding_model_resolver is not None:
            text_provider = await self._multimodal_embedding_model_resolver(
                text_space
            )
        elif self._embedding_model_resolver is not None:
            text_provider = await self._embedding_model_resolver(text_space)
        else:
            text_provider = self._embedding_provider

        if cross_space is None:
            return text_provider, None
        if unified:
            if not hasattr(text_provider, "embed_images"):
                raise RetrievalExecutionError(
                    ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                    diagnostic={"check": "unified_provider_capabilities"},
                )
            return text_provider, text_provider  # type: ignore[return-value]
        if (
            cross_space.model_profile_revision_id is not None
            and self._multimodal_embedding_model_resolver is not None
        ):
            cross_provider = await self._multimodal_embedding_model_resolver(
                cross_space
            )
        else:
            cross_provider = self._multimodal_embedding_provider
        if cross_provider is None:
            raise RetrievalExecutionError(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                diagnostic={"check": "cross_modal_provider_required"},
            )
        return text_provider, cross_provider

    async def _cross_modal_embedding_provider(
        self,
        plan: RetrievalQueryPlan,
    ) -> MultimodalEmbeddingAdapter | None:
        if self._multimodal_embedding_model_resolver is None or not hasattr(
            self._vector_store, "resolve_space"
        ):
            return self._multimodal_embedding_provider
        space = await self._vector_store.resolve_space(
            plan, "cross_modal_retrieval"
        )
        if space is None or space.model_profile_revision_id is None:
            return self._multimodal_embedding_provider
        return await self._multimodal_embedding_model_resolver(space)

    async def _search_text(
        self,
        plan: RetrievalQueryPlan,
        query_embedding: tuple[float, ...],
        provider: EmbeddingModelAdapter,
    ) -> VectorSearchResult | None:
        if provider is self._embedding_provider:
            return await self._vector_store.search(plan, query_embedding)
        return await self._vector_store.search_space(
            plan,
            query_embedding,
            space_role="text_retrieval",
            representation_kinds=("text", "caption_text", "ocr_text", "table_text"),
            expected_space=provider.embedding_space,
        )

    async def _embed_query(
        self,
        query: str,
        provider: EmbeddingModelAdapter | None = None,
    ) -> tuple[float, ...]:
        resolved_provider = provider or self._embedding_provider
        try:
            vector = await resolved_provider.embed_query(query)
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
        definition = resolved_provider.embedding_space
        if (
            definition.distance_metric != "cosine"
            or definition.vector_data_type != "float32"
            or definition.normalization not in {"l2", "client_l2_v1"}
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

    async def _embed_multimodal_query(
        self,
        query: str,
        provider: MultimodalEmbeddingAdapter | None = None,
    ) -> tuple[float, ...]:
        provider = provider or self._multimodal_embedding_provider
        assert provider is not None
        try:
            if hasattr(provider, "embed_query"):
                vector = await provider.embed_query(query)
            else:
                embedded = await provider.embed_texts((query,))
                if len(embedded.vectors) != 1:
                    raise RetrievalExecutionError(
                        ErrorCode.EMBEDDING_RESPONSE_INVALID,
                        diagnostic={"check": "cross_modal_query_cardinality"},
                    )
                vector = embedded.vectors[0]
            validate_embedding_vector(vector, provider.embedding_space)
            return vector
        except RetrievalExecutionError:
            raise
        except IndexingExecutionError as error:
            raise RetrievalExecutionError(
                error.code, diagnostic=error.diagnostic
            ) from error
        except Exception as error:
            raise RetrievalExecutionError(
                ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                diagnostic={"check": "cross_modal_provider_contract"},
            ) from error

    async def _normalize(
        self,
        plan: RetrievalQueryPlan,
        result: VectorSearchResult,
        *,
        query: str,
        profile: RetrievalExecutionProfile,
    ) -> tuple[tuple[Evidence, ...], int | None, int | None]:
        result_limit = plan.candidate_count or plan.top_k
        if len(result.hits) > result_limit:
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
        self._validate_scope(plan, result)
        if plan.rerank:
            output_limit = self._candidate_evidence_limit(plan)
            ordered_reranked = rerank_hits(
                query,
                result.hits,
                top_k=output_limit,
                vector_weight=profile.rerank_vector_weight,
                lexical_weight=profile.rerank_lexical_weight,
                mmr_lambda=profile.mmr_lambda,
            )
            candidates = tuple(
                RetrievalService._evidence_from_reranked(rank, item)
                for rank, item in enumerate(ordered_reranked, start=1)
            )
            return await self._finish_reranking(query, candidates, plan)
        ordered = sorted(
            result.hits,
            key=lambda hit: (hit.cosine_distance, hit.index_chunk_id.int),
        )[: plan.top_k]
        evidence = tuple(
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
                modality=hit.modality,
                asset=RetrievalService._asset(hit),
                evidence_group_key=hit.evidence_group_key,
                matched_representations=(hit.representation_kind,),
                document_display_name=hit.document_display_name,
                document_original_filename=hit.document_original_filename,
            )
            for rank, hit in enumerate(ordered, start=1)
        )
        return evidence, None, None

    @staticmethod
    def _candidate_evidence_limit(plan: RetrievalQueryPlan) -> int:
        if plan.rerank_mode is RerankMode.LOCAL_MINILM_V1:
            return 20
        return plan.top_k

    async def _finish_reranking(
        self,
        query: str,
        candidates: tuple[Evidence, ...],
        plan: RetrievalQueryPlan,
    ) -> tuple[tuple[Evidence, ...], int | None, int | None]:
        if plan.rerank_mode is not RerankMode.LOCAL_MINILM_V1:
            return candidates[: plan.top_k], None, None
        reranker = self._text_reranker
        if reranker is None or reranker.profile is not plan.rerank_mode:
            raise RetrievalExecutionError(
                ErrorCode.LOCAL_RERANKER_UNAVAILABLE,
                diagnostic={"check": "local_reranker_not_configured"},
            )
        model_positions = tuple(
            index
            for index, item in enumerate(candidates)
            if item.modality in {"text", "table"} and item.text.strip()
        )
        if len(model_positions) > reranker.max_documents:
            raise RetrievalExecutionError(
                ErrorCode.LOCAL_RERANKER_UNAVAILABLE,
                diagnostic={"check": "local_reranker_candidate_limit"},
            )
        if not model_positions:
            return candidates[: plan.top_k], 0, 0
        documents = tuple(
            RerankDocument(
                index_chunk_id=candidates[index].index_chunk_id,
                text=candidates[index].text,
                hierarchy=candidates[index].hierarchy,
                modality=candidates[index].modality,
            )
            for index in model_positions
        )
        try:
            scores = await reranker.score(query, documents)
        except (RerankerAdapterError, OSError, RuntimeError) as error:
            raise RetrievalExecutionError(
                ErrorCode.LOCAL_RERANKER_UNAVAILABLE,
                diagnostic={"check": "local_reranker_inference"},
            ) from error
        score_by_id = {item.index_chunk_id: item for item in scores}
        document_ids = {item.index_chunk_id for item in documents}
        if (
            len(scores) != len(documents)
            or len(score_by_id) != len(scores)
            or set(score_by_id) != document_ids
        ):
            raise RetrievalExecutionError(
                ErrorCode.LOCAL_RERANKER_UNAVAILABLE,
                diagnostic={"check": "local_reranker_response_contract"},
            )
        raw_ranked = sorted(
            (candidates[index] for index in model_positions),
            key=lambda item: (
                -score_by_id[item.index_chunk_id].score,
                item.rank,
                item.index_chunk_id.int,
            ),
        )
        raw_rank_by_id = {
            item.index_chunk_id: rank
            for rank, item in enumerate(raw_ranked, start=1)
        }
        ranked = order_model_scored_evidence(
            (candidates[index] for index in model_positions),
            {
                item.index_chunk_id: item.score
                for item in scores
            },
            mmr_lambda=self._mmr_lambda,
        )
        ranked_values = tuple(
            replace(
                item,
                model_rerank_score=score_by_id[item.index_chunk_id].score,
                model_rerank_rank=raw_rank_by_id[item.index_chunk_id],
                model_rerank_window_count=(
                    score_by_id[item.index_chunk_id].window_count
                ),
                model_rerank_winning_window_index=(
                    score_by_id[item.index_chunk_id].winning_window_index
                ),
            )
            for item in ranked
        )
        reordered = list(candidates)
        for position, item in zip(model_positions, ranked_values, strict=True):
            reordered[position] = item
        final = tuple(
            replace(item, rank=rank)
            for rank, item in enumerate(reordered[: plan.top_k], start=1)
        )
        return (
            final,
            len(documents),
            sum(item.window_count for item in scores),
        )

    @staticmethod
    def _validate_scope(
        plan: RetrievalQueryPlan, result: VectorSearchResult
    ) -> None:
        for hit in result.hits:
            if (
                hit.workspace_id != plan.workspace_id
                or hit.knowledge_base_id != plan.knowledge_base_id
                or hit.index_revision_id != result.resolved_active_revision_id
                or hit.build_status != "ready"
                or hit.serving_status != "serving"
                or not hit.is_current_serving_version
            ):
                raise RetrievalExecutionError(
                    ErrorCode.INTERNAL_SERVER_ERROR,
                    diagnostic={"check": "mandatory_scope"},
                )

    @staticmethod
    def _validate_adjacent_scope(
        query: AdjacentChunkQuery,
        result: AdjacentChunkResult,
    ) -> None:
        anchors = {
            item.index_chunk_id: (rank, item)
            for rank, item in enumerate(query.anchors, start=1)
        }
        ordered_links = [
            (hit.anchor_rank, hit.offset, hit.index_chunk_id.int)
            for hit in result.hits
        ]
        identities = [
            (hit.anchor_index_chunk_id, hit.index_chunk_id)
            for hit in result.hits
        ]
        if (
            ordered_links != sorted(ordered_links)
            or len(identities) != len(set(identities))
        ):
            raise RetrievalExecutionError(
                ErrorCode.INTERNAL_SERVER_ERROR,
                diagnostic={"check": "adjacency_result_order"},
            )
        for hit in result.hits:
            anchor_value = anchors.get(hit.anchor_index_chunk_id)
            if anchor_value is None:
                raise RetrievalExecutionError(
                    ErrorCode.INTERNAL_SERVER_ERROR,
                    diagnostic={"check": "adjacency_anchor_identity"},
                )
            anchor_rank, anchor = anchor_value
            if (
                hit.workspace_id != query.workspace_id
                or hit.knowledge_base_id != query.knowledge_base_id
                or hit.index_revision_id != result.resolved_active_revision_id
                or hit.indexed_document_version_id
                != anchor.indexed_document_version_id
                or hit.ordinal - anchor.ordinal != hit.offset
                or hit.offset not in {-1, 1}
                or hit.anchor_rank != anchor_rank
                or hit.build_status != "ready"
                or hit.serving_status != "serving"
                or not hit.is_current_serving_version
                or hit.modality not in {"text", "table"}
            ):
                raise RetrievalExecutionError(
                    ErrorCode.INTERNAL_SERVER_ERROR,
                    diagnostic={"check": "adjacency_mandatory_scope"},
                )

    @classmethod
    def _validate_multimodal_results(
        cls,
        plan: RetrievalQueryPlan,
        text_result: VectorSearchResult,
        cross_plan: RetrievalQueryPlan,
        cross_result: VectorSearchResult,
    ) -> None:
        cls._validate_scope(plan, text_result)
        cls._validate_scope(cross_plan, cross_result)
        if (
            cross_result.resolved_active_revision_id
            != text_result.resolved_active_revision_id
        ):
            raise RetrievalExecutionError(
                ErrorCode.INTERNAL_SERVER_ERROR,
                diagnostic={"check": "multimodal_revision_snapshot"},
            )

    @staticmethod
    def _validate_lexical_result(
        plan: RetrievalQueryPlan,
        dense_result: VectorSearchResult,
        lexical_result: LexicalSearchResult,
    ) -> None:
        if (
            lexical_result.resolved_active_revision_id
            != dense_result.resolved_active_revision_id
            or lexical_result.analyzer_version != LEXICAL_ANALYZER_VERSION
        ):
            raise RetrievalExecutionError(
                ErrorCode.INDEX_REVISION_INCOMPATIBLE,
                diagnostic={"check": "lexical_revision_snapshot"},
            )
        for hit in lexical_result.hits:
            if (
                hit.workspace_id != plan.workspace_id
                or hit.knowledge_base_id != plan.knowledge_base_id
                or hit.index_revision_id
                != lexical_result.resolved_active_revision_id
                or hit.build_status != "ready"
                or hit.serving_status != "serving"
                or not hit.is_current_serving_version
                or hit.lexical_rank is None
            ):
                raise RetrievalExecutionError(
                    ErrorCode.INTERNAL_SERVER_ERROR,
                    diagnostic={"check": "lexical_mandatory_scope"},
                )

    @staticmethod
    def _validate_relations(
        plan: RetrievalQueryPlan,
        result: VectorSearchResult,
        relations: tuple[IndexChunkAssetRelationSnapshot, ...],
    ) -> None:
        for relation in relations:
            if (
                relation.workspace_id != plan.workspace_id
                or relation.kb_id != plan.knowledge_base_id
                or relation.index_revision_id
                != result.resolved_active_revision_id
            ):
                raise RetrievalExecutionError(
                    ErrorCode.INTERNAL_SERVER_ERROR,
                    diagnostic={"check": "relation_hydration_scope"},
                )

    @staticmethod
    def _evidence_from_reranked(rank: int, item: RerankedHit) -> Evidence:
        hit = item.hit
        return Evidence(
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
            score=item.score,
            score_kind=EvidenceScoreKind.HYBRID_RERANK,
            vector_similarity=item.vector_similarity,
            lexical_score=item.lexical_score,
            lexical_coverage=item.lexical_coverage,
            modality=hit.modality,
            asset=RetrievalService._asset(hit),
            evidence_group_key=hit.evidence_group_key,
            matched_representations=(hit.representation_kind,),
            document_display_name=hit.document_display_name,
            document_original_filename=hit.document_original_filename,
        )

    @staticmethod
    def _asset(hit) -> EvidenceAsset | None:
        if hit.index_asset_id is None:
            return None
        return EvidenceAsset(
            id=hit.index_asset_id,
            media_type=hit.asset_media_type or "application/octet-stream",
            checksum_sha256=hit.asset_checksum_sha256 or "",
            content_url=f"/api/v1/index-assets/{hit.index_asset_id}/content",
            width=hit.asset_width,
            height=hit.asset_height,
        )

    @staticmethod
    def _relation_asset(
        relation: IndexChunkAssetRelationSnapshot,
    ) -> EvidenceAsset:
        return EvidenceAsset(
            id=relation.asset_id,
            media_type=relation.asset_media_type,
            checksum_sha256=relation.asset_checksum_sha256,
            content_url=f"/api/v1/index-assets/{relation.asset_id}/content",
            width=relation.asset_width,
            height=relation.asset_height,
        )


def _providers_share_space(
    text_provider: EmbeddingModelAdapter,
    cross_provider: MultimodalEmbeddingAdapter,
) -> bool:
    text_space = text_provider.embedding_space
    cross_space = cross_provider.embedding_space
    return (
        text_space.model_profile_revision_id is not None
        and text_space.model_profile_revision_id
        == cross_space.model_profile_revision_id
        and text_space.compatibility_fingerprint
        == cross_space.compatibility_fingerprint
    )


def _minimum_rank(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


async def _gather_cancel_on_error(
    *awaitables: Awaitable[Any],
) -> tuple[Any, ...]:
    """Run lanes concurrently and settle every sibling before propagating."""

    tasks = tuple(asyncio.ensure_future(item) for item in awaitables)
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _preferred_asset_relations(
    relations: Iterable[IndexChunkAssetRelationSnapshot],
) -> tuple[IndexChunkAssetRelationSnapshot, ...]:
    preferred: dict[UUID, IndexChunkAssetRelationSnapshot] = {}
    for relation in sorted(relations, key=_relation_priority):
        preferred.setdefault(relation.asset_id, relation)
    return tuple(preferred.values())


def _relation_priority(
    relation: IndexChunkAssetRelationSnapshot,
) -> tuple[int, int, int]:
    priority = {
        ChunkAssetRelationType.EXPLICIT_FIGURE_REFERENCE: 0,
        ChunkAssetRelationType.CAPTION_OF: 1,
        ChunkAssetRelationType.INLINE_FIGURE: 2,
        ChunkAssetRelationType.OCR_OF: 3,
        ChunkAssetRelationType.TABLE_OF: 4,
        ChunkAssetRelationType.SPATIAL_NEIGHBOR: 5,
        ChunkAssetRelationType.SAME_PAGE: 6,
    }
    return (
        priority[ChunkAssetRelationType(relation.relation_type)],
        relation.ordinal,
        relation.id.int,
    )
