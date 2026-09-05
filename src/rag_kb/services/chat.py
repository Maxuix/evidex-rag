"""Durable chat session and run-creation application service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from rag_kb.domain.chat_scope import ChatKnowledgeBaseSnapshot, normalize_knowledge_base_ids
from typing import TYPE_CHECKING, Any
from uuid import UUID

from rag_kb.domain import (
    CHAT_AGENT_VERSION,
    ChatAgentBudget,
    ChatMessage,
    ChatRun,
    ChatSession,
    ChatSessionBusyError,
    ModelKind,
    ModelProfileBundle,
    ModelValidationStatus,
    IdempotencyKeyReusedError,
    IdempotencyScope,
    Page,
    ResourceNotFoundError,
    ResourceStateConflictError,
    RerankMode,
    RetrievalStrategy,
    canonical_request_hash,
)
from rag_kb.retrieval.profile import (
    RetrievalExecutionProfile,
    adaptive_graphiti_profile,
    graph_profile,
)
from rag_kb.memory import (
    ConversationContextSelector,
    serialize_conversation_context,
)
from rag_kb.services.chat_visuals import (
    DEFAULT_CHAT_MAX_VISUAL_IMAGE_BYTES,
    DEFAULT_CHAT_MAX_VISUAL_IMAGES,
    DEFAULT_CHAT_MAX_VISUAL_PIXELS,
    DEFAULT_CHAT_MAX_VISUAL_TOTAL_BYTES,
    DEFAULT_CHAT_VISUAL_MEDIA_PROFILE,
)
from rag_kb.uow import execute_in_transaction

if TYPE_CHECKING:
    from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWork, SqlAlchemyUnitOfWorkFactory


CREATE_CHAT_RUN_ENDPOINT = "POST /api/v1/chat/runs"


class ChatService:
    """Create durable chat facts without executing retrieval or model work."""

    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWorkFactory,
        *,
        model_configuration: dict[str, Any],
        default_rerank: bool = False,
        retrieval_profile_factory: Callable[
            [RetrievalStrategy, int, RerankMode], RetrievalExecutionProfile
        ],
        context_strategy: str = "recent_completed_turns_v1",
        context_max_turns: int = 6,
        context_max_tokens: int = 4000,
        context_tokenizer: str = "cl100k_base",
        allow_legacy_model_configuration: bool = True,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._model_configuration = dict(model_configuration)
        self._allow_legacy_model_configuration = allow_legacy_model_configuration
        self._default_rerank_mode = (
            RerankMode.CLASSIC if default_rerank else RerankMode.NONE
        )
        self._retrieval_profile_factory = retrieval_profile_factory
        if (
            context_strategy != "recent_completed_turns_v1"
            or context_max_turns != 6
            or context_max_tokens != 4000
            or context_tokenizer != "cl100k_base"
        ):
            raise ValueError("unsupported Session context policy")
        self._context_selector = ConversationContextSelector(
            max_turns=context_max_turns,
            token_budget=context_max_tokens,
            tokenizer=context_tokenizer,
        )
        self._context_max_turns = context_max_turns

    async def create_session(
        self,
        *,
        kb_id: UUID | None = None,
        kb_ids: tuple[UUID, ...] | None = None,
        title: str | None,
    ) -> ChatSession:
        selected = normalize_knowledge_base_ids(kb_ids, kb_id)
        async def persist(uow: SqlAlchemyUnitOfWork) -> ChatSession:
            for identifier in selected:
                if await uow.knowledge_bases.chat_scope_snapshot(identifier) is None:
                    raise ResourceNotFoundError("knowledge base was not found")
            return await uow.chat.create_session(kb_ids=selected, title=title)

        return await execute_in_transaction(self._unit_of_work, persist)

    async def update_session_scope(self, session_id: UUID, kb_ids: tuple[UUID, ...]) -> ChatSession:
        selected = normalize_knowledge_base_ids(kb_ids)
        async def persist(uow: SqlAlchemyUnitOfWork) -> ChatSession:
            if await uow.chat.lock_session(session_id) is None:
                raise ResourceNotFoundError("chat session was not found")
            for identifier in selected:
                if await uow.knowledge_bases.chat_scope_snapshot(identifier) is None:
                    raise ResourceNotFoundError("knowledge base was not found")
            return await uow.chat.set_session_scope(session_id, selected)
        return await execute_in_transaction(self._unit_of_work, persist)

    async def list_sessions(
        self,
        *,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
        kb_id: UUID | None = None,
    ) -> Page[ChatSession]:
        async def load(uow: SqlAlchemyUnitOfWork) -> Page[ChatSession]:
            if kb_id is not None and await uow.knowledge_bases.get(kb_id) is None:
                raise ResourceNotFoundError("knowledge base was not found")
            return await uow.chat.list_sessions(
                limit=limit,
                sort=sort,
                after=after,
                kb_id=kb_id,
            )

        return await execute_in_transaction(
            self._unit_of_work, load
        )

    async def list_messages(
        self,
        session_id: UUID,
        *,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[ChatMessage]:
        async def load(uow: SqlAlchemyUnitOfWork) -> Page[ChatMessage]:
            result = await uow.chat.list_messages(
                session_id=session_id,
                limit=limit,
                sort=sort,
                after=after,
            )
            if result is None:
                raise ResourceNotFoundError("chat session was not found")
            return result

        return await execute_in_transaction(
            self._unit_of_work, load
        )

    async def get_run(self, run_id: UUID) -> ChatRun:
        async def load(uow: SqlAlchemyUnitOfWork) -> ChatRun:
            result = await uow.chat.get_run(run_id)
            if result is None:
                raise ResourceNotFoundError("chat run was not found")
            return result

        return await execute_in_transaction(
            self._unit_of_work, load
        )

    async def create_run(
        self,
        idempotency_key: UUID,
        *,
        session_id: UUID,
        kb_id: UUID | None = None,
        kb_ids: tuple[UUID, ...] | None = None,
        message: str,
        retrieval_mode: str,
        top_k: int,
        rerank_mode: RerankMode | None = None,
        model_profile_revision_id: UUID | None = None,
    ) -> ChatRun:
        selected = normalize_knowledge_base_ids(kb_ids, kb_id)
        if retrieval_mode not in {"text", "auto", "graph"}:
            raise ResourceStateConflictError("retrieval mode is unsupported")
        if retrieval_mode == "graph" and rerank_mode is None:
            raise ResourceStateConflictError(
                "graph retrieval requires an explicit reranker"
            )
        resolved_rerank_mode = (
            self._default_rerank_mode
            if rerank_mode is None
            else RerankMode(rerank_mode)
        )
        if retrieval_mode == "graph":
            if not 4 <= top_k <= 20:
                raise ResourceStateConflictError(
                    "graph retrieval top_k must be between 4 and 20"
                )
            if resolved_rerank_mode not in {
                RerankMode.CLASSIC,
                RerankMode.LOCAL_MINILM_V1,
            }:
                raise ResourceStateConflictError(
                    "graph retrieval requires an enabled reranker"
                )
        if (
            resolved_rerank_mode is RerankMode.LOCAL_MINILM_V1
            and top_k > 20
        ):
            raise ResourceStateConflictError(
                "local reranking supports top_k up to 20"
            )
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("message must contain non-whitespace characters")

        scope = IdempotencyScope(
            CREATE_CHAT_RUN_ENDPOINT,
            idempotency_key,
        )
        requested_retrieval = {
            "mode": retrieval_mode,
            "top_k": top_k,
            "rerank_mode": resolved_rerank_mode.value,
        }
        retrieval_strategy = (
            graph_profile(top_k=top_k, rerank_mode=resolved_rerank_mode).as_dict()
            if retrieval_mode == "graph"
            else adaptive_graphiti_profile(
                top_k=top_k, rerank_mode=resolved_rerank_mode
            ).as_dict()
            if retrieval_mode == "auto"
            else self._retrieval_profile_factory(
                RetrievalStrategy.EXACT_VECTOR, top_k, resolved_rerank_mode
            ).as_dict()
        )
        request_hash = canonical_request_hash(
            {
                "session_id": str(session_id),
                "knowledge_base_ids": [str(identifier) for identifier in selected],
                "message": normalized_message,
                "retrieval": requested_retrieval,
                "model_profile_revision_id": (
                    str(model_profile_revision_id)
                    if model_profile_revision_id is not None
                    else None
                ),
            }
        )

        legacy_request_hash = canonical_request_hash({
            "session_id": str(session_id), "knowledge_base_id": str(selected[0]),
            "message": normalized_message, "retrieval": requested_retrieval,
            "model_profile_revision_id": str(model_profile_revision_id) if model_profile_revision_id else None,
        }) if len(selected) == 1 else None

        async def persist(uow: SqlAlchemyUnitOfWork) -> ChatRun:
            await uow.chat.lock_idempotency(scope)
            prior = await uow.chat.get_run_by_scope(scope)
            if prior is not None:
                if prior.request_hash not in ({request_hash, legacy_request_hash} - {None}):
                    raise IdempotencyKeyReusedError(
                        "idempotency key was already used with a different request"
                    )
                return prior

            session = await uow.chat.lock_session(session_id)
            if session is None:
                raise ResourceNotFoundError("chat session was not found")
            if await uow.chat.has_nonterminal_run(session_id):
                raise ChatSessionBusyError(
                    "chat session already has a queued or running run"
                )
            snapshots = []
            for identifier in selected:
                snapshot = await uow.knowledge_bases.chat_scope_snapshot(identifier)
                if snapshot is None:
                    raise ResourceNotFoundError("knowledge base was not found")
                defaults = snapshot.retrieval_strategy
                target_top_k = int(defaults.get("top_k", 10)) if len(selected) > 1 else top_k
                target_rerank = RerankMode(defaults.get("rerank_mode", "classic")) if len(selected) > 1 else resolved_rerank_mode
                if retrieval_mode == "graph":
                    # The manual Graph control has its own validated 4–20 range.
                    target_top_k, target_rerank = top_k, resolved_rerank_mode
                profile = (
                    graph_profile(top_k=target_top_k, rerank_mode=target_rerank)
                    if retrieval_mode == "graph" else
                    adaptive_graphiti_profile(top_k=target_top_k, rerank_mode=target_rerank)
                    if retrieval_mode == "auto" else
                    self._retrieval_profile_factory(RetrievalStrategy.EXACT_VECTOR, target_top_k, target_rerank)
                )
                graph = await uow.graph.get_config(identifier) if retrieval_mode != "text" else None
                snapshots.append(replace(snapshot, retrieval_strategy=profile.as_dict(),
                    graph_build_id=graph.active_build_id if graph is not None else None))
            await uow.chat.set_session_scope(session_id, selected)
            model_configuration = await self._resolve_model_configuration(
                uow,
                model_profile_revision_id,
            )
            retrieval_strategy_snapshot = dict(retrieval_strategy)
            recent_turns = await uow.chat.list_completed_turns(
                session_id=session_id,
                limit=self._context_max_turns + 1,
            )
            conversation_context = self._context_selector.select(recent_turns)
            return await uow.chat.create_run(
                scope=scope,
                request_hash=request_hash,
                kb_id=selected[0] if len(selected) == 1 else None,
                session_id=session_id,
                index_revision_id=snapshots[0].index_revision_id if len(snapshots) == 1 else None,
                knowledge_bases=tuple(snapshots),
                message=normalized_message,
                retrieval_strategy=retrieval_strategy_snapshot,
                model_configuration=model_configuration,
                conversation_context=serialize_conversation_context(
                    conversation_context
                ),
                agent_configuration={
                    "version": CHAT_AGENT_VERSION,
                    "budget": ChatAgentBudget().as_dict(),
                },
            )

        return await execute_in_transaction(self._unit_of_work, persist)

    async def _resolve_model_configuration(
        self,
        uow: SqlAlchemyUnitOfWork,
        requested_revision_id: UUID | None,
    ) -> dict[str, Any]:
        revision_id = requested_revision_id
        repository = getattr(uow, "model_settings", None)
        if repository is None:
            if revision_id is not None:
                raise ResourceStateConflictError("model settings are unavailable")
            return dict(self._model_configuration)
        if revision_id is None:
            selection = await repository.get_selection()
            revision_id = selection.chat_profile_revision_id
        if revision_id is None:
            if (
                not self._allow_legacy_model_configuration
                or not self._model_configuration
            ):
                raise ResourceStateConflictError(
                    "a chat model must be selected before creating a run"
                )
            return dict(self._model_configuration)
        bundle = await repository.get_profile_revision(revision_id)
        if bundle is None:
            raise ResourceNotFoundError("chat model profile revision was not found")
        if bundle.profile.kind is not ModelKind.CHAT:
            raise ResourceStateConflictError("model profile is not a chat model")
        if not bundle.profile.enabled or not bundle.provider.enabled:
            raise ResourceStateConflictError("chat model profile is disabled")
        if (
            bundle.current_revision.validation_status
            is not ModelValidationStatus.VALID
        ):
            raise ResourceStateConflictError("chat model profile is not validated")
        return _chat_profile_configuration(bundle, self._model_configuration)

def chat_model_configuration(settings: Any) -> dict[str, Any]:
    """Return the reproducibility snapshot without endpoint URLs or secrets."""

    return {
        "provider_identity": settings.provider_identity,
        "logical_endpoint_identity": settings.logical_endpoint_identity,
        "requested_model": settings.model,
        "resolved_model": settings.resolved_model,
        "model_version": settings.model_version,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "structured_output_mode": settings.structured_output_mode,
        "thinking_enabled": settings.thinking_enabled,
        "vision_enabled": settings.vision_enabled,
        "max_visual_images": settings.max_visual_images,
        "max_visual_image_bytes": settings.max_visual_image_bytes,
        "max_visual_total_bytes": settings.max_visual_total_bytes,
        "max_visual_pixels": settings.max_visual_pixels,
        "visual_media_profile": settings.visual_media_profile,
        "configuration_fingerprint": settings.configuration_fingerprint,
        "capability_fingerprint": settings.capability_fingerprint,
    }


def _chat_profile_configuration(
    bundle: ModelProfileBundle,
    safety_defaults: dict[str, Any],
) -> dict[str, Any]:
    revision = bundle.current_revision
    parameters = dict(revision.configuration)
    return {
        "source": "user_model_profile",
        "model_profile_id": str(bundle.profile.id),
        "model_profile_name": bundle.profile.name,
        "model_profile_revision_id": str(revision.id),
        "model_profile_revision": revision.revision,
        "provider_id": str(bundle.provider.id),
        "provider_revision_id": str(bundle.provider_revision.id),
        "provider_identity": bundle.provider.name,
        "requested_model": revision.model,
        "resolved_model": revision.model,
        "temperature": parameters.get("temperature", 0.2),
        "top_p": parameters.get("top_p", 0.9),
        "sampling_top_k": parameters.get("sampling_top_k", 40),
        "max_tokens": parameters.get("max_output_tokens", 4096),
        "reasoning_effort": parameters.get("reasoning_effort", "off"),
        "thinking_enabled": parameters.get("reasoning_effort", "off") != "off",
        "structured_output_mode": parameters.get(
            "structured_output_mode", "json_object"
        ),
        "vision_enabled": parameters.get("vision_enabled", False),
        "configuration_fingerprint": revision.configuration_fingerprint,
        "capability_fingerprint": revision.capability_fingerprint,
        "max_visual_images": safety_defaults.get(
            "max_visual_images", DEFAULT_CHAT_MAX_VISUAL_IMAGES
        ),
        "max_visual_image_bytes": safety_defaults.get(
            "max_visual_image_bytes", DEFAULT_CHAT_MAX_VISUAL_IMAGE_BYTES
        ),
        "max_visual_total_bytes": safety_defaults.get(
            "max_visual_total_bytes", DEFAULT_CHAT_MAX_VISUAL_TOTAL_BYTES
        ),
        "max_visual_pixels": safety_defaults.get(
            "max_visual_pixels", DEFAULT_CHAT_MAX_VISUAL_PIXELS
        ),
        "visual_media_profile": safety_defaults.get(
            "visual_media_profile", DEFAULT_CHAT_VISUAL_MEDIA_PROFILE
        ),
    }
