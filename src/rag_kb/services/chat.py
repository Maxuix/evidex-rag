"""Durable chat session and run-creation application service."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from rag_kb.auth import AccessPolicy, AuthContext
from rag_kb.domain import (
    AnswerStyle,
    CHAT_WORKFLOW_VERSION,
    ChatMessage,
    ChatRun,
    ChatSession,
    ChatSessionBusyError,
    ChatWorkflowMode,
    ContextualizedQuery,
    ErrorCode,
    ModelKind,
    ModelProfileBundle,
    ModelValidationStatus,
    QueryContextStatus,
    QueryRewriteSource,
    CONTEXTUAL_QUERY_VERSION,
    RetrievalExecutionError,
    IdempotencyKeyReusedError,
    IdempotencyScope,
    InsufficiencyPolicy,
    Page,
    ResourceNotFoundError,
    ResourceStateConflictError,
    RetrievalStrategy,
    canonical_request_hash,
    initial_chat_workflow,
    resolve_p1_policy,
)
from rag_kb.retrieval.profile import RetrievalExecutionProfile
from rag_kb.memory import (
    ConversationContextSelector,
    serialize_contextualized_query,
    serialize_conversation_context,
)
from rag_kb.uow import UnitOfWork, UnitOfWorkFactory, UnitOfWorkPurpose, execute_in_transaction


CREATE_CHAT_RUN_ENDPOINT = "POST /api/v1/chat/runs"


class ChatService:
    """Create durable chat facts without executing retrieval or model work."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        access_policy: AccessPolicy,
        *,
        model_configuration: dict[str, Any],
        default_rerank: bool = False,
        hybrid_enabled: bool = False,
        agent_enabled: bool = False,
        auto_enabled: bool = False,
        retrieval_profile_factory: Callable[
            [RetrievalStrategy, int, bool], RetrievalExecutionProfile
        ],
        context_strategy: str = "recent_completed_turns_v1",
        context_max_turns: int = 6,
        context_max_tokens: int = 4000,
        context_tokenizer: str = "cl100k_base",
        allow_legacy_model_configuration: bool = True,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._access_policy = access_policy
        self._model_configuration = dict(model_configuration)
        self._allow_legacy_model_configuration = allow_legacy_model_configuration
        self._default_rerank = default_rerank
        self._hybrid_enabled = hybrid_enabled
        self._agent_enabled = agent_enabled
        self._auto_enabled = auto_enabled
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

    def workflow_capabilities_snapshot(self) -> dict[str, Any]:
        return {
            "version": CHAT_WORKFLOW_VERSION,
            "default_mode": ChatWorkflowMode.SIMPLE.value,
            "modes": (
                {"mode": ChatWorkflowMode.SIMPLE.value, "enabled": True},
                {
                    "mode": ChatWorkflowMode.AGENT.value,
                    "enabled": self._agent_enabled,
                },
                {
                    "mode": ChatWorkflowMode.AUTO.value,
                    "enabled": self._auto_enabled,
                },
            ),
        }

    async def create_session(
        self,
        context: AuthContext,
        *,
        kb_id: UUID,
        title: str | None,
    ) -> ChatSession:
        self._authorize(context)

        async def persist(uow: UnitOfWork) -> ChatSession:
            _require_scope(uow, context)
            if await uow.knowledge_bases.get(kb_id) is None:
                raise ResourceNotFoundError("knowledge base was not found")
            return await uow.chat.create_session(
                kb_id=kb_id,
                principal_id=context.principal_id,
                title=title,
            )

        return await execute_in_transaction(self._unit_of_work, persist)

    async def list_sessions(
        self,
        context: AuthContext,
        *,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
        kb_id: UUID | None = None,
    ) -> Page[ChatSession]:
        self._authorize(context)

        async def load(uow: UnitOfWork) -> Page[ChatSession]:
            _require_scope(uow, context)
            if kb_id is not None and await uow.knowledge_bases.get(kb_id) is None:
                raise ResourceNotFoundError("knowledge base was not found")
            return await uow.chat.list_sessions(
                principal_id=context.principal_id,
                limit=limit,
                sort=sort,
                after=after,
                kb_id=kb_id,
            )

        return await execute_in_transaction(
            self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST
        )

    async def list_messages(
        self,
        context: AuthContext,
        session_id: UUID,
        *,
        limit: int,
        sort: str,
        after: tuple[str, ...] | None,
    ) -> Page[ChatMessage]:
        self._authorize(context)

        async def load(uow: UnitOfWork) -> Page[ChatMessage]:
            _require_scope(uow, context)
            result = await uow.chat.list_messages(
                session_id=session_id,
                principal_id=context.principal_id,
                limit=limit,
                sort=sort,
                after=after,
            )
            if result is None:
                raise ResourceNotFoundError("chat session was not found")
            return result

        return await execute_in_transaction(
            self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST
        )

    async def get_run(self, context: AuthContext, run_id: UUID) -> ChatRun:
        self._authorize(context)

        async def load(uow: UnitOfWork) -> ChatRun:
            _require_scope(uow, context)
            result = await uow.chat.get_run(
                run_id,
                principal_id=context.principal_id,
                client_id=context.client_id,
            )
            if result is None:
                raise ResourceNotFoundError("chat run was not found")
            return result

        return await execute_in_transaction(
            self._unit_of_work, load, purpose=UnitOfWorkPurpose.REQUEST
        )

    async def create_run(
        self,
        context: AuthContext,
        idempotency_key: UUID,
        *,
        session_id: UUID,
        kb_id: UUID,
        message: str,
        answer_style: AnswerStyle | None,
        insufficiency_policy: InsufficiencyPolicy | None,
        retrieval_mode: str,
        top_k: int,
        rerank: bool | None = None,
        workflow_mode: ChatWorkflowMode = ChatWorkflowMode.SIMPLE,
        model_profile_revision_id: UUID | None = None,
    ) -> ChatRun:
        self._authorize(context)
        if workflow_mode is ChatWorkflowMode.AGENT and not self._agent_enabled:
            raise RetrievalExecutionError(
                ErrorCode.CAPABILITY_NOT_ENABLED,
                diagnostic={"capability": "chat_agent"},
            )
        if workflow_mode is ChatWorkflowMode.AUTO and not self._auto_enabled:
            raise RetrievalExecutionError(
                ErrorCode.CAPABILITY_NOT_ENABLED,
                diagnostic={"capability": "chat_auto"},
            )
        if retrieval_mode not in {"vector", "hybrid"}:
            raise ResourceStateConflictError("retrieval mode is unsupported")
        if retrieval_mode == "hybrid" and not self._hybrid_enabled:
            raise RetrievalExecutionError(
                ErrorCode.CAPABILITY_NOT_ENABLED,
                diagnostic={"capability": "hybrid"},
            )
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("message must contain non-whitespace characters")

        scope = IdempotencyScope(
            context.principal_id,
            context.client_id,
            CREATE_CHAT_RUN_ENDPOINT,
            idempotency_key,
        )
        requested_policy: dict[str, str] = {}
        if answer_style is not None:
            requested_policy["answer_style"] = answer_style.value
        if insufficiency_policy is not None:
            requested_policy["insufficiency_policy"] = insufficiency_policy.value
        resolved_rerank = self._default_rerank if rerank is None else rerank
        requested_retrieval = {
            "mode": retrieval_mode,
            "top_k": top_k,
            "rerank": resolved_rerank,
        }
        strategy = (
            RetrievalStrategy.HYBRID
            if retrieval_mode == "hybrid"
            else RetrievalStrategy.EXACT_VECTOR
        )
        retrieval_strategy = self._retrieval_profile_factory(
            strategy, top_k, resolved_rerank
        ).as_dict()
        workflow_configuration, workflow_state = initial_chat_workflow(
            workflow_mode
        )
        request_hash = canonical_request_hash(
            {
                "session_id": str(session_id),
                "knowledge_base_id": str(kb_id),
                "message": normalized_message,
                "answer_policy": requested_policy,
                "workflow": {"mode": workflow_mode.value},
                "retrieval": requested_retrieval,
                "model_profile_revision_id": (
                    str(model_profile_revision_id)
                    if model_profile_revision_id is not None
                    else None
                ),
            }
        )

        async def persist(uow: UnitOfWork) -> ChatRun:
            _require_scope(uow, context)
            await uow.chat.lock_idempotency(scope)
            prior = await uow.chat.get_run_by_scope(scope)
            if prior is not None:
                if prior.request_hash != request_hash:
                    raise IdempotencyKeyReusedError(
                        "idempotency key was already used with a different request"
                    )
                return prior

            session = await uow.chat.lock_session(
                session_id, principal_id=context.principal_id
            )
            if session is None:
                raise ResourceNotFoundError("chat session was not found")
            if session.kb_id != kb_id:
                raise ResourceStateConflictError(
                    "chat session belongs to a different knowledge base"
                )
            if await uow.chat.has_nonterminal_run(session_id):
                raise ChatSessionBusyError(
                    "chat session already has a queued or running run"
                )
            knowledge_base = await uow.knowledge_bases.get(kb_id)
            if knowledge_base is None:
                raise ResourceNotFoundError("knowledge base was not found")
            model_configuration = await self._resolve_model_configuration(
                uow,
                model_profile_revision_id,
            )
            effective_policy = resolve_p1_policy(
                requested_policy=requested_policy,
                knowledge_base_defaults=knowledge_base.answer_policy_defaults,
            ).as_dict()
            recent_turns = await uow.chat.list_completed_turns(
                session_id=session_id,
                principal_id=context.principal_id,
                kb_id=kb_id,
                limit=self._context_max_turns + 1,
            )
            conversation_context = self._context_selector.select(recent_turns)
            original_query = (
                ContextualizedQuery(
                    version=CONTEXTUAL_QUERY_VERSION,
                    status=QueryContextStatus.ORIGINAL,
                    original_query=normalized_message,
                    standalone_query=normalized_message,
                    context_hash=conversation_context.content_hash,
                    rewrite_source=QueryRewriteSource.ORIGINAL,
                )
                if not conversation_context.turns
                else None
            )
            return await uow.chat.create_run(
                scope=scope,
                request_hash=request_hash,
                kb_id=kb_id,
                session_id=session_id,
                index_revision_id=knowledge_base.active_index_revision_id,
                message=normalized_message,
                requested_policy=requested_policy,
                effective_policy=effective_policy,
                retrieval_strategy=retrieval_strategy,
                model_configuration=model_configuration,
                workflow_configuration=workflow_configuration.as_dict(),
                workflow_state=workflow_state.as_dict(),
                conversation_context=serialize_conversation_context(
                    conversation_context
                ),
                contextualized_query=(
                    serialize_contextualized_query(original_query)
                    if original_query is not None
                    else None
                ),
            )

        return await execute_in_transaction(self._unit_of_work, persist)

    async def _resolve_model_configuration(
        self,
        uow: UnitOfWork,
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

    def _authorize(self, context: AuthContext) -> None:
        self._access_policy.metadata_filter(context)


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
        "max_visual_images": safety_defaults.get("max_visual_images", 4),
        "max_visual_image_bytes": safety_defaults.get(
            "max_visual_image_bytes", 5 * 1024 * 1024
        ),
        "max_visual_total_bytes": safety_defaults.get(
            "max_visual_total_bytes", 12 * 1024 * 1024
        ),
        "max_visual_pixels": safety_defaults.get("max_visual_pixels", 16_000_000),
        "visual_media_profile": safety_defaults.get("visual_media_profile"),
    }


def _require_scope(uow: UnitOfWork, context: AuthContext) -> None:
    if uow.workspace_id != context.workspace_id:
        raise RuntimeError("Unit of Work workspace does not match AuthContext")
