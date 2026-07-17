"""Durable chat session and run-creation application service."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from rag_kb.auth import AccessPolicy, AuthContext
from rag_kb.domain import (
    AnswerStyle,
    ChatMessage,
    ChatRun,
    ChatSession,
    IdempotencyKeyReusedError,
    IdempotencyScope,
    InsufficiencyPolicy,
    Page,
    ResourceNotFoundError,
    ResourceStateConflictError,
    canonical_request_hash,
    resolve_p1_policy,
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
    ) -> None:
        self._unit_of_work = unit_of_work
        self._access_policy = access_policy
        self._model_configuration = dict(model_configuration)

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
    ) -> Page[ChatSession]:
        self._authorize(context)

        async def load(uow: UnitOfWork) -> Page[ChatSession]:
            _require_scope(uow, context)
            return await uow.chat.list_sessions(
                principal_id=context.principal_id,
                limit=limit,
                sort=sort,
                after=after,
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
    ) -> ChatRun:
        self._authorize(context)
        if retrieval_mode != "vector":
            raise ResourceStateConflictError("only vector retrieval is enabled")
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
        requested_retrieval = {"mode": retrieval_mode, "top_k": top_k}
        retrieval_strategy = {
            "strategy": "exact_vector",
            "top_k": top_k,
            "rerank": False,
        }
        request_hash = canonical_request_hash(
            {
                "session_id": str(session_id),
                "knowledge_base_id": str(kb_id),
                "message": normalized_message,
                "answer_policy": requested_policy,
                "retrieval": requested_retrieval,
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

            session = await uow.chat.get_session(
                session_id, principal_id=context.principal_id
            )
            if session is None:
                raise ResourceNotFoundError("chat session was not found")
            if session.kb_id != kb_id:
                raise ResourceStateConflictError(
                    "chat session belongs to a different knowledge base"
                )
            knowledge_base = await uow.knowledge_bases.get(kb_id)
            if knowledge_base is None:
                raise ResourceNotFoundError("knowledge base was not found")
            effective_policy = resolve_p1_policy(
                requested_policy=requested_policy,
                knowledge_base_defaults=knowledge_base.answer_policy_defaults,
            ).as_dict()
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
                model_configuration=self._model_configuration,
            )

        return await execute_in_transaction(self._unit_of_work, persist)

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
        "configuration_fingerprint": settings.configuration_fingerprint,
        "capability_fingerprint": settings.capability_fingerprint,
    }


def _require_scope(uow: UnitOfWork, context: AuthContext) -> None:
    if uow.workspace_id != context.workspace_id:
        raise RuntimeError("Unit of Work workspace does not match AuthContext")
