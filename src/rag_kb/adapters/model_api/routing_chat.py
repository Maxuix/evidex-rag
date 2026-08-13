"""Select an immutable user-configured chat model for each request."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from uuid import UUID

from rag_kb.domain import ChatModelRequest, ChatModelResponse
from rag_kb.ports.model_api import ChatModelAdapter


ChatModelLoader = Callable[[UUID], Awaitable[ChatModelAdapter]]


class RoutingChatModelAdapter:
    """Route new requests by profile revision and retain a legacy fallback."""

    def __init__(
        self,
        loader: ChatModelLoader,
        *,
        legacy_fallback: ChatModelAdapter,
    ) -> None:
        self._loader = loader
        self._legacy_fallback = legacy_fallback
        self._models: dict[UUID, ChatModelAdapter] = {}
        self._lock = asyncio.Lock()

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        model = await self._resolve(request.model_profile_revision_id)
        return await model.complete(request)

    async def _resolve(self, revision_id: UUID | None) -> ChatModelAdapter:
        if revision_id is None:
            return self._legacy_fallback
        cached = self._models.get(revision_id)
        if cached is not None:
            return cached
        async with self._lock:
            cached = self._models.get(revision_id)
            if cached is None:
                cached = await self._loader(revision_id)
                self._models[revision_id] = cached
            return cached
