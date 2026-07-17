"""Stable workflow boundary consumed by the Chat scheduler."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_kb.domain import ChatExecutionCommand, ChatPipelineState


@runtime_checkable
class GraphRunner(Protocol):
    async def execute(self, command: ChatExecutionCommand) -> ChatPipelineState: ...
