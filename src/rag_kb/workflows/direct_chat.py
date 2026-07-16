"""Workflow runner boundary for the direct P1A chat pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_kb.domain import ChatExecutionCommand, ChatPipelineState
from rag_kb.services.chat_pipeline import DirectChatPipeline


@runtime_checkable
class GraphRunner(Protocol):
    async def execute(self, command: ChatExecutionCommand) -> ChatPipelineState: ...


class DirectGraphRunner:
    def __init__(self, pipeline: DirectChatPipeline) -> None:
        self._pipeline = pipeline

    async def execute(self, command: ChatExecutionCommand) -> ChatPipelineState:
        return await self._pipeline.execute(command)
