"""Run-local bounded dispatch over the existing single-KB retriever."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from rag_kb.domain.chat_pipeline import ChatExecutionContext
from rag_kb.domain.chat_scope import ChatKnowledgeBaseSnapshot
from rag_kb.retrieval.profile import parse_chat_retrieval_snapshot


def target_context(context: ChatExecutionContext, snapshot: ChatKnowledgeBaseSnapshot) -> ChatExecutionContext:
    return replace(context, knowledge_base_id=snapshot.knowledge_base_id,
                   index_revision_id=snapshot.index_revision_id,
                   retrieval_strategy=snapshot.retrieval_strategy,
                   knowledge_bases=(snapshot,))


async def gather_owned(*awaitables):
    """Cancel and settle sibling work when a caller cancels or a task fails."""
    tasks = [asyncio.create_task(item) for item in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class BoundedRetriever:
    """A single limiter shared by all tools, queries and scopes in one run."""

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever
        self._semaphore = asyncio.Semaphore(4)

    async def invoke(self, name: str, context: ChatExecutionContext, *args, **kwargs):
        async with self._semaphore:
            method = getattr(self._retriever, name)
            validator = getattr(self._retriever, "validate_scope", None)
            check = name in {"semantic_search", "keyword_search", "search_graph_relations", "read_chunk_context", "list_documents"}
            graph = name == "search_graph_relations" or (name == "semantic_search" and parse_chat_retrieval_snapshot(context.retrieval_strategy)[3] == "manual_graph")
            if validator is not None and check:
                await validator(context, check_graph=graph)
            value = await method(context, *args, **kwargs)
            if validator is not None and check:
                await validator(context, check_graph=graph)
            return value

    async def semantic_search(self, context, *args, **kwargs):
        return await self.invoke("semantic_search", context, *args, **kwargs)

    async def keyword_search(self, context, *args, **kwargs):
        return await self.invoke("keyword_search", context, *args, **kwargs)

    async def keyword_search_capable(self, context):
        return await self.invoke("keyword_search_capable", context)

    async def graph_relations_capable(self, context):
        return await self.invoke("graph_relations_capable", context)

    async def search_graph_relations(self, context, *args, **kwargs):
        return await self.invoke("search_graph_relations", context, *args, **kwargs)

    async def read_chunk_context(self, context, *args, **kwargs):
        return await self.invoke("read_chunk_context", context, *args, **kwargs)

    async def list_documents(self, context, *args, **kwargs):
        return await self.invoke("list_documents", context, *args, **kwargs)
