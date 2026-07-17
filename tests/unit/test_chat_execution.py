from __future__ import annotations

import unittest
from datetime import UTC, datetime
from uuid import uuid4

from rag_kb.domain import (
    ChatExecutionContext,
    ChatPipelineExecutionError,
    ChatRunLease,
    ErrorCode,
    EvidencePack,
    RetrievalStrategy,
)
from rag_kb.services.chat_execution import ChatEvidenceRetriever


def _context() -> ChatExecutionContext:
    run_id = uuid4()
    workspace_id = uuid4()
    return ChatExecutionContext(
        lease=ChatRunLease(
            run_id=run_id,
            workspace_id=workspace_id,
            claimed_by="worker",
            attempt=1,
            claimed_at=datetime.now(UTC),
        ),
        run_id=run_id,
        workspace_id=workspace_id,
        knowledge_base_id=uuid4(),
        session_id=uuid4(),
        user_message_id=uuid4(),
        assistant_message_id=uuid4(),
        index_revision_id=uuid4(),
        principal_id="principal",
        client_id="client",
        query="What is frozen?",
        effective_policy={"grounding_policy": "evidence_only"},
        retrieval_strategy={
            "strategy": "exact_vector",
            "top_k": 3,
            "rerank": False,
        },
        model_configuration={"requested_model": "fixed"},
        attempt=1,
    )


class ChatExecutionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieval_fails_closed_when_active_revision_moved(self) -> None:
        context = _context()

        class Retrieval:
            async def retrieve(self, auth, request):
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=uuid4(),
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                )

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            retriever = ChatEvidenceRetriever(Retrieval())  # type: ignore[arg-type]
            await retriever.retrieve(context)

        self.assertEqual(raised.exception.code, ErrorCode.CHAT_REVISION_MISMATCH)


if __name__ == "__main__":
    unittest.main()
