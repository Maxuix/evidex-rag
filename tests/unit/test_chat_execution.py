from __future__ import annotations

import unittest
from datetime import UTC, datetime
from uuid import uuid4

from rag_kb.domain import (
    CONTEXTUAL_QUERY_VERSION,
    ChatExecutionContext,
    ChatModelCallRecord,
    ChatModelOperation,
    ChatPipelineExecutionError,
    ChatRunLease,
    ErrorCode,
    Evidence,
    EvidencePack,
    RetrievalDebug,
    RetrievalQueryPlan,
    RetrievalStrategy,
    ContextualizedQuery,
    QueryContextStatus,
    QueryRewriteSource,
)
from rag_kb.services.chat_execution import ChatEvidenceRetriever
from rag_kb.retrieval.profile import exact_profile


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
        retrieval_strategy=exact_profile(top_k=3, rerank=False).as_dict(),
        model_configuration={"requested_model": "fixed"},
        attempt=1,
    )


class ChatExecutionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieval_uses_only_the_standalone_query(self) -> None:
        context = _context()

        class Retrieval:
            request = None

            async def retrieve(self, auth, request):
                del auth
                self.request = request
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=context.index_revision_id,
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                )

        retrieval = Retrieval()
        query_context = ContextualizedQuery(
            version=CONTEXTUAL_QUERY_VERSION,
            status=QueryContextStatus.CONTEXTUALIZED,
            original_query=context.query,
            standalone_query="A fully standalone retrieval query",
            context_hash=context.conversation_context.content_hash,
            model_calls=(
                ChatModelCallRecord(
                    operation=ChatModelOperation.CONTEXTUALIZE_QUERY,
                    model="fixed-model",
                    provider_request_id="context-call",
                    usage={},
                ),
            ),
            created_at=datetime.now(UTC),
            origin_attempt=1,
            rewrite_source=QueryRewriteSource.MODEL,
        )

        await ChatEvidenceRetriever(retrieval).retrieve(context, query_context)  # type: ignore[arg-type]

        self.assertEqual(
            retrieval.request.query, "A fully standalone retrieval query"
        )

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

    async def test_native_image_only_evidence_is_retained_for_visual_preparation(
        self,
    ) -> None:
        context = _context()
        visual = Evidence(
            rank=1,
            index_chunk_id=uuid4(),
            indexed_document_version_id=uuid4(),
            document_id=uuid4(),
            document_version_id=uuid4(),
            index_revision_id=context.index_revision_id,
            ordinal=0,
            text="",
            source_location={"page_number": 1},
            hierarchy={},
            source_metadata={},
            score=0.5,
            document_display_name="Guide",
            document_original_filename="guide.png",
            modality="image",
            matched_representations=("native_image",),
        )

        class Retrieval:
            async def retrieve(self, auth, request):
                del auth
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=context.index_revision_id,
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                    evidence=(visual,),
                )

        retriever = ChatEvidenceRetriever(Retrieval())  # type: ignore[arg-type]
        result = await retriever.retrieve(context)

        self.assertEqual(result.evidence, (visual,))

    async def test_retrieval_preserves_internal_debug_for_terminal_diagnostics(
        self,
    ) -> None:
        context = _context()
        debug = RetrievalDebug(
            query_plan=RetrievalQueryPlan(
                workspace_id=context.workspace_id,
                knowledge_base_id=context.knowledge_base_id,
                strategy=RetrievalStrategy.EXACT_VECTOR,
                top_k=3,
            ),
            resolved_active_revision_id=context.index_revision_id,
            result_count=0,
            text_candidate_count=4,
            cross_modal_candidate_count=2,
            hydrated_relation_count=1,
            evidence_group_count=1,
        )

        class Retrieval:
            async def retrieve(self, auth, request):
                del auth
                return EvidencePack(
                    knowledge_base_id=request.knowledge_base_id,
                    index_revision_id=context.index_revision_id,
                    strategy=RetrievalStrategy.EXACT_VECTOR,
                    debug=debug,
                )

        result = await ChatEvidenceRetriever(Retrieval()).retrieve(context)  # type: ignore[arg-type]

        self.assertIs(result.debug, debug)


if __name__ == "__main__":
    unittest.main()
