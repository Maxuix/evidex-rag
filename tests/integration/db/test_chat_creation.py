from __future__ import annotations

import asyncio
import os
import unittest
from uuid import UUID, uuid4

import asyncpg

from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.domain import (
    AnswerStyle,
    EmbeddingSpaceDefinition,
    IdempotencyKeyReusedError,
    IndexProfileDefinition,
    InsufficiencyPolicy,
    ResourceNotFoundError,
    ResourceStateConflictError,
)
from rag_kb.services import ChatService, KnowledgeBaseService
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")
WORKSPACE = UUID("01900000-0000-7000-8000-000000000a01")


@unittest.skipUnless(
    MIGRATION_DSN and RUNTIME_SQLALCHEMY_DSN,
    "database integration DSNs are not configured",
)
class ChatCreationDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute("TRUNCATE TABLE workspace CASCADE")
        finally:
            await connection.close()
        self.database = create_database_resources(
            RUNTIME_SQLALCHEMY_DSN,
            pool_size=4,
            max_overflow=0,
            process=DatabaseProcess.API,
        )
        self.factory = SqlAlchemyUnitOfWorkFactory(self.database.sessions, WORKSPACE)
        self.policy = SingleWorkspaceAccessPolicy(WORKSPACE)
        self.context = AuthContext("principal", "client", WORKSPACE)
        self.knowledge_bases = KnowledgeBaseService(
            self.factory,
            self.policy,
            embedding_space=_embedding(),
            index_profile=_profile(),
        )
        self.chat = ChatService(
            self.factory,
            self.policy,
            model_configuration=_model_configuration(),
        )

    async def asyncTearDown(self) -> None:
        await self.database.close()

    async def test_concurrent_lost_response_replay_is_one_atomic_run(self) -> None:
        kb = await self._create_kb("primary")
        session = await self.chat.create_session(
            self.context, kb_id=kb.id, title="Incident response"
        )
        key = uuid4()

        first, replay = await asyncio.gather(
            self._create_run(session.id, kb.id, key),
            self._create_run(session.id, kb.id, key),
        )
        self.assertEqual(first.id, replay.id)
        self.assertEqual(first.user_message_id, replay.user_message_id)
        self.assertEqual(first.assistant_message_id, replay.assistant_message_id)
        self.assertEqual(first.status, "queued")
        self.assertEqual(first.assistant_status, "generating")
        self.assertEqual(first.assistant_content, "")
        self.assertEqual(first.effective_policy["grounding_policy"], "evidence_only")
        self.assertEqual(first.effective_policy["answer_style"], "summary")
        self.assertEqual(first.index_revision_id, kb.active_index_revision_id)
        self.assertNotIn("api_key", first.model_configuration)
        self.assertNotIn("base_url", first.model_configuration)

        with self.assertRaises(IdempotencyKeyReusedError):
            await self.chat.create_run(
                self.context,
                key,
                session_id=session.id,
                kb_id=kb.id,
                message="different question",
                answer_style=AnswerStyle.SUMMARY,
                insufficiency_policy=InsufficiencyPolicy.PARTIAL_ANSWER,
                retrieval_mode="vector",
                top_k=8,
            )

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            counts = await connection.fetchrow(
                """
                SELECT
                    (SELECT count(*) FROM chat_session) AS sessions,
                    (SELECT count(*) FROM chat_run) AS runs,
                    (SELECT count(*) FROM chat_message WHERE role = 'user') AS users,
                    (SELECT count(*) FROM chat_message WHERE role = 'assistant') AS assistants,
                    (SELECT count(*) FROM citation) AS citations
                """
            )
        finally:
            await connection.close()
        self.assertEqual(tuple(counts), (1, 1, 1, 1, 0))

    async def test_history_status_and_authorization_are_principal_bound(self) -> None:
        kb = await self._create_kb("authorized")
        other_kb = await self._create_kb("other")
        session = await self.chat.create_session(
            self.context, kb_id=kb.id, title=None
        )
        run = await self._create_run(session.id, kb.id, uuid4())

        status = await self.chat.get_run(self.context, run.id)
        messages = await self.chat.list_messages(
            self.context,
            session.id,
            limit=10,
            sort="created_at",
            after=None,
        )
        sessions = await self.chat.list_sessions(
            self.context,
            limit=10,
            sort="-updated_at",
            after=None,
        )
        self.assertEqual(status.id, run.id)
        self.assertEqual([item.role for item in messages.items], ["user", "assistant"])
        self.assertEqual(sessions.items[0].id, session.id)

        with self.assertRaises(ResourceStateConflictError):
            await self.chat.create_run(
                self.context,
                uuid4(),
                session_id=session.id,
                kb_id=other_kb.id,
                message="wrong knowledge base",
                answer_style=None,
                insufficiency_policy=None,
                retrieval_mode="vector",
                top_k=10,
            )

        other_principal = AuthContext("other-principal", "client", WORKSPACE)
        with self.assertRaises(ResourceNotFoundError):
            await self.chat.get_run(other_principal, run.id)
        with self.assertRaises(ResourceNotFoundError):
            await self.chat.list_messages(
                other_principal,
                session.id,
                limit=10,
                sort="created_at",
                after=None,
            )

    async def _create_kb(self, name: str):
        return await self.knowledge_bases.create(
            self.context,
            uuid4(),
            name=name,
            retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
        )

    async def _create_run(self, session_id: UUID, kb_id: UUID, key: UUID):
        return await self.chat.create_run(
            self.context,
            key,
            session_id=session_id,
            kb_id=kb_id,
            message="How should RUN-ORD-14 be handled?",
            answer_style=AnswerStyle.SUMMARY,
            insufficiency_policy=InsufficiencyPolicy.PARTIAL_ANSWER,
            retrieval_mode="vector",
            top_k=8,
        )


def _embedding() -> EmbeddingSpaceDefinition:
    return EmbeddingSpaceDefinition(
        provider_identity="provider",
        endpoint_identity="embedding-endpoint",
        requested_model="embedding-model",
        resolved_model="embedding-model",
        model_version="v1",
        deployment_revision=None,
        dimension=1024,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization="l2",
        configuration_fingerprint="sha256:" + "a" * 64,
        tokenizer_fingerprint=None,
        compatibility_fingerprint="sha256:" + "b" * 64,
    )


def _profile() -> IndexProfileDefinition:
    return IndexProfileDefinition(
        parser_config={"profile": "plain_text_test_v1"},
        chunking_config={
            "profile": "paragraph_window_v1",
            "max_characters": 2000,
            "overlap_characters": 200,
        },
    )


def _model_configuration() -> dict[str, str]:
    return {
        "provider_identity": "chat-provider",
        "logical_endpoint_identity": "chat-endpoint",
        "requested_model": "chat-model",
        "resolved_model": "chat-model",
        "model_version": "v1",
        "structured_output_mode": "json_object",
        "configuration_fingerprint": "sha256:" + "c" * 64,
        "capability_fingerprint": "sha256:" + "d" * 64,
    }


if __name__ == "__main__":
    unittest.main()
