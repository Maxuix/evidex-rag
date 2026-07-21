from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import UUID, uuid4

from pydantic import ValidationError

from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.domain import (
    AnswerPolicyNotSupportedError,
    AnswerStyle,
    ChatSessionBusyError,
    ConversationTurn,
    InsufficiencyPolicy,
    resolve_p1_policy,
)
from rag_kb.memory import hydrate_conversation_context
from rag_kb.schemas import ChatRunCreate
from rag_kb.services import ChatService, chat_model_configuration


class ChatCreationContractTests(unittest.TestCase):
    def test_safe_defaults_and_all_four_override_pairs_are_complete(self) -> None:
        default = resolve_p1_policy(
            requested_policy={},
            knowledge_base_defaults={
                "answer_style": "concise",
                "insufficiency_policy": "refuse",
            },
        ).as_dict()
        self.assertEqual(default["answer_style"], "concise")
        self.assertEqual(default["insufficiency_policy"], "refuse")
        self.assertEqual(default["grounding_policy"], "evidence_only")
        self.assertTrue(default["citation_required"])
        self.assertEqual(default["citation_granularity"], "claim_level")
        self.assertEqual(default["answer_task"], "answer")

        pairs = {
            (
                resolve_p1_policy(
                    requested_policy={
                        "answer_style": style,
                        "insufficiency_policy": insufficiency,
                    },
                    knowledge_base_defaults={
                        "answer_style": "concise",
                        "insufficiency_policy": "refuse",
                    },
                ).answer_style,
                resolve_p1_policy(
                    requested_policy={
                        "answer_style": style,
                        "insufficiency_policy": insufficiency,
                    },
                    knowledge_base_defaults={
                        "answer_style": "concise",
                        "insufficiency_policy": "refuse",
                    },
                ).insufficiency_policy,
            )
            for style in AnswerStyle
            for insufficiency in InsufficiencyPolicy
        }
        self.assertEqual(len(pairs), 4)

    def test_request_overrides_kb_defaults_while_server_constraints_remain_fixed(self) -> None:
        resolved = resolve_p1_policy(
            requested_policy={"answer_style": "concise"},
            knowledge_base_defaults={
                "answer_style": "summary",
                "insufficiency_policy": "partial_answer",
            },
        )
        self.assertIs(resolved.answer_style, AnswerStyle.CONCISE)
        self.assertIs(
            resolved.insufficiency_policy, InsufficiencyPolicy.PARTIAL_ANSWER
        )
        self.assertEqual(resolved.grounding_policy, "evidence_only")
        self.assertTrue(resolved.citation_required)
        self.assertEqual(resolved.citation_granularity, "claim_level")
        self.assertEqual(resolved.answer_task, "answer")
        self.assertEqual(resolved.policy_version, "p1")

    def test_resolver_rejects_unknown_dimensions_and_invalid_defaults(self) -> None:
        cases = (
            (
                {"grounding_policy": "model_knowledge_allowed"},
                {"answer_style": "concise", "insufficiency_policy": "refuse"},
            ),
            ({}, {"answer_style": "detailed", "insufficiency_policy": "refuse"}),
            ({}, {}),
        )
        for requested, defaults in cases:
            with self.subTest(requested=requested, defaults=defaults), self.assertRaises(
                AnswerPolicyNotSupportedError
            ):
                resolve_p1_policy(
                    requested_policy=requested,
                    knowledge_base_defaults=defaults,
                )

    def test_public_request_normalizes_content_and_rejects_policy_weakening(self) -> None:
        request = ChatRunCreate.model_validate(
            {
                "session_id": "01900000-0000-7000-8000-000000000101",
                "knowledge_base_id": "01900000-0000-7000-8000-000000000102",
                "message": "  查询 RUN-ORD-14  ",
                "answer_policy": {
                    "answer_style": "summary",
                    "insufficiency_policy": "partial_answer",
                },
                "retrieval": {"mode": "vector", "top_k": 8},
            }
        )
        self.assertEqual(request.message, "查询 RUN-ORD-14")
        self.assertIs(request.answer_policy.answer_style, AnswerStyle.SUMMARY)

        for field_name in (
            "grounding_policy",
            "citation_required",
            "citation_granularity",
            "answer_task",
            "policy_version",
        ):
            with self.subTest(field_name=field_name), self.assertRaises(
                ValidationError
            ) as captured:
                ChatRunCreate.model_validate(
                    {
                        "session_id": "01900000-0000-7000-8000-000000000101",
                        "knowledge_base_id": "01900000-0000-7000-8000-000000000102",
                        "message": "question",
                        "answer_policy": {field_name: "client-controlled"},
                    }
                )
            self.assertEqual(
                captured.exception.errors()[0]["type"],
                "answer_policy_not_supported",
            )

    def test_model_snapshot_excludes_url_key_and_runtime_controls(self) -> None:
        settings = SimpleNamespace(
            provider_identity="provider",
            logical_endpoint_identity="logical-endpoint",
            model="requested",
            resolved_model="resolved",
            model_version="version",
            temperature=0.1,
            max_tokens=2048,
            structured_output_mode="json_object",
            configuration_fingerprint="sha256:configuration",
            capability_fingerprint="sha256:capability",
            base_url="https://secret-host.example/v1",
            api_key="secret",
            timeout_seconds=30,
            max_retries=2,
        )
        snapshot = chat_model_configuration(settings)
        self.assertEqual(snapshot["resolved_model"], "resolved")
        self.assertEqual(snapshot["temperature"], 0.1)
        self.assertEqual(snapshot["max_tokens"], 2048)
        self.assertNotIn("base_url", snapshot)
        self.assertNotIn("api_key", snapshot)
        self.assertNotIn("timeout_seconds", snapshot)
        self.assertNotIn("max_retries", snapshot)


class ChatCreationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_creation_locks_session_and_freezes_completed_turns(self) -> None:
        workspace_id = uuid4()
        kb_id = uuid4()
        session_id = uuid4()
        turns = tuple(
            ConversationTurn(uuid4(), f"user {index}", uuid4(), f"assistant {index}")
            for index in range(2)
        )
        chat = _ChatRepository(kb_id=kb_id, turns=tuple(reversed(turns)))
        service = ChatService(
            _Factory(workspace_id, chat, kb_id),
            SingleWorkspaceAccessPolicy(workspace_id),
            model_configuration={"resolved_model": "fixed-model"},
        )

        created = await service.create_run(
            AuthContext("principal", "client", workspace_id),
            uuid4(),
            session_id=session_id,
            kb_id=kb_id,
            message="What about it?",
            answer_style=None,
            insufficiency_policy=None,
            retrieval_mode="vector",
            top_k=3,
        )

        snapshot = hydrate_conversation_context(created["conversation_context"])
        self.assertEqual(snapshot.turns, turns)
        self.assertIsNone(created["contextualized_query"])
        self.assertEqual(chat.events[:3], ["idempotency", "lock_session", "busy"])

    async def test_busy_session_is_rejected_before_history_or_insert(self) -> None:
        workspace_id = uuid4()
        kb_id = uuid4()
        chat = _ChatRepository(kb_id=kb_id, busy=True)
        service = ChatService(
            _Factory(workspace_id, chat, kb_id),
            SingleWorkspaceAccessPolicy(workspace_id),
            model_configuration={"resolved_model": "fixed-model"},
        )

        with self.assertRaises(ChatSessionBusyError):
            await service.create_run(
                AuthContext("principal", "client", workspace_id),
                uuid4(),
                session_id=uuid4(),
                kb_id=kb_id,
                message="question",
                answer_style=None,
                insufficiency_policy=None,
                retrieval_mode="vector",
                top_k=3,
            )

        self.assertNotIn("history", chat.events)
        self.assertNotIn("create", chat.events)


class _ChatRepository:
    def __init__(self, *, kb_id, turns=(), busy=False) -> None:
        self.kb_id = kb_id
        self.turns = turns
        self.busy = busy
        self.events = []

    async def lock_idempotency(self, scope):
        del scope
        self.events.append("idempotency")

    async def get_run_by_scope(self, scope):
        del scope
        return None

    async def lock_session(self, session_id, *, principal_id):
        del principal_id
        self.events.append("lock_session")
        return SimpleNamespace(id=session_id, kb_id=self.kb_id)

    async def has_nonterminal_run(self, session_id):
        del session_id
        self.events.append("busy")
        return self.busy

    async def list_completed_turns(self, **values):
        del values
        self.events.append("history")
        return self.turns

    async def create_run(self, **values):
        self.events.append("create")
        return values


class _KnowledgeBases:
    def __init__(self, kb_id) -> None:
        self.kb_id = kb_id

    async def get(self, kb_id):
        if kb_id != self.kb_id:
            return None
        return SimpleNamespace(
            active_index_revision_id=uuid4(),
            answer_policy_defaults={
                "answer_style": "concise",
                "insufficiency_policy": "refuse",
            },
        )


class _UnitOfWork:
    def __init__(self, workspace_id, chat, kb_id) -> None:
        self.workspace_id = workspace_id
        self.chat = chat
        self.knowledge_bases = _KnowledgeBases(kb_id)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def commit(self):
        return None


class _Factory:
    def __init__(self, workspace_id, chat, kb_id) -> None:
        self.values = (workspace_id, chat, kb_id)

    def __call__(self, **kwargs):
        del kwargs
        return _UnitOfWork(*self.values)


if __name__ == "__main__":
    unittest.main()
