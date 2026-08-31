from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.domain import (
    ChatSessionBusyError,
    ConversationTurn,
    ErrorCode,
    Page,
    RerankMode,
    RetrievalStrategy,
    RetrievalExecutionError,
)
from rag_kb.memory import hydrate_conversation_context
from rag_kb.repositories.sqlalchemy_chat import SqlAlchemyChatRepository
from rag_kb.schemas import ChatRunCreate
from rag_kb.services.chat import (
    ChatService,
    _chat_profile_configuration,
    chat_model_configuration,
)
from rag_kb.retrieval.profile import (
    HYBRID_PROFILE_VERSION,
    exact_profile,
)


class ChatCreationContractTests(unittest.TestCase):
    def test_request_normalizes_content_and_rejects_retired_policy(self) -> None:
        payload = {
            "session_id": str(uuid4()),
            "knowledge_base_id": str(uuid4()),
            "message": "  查询 RUN-ORD-14  ",
        }
        self.assertEqual(ChatRunCreate.model_validate(payload).message, "查询 RUN-ORD-14")
        self.assertNotIn("answer_policy", ChatRunCreate.model_json_schema()["properties"])
        for old in ({}, {"answer_style": "summary"}, {"insufficiency_policy": "refuse"}):
            with self.subTest(old=old), self.assertRaises(ValidationError) as captured:
                ChatRunCreate.model_validate({**payload, "answer_policy": old})
            self.assertEqual(captured.exception.errors()[0]["type"], "extra_forbidden")

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
            thinking_enabled=False,
            vision_enabled=True,
            max_visual_images=2,
            max_visual_image_bytes=5_242_880,
            max_visual_total_bytes=12_582_912,
            max_visual_pixels=16_000_000,
            visual_media_profile="jpeg_png_webp_v1",
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
        self.assertTrue(snapshot["vision_enabled"])
        self.assertEqual(snapshot["max_visual_images"], 2)
        self.assertNotIn("base_url", snapshot)
        self.assertNotIn("api_key", snapshot)
        self.assertNotIn("timeout_seconds", snapshot)
        self.assertNotIn("max_retries", snapshot)

    def test_user_profile_uses_visual_safety_defaults_without_legacy_settings(
        self,
    ) -> None:
        bundle = SimpleNamespace(
            profile=SimpleNamespace(id=uuid4(), name="Mimo v2.5"),
            current_revision=SimpleNamespace(
                id=uuid4(),
                revision=1,
                model="mimo-v2.5",
                configuration={},
                configuration_fingerprint="sha256:configuration",
                capability_fingerprint="sha256:capability",
            ),
            provider=SimpleNamespace(id=uuid4(), name="OpenCode Go"),
            provider_revision=SimpleNamespace(id=uuid4()),
        )

        snapshot = _chat_profile_configuration(bundle, {})

        self.assertEqual(snapshot["max_visual_images"], 2)
        self.assertEqual(snapshot["max_visual_image_bytes"], 5_242_880)
        self.assertEqual(snapshot["max_visual_total_bytes"], 12_582_912)
        self.assertEqual(snapshot["max_visual_pixels"], 16_000_000)
        self.assertEqual(snapshot["visual_media_profile"], "jpeg_png_webp_v1")


class ChatCreationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_reranker_is_frozen_for_native_agent(self) -> None:
        workspace_id = uuid4()
        kb_id = uuid4()
        chat = _ChatRepository(kb_id=kb_id)
        service = ChatService(
            _Factory(workspace_id, chat, kb_id),
            SingleWorkspaceAccessPolicy(workspace_id),
            model_configuration={"resolved_model": "fixed-model"},
            retrieval_profile_factory=_profile_factory,
        )
        created = await service.create_run(
            AuthContext("principal", "client", workspace_id),
            uuid4(),
            session_id=uuid4(),
            kb_id=kb_id,
            message="query",
            retrieval_mode="vector",
            top_k=5,
            rerank_mode=RerankMode.LOCAL_MINILM_V1,
        )
        self.assertEqual(
            created["retrieval_strategy"]["rerank_mode"],
            "local_minilm_v1",
        )

    async def test_disabled_hybrid_run_uses_capability_error(self) -> None:
        workspace_id = uuid4()
        kb_id = uuid4()
        chat = _ChatRepository(kb_id=kb_id)
        service = ChatService(
            _Factory(workspace_id, chat, kb_id),
            SingleWorkspaceAccessPolicy(workspace_id),
            model_configuration={"resolved_model": "fixed-model"},
            retrieval_profile_factory=_profile_factory,
        )

        with self.assertRaises(RetrievalExecutionError) as failure:
            await service.create_run(
                AuthContext("principal", "client", workspace_id),
                uuid4(),
                session_id=uuid4(),
                kb_id=kb_id,
                message="query",
                retrieval_mode="hybrid",
                top_k=5,
            )
        self.assertEqual(failure.exception.code, ErrorCode.CAPABILITY_NOT_ENABLED)

    async def test_hybrid_run_freezes_only_the_selected_retrieval_preset(
        self,
    ) -> None:
        workspace_id = uuid4()
        kb_id = uuid4()
        chat = _ChatRepository(kb_id=kb_id)

        def profile_factory(strategy, top_k, rerank_mode):
            self.assertIs(strategy, RetrievalStrategy.HYBRID)
            return replace(
                exact_profile(top_k=top_k, rerank_mode=rerank_mode),
                profile_version=HYBRID_PROFILE_VERSION,
                strategy=RetrievalStrategy.HYBRID,
                lexical_analyzer_version="lexical_simple_cjk_bigram_v1",
                lexical_query_version="lexical_or_query_v1",
                dense_candidate_count=17,
                lexical_candidate_count=23,
                dense_weight_micros=1_000_000,
                lexical_weight_micros=1_000_000,
                min_rerank_score=0.45,
            )

        service = ChatService(
            _Factory(workspace_id, chat, kb_id),
            SingleWorkspaceAccessPolicy(workspace_id),
            model_configuration={"resolved_model": "fixed-model"},
            hybrid_enabled=True,
            retrieval_profile_factory=profile_factory,
        )

        created = await service.create_run(
            AuthContext("principal", "client", workspace_id),
            uuid4(),
            session_id=uuid4(),
            kb_id=kb_id,
            message="查询 ABC-42",
            retrieval_mode="hybrid",
            top_k=4,
            rerank_mode=RerankMode.CLASSIC,
        )

        snapshot = created["retrieval_strategy"]
        self.assertEqual(
            snapshot,
            {
                "profile_version": HYBRID_PROFILE_VERSION,
                "strategy": "hybrid",
                "top_k": 4,
                "rerank_mode": "classic",
            },
        )

    async def test_graph_run_freezes_classic_outer_profile_without_hybrid_capability(
        self,
    ) -> None:
        workspace_id = uuid4()
        kb_id = uuid4()
        chat = _ChatRepository(kb_id=kb_id)
        service = ChatService(
            _Factory(workspace_id, chat, kb_id),
            SingleWorkspaceAccessPolicy(workspace_id),
            model_configuration={"resolved_model": "fixed-model"},
            hybrid_enabled=False,
            retrieval_profile_factory=_profile_factory,
        )

        created = await service.create_run(
            AuthContext("principal", "client", workspace_id),
            uuid4(),
            session_id=uuid4(),
            kb_id=kb_id,
            message="Atlas Labs",
            retrieval_mode="graph",
            top_k=4,
            rerank_mode=RerankMode.CLASSIC,
        )

        self.assertEqual(
            created["retrieval_strategy"],
            {
                "profile_version": "graphiti_path_augmented_v3",
                "strategy": "hybrid",
                "top_k": 4,
                "rerank_mode": "classic",
                "augmentation": "graphiti_path_v3",
            },
        )

    async def test_auto_run_freezes_exact_vector_adaptive_profile(self) -> None:
        workspace_id = uuid4()
        kb_id = uuid4()
        chat = _ChatRepository(kb_id=kb_id)
        service = ChatService(
            _Factory(workspace_id, chat, kb_id),
            SingleWorkspaceAccessPolicy(workspace_id),
            model_configuration={"resolved_model": "fixed-model"},
            retrieval_profile_factory=_profile_factory,
        )

        created = await service.create_run(
            AuthContext("principal", "client", workspace_id),
            uuid4(),
            session_id=uuid4(),
            kb_id=kb_id,
            message="Atlas Labs",
            retrieval_mode="auto",
            top_k=8,
            rerank_mode=RerankMode.LOCAL_MINILM_V1,
        )

        self.assertEqual(
            created["retrieval_strategy"],
            {
                "profile_version": "adaptive_graphiti_v3",
                "strategy": "exact_vector",
                "top_k": 8,
                "rerank_mode": "local_minilm_v1",
                "router": "native_agent_graph_tool_v1",
                "augmentation": "graphiti_path_v3",
                "graph_edge_limit": 16,
                "graph_source_chunk_target": 12,
                "graph_source_chunk_limit": 16,
                "graph_call_timeout_seconds": 90,
            },
        )

    async def test_session_listing_filters_by_authorized_knowledge_base(self) -> None:
        workspace_id = uuid4()
        kb_id = uuid4()
        chat = _ChatRepository(kb_id=kb_id)
        service = ChatService(
            _Factory(workspace_id, chat, kb_id),
            SingleWorkspaceAccessPolicy(workspace_id),
            model_configuration={"resolved_model": "fixed-model"},
            retrieval_profile_factory=_profile_factory,
        )

        result = await service.list_sessions(
            AuthContext("principal", "client", workspace_id),
            limit=20,
            sort="-updated_at",
            after=None,
            kb_id=kb_id,
        )

        self.assertEqual(result.items, ())
        self.assertEqual(chat.list_kb_id, kb_id)

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
            retrieval_profile_factory=_profile_factory,
        )

        created = await service.create_run(
            AuthContext("principal", "client", workspace_id),
            uuid4(),
            session_id=session_id,
            kb_id=kb_id,
            message="What about it?",
            retrieval_mode="vector",
            top_k=3,
        )

        self.assertNotIn("requested_policy", created)
        self.assertNotIn("effective_policy", created)
        snapshot = hydrate_conversation_context(created["conversation_context"])
        self.assertEqual(snapshot.turns, turns)
        self.assertEqual(created["contextualized_query"]["status"], "original")
        self.assertEqual(
            created["contextualized_query"]["standalone_query"], "What about it?"
        )
        self.assertEqual(
            created["contextualized_query"]["context_hash"], snapshot.content_hash
        )
        self.assertEqual(chat.events[:3], ["idempotency", "lock_session", "busy"])

    async def test_busy_session_is_rejected_before_history_or_insert(self) -> None:
        workspace_id = uuid4()
        kb_id = uuid4()
        chat = _ChatRepository(kb_id=kb_id, busy=True)
        service = ChatService(
            _Factory(workspace_id, chat, kb_id),
            SingleWorkspaceAccessPolicy(workspace_id),
            model_configuration={"resolved_model": "fixed-model"},
            retrieval_profile_factory=_profile_factory,
        )

        with self.assertRaises(ChatSessionBusyError):
            await service.create_run(
                AuthContext("principal", "client", workspace_id),
                uuid4(),
                session_id=uuid4(),
                kb_id=kb_id,
                message="question",
                retrieval_mode="vector",
                top_k=3,
            )

        self.assertNotIn("history", chat.events)
        self.assertNotIn("create", chat.events)


class ChatHistoryRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_turn_query_compiles_against_persisted_relationships(
        self,
    ) -> None:
        session = _CompileOnlySession()
        repository = SqlAlchemyChatRepository(session, uuid4())

        turns = await repository.list_completed_turns(
            session_id=uuid4(),
            principal_id="principal",
            kb_id=uuid4(),
            limit=7,
        )

        self.assertEqual(turns, ())
        self.assertIn(".chat_run_id = chat_run.id", session.sql)
        self.assertNotIn("chat_run.assistant_message_id", session.sql)
        self.assertIn("status", session.sql)


class _ChatRepository:
    def __init__(self, *, kb_id, turns=(), busy=False) -> None:
        self.kb_id = kb_id
        self.turns = turns
        self.busy = busy
        self.events = []
        self.list_kb_id = None

    async def list_sessions(self, **values):
        self.list_kb_id = values["kb_id"]
        return Page(items=())

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


class _CompileOnlySession:
    def __init__(self) -> None:
        self.sql = ""

    async def execute(self, statement):
        self.sql = str(statement.compile(dialect=postgresql.dialect()))
        return SimpleNamespace(all=lambda: [])


def _profile_factory(strategy, top_k, rerank_mode):
    if strategy is not RetrievalStrategy.EXACT_VECTOR:
        raise ValueError("test factory only supports exact retrieval")
    return exact_profile(top_k=top_k, rerank_mode=rerank_mode)


class _KnowledgeBases:
    def __init__(self, kb_id) -> None:
        self.kb_id = kb_id

    async def get(self, kb_id):
        if kb_id != self.kb_id:
            return None
        return SimpleNamespace(
            active_index_revision_id=uuid4(),
            answer_policy_defaults={"answer_style": "retired-value", "policy_version": "unknown"},
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
