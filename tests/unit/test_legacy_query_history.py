from __future__ import annotations

import copy
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from rag_kb.domain import ChatRunLease
from rag_kb.memory import empty_conversation_context, serialize_conversation_context
from rag_kb.repositories.sqlalchemy_chat import (
    SqlAlchemyChatRepository,
    _contextualization_calls,
    _historical_query_timing,
    _merge_usage,
)


class LegacyQueryHistoryTests(unittest.TestCase):
    def test_new_runs_have_no_query_rewrite_usage_or_diagnostics(self) -> None:
        run = SimpleNamespace(contextualized_query=None)
        self.assertEqual(_contextualization_calls(run), {})
        self.assertEqual(_historical_query_timing(run), {})

    def test_legacy_calls_keep_identity_and_usage_without_version_or_roundtrip_checks(self) -> None:
        call = {
            "operation": "contextualize_query", "model": "historical-model",
            "provider_request_id": "old-request", "usage": {"total_tokens": 12},
        }
        snapshot = {
            "version": "retired-version", "origin_attempt": 2,
            "status": "contextualized", "rewrite_source": "repair",
            "model_calls": [call], "unused_legacy_field": "kept",
        }
        before = copy.deepcopy(snapshot)
        run = SimpleNamespace(contextualized_query=snapshot)
        calls = _contextualization_calls(run)
        self.assertEqual(calls, {"2:1:contextualize_query": {
            "attempt": 2, "sequence": 1, **call,
        }})
        usage = _merge_usage(None, calls)
        self.assertEqual(usage["totals"], {"total_tokens": 12})
        self.assertEqual(_merge_usage(usage, calls), usage)
        self.assertEqual(_historical_query_timing(run)["query_rewrite"]["source"], "repair")
        self.assertEqual(snapshot, before)

    def test_invalid_legacy_usage_does_not_block_or_corrupt_current_accounting(self) -> None:
        valid = {"operation": "contextualize_query", "model": "old", "usage": {"total_tokens": 3}}
        for bad_usage in ({"total_tokens": -1}, {"total_tokens": True}, {"total_tokens": "3"}, None):
            with self.subTest(usage=bad_usage):
                run = SimpleNamespace(contextualized_query={
                    "origin_attempt": 1,
                    "model_calls": [{**valid, "usage": bad_usage}, valid],
                })
                calls = _contextualization_calls(run)
                self.assertEqual(list(calls), ["1:2:contextualize_query"])
                self.assertEqual(_merge_usage(None, calls)["totals"], {"total_tokens": 3})
        for snapshot in ({}, {"origin_attempt": True, "model_calls": [valid]}, {"model_calls": "bad"}):
            self.assertEqual(_contextualization_calls(SimpleNamespace(contextualized_query=snapshot)), {})


class QueryExecutionIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_ignores_retired_query_but_still_validates_session_context(self) -> None:
        run_id, workspace_id = uuid4(), uuid4()
        lease = ChatRunLease(run_id, workspace_id, 1, datetime.now(UTC))
        run = SimpleNamespace(
            id=run_id, workspace_id=workspace_id, kb_id=uuid4(), session_id=uuid4(),
            user_message_id=uuid4(), index_revision_id=uuid4(), principal_id="local",
            client_id="web", attempt=1, retrieval_strategy={}, model_configuration={},
            agent_configuration={},
            conversation_context=serialize_conversation_context(empty_conversation_context()),
            contextualized_query={"version": "unknown", "standalone_query": "must not execute"},
        )
        run.knowledge_bases = [SimpleNamespace(kb_id=run.kb_id, name="legacy", description="", index_revision_id=run.index_revision_id, retrieval_strategy={}, graph_build_id=None, status="ready")]
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(
            one_or_none=lambda: (run, SimpleNamespace(content="Current user question"), SimpleNamespace(id=uuid4())),
        )))
        repo = SqlAlchemyChatRepository(session, workspace_id)
        context = await repo.load_execution_context(lease)
        self.assertIsNotNone(context)
        self.assertEqual(context.query, "Current user question")
        self.assertFalse(hasattr(context, "contextualized_query"))
        run.conversation_context = {"version": "corrupt"}
        self.assertIsNone(await repo.load_execution_context(lease))
