from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import os
import unittest
from uuid import UUID, uuid4

import asyncpg

from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.document_processing import index_profile
from rag_kb.domain import (
    AnswerControlReason,
    AnswerDraftCandidate,
    AnswerDraftSource,
    AnswerOutcome,
    AnswerStyle,
    AnswerValidationRecord,
    ChatAnsweringState,
    ChatExecutionCommand,
    ChatModelCallRecord,
    ChatModelOperation,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ChatTerminalSuccessCommand,
    ChatTerminalWriteStatus,
    EmbeddingSpaceDefinition,
    ErrorCode,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceEnvelope,
    EvidencePack,
    IdempotencyKeyReusedError,
    IndexProfileDefinition,
    InsufficiencyPolicy,
    RenderedAnswer,
    RenderedCitation,
    ResourceNotFoundError,
    ResourceStateConflictError,
    RetrievalStrategy,
    ValidatedAnswer,
)
from rag_kb.services import (
    ChatExecutionContextLoader,
    ChatFailureSettlementService,
    ChatResultPersistenceStep,
    ChatRunCoordinator,
    ChatService,
    ChatTerminalWatcher,
    KnowledgeBaseService,
)
from rag_kb.scheduling import ChatRunScheduler, RetryPolicy
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory
from rag_kb.uow import execute_in_transaction


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

    async def test_kb_defaults_request_precedence_and_frozen_replay(self) -> None:
        kb = await self.knowledge_bases.create(
            self.context,
            uuid4(),
            name="policy-defaults",
            retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
            answer_policy_defaults={
                "answer_style": "summary",
                "insufficiency_policy": "partial_answer",
            },
        )
        session = await self.chat.create_session(
            self.context, kb_id=kb.id, title="Policy precedence"
        )
        key = uuid4()
        from_defaults = await self.chat.create_run(
            self.context,
            key,
            session_id=session.id,
            kb_id=kb.id,
            message="Use knowledge-base policy defaults",
            answer_style=None,
            insufficiency_policy=None,
            retrieval_mode="vector",
            top_k=10,
        )
        self.assertEqual(from_defaults.effective_policy["answer_style"], "summary")
        self.assertEqual(
            from_defaults.effective_policy["insufficiency_policy"],
            "partial_answer",
        )

        await self.knowledge_bases.update(
            self.context,
            uuid4(),
            kb.id,
            name=None,
            retrieval_defaults=None,
            answer_policy_defaults={
                "answer_style": "concise",
                "insufficiency_policy": "refuse",
            },
        )
        replay, concurrent_replay = await asyncio.gather(
            self.chat.create_run(
                self.context,
                key,
                session_id=session.id,
                kb_id=kb.id,
                message="Use knowledge-base policy defaults",
                answer_style=None,
                insufficiency_policy=None,
                retrieval_mode="vector",
                top_k=10,
            ),
            self.chat.create_run(
                self.context,
                key,
                session_id=session.id,
                kb_id=kb.id,
                message="Use knowledge-base policy defaults",
                answer_style=None,
                insufficiency_policy=None,
                retrieval_mode="vector",
                top_k=10,
            ),
        )
        self.assertEqual(replay.id, from_defaults.id)
        self.assertEqual(concurrent_replay.id, from_defaults.id)
        self.assertEqual(replay.effective_policy, from_defaults.effective_policy)

        request_override = await self.chat.create_run(
            self.context,
            uuid4(),
            session_id=session.id,
            kb_id=kb.id,
            message="Override only the answer style",
            answer_style=AnswerStyle.SUMMARY,
            insufficiency_policy=None,
            retrieval_mode="vector",
            top_k=10,
        )
        self.assertEqual(request_override.effective_policy["answer_style"], "summary")
        self.assertEqual(
            request_override.effective_policy["insufficiency_policy"], "refuse"
        )
        self.assertEqual(
            request_override.effective_policy["grounding_policy"], "evidence_only"
        )
        self.assertTrue(request_override.effective_policy["citation_required"])

    async def test_competing_claims_and_lease_cas_load_one_frozen_context(self) -> None:
        kb = await self._create_kb("claimable")
        session = await self.chat.create_session(
            self.context, kb_id=kb.id, title="Claim behavior"
        )
        run = await self._create_run(session.id, kb.id, uuid4())
        coordinator = ChatRunCoordinator(self.factory)
        observed_at = datetime.now(UTC)

        claims = await asyncio.gather(
            coordinator.claim(
                worker_id="worker-a", observed_at=observed_at, max_attempts=3
            ),
            coordinator.claim(
                worker_id="worker-b", observed_at=observed_at, max_attempts=3
            ),
        )
        leases = [lease for lease in claims if lease is not None]
        self.assertEqual(len(leases), 1)
        lease = leases[0]
        self.assertEqual(lease.run_id, run.id)
        self.assertEqual(lease.attempt, 1)

        execution_context = await ChatExecutionContextLoader(self.factory).load(
            ChatExecutionCommand(lease)
        )
        self.assertEqual(execution_context.query, "How should RUN-ORD-14 be handled?")
        self.assertEqual(execution_context.index_revision_id, run.index_revision_id)
        self.assertEqual(execution_context.effective_policy, run.effective_policy)
        self.assertEqual(execution_context.retrieval_strategy, run.retrieval_strategy)
        self.assertEqual(execution_context.model_configuration, run.model_configuration)
        self.assertEqual(self.database.engine.pool.checkedout(), 0)

        heartbeat_at = observed_at + timedelta(seconds=1)
        self.assertTrue(
            await coordinator.heartbeat(lease, observed_at=heartbeat_at)
        )
        self.assertFalse(
            await coordinator.heartbeat(
                replace(lease, attempt=2),
                observed_at=heartbeat_at + timedelta(seconds=1),
            )
        )

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            persisted = await connection.fetchrow(
                """
                SELECT status, attempt, claimed_by, heartbeat_at
                FROM chat_run WHERE id = $1
                """,
                run.id,
            )
        finally:
            await connection.close()
        self.assertEqual(persisted["status"], "running")
        self.assertEqual(persisted["attempt"], 1)
        self.assertEqual(persisted["claimed_by"], lease.claimed_by)
        self.assertEqual(persisted["heartbeat_at"], heartbeat_at)

    async def test_terminal_success_is_atomic_and_lost_response_replay_is_exact(
        self,
    ) -> None:
        kb = await self._create_kb("terminal-success")
        session = await self.chat.create_session(self.context, kb_id=kb.id, title=None)
        run = await self._create_run(session.id, kb.id, uuid4())
        observed_at = datetime.now(UTC)
        lease = await ChatRunCoordinator(self.factory).claim(
            worker_id="worker-success", observed_at=observed_at, max_attempts=3
        )
        self.assertIsNotNone(lease)
        context = await ChatExecutionContextLoader(self.factory).load(
            ChatExecutionCommand(lease)
        )
        state = _refusal_state(context)
        persister = ChatResultPersistenceStep(
            self.factory, clock=lambda: observed_at + timedelta(seconds=2)
        )

        first = await persister.run(state)
        replay = await persister.run(state)

        self.assertIs(first, state)
        self.assertIs(replay, state)
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            persisted = await connection.fetchrow(
                """
                SELECT r.status, r.claimed_by, r.completed_at, r.usage, r.timing,
                       m.assistant_status, m.content,
                       (SELECT count(*) FROM citation c
                        WHERE c.assistant_message_id = m.id) AS citations
                FROM chat_run r
                JOIN chat_message m ON m.chat_run_id = r.id
                WHERE r.id = $1
                """,
                run.id,
            )
        finally:
            await connection.close()
        self.assertEqual(persisted["status"], "completed")
        self.assertEqual(persisted["assistant_status"], "completed")
        self.assertEqual(persisted["content"], "无法基于当前证据回答。")
        self.assertIsNone(persisted["claimed_by"])
        self.assertEqual(persisted["citations"], 0)
        usage = json.loads(persisted["usage"])
        timing = json.loads(persisted["timing"])
        self.assertEqual(len(usage["calls"]), 1)
        self.assertEqual(usage["totals"]["input_tokens"], 4)
        self.assertEqual(
            timing["attempts"]["1"]["validation"]["safe_fallback"],
            False,
        )

        stale = await ChatFailureSettlementService(
            self.factory,
            max_attempts=3,
            base_delay_seconds=1,
            max_delay_seconds=4,
            clock=lambda: observed_at + timedelta(seconds=3),
        ).settle(
            lease,
            ChatPipelineExecutionError(
                ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                phase=ChatPipelinePhase.GENERATE_OR_REFUSE,
            ),
        )
        self.assertIs(stale, ChatTerminalWriteStatus.STALE)

    async def test_terminal_write_rolls_back_before_commit(self) -> None:
        kb = await self._create_kb("terminal-rollback")
        session = await self.chat.create_session(self.context, kb_id=kb.id, title=None)
        run = await self._create_run(session.id, kb.id, uuid4())
        observed_at = datetime.now(UTC)
        lease = await ChatRunCoordinator(self.factory).claim(
            worker_id="worker-rollback", observed_at=observed_at, max_attempts=3
        )
        context = await ChatExecutionContextLoader(self.factory).load(
            ChatExecutionCommand(lease)
        )
        state = _refusal_state(context)
        command = ChatTerminalSuccessCommand(
            lease=lease,
            assistant_message_id=context.assistant_message_id,
            rendered=state.answering.rendered,
            validation=state.answering.validation,
            model_calls=state.answering.model_calls,
            finished_at=observed_at + timedelta(seconds=1),
        )

        with self.assertRaisesRegex(RuntimeError, "before commit"):
            async with self.factory() as uow:
                self.assertIs(
                    await uow.chat.complete_owned_run(command),
                    ChatTerminalWriteStatus.APPLIED,
                )
                raise RuntimeError("before commit")

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            persisted = await connection.fetchrow(
                """
                SELECT r.status, r.claimed_by, r.usage, r.timing,
                       m.assistant_status, m.content,
                       (SELECT count(*) FROM citation c
                        WHERE c.assistant_message_id = m.id) AS citations
                FROM chat_run r
                JOIN chat_message m ON m.chat_run_id = r.id
                WHERE r.id = $1
                """,
                run.id,
            )
        finally:
            await connection.close()
        self.assertEqual(persisted["status"], "running")
        self.assertEqual(persisted["claimed_by"], lease.claimed_by)
        self.assertEqual(persisted["assistant_status"], "generating")
        self.assertEqual(persisted["content"], "")
        self.assertIsNone(persisted["usage"])
        self.assertIsNone(persisted["timing"])
        self.assertEqual(persisted["citations"], 0)

    async def test_terminal_success_persists_ordered_citation_snapshots(self) -> None:
        kb = await self._create_kb("terminal-citations")
        session = await self.chat.create_session(self.context, kb_id=kb.id, title=None)
        run = await self._create_run(session.id, kb.id, uuid4())
        document_id, version_id, chunk_ids = await self._seed_citation_chunks(kb)
        observed_at = datetime.now(UTC)
        lease = await ChatRunCoordinator(self.factory).claim(
            worker_id="worker-citations", observed_at=observed_at, max_attempts=3
        )
        context = await ChatExecutionContextLoader(self.factory).load(
            ChatExecutionCommand(lease)
        )
        rendered = RenderedAnswer(
            outcome=AnswerOutcome.ANSWERED,
            content="第二段证据 [1]\n第一段证据 [2]",
            citations=(
                RenderedCitation(
                    ordinal=0,
                    citation_id="cite_2",
                    index_chunk_id=chunk_ids[1],
                    document_id=document_id,
                    document_version_id=version_id,
                    quoted_text="第二段证据",
                    source_location={"paragraph": 2},
                    score=0.8,
                ),
                RenderedCitation(
                    ordinal=1,
                    citation_id="cite_1",
                    index_chunk_id=chunk_ids[0],
                    document_id=document_id,
                    document_version_id=version_id,
                    quoted_text="第一段证据",
                    source_location={"paragraph": 1},
                    score=0.9,
                ),
            ),
        )
        command = ChatTerminalSuccessCommand(
            lease=lease,
            assistant_message_id=context.assistant_message_id,
            rendered=rendered,
            validation=AnswerValidationRecord(initial_issues=()),
            model_calls=(_model_call("request-citations"),),
            finished_at=observed_at + timedelta(seconds=1),
        )

        async def complete(uow):
            return await uow.chat.complete_owned_run(command)

        self.assertIs(
            await execute_in_transaction(self.factory, complete),
            ChatTerminalWriteStatus.APPLIED,
        )
        self.assertIs(
            await execute_in_transaction(self.factory, complete),
            ChatTerminalWriteStatus.IDEMPOTENT,
        )
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            rows = await connection.fetch(
                """
                SELECT ordinal, index_chunk_id, document_id_snapshot,
                       document_version_id_snapshot, quoted_text,
                       source_location, score
                FROM citation
                WHERE assistant_message_id = $1
                ORDER BY ordinal
                """,
                context.assistant_message_id,
            )
        finally:
            await connection.close()
        self.assertEqual([row["ordinal"] for row in rows], [0, 1])
        self.assertEqual(
            [row["index_chunk_id"] for row in rows],
            [chunk_ids[1], chunk_ids[0]],
        )
        self.assertEqual(
            [row["quoted_text"] for row in rows], ["第二段证据", "第一段证据"]
        )
        self.assertEqual({row["document_id_snapshot"] for row in rows}, {document_id})
        self.assertEqual(
            {row["document_version_id_snapshot"] for row in rows}, {version_id}
        )
        authoritative = await self.chat.get_run(self.context, run.id)
        self.assertEqual(
            [item.index_chunk_id for item in authoritative.citations],
            [chunk_ids[1], chunk_ids[0]],
        )
        self.assertEqual(
            [item.quoted_text for item in authoritative.citations],
            ["第二段证据", "第一段证据"],
        )

    async def test_terminal_watcher_releases_database_connection_while_waiting(
        self,
    ) -> None:
        kb = await self._create_kb("terminal-watcher")
        session = await self.chat.create_session(self.context, kb_id=kb.id, title=None)
        created = await self._create_run(session.id, kb.id, uuid4())
        initial = await self.chat.get_run(self.context, created.id)
        observed_at = datetime.now(UTC)

        async def finish_during_wait(delay: float) -> None:
            self.assertGreater(delay, 0)
            self.assertEqual(self.database.engine.pool.checkedout(), 0)
            lease = await ChatRunCoordinator(self.factory).claim(
                worker_id="worker-watcher",
                observed_at=observed_at,
                max_attempts=3,
            )
            await ChatFailureSettlementService(
                self.factory,
                max_attempts=3,
                base_delay_seconds=1,
                max_delay_seconds=4,
                clock=lambda: observed_at + timedelta(seconds=1),
            ).settle(
                lease,
                ChatPipelineExecutionError(
                    ErrorCode.CHAT_REVISION_MISMATCH,
                    phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                ),
            )
            self.assertEqual(self.database.engine.pool.checkedout(), 0)

        async def connected() -> bool:
            return False

        watcher = ChatTerminalWatcher(
            self.chat,
            poll_interval_seconds=0.01,
            jitter_ratio=0,
            max_duration_seconds=1,
            sleep=finish_during_wait,
        )
        results = [
            item
            async for item in watcher.watch(
                self.context,
                created.id,
                initial=initial,
                disconnected=connected,
            )
        ]

        self.assertEqual([item.status for item in results], ["failed"])
        self.assertEqual(self.database.engine.pool.checkedout(), 0)

    async def test_chat_scheduler_claims_and_completes_independently(self) -> None:
        kb = await self._create_kb("scheduled-chat")
        session = await self.chat.create_session(self.context, kb_id=kb.id, title=None)
        created = await self._create_run(session.id, kb.id, uuid4())
        coordinator = _CountingCoordinator(ChatRunCoordinator(self.factory))
        pipeline = _PersistingPipeline(self.factory)
        scheduler = ChatRunScheduler(
            coordinator,
            pipeline,
            ChatFailureSettlementService(
                self.factory,
                max_attempts=3,
                base_delay_seconds=1,
                max_delay_seconds=4,
            ),
            worker_id="worker-scheduled-chat",
            heartbeat_interval_seconds=0.005,
            stale_after_seconds=1,
            retry_policy=RetryPolicy(3, 1, 4),
            reconciliation_batch_size=10,
        )

        self.assertIsNotNone(await scheduler.oldest_claimable_at())
        lease = await scheduler.claim_once()
        self.assertIsNotNone(lease)
        await scheduler.execute(lease, asyncio.Event())

        terminal = await self.chat.get_run(self.context, created.id)
        self.assertEqual(terminal.status, "completed")
        self.assertEqual(terminal.assistant_status, "completed")
        self.assertEqual(terminal.assistant_content, "无法基于当前证据回答。")
        self.assertGreaterEqual(coordinator.heartbeats, 1)
        self.assertEqual(self.database.engine.pool.checkedout(), 0)

    async def test_stale_chat_reconciliation_requeues_then_exhausts(self) -> None:
        kb = await self._create_kb("stale-chat")
        session = await self.chat.create_session(self.context, kb_id=kb.id, title=None)
        created = await self._create_run(session.id, kb.id, uuid4())
        coordinator = ChatRunCoordinator(self.factory)
        started_at = datetime.now(UTC)
        first = await coordinator.claim(
            worker_id="worker-stale",
            observed_at=started_at,
            max_attempts=2,
        )
        observed_at = started_at + timedelta(seconds=10)

        first_result = await coordinator.reconcile_stale(
            stale_before=observed_at - timedelta(seconds=1),
            observed_at=observed_at,
            max_attempts=2,
            retry_at_by_attempt=(
                observed_at + timedelta(seconds=1),
                observed_at + timedelta(seconds=2),
            ),
            limit=10,
        )
        self.assertEqual((first_result.requeued, first_result.failed), (1, 0))
        second = await coordinator.claim(
            worker_id="worker-stale",
            observed_at=observed_at + timedelta(seconds=2),
            max_attempts=2,
        )
        self.assertEqual(second.attempt, 2)
        finished_at = observed_at + timedelta(seconds=5)
        second_result = await coordinator.reconcile_stale(
            stale_before=finished_at,
            observed_at=finished_at,
            max_attempts=2,
            retry_at_by_attempt=(
                finished_at + timedelta(seconds=1),
                finished_at + timedelta(seconds=2),
            ),
            limit=10,
        )

        self.assertEqual((second_result.requeued, second_result.failed), (0, 1))
        terminal = await self.chat.get_run(self.context, created.id)
        self.assertEqual(terminal.status, "failed")
        self.assertEqual(terminal.assistant_status, "failed")
        self.assertEqual(terminal.error_code, "CHAT_STALE_WORKER")
        self.assertTrue(terminal.error_retryable)
        self.assertEqual(
            set(terminal.timing["attempts"]),
            {"1", "2"},
        )
        self.assertEqual(self.database.engine.pool.checkedout(), 0)

    async def test_retry_then_exhausted_failure_accumulates_attempt_ledgers(
        self,
    ) -> None:
        kb = await self._create_kb("terminal-failure")
        session = await self.chat.create_session(self.context, kb_id=kb.id, title=None)
        run = await self._create_run(session.id, kb.id, uuid4())
        observed_at = datetime.now(UTC)
        coordinator = ChatRunCoordinator(self.factory)
        first_lease = await coordinator.claim(
            worker_id="worker-failure", observed_at=observed_at, max_attempts=2
        )
        service = ChatFailureSettlementService(
            self.factory,
            max_attempts=2,
            base_delay_seconds=1,
            max_delay_seconds=4,
            clock=lambda: observed_at + timedelta(seconds=1),
        )
        first_error = ChatPipelineExecutionError(
            ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
            phase=ChatPipelinePhase.ASSESS_EVIDENCE,
            diagnostic={"http_status": 503, "raw_content": "must not persist"},
            model_calls=(_model_call("request-attempt-1"),),
        )
        self.assertIs(
            await service.settle(first_lease, first_error),
            ChatTerminalWriteStatus.APPLIED,
        )
        self.assertIs(
            await service.settle(first_lease, first_error),
            ChatTerminalWriteStatus.IDEMPOTENT,
        )
        second_lease = await coordinator.claim(
            worker_id="worker-failure",
            observed_at=observed_at + timedelta(seconds=3),
            max_attempts=2,
        )
        self.assertEqual(second_lease.attempt, 2)
        exhausted = ChatFailureSettlementService(
            self.factory,
            max_attempts=2,
            base_delay_seconds=1,
            max_delay_seconds=4,
            clock=lambda: observed_at + timedelta(seconds=4),
        )
        self.assertIs(
            await exhausted.settle(
                second_lease,
                ChatPipelineExecutionError(
                    ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED,
                    phase=ChatPipelinePhase.VALIDATE_STRUCTURE,
                    diagnostic={"check": "task_deadline"},
                    model_calls=(_model_call("request-attempt-2"),),
                ),
            ),
            ChatTerminalWriteStatus.APPLIED,
        )

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            persisted = await connection.fetchrow(
                """
                SELECT r.status, r.attempt, r.next_attempt_at, r.error_code,
                       r.error_retryable, r.error_detail, r.usage, r.timing,
                       m.assistant_status, m.content
                FROM chat_run r JOIN chat_message m ON m.chat_run_id = r.id
                WHERE r.id = $1
                """,
                run.id,
            )
        finally:
            await connection.close()
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["attempt"], 2)
        self.assertIsNone(persisted["next_attempt_at"])
        self.assertEqual(
            persisted["error_code"], "CHAT_PIPELINE_DEADLINE_EXCEEDED"
        )
        self.assertTrue(persisted["error_retryable"])
        self.assertNotIn("raw_content", persisted["error_detail"])
        self.assertEqual(persisted["assistant_status"], "failed")
        self.assertEqual(persisted["content"], "")
        usage = json.loads(persisted["usage"])
        timing = json.loads(persisted["timing"])
        self.assertEqual(len(usage["calls"]), 2)
        self.assertEqual(set(timing["attempts"]), {"1", "2"})
        self.assertEqual(
            timing["attempts"]["1"]["result"], "requeued"
        )
        self.assertEqual(
            timing["attempts"]["2"]["result"], "failed"
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

    async def _seed_citation_chunks(self, kb):
        document_id = uuid4()
        version_id = uuid4()
        indexed_version_id = uuid4()
        chunk_ids = (uuid4(), uuid4())
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO document
                        (id, workspace_id, kb_id, display_name)
                    VALUES ($1, $2, $3, 'citation source')
                    """,
                    document_id,
                    WORKSPACE,
                    kb.id,
                )
                await connection.execute(
                    """
                    INSERT INTO document_version
                        (id, workspace_id, kb_id, document_id, version_number,
                         source_status, checksum_sha256, storage_uri,
                         original_filename, media_type, size_bytes)
                    VALUES ($1, $2, $3, $4, 1, 'available', $5, $6,
                            'source.txt', 'text/plain', 20)
                    """,
                    version_id,
                    WORKSPACE,
                    kb.id,
                    document_id,
                    "a" * 64,
                    f"local://{version_id}",
                )
                await connection.execute(
                    "UPDATE document SET current_version_id = $1 WHERE id = $2",
                    version_id,
                    document_id,
                )
                await connection.execute(
                    """
                    INSERT INTO indexed_document_version
                        (id, workspace_id, kb_id, document_id,
                         document_version_id, index_revision_id,
                         source_change_seq, build_status, serving_status)
                    VALUES ($1, $2, $3, $4, $5, $6, 1, 'ready', 'serving')
                    """,
                    indexed_version_id,
                    WORKSPACE,
                    kb.id,
                    document_id,
                    version_id,
                    kb.active_index_revision_id,
                )
                for ordinal, (chunk_id, content) in enumerate(
                    zip(chunk_ids, ("第一段证据", "第二段证据"), strict=True)
                ):
                    await connection.execute(
                        """
                        INSERT INTO index_chunk
                            (id, workspace_id, kb_id,
                             indexed_document_version_id, ordinal, content,
                             content_hash, token_count, source_location,
                             hierarchy, source_metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, 4,
                                CAST($8 AS jsonb), '{}'::jsonb, '{}'::jsonb)
                        """,
                        chunk_id,
                        WORKSPACE,
                        kb.id,
                        indexed_version_id,
                        ordinal,
                        content,
                        str(ordinal) * 64,
                        json.dumps({"paragraph": ordinal + 1}),
                    )
        finally:
            await connection.close()
        return document_id, version_id, chunk_ids


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
    return index_profile()


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


def _model_call(request_id: str) -> ChatModelCallRecord:
    return ChatModelCallRecord(
        operation=ChatModelOperation.ASSESS_EVIDENCE,
        model="chat-model",
        provider_request_id=request_id,
        usage={"input_tokens": 4, "output_tokens": 2},
    )


def _refusal_state(context) -> ChatPipelineState:
    evidence = EvidenceEnvelope(
        knowledge_base_id=context.knowledge_base_id,
        index_revision_id=context.index_revision_id,
        items=(),
    )
    assessment = EvidenceAssessment(
        coverage=EvidenceCoverage.NONE,
        usable_citation_ids=(),
        supported_aspects=(),
        missing_aspects=(),
    )
    draft = AnswerDraftCandidate(
        raw_json='{"outcome":"refused","claims":[],"missing_aspects":[]}',
        expected_outcome=AnswerOutcome.REFUSED,
        source=AnswerDraftSource.DETERMINISTIC,
        control_reason=AnswerControlReason.NO_USABLE_EVIDENCE,
    )
    validated = ValidatedAnswer(
        outcome=AnswerOutcome.REFUSED,
        claims=(),
        missing_aspects=(),
        source=AnswerDraftSource.DETERMINISTIC,
        control_reason=AnswerControlReason.NO_USABLE_EVIDENCE,
    )
    return ChatPipelineState(
        context=context,
        evidence_pack=EvidencePack(
            knowledge_base_id=context.knowledge_base_id,
            index_revision_id=context.index_revision_id,
            strategy=RetrievalStrategy.EXACT_VECTOR,
        ),
        answering=ChatAnsweringState(
            evidence=evidence,
            assessment=assessment,
            draft=draft,
            model_calls=(_model_call("request-success"),),
            validated=validated,
            rendered=RenderedAnswer(
                outcome=AnswerOutcome.REFUSED,
                content="无法基于当前证据回答。",
                citations=(),
            ),
            validation=AnswerValidationRecord(initial_issues=()),
        ),
    )


class _PersistingPipeline:
    def __init__(self, factory) -> None:
        self.factory = factory

    async def execute(self, command):
        context = await ChatExecutionContextLoader(self.factory).load(command)
        await asyncio.sleep(0.02)
        state = _refusal_state(context)
        return await ChatResultPersistenceStep(self.factory).run(state)


class _CountingCoordinator:
    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator
        self.heartbeats = 0

    async def oldest_claimable_at(self, **values):
        return await self.coordinator.oldest_claimable_at(**values)

    async def claim(self, **values):
        return await self.coordinator.claim(**values)

    async def heartbeat(self, lease, **values):
        self.heartbeats += 1
        return await self.coordinator.heartbeat(lease, **values)

    async def reconcile_stale(self, **values):
        return await self.coordinator.reconcile_stale(**values)


if __name__ == "__main__":
    unittest.main()
