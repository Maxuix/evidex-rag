from __future__ import annotations

import asyncio
import io
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg

from rag_kb.adapters import (
    FixedPgVectorSpace,
    LocalFileStore,
    UnstructuredProcessor,
)
from rag_kb.auth import AuthContext, SingleWorkspaceAccessPolicy
from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.document_processing import count_chunk_tokens, index_profile
from rag_kb.domain import (
    AnswerStyle,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChunkingPreset,
    EmbeddingBatch,
    EmbeddingSpaceDefinition,
    ErrorCode,
    IndexingCommand,
    IndexingExecutionError,
    IndexingPhase,
    InsufficiencyPolicy,
    ParserLimits,
    PromotionCommand,
    PromotionReason,
    ResourceStateConflictError,
)
from rag_kb.indexing import CandidatePromotionService, IndexingPipeline
from rag_kb.scheduling import (
    ChatRunScheduler,
    FairWorkerScheduler,
    IndexingJobScheduler,
    RetryPolicy,
    WeightedLaneSelector,
)
from rag_kb.services import (
    ChatFailureSettlementService,
    ChatRunCoordinator,
    ChatService,
    IndexingJobService,
    SourceFileService,
)
from rag_kb.services.content import DocumentService, KnowledgeBaseService
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory
from rag_kb.uow import UnitOfWorkPurpose, execute_in_transaction


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")
WORKSPACE = UUID("01900000-0000-7000-8000-000000000401")


@unittest.skipUnless(
    MIGRATION_DSN and RUNTIME_SQLALCHEMY_DSN,
    "database integration DSNs are not configured",
)
class IndexingPipelineDatabaseTests(unittest.IsolatedAsyncioTestCase):
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
            process=DatabaseProcess.WORKER,
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
        self.documents = DocumentService(self.factory, self.policy)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "staging").mkdir()
        (self.root / "final").mkdir()
        self.store = LocalFileStore(self.root / "staging", self.root / "final")
        self.processor = UnstructuredProcessor(ParserLimits())

    async def asyncTearDown(self) -> None:
        await self.database.close()
        self.temporary.cleanup()

    async def test_txt_markdown_and_completed_replay_produce_serving_sets(self) -> None:
        kb = await self._create_kb()
        uploaded = (
            await self._upload(kb.id, "guide.txt", "text/plain", b"first\n\nsecond"),
            await self._upload(
                kb.id,
                "handbook.md",
                "text/markdown",
                b"# Handbook\r\n\r\nrestart-safe evidence",
            ),
        )
        provider = _Provider()
        pipeline = self._pipeline(provider)
        not_ready = await CandidatePromotionService(self.factory).promote(
            _promotion_command(uploaded[0])
        )
        self.assertEqual(
            (not_ready.status, not_ready.reason),
            ("not_ready", PromotionReason.NOT_READY),
        )
        results = []
        for item in uploaded:
            results.append(await pipeline.execute(_command(item)))
        replay = await pipeline.execute(_command(uploaded[0]))

        self.assertTrue(all(result.status == "ready" for result in results))
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.chunk_count, results[0].chunk_count)
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            state = await connection.fetchrow(
                """
                SELECT
                    (SELECT count(*) FROM indexed_document_version
                      WHERE build_status = 'ready' AND serving_status = 'candidate') AS ready_candidates,
                    (SELECT count(*) FROM indexed_document_version
                      WHERE serving_status = 'serving') AS serving,
                    (SELECT count(*) FROM indexing_job WHERE status = 'completed') AS completed_jobs,
                    (SELECT count(*) FROM index_chunk) AS chunks,
                    (SELECT count(*) FROM vector_record_1024) AS vectors,
                    (SELECT bool_and(vector_dims(embedding) = 1024)
                       FROM vector_record_1024) AS dimensions_valid
                """
            )
        finally:
            await connection.close()
        self.assertEqual(state["ready_candidates"], 0)
        self.assertEqual(state["serving"], 2)
        self.assertEqual(state["completed_jobs"], 2)
        self.assertEqual(state["chunks"], state["vectors"])
        self.assertTrue(state["dimensions_valid"])

    async def test_semantic_plan_is_persisted_and_reused_after_final_failure(
        self,
    ) -> None:
        kb = await self._create_kb(ChunkingPreset.SEMANTIC_BALANCED_V1)
        uploaded = await self._upload(
            kb.id,
            "semantic.txt",
            "text/plain",
            ("A bounded semantic sentence with evidence. " * 240).encode(),
        )
        provider = _FailFirstFinalProvider()
        pipeline = self._pipeline(provider)

        with self.assertRaises(IndexingExecutionError) as failure:
            await pipeline.execute(_command(uploaded))
        self.assertEqual(
            failure.exception.code,
            ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
        )
        analysis_calls = provider.analysis_calls
        self.assertGreater(analysis_calls, 0)

        result = await pipeline.execute(_command(uploaded))
        self.assertEqual(result.status, "ready")
        self.assertEqual(provider.analysis_calls, analysis_calls)
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            facts = await connection.fetchrow(
                """
                SELECT
                    (SELECT count(*) FROM index_chunk_plan
                      WHERE indexed_document_version_id = $1) AS plans,
                    (SELECT count(*) FROM index_chunk
                      WHERE indexed_document_version_id = $1) AS chunks,
                    (SELECT count(*) FROM vector_record_1024 vector
                      JOIN index_chunk chunk ON chunk.id = vector.index_chunk_id
                     WHERE chunk.indexed_document_version_id = $1) AS vectors
                """,
                uploaded.indexed_document_version_id,
            )
        finally:
            await connection.close()
        self.assertEqual(
            tuple(facts),
            (1, result.chunk_count, result.chunk_count),
        )

    async def test_partial_provider_failure_is_durable_and_replay_converges(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(
            kb.id, "large.txt", "text/plain", b"word " * 1000
        )
        provider = _Provider(max_batch_size=1, fail_call=2)
        pipeline = self._pipeline(provider)

        with self.assertRaises(IndexingExecutionError) as failure:
            await pipeline.execute(_command(uploaded))
        self.assertEqual(failure.exception.code, ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE)
        failed = await self._target_state(uploaded.indexed_document_version_id)
        self.assertEqual(
            tuple(failed),
            ("failed", "candidate", "failed", "embedding", "EMBEDDING_PROVIDER_UNAVAILABLE", 1, 1),
        )

        provider.fail_call = None
        provider.calls = 0
        completed = await pipeline.execute(_command(uploaded))
        self.assertEqual((completed.status, completed.chunk_count), ("ready", 2))
        replayed = await self._target_state(uploaded.indexed_document_version_id)
        self.assertEqual(
            tuple(replayed),
            ("ready", "serving", "completed", "completed", None, 2, 2),
        )

    async def test_new_version_switches_atomically_after_ready(self) -> None:
        kb = await self._create_kb()
        first = await self._upload(kb.id, "guide.txt", "text/plain", b"version one")
        pipeline = self._pipeline(_Provider())
        await pipeline.execute(_command(first))

        second = await self._upload(
            kb.id,
            "guide.txt",
            "text/plain",
            b"version two",
            document_id=first.document.id,
        )
        self.assertEqual(
            await self._serving_states(first.document.id),
            (
                (first.indexed_document_version_id, "serving"),
                (second.indexed_document_version_id, "candidate"),
            ),
        )

        promoted = await pipeline.execute(_command(second))

        self.assertEqual(promoted.serving_status, "serving")
        self.assertEqual(
            await self._serving_states(first.document.id),
            (
                (first.indexed_document_version_id, "retired"),
                (second.indexed_document_version_id, "serving"),
            ),
        )

    async def test_reverse_completion_cannot_restore_superseded_version(self) -> None:
        kb = await self._create_kb()
        first = await self._upload(kb.id, "guide.txt", "text/plain", b"version one")
        second = await self._upload(
            kb.id,
            "guide.txt",
            "text/plain",
            b"version two",
            document_id=first.document.id,
        )
        pipeline = self._pipeline(_Provider())

        current = await pipeline.execute(_command(second))
        stale = await pipeline.execute(_command(first))
        stale_replay = await pipeline.execute(_command(first))

        self.assertEqual(
            (
                current.serving_status,
                stale.serving_status,
                stale_replay.serving_status,
            ),
            ("serving", "retired", "retired"),
        )
        self.assertTrue(stale_replay.replayed)
        self.assertEqual(
            await self._serving_states(first.document.id),
            (
                (first.indexed_document_version_id, "retired"),
                (second.indexed_document_version_id, "serving"),
            ),
        )

    async def test_late_worker_after_delete_cannot_restore_serving_state(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        await self.documents.delete(self.context, uuid4(), uploaded.document.id)

        result = await self._pipeline(_Provider()).execute(_command(uploaded))

        self.assertEqual(
            (result.status, result.serving_status),
            ("cancelled", "retired"),
        )
        self.assertEqual(
            await self._serving_states(uploaded.document.id),
            ((uploaded.indexed_document_version_id, "retired"),),
        )

    async def test_concurrent_promotions_converge_on_current_version(self) -> None:
        kb = await self._create_kb()
        first = await self._upload(kb.id, "guide.txt", "text/plain", b"version one")
        second = await self._upload(
            kb.id,
            "guide.txt",
            "text/plain",
            b"version two",
            document_id=first.document.id,
        )
        await self._mark_ready(first, second)
        promotion = CandidatePromotionService(self.factory)

        first_result, second_result = await asyncio.gather(
            promotion.promote(_promotion_command(first)),
            promotion.promote(_promotion_command(second)),
        )

        self.assertEqual(first_result.reason, PromotionReason.SUPERSEDED)
        self.assertEqual(second_result.status, "serving")
        self.assertEqual(
            await self._serving_states(first.document.id),
            (
                (first.indexed_document_version_id, "retired"),
                (second.indexed_document_version_id, "serving"),
            ),
        )

    async def test_promotion_delete_race_always_ends_retired(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        await self._mark_ready(uploaded)
        promotion = CandidatePromotionService(self.factory)

        await asyncio.gather(
            promotion.promote(_promotion_command(uploaded)),
            self.documents.delete(self.context, uuid4(), uploaded.document.id),
        )

        self.assertEqual(
            await self._serving_states(uploaded.document.id),
            ((uploaded.indexed_document_version_id, "retired"),),
        )

    async def test_later_source_change_retires_otherwise_eligible_candidate(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        await self._mark_ready(uploaded)
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                await connection.execute(
                    "UPDATE knowledge_base SET source_change_seq = 2 WHERE id = $1",
                    kb.id,
                )
                await connection.execute(
                    """
                    INSERT INTO source_change(
                        workspace_id, kb_id, source_change_seq, document_id,
                        document_version_id, change_kind
                    ) VALUES ($1, $2, 2, $3, $4, 'upsert')
                    """,
                    WORKSPACE,
                    kb.id,
                    uploaded.document.id,
                    uploaded.document_version_id,
                )
        finally:
            await connection.close()

        result = await CandidatePromotionService(self.factory).promote(
            _promotion_command(uploaded)
        )

        self.assertEqual(result.reason, PromotionReason.LATER_SOURCE_CHANGE)
        self.assertEqual(result.status, "retired")

    async def test_claim_heartbeat_and_due_retry_use_owner_attempt_cas(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        observed = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)
        first = self._scheduler_for(
            self._pipeline(_Provider()),
            worker_id="worker-a",
            clock=lambda: observed,
        )
        second = self._scheduler_for(
            self._pipeline(_Provider()),
            worker_id="worker-b",
            clock=lambda: observed,
        )

        lease = await first.claim_once()
        self.assertIsNotNone(lease)
        assert lease is not None
        self.assertEqual(lease.attempt, 1)
        self.assertIsNone(await second.claim_once())
        wrong_owner = replace(lease, claimed_by="worker-b")
        self.assertFalse(
            await execute_in_transaction(
                self.factory,
                lambda uow: uow.indexing.heartbeat(
                    wrong_owner,
                    observed_at=observed + timedelta(seconds=1),
                ),
                purpose=UnitOfWorkPurpose.HEARTBEAT,
            )
        )
        self.assertTrue(
            await execute_in_transaction(
                self.factory,
                lambda uow: uow.indexing.heartbeat(
                    lease,
                    observed_at=observed + timedelta(seconds=1),
                ),
                purpose=UnitOfWorkPurpose.HEARTBEAT,
            )
        )
        due = observed + timedelta(seconds=10)
        self.assertTrue(
            await execute_in_transaction(
                self.factory,
                lambda uow: uow.indexing.reschedule(
                    lease,
                    observed_at=observed + timedelta(seconds=2),
                    next_attempt_at=due,
                    error_code=ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE.value,
                    error_detail={"attempt": 1},
                ),
                purpose=UnitOfWorkPurpose.RECONCILIATION,
            )
        )
        self.assertIsNone(await second.claim_once())

        due_scheduler = self._scheduler_for(
            self._pipeline(_Provider()),
            worker_id="worker-b",
            clock=lambda: due,
        )
        retry = await due_scheduler.claim_once()
        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertEqual((retry.claimed_by, retry.attempt), ("worker-b", 2))

    async def test_stale_reconciliation_requeues_then_exhausts(self) -> None:
        kb = await self._create_kb()
        await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        current = [datetime(2026, 7, 14, 9, 0, tzinfo=UTC)]
        scheduler = self._scheduler_for(
            self._pipeline(_Provider()),
            worker_id="worker-a",
            clock=lambda: current[0],
            max_attempts=2,
            stale_after=10,
        )

        first = await scheduler.claim_once()
        self.assertIsNotNone(first)
        current[0] += timedelta(seconds=11)
        recovered = await scheduler.reconcile_once()
        self.assertEqual((recovered.requeued, recovered.failed), (1, 0))

        current[0] += timedelta(seconds=2)
        second = await scheduler.claim_once()
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second.attempt, 2)
        current[0] += timedelta(seconds=11)
        exhausted = await scheduler.reconcile_once()
        self.assertEqual((exhausted.requeued, exhausted.failed), (0, 1))
        state = await self._job_claim_state(second.job_id)
        self.assertEqual(
            (state["status"], state["attempt"], state["error_code"]),
            ("failed", 2, ErrorCode.INDEXING_STALE_WORKER.value),
        )
        self.assertIsNone(state["claimed_by"])

    async def test_two_schedulers_execute_one_job_once_and_promote(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        provider = _Provider()
        pipeline = self._pipeline(provider)
        first = self._scheduler_for(pipeline, worker_id="worker-a")
        second = self._scheduler_for(pipeline, worker_id="worker-b")
        stopped = asyncio.Event()
        tasks = (
            asyncio.create_task(first.run(stopped)),
            asyncio.create_task(second.run(stopped)),
        )
        try:
            for _ in range(1000):
                state = await self._target_state(
                    uploaded.indexed_document_version_id
                )
                if state[1] == "serving" and state[2] == "completed":
                    break
                await asyncio.sleep(0.01)
            else:
                claim = await self._job_claim_state(uploaded.job_id)
                self.fail(f"schedulers did not complete the queued job: {dict(claim)}")
        finally:
            stopped.set()
            await asyncio.gather(*tasks)

        claim = await self._job_claim_state(uploaded.job_id)
        self.assertEqual((claim["status"], claim["attempt"]), ("completed", 1))
        self.assertIsNone(claim["claimed_by"])
        self.assertEqual(provider.calls, 1)

    async def test_fair_worker_starts_chat_under_indexing_load_without_starvation(
        self,
    ) -> None:
        kb = await self._create_kb()
        uploads = [
            await self._upload(
                kb.id,
                f"load-{ordinal}.txt",
                "text/plain",
                f"load {ordinal}".encode(),
            )
            for ordinal in range(3)
        ]
        chat = ChatService(
            self.factory,
            self.policy,
            model_configuration={
                "provider_identity": "chat-provider",
                "logical_endpoint_identity": "chat-endpoint",
                "requested_model": "chat-model",
                "resolved_model": "chat-model",
                "model_version": "v1",
                "structured_output_mode": "json_object",
                "configuration_fingerprint": "sha256:" + "c" * 64,
                "capability_fingerprint": "sha256:" + "d" * 64,
            },
        )
        session = await chat.create_session(self.context, kb_id=kb.id, title=None)
        run = await chat.create_run(
            self.context,
            uuid4(),
            session_id=session.id,
            kb_id=kb.id,
            message="start while indexing is busy",
            answer_style=AnswerStyle.CONCISE,
            insufficiency_policy=InsufficiencyPolicy.REFUSE,
            retrieval_mode="vector",
            top_k=5,
        )
        retry = RetryPolicy(3, 0.01, 0.02)
        provider = _Provider(delay=0.05)
        indexing = self._scheduler_for(
            self._pipeline(provider),
            worker_id="worker-fair",
        )
        chat_scheduler = ChatRunScheduler(
            ChatRunCoordinator(self.factory),
            _FailingChatPipeline(),
            ChatFailureSettlementService(
                self.factory,
                max_attempts=3,
                base_delay_seconds=0.01,
                max_delay_seconds=0.02,
            ),
            worker_id="worker-fair",
            heartbeat_interval_seconds=0.01,
            stale_after_seconds=1,
            retry_policy=retry,
            reconciliation_batch_size=10,
        )
        fair = FairWorkerScheduler(
            chat_scheduler,
            indexing,
            WeightedLaneSelector(
                chat_weight=3,
                indexing_weight=1,
                aging_seconds=0.05,
            ),
            chat_concurrency=1,
            indexing_concurrency=1,
            poll_interval_seconds=0.01,
        )
        stopped = asyncio.Event()
        started = asyncio.get_running_loop().time()
        task = asyncio.create_task(fair.run(stopped))
        try:
            for _ in range(400):
                state = await chat.get_run(self.context, run.id)
                if state.status == "failed" and provider.calls >= 1:
                    break
                await asyncio.sleep(0.01)
            else:
                self.fail("fair worker did not advance both lanes")
        finally:
            stopped.set()
            await task

        elapsed = asyncio.get_running_loop().time() - started
        self.assertLess(elapsed, 2.0)
        self.assertEqual(state.error_code, ErrorCode.CHAT_REVISION_MISMATCH.value)
        self.assertGreaterEqual(provider.calls, 1)
        claim_states = [
            await self._job_claim_state(item.job_id) for item in uploads
        ]
        self.assertTrue(
            any(item["attempt"] >= 1 for item in claim_states)
        )
        self.assertEqual(self.database.engine.pool.checkedout(), 0)

    async def test_scheduler_recovers_completed_candidate_before_promotion(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        await self._mark_ready(uploaded)
        provider = _Provider()
        scheduler = self._scheduler_for(
            self._pipeline(provider),
            worker_id="worker-a",
        )
        stopped = asyncio.Event()
        task = asyncio.create_task(scheduler.run(stopped))
        try:
            for _ in range(500):
                state = await self._target_state(
                    uploaded.indexed_document_version_id
                )
                if state[1] == "serving":
                    break
                await asyncio.sleep(0.01)
            else:
                self.fail("completed candidate was not promoted by replay")
        finally:
            stopped.set()
            await task

        self.assertEqual(provider.calls, 0)
        claim = await self._job_claim_state(uploaded.job_id)
        self.assertEqual(claim["status"], "completed")
        self.assertIsNone(claim["claimed_by"])

    async def test_status_and_idempotent_explicit_retry_reuse_the_same_target(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        provider = _Provider(fail_call=1)
        with self.assertRaises(IndexingExecutionError):
            await self._pipeline(provider).execute(_command(uploaded))
        service = IndexingJobService(self.factory, self.policy)

        failed = await service.get(self.context, uploaded.job_id)
        self.assertEqual((failed.job_status, failed.build_status), ("failed", "failed"))
        self.assertTrue(failed.can_retry)
        key = uuid4()
        retried = await service.retry(self.context, key, uploaded.job_id)
        replay = await service.retry(self.context, key, uploaded.job_id)

        self.assertEqual(retried.job_id, uploaded.job_id)
        self.assertEqual(retried.indexed_document_version_id, uploaded.indexed_document_version_id)
        self.assertEqual((retried.job_status, retried.build_status, retried.attempt), ("queued", "queued", 0))
        self.assertEqual(replay.job_id, retried.job_id)
        with self.assertRaises(ResourceStateConflictError):
            await service.retry(self.context, uuid4(), uploaded.job_id)

    async def test_cleanup_removes_only_retired_derived_data_and_expired_job(self) -> None:
        kb = await self._create_kb()
        first = await self._upload(kb.id, "guide.txt", "text/plain", b"version one")
        pipeline = self._pipeline(_Provider())
        await pipeline.execute(_command(first))
        second = await self._upload(
            kb.id,
            "guide.txt",
            "text/plain",
            b"version two",
            document_id=first.document.id,
        )
        await pipeline.execute(_command(second))
        future = datetime.now(UTC) + timedelta(days=30)
        past = datetime.now(UTC) - timedelta(days=30)

        cleaned = await execute_in_transaction(
            self.factory,
            lambda uow: uow.indexing.cleanup_retired(
                data_before=future,
                tasks_before=past,
                limit=10,
            ),
            purpose=UnitOfWorkPurpose.RECONCILIATION,
        )
        replay = await execute_in_transaction(
            self.factory,
            lambda uow: uow.indexing.cleanup_retired(
                data_before=future,
                tasks_before=past,
                limit=10,
            ),
            purpose=UnitOfWorkPurpose.RECONCILIATION,
        )
        self.assertGreater(cleaned.chunks_deleted, 0)
        self.assertEqual(cleaned.chunks_deleted, cleaned.vectors_deleted)
        self.assertEqual(replay.chunks_deleted, 0)
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            counts = await connection.fetchrow(
                """
                SELECT
                  (SELECT count(*) FROM index_chunk WHERE indexed_document_version_id = $1) AS retired_chunks,
                  (SELECT count(*) FROM index_chunk WHERE indexed_document_version_id = $2) AS serving_chunks
                """,
                first.indexed_document_version_id,
                second.indexed_document_version_id,
            )
        finally:
            await connection.close()
        self.assertEqual(counts["retired_chunks"], 0)
        self.assertGreater(counts["serving_chunks"], 0)

        expired = await execute_in_transaction(
            self.factory,
            lambda uow: uow.indexing.cleanup_retired(
                data_before=past,
                tasks_before=future,
                limit=10,
            ),
            purpose=UnitOfWorkPurpose.RECONCILIATION,
        )
        self.assertEqual(expired.jobs_deleted, 1)
        self.assertIsNone(
            await execute_in_transaction(
                self.factory,
                lambda uow: uow.indexing.get_job(first.job_id),
                purpose=UnitOfWorkPurpose.REQUEST,
            )
        )
        serving = await execute_in_transaction(
            self.factory,
            lambda uow: uow.indexing.get_job(second.job_id),
            purpose=UnitOfWorkPurpose.REQUEST,
        )
        self.assertIsNotNone(serving)
        self.assertEqual(serving.serving_status, "serving")

    async def test_embedding_mismatch_fails_before_any_derived_write(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        provider = _Provider(
            embedding_space=replace(
                _embedding(), configuration_fingerprint="sha256:changed"
            )
        )

        with self.assertRaises(IndexingExecutionError) as failure:
            await self._pipeline(provider).execute(_command(uploaded))
        self.assertEqual(failure.exception.code, ErrorCode.EMBEDDING_SPACE_MISMATCH)
        state = await self._target_state(uploaded.indexed_document_version_id)
        self.assertEqual(
            tuple(state),
            ("failed", "candidate", "failed", "embedding", "EMBEDDING_SPACE_MISMATCH", 0, 0),
        )

    async def test_conflicting_stable_key_records_persistence_failure(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute(
                """
                INSERT INTO index_chunk(
                    id, workspace_id, kb_id, indexed_document_version_id, ordinal,
                    content, content_hash, token_count, source_location, hierarchy,
                    source_metadata, unit_key, modality
                ) VALUES (
                    $1, $2, $3, $4, 0, 'conflict', $5, 8, '{}', '{}', '{}',
                    'conflicting-unit', 'text'
                )
                """,
                uuid4(),
                WORKSPACE,
                kb.id,
                uploaded.indexed_document_version_id,
                "0" * 64,
            )
        finally:
            await connection.close()

        with self.assertRaises(IndexingExecutionError) as failure:
            await self._pipeline(_Provider()).execute(_command(uploaded))
        self.assertEqual(failure.exception.code, ErrorCode.INDEX_PERSISTENCE_FAILED)
        state = await self._target_state(uploaded.indexed_document_version_id)
        self.assertEqual(
            tuple(state),
            ("failed", "candidate", "failed", "persisting", "INDEX_PERSISTENCE_FAILED", 1, 0),
        )

    async def test_missing_source_records_source_phase_without_partial_content(self) -> None:
        kb = await self._create_kb()
        uploaded = await self._upload(kb.id, "guide.txt", "text/plain", b"safe")
        identity = self.store.parse_uri(uploaded.document.current_version.storage_uri)
        await self.store.delete(identity)

        with self.assertRaises(IndexingExecutionError) as failure:
            await self._pipeline(_Provider()).execute(_command(uploaded))
        self.assertEqual(failure.exception.code, ErrorCode.SOURCE_FILE_MISSING)
        state = await self._target_state(uploaded.indexed_document_version_id)
        self.assertEqual(
            tuple(state),
            ("failed", "candidate", "failed", "source_read", "SOURCE_FILE_MISSING", 0, 0),
        )

    def _pipeline(self, provider) -> IndexingPipeline:
        return IndexingPipeline(
            self.factory,
            self.store,
            self.processor,
            provider,
            FixedPgVectorSpace(_embedding()),
        )

    def _scheduler_for(
        self,
        pipeline,
        *,
        worker_id,
        clock=None,
        max_attempts=3,
        stale_after=1,
    ):
        return IndexingJobScheduler(
            self.factory,
            pipeline,
            worker_id=worker_id,
            concurrency=1,
            poll_interval_seconds=0.01,
            heartbeat_interval_seconds=0.02,
            stale_after_seconds=stale_after,
            deadline_seconds=10,
            retry_policy=RetryPolicy(max_attempts, 1, 2),
            reconciliation_batch_size=10,
            clock=clock,
        )

    async def _create_kb(
        self,
        preset: ChunkingPreset = ChunkingPreset.STRUCTURAL_BALANCED_V2,
    ):
        return await self.knowledge_bases.create(
            self.context,
            uuid4(),
            name=f"kb-{uuid4()}",
            chunking_preset=preset,
            retrieval_defaults={"strategy": "exact_vector", "top_k": 10},
        )

    async def _upload(
        self,
        kb_id,
        filename,
        media_type,
        content,
        *,
        document_id=None,
    ):
        return await SourceFileService(self.documents, self.store).store_and_activate(
            self.context,
            uuid4(),
            kb_id=kb_id,
            document_id=document_id,
            display_name=filename,
            original_filename=filename,
            media_type=media_type,
            source=io.BytesIO(content),
        )

    async def _target_state(self, target_id):
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            return await connection.fetchrow(
                """
                SELECT idv.build_status::text,
                       idv.serving_status::text,
                       job.status::text,
                       job.phase,
                       job.error_code,
                       (SELECT count(*) FROM index_chunk c
                         WHERE c.indexed_document_version_id = idv.id),
                       (SELECT count(*) FROM vector_record_1024 v
                         JOIN index_chunk c ON c.id = v.index_chunk_id
                        WHERE c.indexed_document_version_id = idv.id)
                  FROM indexed_document_version idv
                  JOIN indexing_job job ON job.indexed_document_version_id = idv.id
                 WHERE idv.id = $1
                """,
                target_id,
            )
        finally:
            await connection.close()

    async def _job_claim_state(self, job_id):
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            return await connection.fetchrow(
                """
                SELECT status::text, attempt, claimed_by, claimed_at,
                       heartbeat_at, next_attempt_at, error_code
                  FROM indexing_job
                 WHERE id = $1
                """,
                job_id,
            )
        finally:
            await connection.close()

    async def _serving_states(self, document_id):
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            rows = await connection.fetch(
                """
                SELECT id, serving_status::text
                  FROM indexed_document_version
                 WHERE document_id = $1
                 ORDER BY source_change_seq
                """,
                document_id,
            )
            return tuple((row["id"], row["serving_status"]) for row in rows)
        finally:
            await connection.close()

    async def _mark_ready(self, *uploaded):
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            async with connection.transaction():
                for item in uploaded:
                    await connection.execute(
                        """
                        UPDATE indexed_document_version
                           SET build_status = 'ready'
                         WHERE id = $1
                        """,
                        item.indexed_document_version_id,
                    )
                    await connection.execute(
                        """
                        UPDATE indexing_job
                           SET status = 'completed', phase = 'completed'
                         WHERE id = $1
                        """,
                        item.job_id,
                    )
        finally:
            await connection.close()


class _Provider:
    def __init__(
        self,
        *,
        embedding_space=None,
        max_batch_size=10,
        fail_call=None,
        delay=0,
    ) -> None:
        self.embedding_space = embedding_space or _embedding()
        self.max_batch_size = max_batch_size
        self.fail_call = fail_call
        self.delay = delay
        self.calls = 0

    async def embed_documents(self, texts):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.calls == self.fail_call:
            raise IndexingExecutionError(
                ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                phase=IndexingPhase.EMBEDDING,
                diagnostic={"retry_exhausted": True},
            )
        return EmbeddingBatch(tuple(_vector() for _ in texts))


class _FailFirstFinalProvider:
    def __init__(self) -> None:
        self.embedding_space = _embedding()
        self.max_batch_size = 10
        self.analysis_calls = 0
        self.final_calls = 0
        self.failed = False

    async def embed_documents(self, texts):
        is_final = any(count_chunk_tokens(text) > 160 for text in texts)
        if is_final:
            self.final_calls += 1
            if not self.failed:
                self.failed = True
                raise IndexingExecutionError(
                    ErrorCode.EMBEDDING_PROVIDER_UNAVAILABLE,
                    phase=IndexingPhase.EMBEDDING,
                    diagnostic={"retry_exhausted": True},
                )
        else:
            self.analysis_calls += 1
        return EmbeddingBatch(tuple(_vector() for _ in texts))


class _FailingChatPipeline:
    async def execute(self, command):
        del command
        await asyncio.sleep(0.02)
        raise ChatPipelineExecutionError(
            ErrorCode.CHAT_REVISION_MISMATCH,
            phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
            diagnostic={"check": "load_test"},
        )


def _command(uploaded):
    return IndexingCommand(
        uploaded.job_id,
        uploaded.indexed_document_version_id,
    )


def _promotion_command(uploaded):
    return PromotionCommand(
        uploaded.job_id,
        uploaded.indexed_document_version_id,
    )


def _embedding():
    return EmbeddingSpaceDefinition(
        provider_identity="alibaba-cloud-model-studio-qwen",
        endpoint_identity="alibaba-model-studio-beijing-embedding",
        requested_model="qwen3.7-text-embedding",
        resolved_model="qwen3.7-text-embedding",
        model_version="qwen3.7-text-embedding",
        deployment_revision=None,
        dimension=1024,
        distance_metric="cosine",
        vector_data_type="float32",
        normalization="l2",
        configuration_fingerprint="sha256:5f774411565f9aaef04c7a9762bf6e245589cff064c396a8c5b38eb9098ac18f",
        tokenizer_fingerprint=None,
        compatibility_fingerprint="sha256:398af80b01c3e440c0edf5871de60f80fdab255f453bfa6c685c65e2f9be61c7",
    )


def _profile():
    return index_profile()


def _vector():
    return (1.0,) + (0.0,) * 1023


if __name__ == "__main__":
    unittest.main()
