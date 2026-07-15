from __future__ import annotations

import json
import os
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import asyncpg

from rag_kb.db import DatabaseProcess, create_database_resources
from rag_kb.domain import (
    EvaluationCaseDefinition,
    EvaluationCaseResult,
    EvaluationConflictError,
    EvaluationDatasetDefinition,
    EvaluationRunDefinition,
    EvaluationRunState,
)
from rag_kb.services import EvaluationPersistenceService
from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWorkFactory


MIGRATION_DSN = os.environ.get("RAG_KB_TEST_MIGRATION_DSN")
RUNTIME_SQLALCHEMY_DSN = os.environ.get("RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN")
WORKSPACE_ID = UUID("01900000-0000-7000-8000-000000004001")
KB_ID = UUID("01900000-0000-7000-8000-000000004002")
REVISION_ID = UUID("01900000-0000-7000-8000-000000004003")
SPACE_ID = UUID("01900000-0000-7000-8000-000000004004")
RUN_ID = UUID("01900000-0000-7000-8000-000000004005")
STARTED_AT = datetime(2026, 7, 15, 3, 0, tzinfo=UTC)


@unittest.skipUnless(
    MIGRATION_DSN and RUNTIME_SQLALCHEMY_DSN,
    "database integration DSNs are not configured",
)
class EvaluationPersistenceDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            await connection.execute("TRUNCATE TABLE workspace CASCADE")
            await _seed_scope(connection)
        finally:
            await connection.close()
        self.database = create_database_resources(
            RUNTIME_SQLALCHEMY_DSN,
            pool_size=2,
            max_overflow=0,
            process=DatabaseProcess.WORKER,
        )
        factory = SqlAlchemyUnitOfWorkFactory(self.database.sessions, WORKSPACE_ID)
        self.service = EvaluationPersistenceService(factory)

    async def asyncTearDown(self) -> None:
        await self.database.close()

    async def test_completed_run_and_results_replay_exactly(self) -> None:
        definition = _run_definition()
        results = _results()

        started = await self.service.start(definition)
        completed = await self.service.complete(
            definition,
            results,
            completed_at=STARTED_AT + timedelta(seconds=1),
        )
        replay = await self.service.complete(
            definition,
            results,
            completed_at=STARTED_AT + timedelta(seconds=1),
        )

        self.assertEqual(started.state, EvaluationRunState.RUNNING)
        self.assertEqual(completed.state, EvaluationRunState.COMPLETED)
        self.assertEqual(completed.result_count, 2)
        self.assertEqual(replay, completed)
        loaded = await self.service.get(RUN_ID)
        self.assertEqual(loaded, completed)

        connection = await asyncpg.connect(MIGRATION_DSN)
        try:
            rows = await connection.fetch(
                """
                SELECT eval_case.case_key, eval_result.metrics, eval_result.evidence
                FROM eval_result
                JOIN eval_case ON eval_case.id = eval_result.eval_case_id
                WHERE eval_result.eval_run_id = $1
                ORDER BY eval_case.case_key
                """,
                RUN_ID,
            )
        finally:
            await connection.close()
        self.assertEqual([row["case_key"] for row in rows], ["CASE-1", "CASE-2"])
        metrics = json.loads(rows[0]["metrics"])
        evidence = json.loads(rows[0]["evidence"])
        self.assertEqual(metrics["reciprocal_rank"], 1.0)
        self.assertEqual(evidence["result_count"], 1)

    async def test_replay_conflicts_and_terminal_direction_fail_closed(self) -> None:
        definition = _run_definition()
        await self.service.complete(
            definition,
            _results(),
            completed_at=STARTED_AT + timedelta(seconds=1),
        )

        changed = replace(
            _results()[0],
            metrics={"reciprocal_rank": 0.5},
        )
        with self.assertRaises(EvaluationConflictError):
            await self.service.complete(
                definition,
                (changed, _results()[1]),
                completed_at=STARTED_AT + timedelta(seconds=1),
            )
        with self.assertRaises(EvaluationConflictError):
            await self.service.fail(
                definition,
                completed_at=STARTED_AT + timedelta(seconds=2),
                error_code="evaluation_failed",
                error_detail={"phase": "report"},
            )
        with self.assertRaises(EvaluationConflictError):
            await self.service.complete(
                definition,
                _results(),
                completed_at=STARTED_AT + timedelta(seconds=3),
            )

        changed_dataset = replace(
            definition.dataset,
            metadata={"dataset_id": "changed"},
        )
        with self.assertRaises(EvaluationConflictError):
            await self.service.start(replace(definition, dataset=changed_dataset))

    async def test_failed_run_replays_bounded_diagnostics(self) -> None:
        definition = replace(
            _run_definition(),
            run_id=UUID("01900000-0000-7000-8000-000000004006"),
        )

        failed = await self.service.fail(
            definition,
            completed_at=STARTED_AT + timedelta(seconds=2),
            error_code="embedding_provider_unavailable",
            error_detail={"phase": "query_embedding", "retryable": True},
        )
        replay = await self.service.fail(
            definition,
            completed_at=STARTED_AT + timedelta(seconds=2),
            error_code="embedding_provider_unavailable",
            error_detail={"phase": "query_embedding", "retryable": True},
        )

        self.assertEqual(failed.state, EvaluationRunState.FAILED)
        self.assertEqual(failed.error_code, "embedding_provider_unavailable")
        self.assertEqual(replay, failed)
        with self.assertRaises(EvaluationConflictError):
            await self.service.fail(
                definition,
                completed_at=STARTED_AT + timedelta(seconds=3),
                error_code="embedding_provider_unavailable",
                error_detail={"phase": "query_embedding", "retryable": True},
            )


def _run_definition() -> EvaluationRunDefinition:
    dataset = EvaluationDatasetDefinition(
        name="synthetic-v1-golden",
        version="1.0",
        manifest_hash="a" * 64,
        metadata={"dataset_id": "synthetic-v1-golden-v1.0"},
        cases=(
            EvaluationCaseDefinition(
                "CASE-1",
                "Question one?",
                {"relevant": ["SAMPLE-1"]},
                ("en", "answerable"),
            ),
            EvaluationCaseDefinition(
                "CASE-2",
                "Question two?",
                {"relevant": []},
                ("en", "expected_empty"),
            ),
        ),
    )
    return EvaluationRunDefinition(
        run_id=RUN_ID,
        knowledge_base_id=KB_ID,
        index_revision_id=REVISION_ID,
        dataset=dataset,
        run_config={
            "workspace_id": str(WORKSPACE_ID),
            "strategy_id": "exact-vector-cosine-v1",
        },
        started_at=STARTED_AT,
    )


def _results() -> tuple[EvaluationCaseResult, ...]:
    return (
        EvaluationCaseResult(
            "CASE-1",
            {"result_count": 1, "sample_ids": ["SAMPLE-1"]},
            {"reciprocal_rank": 1.0, "recall_at_5": 1.0},
        ),
        EvaluationCaseResult(
            "CASE-2",
            {"result_count": 0, "sample_ids": []},
            {"reciprocal_rank": 0.0, "expected_empty_safe": True},
        ),
    )


async def _seed_scope(connection: asyncpg.Connection) -> None:
    async with connection.transaction():
        await connection.execute(
            "INSERT INTO workspace (id, name) VALUES ($1, 'evaluation-workspace')",
            WORKSPACE_ID,
        )
        await connection.execute(
            """
            INSERT INTO embedding_space (
                id, workspace_id, provider_identity, endpoint_identity,
                requested_model, resolved_model, model_version,
                dimension, distance_metric, vector_data_type, normalization,
                configuration_fingerprint, compatibility_fingerprint
            ) VALUES (
                $1, $2, 'test', 'test', 'test', 'test', 'v1', 1024,
                'cosine', 'float32', 'l2', 'sha256:test-config',
                'sha256:test-evaluation-space'
            )
            """,
            SPACE_ID,
            WORKSPACE_ID,
        )
        await connection.execute(
            """
            INSERT INTO knowledge_base (
                id, workspace_id, name, source_change_seq, retrieval_defaults
            ) VALUES ($1, $2, 'evaluation-kb', 0, '{"strategy":"exact_vector"}')
            """,
            KB_ID,
            WORKSPACE_ID,
        )
        await connection.execute(
            """
            INSERT INTO index_revision (
                id, workspace_id, kb_id, embedding_space_id, status,
                source_snapshot_seq, parser_config, chunking_config
            ) VALUES ($1, $2, $3, $4, 'active', 0, '{}', '{}')
            """,
            REVISION_ID,
            WORKSPACE_ID,
            KB_ID,
            SPACE_ID,
        )
        await connection.execute(
            """
            UPDATE knowledge_base
            SET active_index_revision_id = $1, provisioned_at = now()
            WHERE id = $2
            """,
            REVISION_ID,
            KB_ID,
        )
