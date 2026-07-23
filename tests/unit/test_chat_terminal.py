from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

from rag_kb.domain import (
    AnswerControlReason,
    AnswerDraftCandidate,
    AnswerDraftSource,
    AnswerOutcome,
    AnswerValidationRecord,
    ChatAnsweringState,
    ChatExecutionContext,
    ChatModelCallRecord,
    ChatModelOperation,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ChatRunLease,
    ChatTerminalWriteStatus,
    ErrorCode,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceEnvelope,
    RenderedAnswer,
    ValidatedAnswer,
    VisualEvidenceDecision,
    VisualEvidenceReason,
)
from rag_kb.repositories.sqlalchemy_chat import _serialized_validation
from rag_kb.services import ChatFailureSettlementService, ChatResultPersistenceStep


class ChatTerminalServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_persists_content_safe_retrieval_and_visual_diagnostics(
        self,
    ) -> None:
        observed = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
        repository = _Repository(ChatTerminalWriteStatus.APPLIED)
        state = _completed_state(observed)
        decision = VisualEvidenceDecision(
            visual_unit_id=uuid4(),
            asset_id=uuid4(),
            reason_code=VisualEvidenceReason.REJECTED_LOW_SIMILARITY,
            cross_modal_rank=1,
            priority_micros=100,
        )
        state = replace(
            state,
            evidence_pack=SimpleNamespace(
                debug=SimpleNamespace(
                    result_count=2,
                    text_candidate_count=3,
                    cross_modal_candidate_count=4,
                    hydrated_relation_count=1,
                    evidence_group_count=2,
                )
            ),
            answering=replace(state.answering, visual_decisions=(decision,)),
        )

        await ChatResultPersistenceStep(
            _Factory(repository), clock=lambda: observed + timedelta(seconds=2)
        ).run(state)

        command = repository.success
        self.assertEqual(command.retrieval_diagnostics["text_candidate_count"], 3)
        facts = _serialized_validation(command)
        self.assertEqual(facts["visual_evidence"]["rejected_count"], 1)
        self.assertEqual(
            facts["visual_evidence"]["rejection_counts"],
            {"rejected_low_similarity": 1},
        )
        serialized = str(facts)
        self.assertNotIn("content", serialized)
        self.assertNotIn("storage", serialized)

    async def test_success_passes_complete_terminal_command_and_accepts_replay(
        self,
    ) -> None:
        observed = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
        repository = _Repository(ChatTerminalWriteStatus.IDEMPOTENT)
        state = _completed_state(observed)

        result = await ChatResultPersistenceStep(
            _Factory(repository), clock=lambda: observed + timedelta(seconds=2)
        ).run(state)

        self.assertIs(result, state)
        self.assertEqual(repository.success.lease, state.context.lease)
        self.assertEqual(
            repository.success.rendered.content, "无法基于当前证据回答。"
        )
        self.assertEqual(
            repository.success.finished_at, observed + timedelta(seconds=2)
        )

    async def test_stale_and_database_failures_are_content_safe(self) -> None:
        observed = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
        state = _completed_state(observed)
        with self.assertRaises(ChatPipelineExecutionError) as stale:
            await ChatResultPersistenceStep(
                _Factory(_Repository(ChatTerminalWriteStatus.STALE)),
                clock=lambda: observed + timedelta(seconds=1),
            ).run(state)
        self.assertEqual(stale.exception.code, ErrorCode.CHAT_CONTEXT_INVALID)
        self.assertEqual(stale.exception.model_calls, state.answering.model_calls)

        with self.assertRaises(ChatPipelineExecutionError) as failed:
            await ChatResultPersistenceStep(
                _Factory(_Repository(RuntimeError("raw database detail"))),
                clock=lambda: observed + timedelta(seconds=1),
            ).run(state)
        self.assertEqual(failed.exception.code, ErrorCode.CHAT_PERSISTENCE_FAILED)
        self.assertNotIn("raw", str(failed.exception.diagnostic))

    async def test_retryable_failure_requeues_with_backoff_and_safe_diagnostic(
        self,
    ) -> None:
        observed = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
        lease = _lease(observed, attempt=2)
        repository = _Repository(ChatTerminalWriteStatus.APPLIED)
        service = ChatFailureSettlementService(
            _Factory(repository),
            max_attempts=3,
            base_delay_seconds=5,
            max_delay_seconds=30,
            clock=lambda: observed + timedelta(seconds=4),
        )
        error = ChatPipelineExecutionError(
            ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
            phase=ChatPipelinePhase.GENERATE_OR_REFUSE,
            diagnostic={"http_status": 503, "raw_content": "secret"},
            model_calls=(_call(),),
        )

        result = await service.settle(lease, error)

        self.assertIs(result, ChatTerminalWriteStatus.APPLIED)
        command = repository.failure
        self.assertTrue(command.retryable)
        self.assertFalse(command.exhausted)
        self.assertEqual(command.next_attempt_at, observed + timedelta(seconds=14))
        self.assertEqual(dict(command.diagnostic), {"http_status": 503})
        self.assertEqual(command.model_calls, (_call(),))

    async def test_nonretryable_and_exhausted_failures_have_no_next_attempt(
        self,
    ) -> None:
        observed = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
        cases = (
            (ErrorCode.CHAT_REVISION_MISMATCH, 1, False, False),
            (ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED, 3, True, True),
        )
        for code, attempt, retryable, exhausted in cases:
            with self.subTest(code=code):
                repository = _Repository(ChatTerminalWriteStatus.APPLIED)
                service = ChatFailureSettlementService(
                    _Factory(repository),
                    max_attempts=3,
                    base_delay_seconds=5,
                    max_delay_seconds=30,
                    clock=lambda: observed + timedelta(seconds=1),
                )
                await service.settle(
                    _lease(observed, attempt=attempt),
                    ChatPipelineExecutionError(
                        code,
                        phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                    ),
                )
                self.assertEqual(repository.failure.retryable, retryable)
                self.assertEqual(repository.failure.exhausted, exhausted)
                self.assertIsNone(repository.failure.next_attempt_at)


class _Repository:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.success = None
        self.failure = None

    async def complete_owned_run(self, command):
        self.success = command
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    async def settle_owned_failure(self, command):
        self.failure = command
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class _UnitOfWork:
    def __init__(self, repository: _Repository) -> None:
        self.chat = repository
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def commit(self) -> None:
        self.committed = True


class _Factory:
    def __init__(self, repository: _Repository) -> None:
        self.repository = repository

    def __call__(self, **kwargs):
        return _UnitOfWork(self.repository)


def _lease(observed: datetime, *, attempt: int = 1) -> ChatRunLease:
    return ChatRunLease(
        run_id=uuid4(),
        workspace_id=uuid4(),
        claimed_by="worker-1",
        attempt=attempt,
        claimed_at=observed,
    )


def _call() -> ChatModelCallRecord:
    return ChatModelCallRecord(
        operation=ChatModelOperation.ASSESS_EVIDENCE,
        model="fixed-model",
        provider_request_id="request-1",
        usage={"input_tokens": 4, "output_tokens": 2},
    )


def _completed_state(observed: datetime) -> ChatPipelineState:
    lease = _lease(observed)
    context = ChatExecutionContext(
        lease=lease,
        run_id=lease.run_id,
        workspace_id=lease.workspace_id,
        knowledge_base_id=uuid4(),
        session_id=uuid4(),
        user_message_id=uuid4(),
        assistant_message_id=uuid4(),
        index_revision_id=uuid4(),
        principal_id="principal",
        client_id="client",
        query="question",
        effective_policy={},
        retrieval_strategy={},
        model_configuration={},
        attempt=lease.attempt,
    )
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
    answering = ChatAnsweringState(
        evidence=evidence,
        assessment=assessment,
        draft=draft,
        model_calls=(_call(),),
        validated=validated,
        rendered=RenderedAnswer(
            outcome=AnswerOutcome.REFUSED,
            content="无法基于当前证据回答。",
            citations=(),
        ),
        validation=AnswerValidationRecord(initial_issues=()),
    )
    return ChatPipelineState(
        context=context,
        evidence_pack=SimpleNamespace(),
        answering=answering,
    )
