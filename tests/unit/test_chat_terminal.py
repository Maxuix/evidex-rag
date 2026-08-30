from __future__ import annotations

import asyncio
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
    ChatAnsweringState,
    ChatAgentBudget,
    ChatAgentTrace,
    ChatExecutionCommand,
    ChatExecutionContext,
    ChatModelCallRecord,
    ChatModelOperation,
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ChatPipelineState,
    ChatProgressStage,
    ChatRunLease,
    ChatTerminalWriteStatus,
    ErrorCode,
    EvidenceEnvelope,
    RenderedAnswer,
    ValidatedAnswer,
    VisualEvidenceDecision,
    VisualEvidenceReason,
)
from rag_kb.repositories.sqlalchemy_chat import (
    _serialized_failure,
    _serialized_success,
)
from rag_kb.answering.runner import NativeAgentRunner
from rag_kb.services.chat_terminal import (
    ChatFailureSettlementService,
    ChatResultPersistenceStep,
)


class ChatTerminalServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_runner_reports_only_real_completed_stages(self) -> None:
        observed = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
        state = _completed_state(observed)
        reporter = _Reporter()
        agent = _Agent(state)
        runner = NativeAgentRunner(
            _ContextLoader(state.context),
            agent,
            _Persister(state),
            deadline_seconds=1,
            progress_reporter_factory=lambda *_: reporter,
        )

        result = await runner.execute(ChatExecutionCommand(state.context.lease))

        self.assertIs(result, state)
        self.assertEqual(agent.deadline_seconds, 1)
        completed = reporter.shown[-1][2]
        self.assertEqual(
            completed,
            (
                ChatProgressStage.RETRIEVE_EVIDENCE,
                ChatProgressStage.GENERATE_ANSWER,
            ),
        )
        self.assertNotIn(ChatProgressStage.PREPARE_VISUAL_EVIDENCE, completed)
        self.assertNotIn(ChatProgressStage.VALIDATE_ANSWER, completed)

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
        facts = _serialized_success(command)
        self.assertEqual(facts["control_reason"], "no_usable_evidence")
        self.assertNotIn("validation", facts)
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
        self.assertIsNone(repository.success.agent_trace)

    async def test_success_persists_current_agent_trace_v3(self) -> None:
        observed = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
        state = replace(
            _completed_state(observed),
            artifacts={
                "chat_agent_trace": ChatAgentTrace(
                    events=(),
                    budget=ChatAgentBudget(max_model_rounds=1, max_graph_calls=2),
                    model_rounds=2,
                    retrieval_calls=7,
                    calculation_calls=5,
                    evidence_ref_count=105,
                    outcome="refused",
                )
            },
        )
        repository = _Repository(ChatTerminalWriteStatus.APPLIED)

        await ChatResultPersistenceStep(_Factory(repository)).run(state)

        self.assertEqual(
            repository.success.agent_trace["version"],
            "native_tool_calling_agent_v3",
        )
        self.assertEqual(
            repository.success.agent_trace["budget"],
            ChatAgentBudget(max_model_rounds=1, max_graph_calls=2).as_dict(),
        )
        self.assertEqual(
            repository.success.agent_trace["usage"]["evidence_refs"], 105
        )
        self.assertEqual(
            repository.success.agent_trace["diagnostics"]["stop_reason"],
            "submitted",
        )

    async def test_timeout_retains_nonblocking_agent_checkpoint_per_attempt(self) -> None:
        observed = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
        state = _completed_state(observed)
        runner = NativeAgentRunner(
            _ContextLoader(state.context),
            _BlockingAgent(),
            _Persister(state),
            deadline_seconds=0.001,
            progress_reporter_factory=lambda *_: _Reporter(),
        )

        with self.assertRaises(ChatPipelineExecutionError) as raised:
            await runner.execute(ChatExecutionCommand(state.context.lease))

        error = raised.exception
        self.assertEqual(error.code, ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED)
        self.assertEqual(error.model_calls, (_call(),))
        self.assertIsNotNone(error.agent_trace)
        self.assertTrue(error.agent_trace["diagnostics"]["partial"])
        self.assertTrue(error.agent_trace["diagnostics"]["deadline_exceeded"])

        repository = _Repository(ChatTerminalWriteStatus.APPLIED)
        await ChatFailureSettlementService(
            _Factory(repository),
            max_attempts=1,
            base_delay_seconds=1,
            max_delay_seconds=1,
            clock=lambda: observed + timedelta(seconds=1),
        ).settle(state.context.lease, error)
        self.assertEqual(
            repository.failure.agent_trace["usage"]["model_rounds"], 1
        )
        self.assertEqual(
            _serialized_failure(repository.failure)["agent_trace"]["diagnostics"][
                "stop_reason"
            ],
            "deadline_exceeded",
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

    async def test_provider_nonretryable_diagnostic_prevents_requeue(self) -> None:
        observed = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
        repository = _Repository(ChatTerminalWriteStatus.APPLIED)
        service = ChatFailureSettlementService(
            _Factory(repository),
            max_attempts=3,
            base_delay_seconds=5,
            max_delay_seconds=30,
            clock=lambda: observed + timedelta(seconds=1),
        )

        await service.settle(
            _lease(observed, attempt=1),
            ChatPipelineExecutionError(
                ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                phase=ChatPipelinePhase.RETRIEVE_EVIDENCE,
                diagnostic={"http_status": 400, "retryable": False},
            ),
        )

        self.assertFalse(repository.failure.retryable)
        self.assertFalse(repository.failure.exhausted)
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
        operation=ChatModelOperation.AGENT_ROUND,
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
        usable_citation_ids=(),
        draft=draft,
        model_calls=(_call(),),
        validated=validated,
        rendered=RenderedAnswer(
            outcome=AnswerOutcome.REFUSED,
            content="无法基于当前证据回答。",
            citations=(),
            control_reason=AnswerControlReason.NO_USABLE_EVIDENCE,
        ),
    )
    return ChatPipelineState(
        context=context,
        evidence_pack=SimpleNamespace(evidence=()),
        answering=answering,
    )


class _ContextLoader:
    def __init__(self, context) -> None:
        self.context = context

    async def load(self, command):
        return self.context


class _Agent:
    def __init__(self, state) -> None:
        self.state = state
        self.deadline_seconds = None

    async def run(self, context, *, deadline_seconds=None, progress=None):
        self.deadline_seconds = deadline_seconds
        return self.state


class _BlockingAgent:
    async def run(self, context, *, deadline_seconds=None, progress=None):
        del context, deadline_seconds
        progress.budget = ChatAgentBudget()
        progress.model_calls.append(_call())
        progress.model_rounds = 1
        await asyncio.Event().wait()


class _Persister:
    def __init__(self, state) -> None:
        self.state = state

    async def run(self, state):
        return self.state


class _Reporter:
    def __init__(self) -> None:
        self.shown = []

    async def show(self, stage, activity, *, facts=None, completed=()) -> None:
        self.shown.append((stage, activity, completed))

    async def finish(self, activity) -> None:
        return None
