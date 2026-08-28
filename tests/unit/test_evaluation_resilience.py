from __future__ import annotations

import unittest

from rag_kb.domain import (
    ChatPipelineExecutionError,
    ChatPipelinePhase,
    ErrorCode,
)
from tools.evaluation_resilience import (
    RetryPolicy,
    is_retryable_provider_failure,
    safe_failure_summary,
    stable_error_code,
)


class EvaluationResilienceTests(unittest.TestCase):
    def test_retry_policy_is_bounded_and_deterministic(self) -> None:
        policy = RetryPolicy(
            max_attempts=4,
            backoff_seconds=(1.0, 2.0),
            maximum_backoff_seconds=1.5,
        )

        self.assertEqual(policy.delay_after(1), 1.0)
        self.assertEqual(policy.delay_after(2), 1.5)
        self.assertEqual(policy.delay_after(20), 1.5)

    def test_provider_transport_and_deadline_failures_are_retryable(self) -> None:
        transport = ChatPipelineExecutionError(
            ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
            phase=ChatPipelinePhase.GENERATE_OR_REFUSE,
            diagnostic={"check": "transport", "retryable": True},
        )
        deadline = ChatPipelineExecutionError(
            ErrorCode.CHAT_PIPELINE_DEADLINE_EXCEEDED,
            phase=ChatPipelinePhase.GENERATE_OR_REFUSE,
        )

        self.assertTrue(is_retryable_provider_failure(transport))
        self.assertTrue(is_retryable_provider_failure(deadline))
        self.assertEqual(stable_error_code(transport), "CHAT_PROVIDER_UNAVAILABLE")

    def test_non_provider_failure_is_not_retryable_and_summary_is_safe(self) -> None:
        failure = ChatPipelineExecutionError(
            ErrorCode.CHAT_RESPONSE_INVALID,
            phase=ChatPipelinePhase.GENERATE_OR_REFUSE,
            diagnostic={"check": "schema", "untrusted": "not persisted"},
        )

        self.assertFalse(is_retryable_provider_failure(failure))
        self.assertEqual(
            safe_failure_summary(failure),
            {
                "type": "ChatPipelineExecutionError",
                "code": "CHAT_RESPONSE_INVALID",
                "diagnostic": {"check": "schema"},
            },
        )


if __name__ == "__main__":
    unittest.main()
