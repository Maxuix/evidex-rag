from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rag_kb.domain import ChatPipelineExecutionError, ChatPipelinePhase, ErrorCode
from tools.evaluation_campaign_state import load_or_create
from tools.evaluation_resilience import RetryPolicy
from tools.run_large_evaluation import _run_cases


class LargeEvaluationRunnerResilienceTests(unittest.TestCase):
    def test_retryable_case_resumes_without_advancing_the_phase(self) -> None:
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "campaign.json"
            state = load_or_create(checkpoint, {"campaign": "frozen"})
            calls = 0

            async def producer():
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise ChatPipelineExecutionError(
                        ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
                        phase=ChatPipelinePhase.GENERATE_OR_REFUSE,
                        diagnostic={"check": "transport", "retryable": True},
                    )
                return {"actual_outcome": "answered", "total_tokens": 10}

            asyncio.run(
                _run_cases(
                    checkpoint,
                    state,
                    phase="answers",
                    cases=[("case-1", producer)],
                    retry_policy=RetryPolicy(
                        max_attempts=2,
                        backoff_seconds=(0.0,),
                        maximum_backoff_seconds=0.0,
                    ),
                    heartbeat_interval_seconds=0.01,
                )
            )

            record = state["phases"]["answers"]["case-1"]
            self.assertEqual(calls, 2)
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["attempt_count"], 2)
            self.assertEqual(record["retry_count"], 1)
            self.assertEqual(state["phase_summaries"]["answers"]["planned"], 1)
            self.assertEqual(state["phase_summaries"]["answers"]["completed"], 1)


if __name__ == "__main__":
    unittest.main()
