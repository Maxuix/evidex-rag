from __future__ import annotations

import argparse
import unittest
from uuid import UUID

from rag_kb.domain.chat_pipeline import ChatPipelineExecutionError, ChatPipelinePhase
from rag_kb.domain.errors import ErrorCode
from tools.run_evaluation_feasibility_smoke import (
    FeasibilitySmokeError,
    _parser,
    _safe_failure_summary,
    _simple_identity_override,
)


class EvaluationFeasibilitySmokeTests(unittest.TestCase):
    def _arguments(self, **values: object) -> argparse.Namespace:
        defaults = {
            "simple_workspace_id": None,
            "simple_knowledge_base_id": None,
            "simple_index_revision_id": None,
            "simple_chat_profile_revision_id": None,
        }
        defaults.update(values)
        return argparse.Namespace(**defaults)

    def test_simple_identity_requires_all_four_bound_ids(self) -> None:
        with self.assertRaisesRegex(FeasibilitySmokeError, "simple_identity_incomplete"):
            _simple_identity_override(
                self._arguments(
                    simple_workspace_id=UUID("01900000-0000-7000-8000-000000000001")
                )
            )

    def test_simple_identity_requires_full_binding_for_simple_recovery(self) -> None:
        identity = _simple_identity_override(
            self._arguments(
                simple_workspace_id=UUID("01900000-0000-7000-8000-000000000001"),
                simple_knowledge_base_id=UUID("01900000-0000-7000-8000-000000000002"),
                simple_index_revision_id=UUID("01900000-0000-7000-8000-000000000003"),
                simple_chat_profile_revision_id=UUID("01900000-0000-7000-8000-000000000004"),
            )
        )
        self.assertEqual(
            str(identity["knowledge_base_id"]),
            "01900000-0000-7000-8000-000000000002",
        )
        self.assertIsNone(_simple_identity_override(self._arguments()))
        self.assertEqual(
            _parser()
            .parse_args(["--checkpoint", "a", "--locked-output", "b"])
            .agent_timeout_seconds,
            60.0,
        )

    def test_pipeline_failure_summary_is_content_safe(self) -> None:
        summary = _safe_failure_summary(
            ChatPipelineExecutionError(
                ErrorCode.CHAT_CONTEXT_INVALID,
                phase=ChatPipelinePhase.LOAD_CONTEXT,
                diagnostic={"check": "agent_budget", "untrusted": "not persisted"},
            )
        )
        self.assertEqual(
            summary,
            {
                "type": "ChatPipelineExecutionError",
                "code": "CHAT_CONTEXT_INVALID",
                "phase": "load_context",
                "diagnostic": {"check": "agent_budget"},
            },
        )


if __name__ == "__main__":
    unittest.main()
