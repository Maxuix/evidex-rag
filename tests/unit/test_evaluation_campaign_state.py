from __future__ import annotations

import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest

from tools.evaluation_campaign_state import (
    CampaignStateError,
    begin_case,
    complete_case,
    completed_case_ids,
    load_or_create,
    phase_progress,
)


class EvaluationCampaignStateTests(unittest.TestCase):
    def test_completed_cases_are_skipped_and_interrupted_cases_are_retried(self) -> None:
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "campaign.json"
            binding = {"corpora": [{"dataset_id": "frozen-v1", "sha256": "a" * 64}]}
            state = load_or_create(checkpoint, binding)
            self.assertTrue(begin_case(checkpoint, state, phase="answer", case_id="case-1"))
            complete_case(
                checkpoint,
                state,
                phase="answer",
                case_id="case-1",
                observation={"outcome": "answered", "total_tokens": 123},
            )
            self.assertFalse(begin_case(checkpoint, state, phase="answer", case_id="case-1"))
            self.assertTrue(begin_case(checkpoint, state, phase="answer", case_id="case-2"))

            resumed = load_or_create(checkpoint, binding)
            self.assertEqual(completed_case_ids(resumed, phase="answer"), {"case-1"})
            self.assertTrue(begin_case(checkpoint, resumed, phase="answer", case_id="case-2"))
            self.assertEqual(phase_progress(resumed, phase="answer"), {"started_or_completed": 2, "completed": 1})
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(payload["phases"]["answer"]["case-2"]["attempt_count"], 2)
            self.assertTrue(any(event["event"] == "case_resumed_after_interruption" for event in payload["events"]))
            self.assertEqual(stat.S_IMODE(checkpoint.stat().st_mode), 0o600)

    def test_rejects_resume_against_a_different_frozen_binding(self) -> None:
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "campaign.json"
            load_or_create(checkpoint, {"corpora": ["v1"]})
            with self.assertRaisesRegex(CampaignStateError, "binding_mismatch"):
                load_or_create(checkpoint, {"corpora": ["v2"]})


if __name__ == "__main__":
    unittest.main()
