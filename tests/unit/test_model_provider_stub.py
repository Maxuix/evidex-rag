from __future__ import annotations

import json
import unittest

from tools.model_provider_stub import (
    DIMENSION,
    ProviderScenario,
    deterministic_embedding,
)


class ModelProviderStubTests(unittest.TestCase):
    def test_embeddings_are_normalized_and_failure_is_finite(self) -> None:
        alpha = deterministic_embedding("E2E_ALPHA")
        beta = deterministic_embedding("E2E_BETA")
        self.assertEqual(len(alpha), DIMENSION)
        self.assertEqual(sum(value * value for value in alpha), 1.0)
        self.assertNotEqual(alpha, beta)

        scenario = ProviderScenario(indexing_failures=2)
        payload = {"input": ["E2E_INDEX_FAIL"]}
        self.assertEqual(scenario.embeddings(payload).status, 503)
        self.assertEqual(scenario.embeddings(payload).status, 503)
        recovered = scenario.embeddings(payload)
        self.assertEqual(recovered.status, 200)
        self.assertEqual(len(recovered.body["data"][0]["embedding"]), DIMENSION)

    def test_chat_returns_strict_assessment_and_grounded_draft(self) -> None:
        scenario = ProviderScenario(chat_delay_seconds=0.25)
        assessment = scenario.chat(
            {
                "messages": [
                    {"content": "You assess whether supplied evidence supports a question."},
                    {
                        "content": json.dumps(
                            {
                                "question": "E2E_ALPHA",
                                "evidence": [{"citation_id": "cite_1"}],
                            }
                        )
                    },
                ]
            }
        )
        self.assertEqual(assessment.status, 200)
        content = json.loads(assessment.body["choices"][0]["message"]["content"])
        self.assertEqual(content["coverage"], "sufficient")
        self.assertEqual(content["usable_citation_ids"], ["cite_1"])

        generated = scenario.chat(
            {
                "messages": [
                    {"content": "You produce an unvalidated internal answer draft."},
                    {
                        "content": json.dumps(
                            {
                                "question": "E2E_ALPHA E2E_CHAT_DELAY",
                                "required_outcome": "answered",
                                "evidence": [{"citation_id": "cite_1"}],
                                "missing_aspects": [],
                            }
                        )
                    },
                ]
            }
        )
        self.assertEqual(generated.delay_seconds, 0.25)
        draft = json.loads(generated.body["choices"][0]["message"]["content"])
        self.assertEqual(draft["claims"][0]["citation_ids"], ["cite_1"])

    def test_chat_failure_is_content_safe(self) -> None:
        failed = ProviderScenario().chat(
            {
                "messages": [
                    {"content": "You assess whether supplied evidence supports a question."},
                    {
                        "content": json.dumps(
                            {"question": "E2E_CHAT_FAIL", "evidence": []}
                        )
                    },
                ]
            }
        )
        self.assertEqual(failed.status, 503)
        self.assertNotIn("question", json.dumps(failed.body))


if __name__ == "__main__":
    unittest.main()
