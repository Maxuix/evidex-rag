from __future__ import annotations

import json
import unittest

from deploy.operations_provider.provider import (
    DIMENSION,
    OperationsScenario,
    deterministic_embedding,
)


class OperationsProviderStubTests(unittest.TestCase):
    def test_embedding_delay_and_failure_are_bounded(self) -> None:
        scenario = OperationsScenario(
            indexing_failures=2,
            indexing_delay_seconds=0.25,
        )
        delayed = scenario.embeddings({"input": ["OPS_ALPHA OPS_INDEX_DELAY"]})
        self.assertEqual(delayed.status, 200)
        self.assertEqual(delayed.delay_seconds, 0.25)
        self.assertEqual(len(delayed.body["data"][0]["embedding"]), DIMENSION)

        payload = {"input": ["OPS_INDEX_FAIL"]}
        self.assertEqual(scenario.embeddings(payload).status, 503)
        self.assertEqual(scenario.embeddings(payload).status, 503)
        self.assertEqual(scenario.embeddings(payload).status, 200)

    def test_embeddings_are_deterministic_normalized_and_separated(self) -> None:
        alpha = deterministic_embedding("OPS_ALPHA")
        beta = deterministic_embedding("OPS_BETA")
        self.assertEqual(alpha, deterministic_embedding("OPS_ALPHA"))
        self.assertNotEqual(alpha, beta)
        self.assertEqual(sum(value * value for value in alpha), 1.0)

    def test_chat_delay_preserves_strict_grounded_shapes(self) -> None:
        scenario = OperationsScenario(chat_delay_seconds=0.25)
        assessment = scenario.chat(
            {
                "messages": [
                    {"content": "You assess whether supplied evidence supports a question."},
                    {
                        "content": json.dumps(
                            {
                                "question": "OPS_ALPHA OPS_CHAT_DELAY",
                                "evidence": [{"citation_id": "cite_1"}],
                            }
                        )
                    },
                ]
            }
        )
        self.assertEqual(assessment.status, 200)
        self.assertEqual(assessment.delay_seconds, 0.25)
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
                                "question": "OPS_ALPHA",
                                "required_outcome": "answered",
                                "evidence": [{"citation_id": "cite_1"}],
                            }
                        )
                    },
                ]
            }
        )
        draft = json.loads(generated.body["choices"][0]["message"]["content"])
        self.assertEqual(draft["claims"][0]["citation_ids"], ["cite_1"])

    def test_invalid_requests_are_content_safe(self) -> None:
        value = OperationsScenario().embeddings({"input": "secret content"})
        self.assertEqual(value.status, 400)
        self.assertNotIn("secret", json.dumps(value.body))


if __name__ == "__main__":
    unittest.main()
