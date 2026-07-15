from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import UUID

from pydantic import ValidationError

from rag_kb.domain import (
    AnswerStyle,
    InsufficiencyPolicy,
    resolve_p1_policy,
)
from rag_kb.schemas import ChatRunCreate
from rag_kb.services import chat_model_configuration


class ChatCreationContractTests(unittest.TestCase):
    def test_safe_defaults_and_all_four_override_pairs_are_complete(self) -> None:
        default = resolve_p1_policy(
            answer_style=None,
            insufficiency_policy=None,
        ).as_dict()
        self.assertEqual(default["answer_style"], "concise")
        self.assertEqual(default["insufficiency_policy"], "refuse")
        self.assertEqual(default["grounding_policy"], "evidence_only")
        self.assertTrue(default["citation_required"])
        self.assertEqual(default["citation_granularity"], "claim_level")
        self.assertEqual(default["answer_task"], "answer")

        pairs = {
            (
                resolve_p1_policy(
                    answer_style=style,
                    insufficiency_policy=insufficiency,
                ).answer_style,
                resolve_p1_policy(
                    answer_style=style,
                    insufficiency_policy=insufficiency,
                ).insufficiency_policy,
            )
            for style in AnswerStyle
            for insufficiency in InsufficiencyPolicy
        }
        self.assertEqual(len(pairs), 4)

    def test_public_request_normalizes_content_and_rejects_policy_weakening(self) -> None:
        request = ChatRunCreate.model_validate(
            {
                "session_id": "01900000-0000-7000-8000-000000000101",
                "knowledge_base_id": "01900000-0000-7000-8000-000000000102",
                "message": "  查询 RUN-ORD-14  ",
                "answer_policy": {
                    "answer_style": "summary",
                    "insufficiency_policy": "partial_answer",
                },
                "retrieval": {"mode": "vector", "top_k": 8},
            }
        )
        self.assertEqual(request.message, "查询 RUN-ORD-14")
        self.assertIs(request.answer_policy.answer_style, AnswerStyle.SUMMARY)

        for field_name in (
            "grounding_policy",
            "citation_required",
            "citation_granularity",
            "answer_task",
            "policy_version",
        ):
            with self.subTest(field_name=field_name), self.assertRaises(
                ValidationError
            ):
                ChatRunCreate.model_validate(
                    {
                        "session_id": "01900000-0000-7000-8000-000000000101",
                        "knowledge_base_id": "01900000-0000-7000-8000-000000000102",
                        "message": "question",
                        "answer_policy": {field_name: "client-controlled"},
                    }
                )

    def test_model_snapshot_excludes_url_key_and_runtime_controls(self) -> None:
        settings = SimpleNamespace(
            provider_identity="provider",
            logical_endpoint_identity="logical-endpoint",
            model="requested",
            resolved_model="resolved",
            model_version="version",
            structured_output_mode="json_object",
            configuration_fingerprint="sha256:configuration",
            capability_fingerprint="sha256:capability",
            base_url="https://secret-host.example/v1",
            api_key="secret",
            timeout_seconds=30,
            max_retries=2,
        )
        snapshot = chat_model_configuration(settings)
        self.assertEqual(snapshot["resolved_model"], "resolved")
        self.assertNotIn("base_url", snapshot)
        self.assertNotIn("api_key", snapshot)
        self.assertNotIn("timeout_seconds", snapshot)
        self.assertNotIn("max_retries", snapshot)


if __name__ == "__main__":
    unittest.main()
