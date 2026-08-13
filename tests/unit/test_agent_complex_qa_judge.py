from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
from types import SimpleNamespace
import unittest
from typing import Any
from unittest.mock import AsyncMock, Mock, call, patch
from uuid import UUID, uuid4

from rag_kb.domain import (
    ChatModelExecutionError,
    ChatModelRequest,
    ChatModelResponse,
    ChatToolCall,
    ErrorCode,
    ModelKind,
    ModelProviderProtocol,
    ModelValidationStatus,
)
from tools.agent_complex_qa_judge import (
    ComplexQaLlmJudge,
    build_judge_packet,
    load_frozen_judge_runtime,
    semantic_score,
)


_JUDGE_MODEL = "fixed-judge-model"


class _FakeChatModelAdapter:
    def __init__(
        self,
        *responses: ChatModelResponse | ChatModelExecutionError,
    ) -> None:
        self.responses = list(responses)
        self.requests: list[ChatModelRequest] = []

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected Judge model call")
        value = self.responses.pop(0)
        if isinstance(value, ChatModelExecutionError):
            raise value
        return value


def _assessment(
    aspect_id: str | int,
    *,
    verdict: str = "correct",
    citation_support: str = "supported",
    material_omissions: tuple[str, ...] = (),
    unsupported_statements: tuple[str, ...] = (),
) -> dict[str, object]:
    aspect_number = (
        aspect_id
        if isinstance(aspect_id, int)
        else ord(aspect_id) - ord("a") + 1
    )
    return {
        "aspect_number": aspect_number,
        "verdict": verdict,
        "citation_support": citation_support,
        "material_omissions": list(material_omissions),
        "unsupported_statements": list(unsupported_statements),
    }


def _response(*assessments: dict[str, object]) -> ChatModelResponse:
    return _response_arguments({"aspects": list(assessments)})


def _response_arguments(arguments: dict[str, object]) -> ChatModelResponse:
    return ChatModelResponse(
        content="",
        model=_JUDGE_MODEL,
        finish_reason="tool_calls",
        provider_request_id="offline-test-request",
        usage={"input_tokens": 10, "output_tokens": 5},
        tool_calls=(
            ChatToolCall(
                "submit-judgement",
                "submit_judgement",
                arguments,
            ),
        ),
    )


def _model_error(
    *,
    check: str,
    retryable: bool,
) -> ChatModelExecutionError:
    return ChatModelExecutionError(
        ErrorCode.CHAT_PROVIDER_UNAVAILABLE,
        diagnostic={"check": check, "retryable": retryable},
    )


def _packet(*aspect_ids: str) -> dict[str, Any]:
    return {
        "question": "Compare the supplied facts.",
        "reference": {
            "case_notes": None,
            "aspects": [
                {
                    "aspect_id": aspect_id,
                    "reference_facts": {
                        "reference_answer_cues": [aspect_id],
                    },
                    "gold_evidence": [],
                }
                for aspect_id in aspect_ids
            ],
        },
        "agent_result": {
            "run_completed": True,
            "final_answer": "A grounded answer.",
            "cited_evidence": [],
            "citation_facts": {},
        },
    }


def _judge(model: _FakeChatModelAdapter, profile_id: UUID | None = None):
    return ComplexQaLlmJudge(
        model,
        profile_revision_id=profile_id or uuid4(),
        expected_model=_JUDGE_MODEL,
    )


def _reverse_mapping_order(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _reverse_mapping_order(value[key])
            for key in reversed(tuple(value))
        }
    if isinstance(value, list):
        return [_reverse_mapping_order(item) for item in value]
    return value


def _runtime_bundle(
    *,
    profile_revision_id: UUID,
    provider_revision_id: UUID,
    validation_status: ModelValidationStatus = ModelValidationStatus.VALID,
) -> SimpleNamespace:
    workspace_id = uuid4()
    return SimpleNamespace(
        profile=SimpleNamespace(
            kind=ModelKind.CHAT,
            enabled=True,
        ),
        provider=SimpleNamespace(enabled=True),
        current_revision=SimpleNamespace(
            id=profile_revision_id,
            revision=7,
            validation_status=validation_status,
            configuration={
                "temperature": 0.1,
                "top_p": 0.9,
                "sampling_top_k": None,
                "reasoning_effort": "off",
                "max_output_tokens": 2048,
            },
            model="fixed-judge-model",
            configuration_fingerprint="profile-config",
            capability_fingerprint="profile-capability",
        ),
        provider_revision=SimpleNamespace(
            id=provider_revision_id,
            workspace_id=workspace_id,
            protocol=ModelProviderProtocol.OPENAI_COMPATIBLE,
            base_url="https://provider.invalid/v1",
            secret_reference="secret-ref",
            timeout_seconds=60.0,
            max_retries=1,
            configuration_fingerprint="provider-config",
        ),
    )


@contextmanager
def _runtime_patches(
    *,
    database: object,
    bundle: object,
    secret_store: object,
    adapter: object,
):
    settings = SimpleNamespace(
        database=SimpleNamespace(
            runtime_dsn=SimpleNamespace(
                get_secret_value=Mock(return_value="postgresql://local")
            ),
            worker_statement_timeout_ms=1000,
            lock_timeout_ms=1000,
            idle_in_transaction_session_timeout_ms=1000,
        ),
        identity=SimpleNamespace(workspace_id=uuid4()),
        model_secrets=SimpleNamespace(root_path="/model-secrets"),
    )
    transaction = AsyncMock(return_value=bundle)
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "tools.agent_complex_qa_judge.load_settings",
                return_value=settings,
            )
        )
        stack.enter_context(
            patch(
                "tools.agent_complex_qa_judge.create_database_resources",
                return_value=database,
            )
        )
        stack.enter_context(
            patch(
                "tools.agent_complex_qa_judge.SqlAlchemyUnitOfWorkFactory",
                return_value=object(),
            )
        )
        stack.enter_context(
            patch(
                "tools.agent_complex_qa_judge.execute_in_transaction",
                new=transaction,
            )
        )
        stack.enter_context(
            patch(
                "tools.agent_complex_qa_judge.LocalModelSecretStore",
                return_value=secret_store,
            )
        )
        stack.enter_context(
            patch(
                "tools.agent_complex_qa_judge.LangChainChatModelAdapter",
                return_value=adapter,
            )
        )
        yield transaction


class ComplexQaLlmJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_frozen_runtime_uses_exact_valid_revision_and_closes(self) -> None:
        profile_revision_id = uuid4()
        provider_revision_id = uuid4()
        database = SimpleNamespace(close=AsyncMock(), sessions=object())
        bundle = _runtime_bundle(
            profile_revision_id=profile_revision_id,
            provider_revision_id=provider_revision_id,
        )
        adapter = object()
        secret_store = SimpleNamespace(read=Mock(return_value="provider-secret"))

        with _runtime_patches(
            database=database,
            bundle=bundle,
            secret_store=secret_store,
            adapter=adapter,
        ) as transaction:
            runtime = await load_frozen_judge_runtime(
                profile_revision_id,
                env_file="/dev/null",
            )

        self.assertIs(runtime.model, adapter)
        self.assertEqual(runtime.profile_revision_id, profile_revision_id)
        self.assertEqual(runtime.provider_revision_id, provider_revision_id)
        self.assertEqual(runtime.max_output_tokens, 2048)
        self.assertEqual(runtime.provider_timeout_seconds, 60.0)
        self.assertEqual(runtime.provider_max_retries, 1)
        self.assertEqual(runtime.config()["temperature"], 0.1)
        transaction.assert_awaited_once()
        secret_store.read.assert_called_once_with("secret-ref")
        await runtime.close()
        database.close.assert_awaited_once()

    async def test_frozen_runtime_rejects_invalid_revision_and_closes(self) -> None:
        profile_revision_id = uuid4()
        database = SimpleNamespace(close=AsyncMock(), sessions=object())
        bundle = _runtime_bundle(
            profile_revision_id=profile_revision_id,
            provider_revision_id=uuid4(),
            validation_status=ModelValidationStatus.INVALID,
        )
        secret_store = SimpleNamespace(read=Mock(return_value="provider-secret"))

        with _runtime_patches(
            database=database,
            bundle=bundle,
            secret_store=secret_store,
            adapter=object(),
        ):
            with self.assertRaisesRegex(ValueError, "enabled valid chat model"):
                await load_frozen_judge_runtime(
                    profile_revision_id,
                    env_file="/dev/null",
                )

        database.close.assert_awaited_once()
        secret_store.read.assert_not_called()

    async def test_agreeing_judges_do_not_call_c_and_merge_by_aspect_id(self) -> None:
        model = _FakeChatModelAdapter(
            _response(
                _assessment(
                    "b",
                    verdict="incorrect",
                    citation_support="unsupported",
                    material_omissions=("Missing B",),
                ),
                _assessment("a", material_omissions=("First wording",)),
            ),
            _response(
                _assessment(
                    "a",
                    material_omissions=("first   wording", "Second wording"),
                ),
                _assessment(
                    "b",
                    verdict="incorrect",
                    citation_support="unsupported",
                    material_omissions=("Missing B again",),
                ),
            ),
        )

        result = await _judge(model).judge(_packet("a", "b"))

        self.assertEqual(len(model.requests), 2)
        self.assertEqual(result["disputed_aspect_ids"], [])
        self.assertIsNone(result["calls"]["judge_c"])
        self.assertEqual(
            [item["aspect_id"] for item in result["aspects"]],
            ["a", "b"],
        )
        self.assertEqual(
            [item["verdict"] for item in result["aspects"]],
            ["correct", "incorrect"],
        )
        self.assertEqual(
            result["aspects"][0]["material_omissions"],
            ["First wording", "Second wording"],
        )
        self.assertTrue(
            all(
                item["consensus_source"] == "judge_a_b_agreement"
                for item in result["aspects"]
            )
        )

    async def test_total_timeout_retries_and_records_transport_attempts(self) -> None:
        model = _FakeChatModelAdapter(
            _model_error(check="total_timeout", retryable=False),
            _response(_assessment("a")),
            _response(_assessment("a")),
        )

        with patch(
            "tools.agent_complex_qa_judge.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            result = await _judge(model).judge(_packet("a"))

        self.assertEqual(len(model.requests), 3)
        sleep.assert_awaited_once_with(1.0)
        self.assertEqual(result["calls"]["judge_a"]["transport_attempts"], 2)
        self.assertEqual(result["calls"]["judge_b"]["transport_attempts"], 1)

    async def test_three_transient_failures_exhaust_retry_and_raise(self) -> None:
        model = _FakeChatModelAdapter(
            *(
                _model_error(check="total_timeout", retryable=True)
                for _ in range(3)
            )
        )

        with patch(
            "tools.agent_complex_qa_judge.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep, self.assertRaises(ChatModelExecutionError) as raised:
            await _judge(model).judge(_packet("a"))

        self.assertEqual(raised.exception.diagnostic["check"], "total_timeout")
        self.assertEqual(len(model.requests), 3)
        self.assertEqual(sleep.await_args_list, [call(1.0), call(2.0)])

    async def test_nonretryable_non_timeout_error_is_not_retried(self) -> None:
        model = _FakeChatModelAdapter(
            _model_error(check="provider_rejected", retryable=False),
            _response(_assessment("a")),
        )

        with patch(
            "tools.agent_complex_qa_judge.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep, self.assertRaises(ChatModelExecutionError) as raised:
            await _judge(model).judge(_packet("a"))

        self.assertEqual(raised.exception.diagnostic["retryable"], False)
        self.assertEqual(len(model.requests), 1)
        sleep.assert_not_awaited()

    async def test_only_disputed_aspect_is_sent_to_c_and_c_is_adopted(self) -> None:
        model = _FakeChatModelAdapter(
            _response(
                _assessment("a"),
                _assessment(
                    "b",
                    verdict="partial",
                    citation_support="partially_supported",
                ),
            ),
            _response(
                _assessment(
                    "b",
                    verdict="incorrect",
                    citation_support="unsupported",
                ),
                _assessment("a"),
            ),
            _response(
                _assessment(
                    1,
                    verdict="correct",
                    citation_support="supported",
                    unsupported_statements=("C finding",),
                )
            ),
        )

        result = await _judge(model).judge(_packet("a", "b"))

        self.assertEqual(len(model.requests), 3)
        self.assertEqual(result["disputed_aspect_ids"], ["b"])
        arbitration_packet = json.loads(model.requests[2].messages[1].content)
        self.assertEqual(
            [
                item["aspect_id"]
                for item in arbitration_packet["reference"]["aspects"]
            ],
            ["b"],
        )
        aspect_number_schema = model.requests[2].tools[0].input_schema["properties"][
            "aspects"
        ]["items"]["properties"]["aspect_number"]
        self.assertEqual(aspect_number_schema["type"], "integer")
        by_id = {item["aspect_id"]: item for item in result["aspects"]}
        self.assertEqual(by_id["a"]["consensus_source"], "judge_a_b_agreement")
        self.assertEqual(by_id["b"]["consensus_source"], "judge_c")
        self.assertEqual(by_id["b"]["verdict"], "correct")
        self.assertEqual(by_id["b"]["citation_support"], "supported")
        self.assertEqual(by_id["b"]["unsupported_statements"], ["C finding"])

    async def test_missing_duplicate_and_invalid_verdict_fail_explicitly(self) -> None:
        invalid_responses = {
            "missing": _response(_assessment("a")),
            "duplicate": _response(_assessment("a"), _assessment("a")),
            "invalid_verdict": _response(
                _assessment("a", verdict="mostly_correct"),
                _assessment("b"),
            ),
        }

        for name, response in invalid_responses.items():
            with self.subTest(name=name):
                model = _FakeChatModelAdapter(response, response, response)
                with patch(
                    "tools.agent_complex_qa_judge.asyncio.sleep",
                    new=AsyncMock(),
                ), self.assertRaisesRegex(RuntimeError, "judge"):
                    await _judge(model).judge(_packet("a", "b"))
                self.assertEqual(len(model.requests), 3)

    async def test_generic_schema_is_stable_across_aspect_sets(self) -> None:
        model = _FakeChatModelAdapter(
            _response(_assessment("a")),
            _response(_assessment("a")),
            _response(_assessment(1), _assessment(2)),
            _response(_assessment(2), _assessment(1)),
        )
        judge = _judge(model)

        await judge.judge(_packet("a"))
        await judge.judge(_packet("x", "y"))

        first_schema = model.requests[0].tools[0].input_schema
        second_schema = model.requests[2].tools[0].input_schema
        self.assertEqual(first_schema, second_schema)
        aspect_number_schema = first_schema["properties"]["aspects"]["items"][
            "properties"
        ]["aspect_number"]
        self.assertEqual(
            aspect_number_schema,
            {"type": "integer", "minimum": 1, "maximum": 64},
        )
        self.assertFalse(first_schema["additionalProperties"])

    async def test_extra_structured_result_fields_fail_explicitly(self) -> None:
        invalid_responses = {
            "top_level": _response_arguments(
                {"aspects": [_assessment("a")], "unexpected": True}
            ),
            "aspect": _response(
                {**_assessment("a"), "unexpected": True}
            ),
        }

        for name, response in invalid_responses.items():
            with self.subTest(name=name):
                model = _FakeChatModelAdapter(response, response, response)
                with patch(
                    "tools.agent_complex_qa_judge.asyncio.sleep",
                    new=AsyncMock(),
                ), self.assertRaisesRegex(RuntimeError, "judge result"):
                    await _judge(model).judge(_packet("a"))
                self.assertEqual(len(model.requests), 3)

    async def test_malformed_structured_result_is_retried_then_succeeds(self) -> None:
        model = _FakeChatModelAdapter(
            _response_arguments({"aspects": [], "unexpected": True}),
            _response(_assessment("a")),
            _response(_assessment("a")),
        )

        with patch(
            "tools.agent_complex_qa_judge.asyncio.sleep",
            new=AsyncMock(),
        ):
            result = await _judge(model).judge(_packet("a"))

        self.assertEqual(len(model.requests), 3)
        self.assertEqual(result["calls"]["judge_a"]["transport_attempts"], 2)
        self.assertEqual(result["calls"]["judge_b"]["transport_attempts"], 1)

    async def test_packet_allowlist_excludes_blinding_fields_and_hash_is_stable(
        self,
    ) -> None:
        case = {
            "case_id": "complex-secret",
            "question": "What is the stated amount?",
            "notes": "Use the supplied evidence.",
            "agent_model": "must-not-leak-from-case",
            "legacy_score": {"strict_correct": False},
            "aspects": [
                {
                    "aspect_id": "amount",
                    "answer_variants": ["ten"],
                    "expected_decimal": "10",
                    "numeric_tolerance": "0.1",
                    "source": [
                        {
                            "source_case_id": "base-1",
                            "evidence_locator": {"kind": "base_case_evidence"},
                        }
                    ],
                }
            ],
            "required_citation_document_ids": ["doc-a"],
        }
        reference_cases = {
            "base-1": {
                "document_id": "doc-a",
                "gold": {"answer": "ten"},
                "justification": "The text says ten.",
                "evidence": {
                    "kind": "text_spans",
                    "items": [{"text": "The amount is ten."}],
                },
            }
        }
        run = {
            "status": "completed",
            "answer": "The amount is ten.",
            "agent": {
                "model": "secret-agent-model",
                "retrieval": {"strategy": "hybrid-secret"},
                "trace": {"outcome": "answered"},
            },
            "retrieval_strategy": "hybrid-secret",
            "legacy_score": {"strict_correct": False},
            "citations": [
                {
                    "ordinal": 1,
                    "document_id": "doc-a",
                    "quoted_text": "The amount is ten.",
                    "source_location": {"page": 1},
                    "modality": "text",
                }
            ],
        }

        packet = build_judge_packet(
            case,
            run,
            reference_cases=reference_cases,
            citation_document_id=lambda citation: str(citation["document_id"]),
        )

        self.assertEqual(set(packet), {"question", "reference", "agent_result"})
        self.assertEqual(set(packet["reference"]), {"case_notes", "aspects"})
        self.assertEqual(
            set(packet["agent_result"]),
            {
                "run_completed",
                "final_answer",
                "cited_evidence",
                "citation_facts",
            },
        )
        serialized = json.dumps(packet, ensure_ascii=False, sort_keys=True)
        for forbidden in (
            "secret-agent-model",
            "hybrid-secret",
            "legacy_score",
            "must-not-leak-from-case",
        ):
            self.assertNotIn(forbidden, serialized)

        agreement = _response(_assessment(1))
        model = _FakeChatModelAdapter(
            agreement,
            agreement,
            agreement,
            agreement,
        )
        judge = _judge(model)
        first = await judge.judge(packet)
        second = await judge.judge(_reverse_mapping_order(packet))

        self.assertEqual(first["input_sha256"], second["input_sha256"])
        self.assertEqual(
            model.requests[0].messages[1].content,
            model.requests[2].messages[1].content,
        )


class SemanticScoreTests(unittest.TestCase):
    def test_strict_requires_correct_verdict_and_supported_citation(self) -> None:
        correct = _assessment("a")
        score = semantic_score({"aspects": [correct]})
        self.assertTrue(score["semantic_strict_correct"])
        self.assertTrue(score["semantic_grounding_supported"])

        for field, value in (
            ("verdict", "partial"),
            ("citation_support", "partially_supported"),
        ):
            with self.subTest(field=field):
                changed = {**correct, field: value}
                score = semantic_score({"aspects": [changed]})
                self.assertFalse(score["semantic_strict_correct"])


if __name__ == "__main__":
    unittest.main()
