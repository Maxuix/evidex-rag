from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from tools.evaluate_agent_complex_qa import (
    _answer_checkpoint_path,
    _atomic_write_json,
    apply_llm_judgements,
    _extract_decimal_values,
    _validate_options,
    evaluate_cases,
    _validated_api_base,
    load_evaluation_artifact,
    load_document_identity_map,
    score_complex_case,
    summarize_results,
)
from tools.agent_complex_qa_judge import judge_packet_sha256
from tools.build_document_qa_corpus import COMPLEX_CASE_DEFINITIONS


def _agent(
    version: str = "native_tool_calling_agent_v1",
) -> dict[str, object]:
    return {
        "version": version,
        "trace": {
            "outcome": "answered",
            "usage": {},
        },
    }


def _minimal_case(
    case_id: str = "complex-test",
    *,
    question: str = "What fact is stated?",
    aspect_id: str = "fact",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "question": question,
        "evaluation_group": "evidence_only",
        "required_citation_document_ids": [],
        "aspects": [
            {
                "aspect_id": aspect_id,
                "answer_variants": ["answer"],
                "source": [],
            }
        ],
    }


def _legacy_result(
    case: dict[str, object],
    *,
    status: str = "completed",
) -> dict[str, object]:
    aspects = case["aspects"]
    assert isinstance(aspects, list)
    first_aspect = aspects[0]
    assert isinstance(first_aspect, dict)
    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "status": status,
        "answer": "The answer is stated.",
        "agent": _agent("native_tool_calling_agent_v2"),
        "citations": [],
        "score": {
            "strict_correct": False,
            "at_least_partial": True,
            "agent_protocol_ok": True,
            "required_document_citation_coverage": 1.0,
            "cited_document_ids": [],
            "forbidden_citation_document_ids": [],
            "agent_outcome": "answered",
            "answered_precision_ok": True,
            "aspects": [
                {
                    "aspect_id": first_aspect["aspect_id"],
                    "matched": False,
                }
            ],
        },
    }


class _FakeJudge:
    def __init__(self) -> None:
        self.profile_revision_id = uuid4()
        self.packets: list[dict[str, object]] = []

    async def judge(self, packet):
        self.packets.append(packet)
        aspect_id = packet["reference"]["aspects"][0]["aspect_id"]
        return {
            "schema_version": "native_agent_complex_qa_llm_judge_v2",
            "status": "judged",
            "prompt_version": "agent_complex_qa_semantic_judge_v2",
            "profile_revision_id": str(self.profile_revision_id),
            "input_sha256": "a" * 64,
            "disputed_aspect_ids": [],
            "aspects": [
                {
                    "aspect_id": aspect_id,
                    "verdict": "correct",
                    "citation_support": "supported",
                    "material_omissions": [],
                    "unsupported_statements": [],
                    "consensus_source": "judge_a_b_agreement",
                }
            ],
            "calls": {
                "judge_a": {"usage": {"total_tokens": 10}},
                "judge_b": {"usage": {"total_tokens": 10}},
                "judge_c": None,
            },
        }


class _CacheJudge(_FakeJudge):
    def __init__(self, profile_revision_id=None) -> None:
        super().__init__()
        if profile_revision_id is not None:
            self.profile_revision_id = profile_revision_id

    async def judge(self, packet):
        judgement = await super().judge(packet)
        judgement["input_sha256"] = judge_packet_sha256(packet)
        return judgement


class AgentComplexEvaluationToolTests(unittest.TestCase):
    def test_batch_stops_after_nonretryable_auth_failure_and_marks_remaining_not_run(
        self,
    ) -> None:
        for http_status in (401, 403):
            cases = [
                {"case_id": case_id, "question": case_id, "aspects": []}
                for case_id in ("complex-04", "complex-02", "complex-05")
            ]
            failed = {
                "case_id": "complex-04",
                "status": "failed",
                "error": {"http_status": http_status, "retryable": False},
                "score": {},
            }

            with self.subTest(http_status=http_status), patch(
                "tools.evaluate_agent_complex_qa._evaluate_one",
                return_value=failed,
            ) as evaluate:
                results = evaluate_cases(
                    "http://127.0.0.1:8000/api/v1",
                    "kb-id",
                    cases,
                    strategy="exact_vector",
                    rerank_mode="classic",
                    top_k=10,
                    parallelism=1,
                    timeout_seconds=900,
                    poll_seconds=1,
                    document_identity_map={},
                )

            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(
                [item["status"] for item in results],
                ["failed", "not_run", "not_run"],
            )
            self.assertTrue(
                all(
                    item.get("not_run_reason")
                    == "provider_nonretryable_auth_failure"
                    for item in results[1:]
                )
            )

    def test_v2_artifact_loader_accepts_matching_artifact_and_preserves_case_order(
        self,
    ) -> None:
        cases = [
            _minimal_case("complex-a", question="Question A", aspect_id="a"),
            _minimal_case("complex-b", question="Question B", aspect_id="b"),
        ]
        report = {
            "schema_version": "native_agent_complex_qa_evaluation_v2",
            "config": {"corpus_sha256": "corpus-hash"},
            "cases": [
                _legacy_result(cases[1]),
                _legacy_result(cases[0]),
            ],
        }
        raw = json.dumps(report, sort_keys=True).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation-v2.json"
            path.write_bytes(raw)

            loaded, artifact_sha256 = load_evaluation_artifact(
                path,
                cases=cases,
                corpus_sha256="corpus-hash",
            )

        self.assertEqual(
            [item["case_id"] for item in loaded["cases"]],
            ["complex-a", "complex-b"],
        )
        self.assertEqual(artifact_sha256, hashlib.sha256(raw).hexdigest())

    def test_v2_artifact_loader_rejects_corpus_and_case_mismatches(self) -> None:
        case = _minimal_case()
        report = {
            "schema_version": "native_agent_complex_qa_evaluation_v2",
            "config": {"corpus_sha256": "corpus-hash"},
            "cases": [_legacy_result(case)],
        }
        mismatches = {
            "corpus": (
                {**deepcopy(report), "config": {"corpus_sha256": "other"}},
                "corpus hash",
            ),
            "question": (
                {
                    **deepcopy(report),
                    "cases": [
                        {**deepcopy(report["cases"][0]), "question": "Other"}
                    ],
                },
                "does not match the corpus",
            ),
            "aspect": (
                {
                    **deepcopy(report),
                    "cases": [
                        {
                            **deepcopy(report["cases"][0]),
                            "score": {
                                **deepcopy(report["cases"][0]["score"]),
                                "aspects": [{"aspect_id": "other"}],
                            },
                        }
                    ],
                },
                "aspect IDs",
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation-v2.json"
            for name, (artifact, message) in mismatches.items():
                with self.subTest(name=name):
                    path.write_text(json.dumps(artifact), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_evaluation_artifact(
                            path,
                            cases=[case],
                            corpus_sha256="corpus-hash",
                        )

    def test_v2_artifact_loader_accepts_explicit_corpus_correction(self) -> None:
        case = _minimal_case()
        report = {
            "schema_version": "native_agent_complex_qa_evaluation_v2",
            "config": {"corpus_sha256": "legacy-corpus-hash"},
            "cases": [_legacy_result(case)],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation-v2.json"
            raw = json.dumps(report).encode("utf-8")
            path.write_bytes(raw)
            artifact_sha256 = hashlib.sha256(raw).hexdigest()

            loaded, _ = load_evaluation_artifact(
                path,
                cases=[case],
                corpus_sha256="corrected-corpus-hash",
                compatible_corpus_transitions={
                    (
                        "legacy-corpus-hash",
                        "corrected-corpus-hash",
                    ): frozenset({artifact_sha256})
                },
            )

            with self.assertRaisesRegex(ValueError, "corpus hash"):
                load_evaluation_artifact(
                    path,
                    cases=[case],
                    corpus_sha256="different-target-hash",
                    compatible_corpus_transitions={
                        (
                            "legacy-corpus-hash",
                            "corrected-corpus-hash",
                        ): frozenset({artifact_sha256})
                    },
                )

        self.assertEqual(loaded["config"]["corpus_sha256"], "legacy-corpus-hash")

    def test_completed_result_is_judged_and_preserves_legacy_score(self) -> None:
        case = _minimal_case()
        original = _legacy_result(case)
        legacy_score = deepcopy(original["score"])
        judge = _FakeJudge()

        with tempfile.TemporaryDirectory() as directory:
            corpus_root = Path(directory)
            (corpus_root / "cases.jsonl").write_text("", encoding="utf-8")
            results = asyncio.run(
                apply_llm_judgements(
                    [original],
                    cases=[case],
                    corpus_root=corpus_root,
                    document_identity_map={},
                    document_paths={},
                    judge=judge,
                )
            )

        self.assertEqual(len(judge.packets), 1)
        self.assertEqual(judge.packets[0]["question"], case["question"])
        self.assertEqual(results[0]["legacy_score"], legacy_score)
        self.assertEqual(results[0]["judge"]["status"], "judged")
        self.assertTrue(results[0]["score"]["semantic_strict_correct"])
        self.assertTrue(results[0]["score"]["semantic_at_least_partial"])
        self.assertNotIn("strict_correct", results[0]["score"])

    def test_noncompleted_result_is_not_sent_to_judge(self) -> None:
        case = _minimal_case()
        for status in ("failed", "cancelled", "not_run"):
            with self.subTest(status=status):
                original = _legacy_result(case, status=status)
                legacy_score = deepcopy(original["score"])
                judge = _FakeJudge()
                with tempfile.TemporaryDirectory() as directory:
                    corpus_root = Path(directory)
                    (corpus_root / "cases.jsonl").write_text(
                        "", encoding="utf-8"
                    )
                    results = asyncio.run(
                        apply_llm_judgements(
                            [original],
                            cases=[case],
                            corpus_root=corpus_root,
                            document_identity_map={},
                            document_paths={},
                            judge=judge,
                        )
                    )

                self.assertEqual(judge.packets, [])
                self.assertEqual(results[0]["legacy_score"], legacy_score)
                self.assertEqual(results[0]["judge"]["status"], "not_judged")
                self.assertFalse(
                    results[0]["score"]["semantic_strict_correct"]
                )

    def test_judge_cache_miss_writes_and_hit_skips_judge(self) -> None:
        case = _minimal_case("complex-01")
        original = _legacy_result(case)
        profile_revision_id = uuid4()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_root = root / "corpus"
            corpus_root.mkdir()
            (corpus_root / "cases.jsonl").write_text("", encoding="utf-8")
            cache_dir = root / "judge-cache"

            miss_judge = _CacheJudge(profile_revision_id)
            first = asyncio.run(
                apply_llm_judgements(
                    [original],
                    cases=[case],
                    corpus_root=corpus_root,
                    document_identity_map={},
                    document_paths={},
                    judge=miss_judge,
                    cache_dir=cache_dir,
                )
            )

            cache_files = list(cache_dir.rglob("*.json"))
            self.assertEqual(len(miss_judge.packets), 1)
            self.assertEqual(len(cache_files), 1)
            cached = json.loads(cache_files[0].read_text(encoding="utf-8"))
            self.assertEqual(cached["case_id"], "complex-01")
            self.assertEqual(cached["judgement"], first[0]["judge"])
            self.assertEqual(list(cache_dir.rglob("*.tmp")), [])

            hit_judge = _CacheJudge(profile_revision_id)
            second = asyncio.run(
                apply_llm_judgements(
                    [original],
                    cases=[case],
                    corpus_root=corpus_root,
                    document_identity_map={},
                    document_paths={},
                    judge=hit_judge,
                    cache_dir=cache_dir,
                )
            )

            self.assertEqual(hit_judge.packets, [])
            self.assertEqual(second[0]["judge"], first[0]["judge"])
            self.assertEqual(len(list(cache_dir.rglob("*.json"))), 1)

    def test_judge_cache_ignores_stale_profile_or_packet_hash(self) -> None:
        case = _minimal_case("complex-01")
        original = _legacy_result(case)
        profile_revision_id = uuid4()
        stale_values = {
            "profile_revision_id": str(uuid4()),
            "input_sha256": "0" * 64,
        }

        for field, stale_value in stale_values.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                corpus_root = root / "corpus"
                corpus_root.mkdir()
                (corpus_root / "cases.jsonl").write_text(
                    "",
                    encoding="utf-8",
                )
                cache_dir = root / "judge-cache"
                seed_judge = _CacheJudge(profile_revision_id)
                asyncio.run(
                    apply_llm_judgements(
                        [original],
                        cases=[case],
                        corpus_root=corpus_root,
                        document_identity_map={},
                        document_paths={},
                        judge=seed_judge,
                        cache_dir=cache_dir,
                    )
                )
                cache_path = next(cache_dir.rglob("*.json"))
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                cached["judgement"][field] = stale_value
                cache_path.write_text(json.dumps(cached), encoding="utf-8")

                retry_judge = _CacheJudge(profile_revision_id)
                results = asyncio.run(
                    apply_llm_judgements(
                        [original],
                        cases=[case],
                        corpus_root=corpus_root,
                        document_identity_map={},
                        document_paths={},
                        judge=retry_judge,
                        cache_dir=cache_dir,
                    )
                )

                self.assertEqual(len(retry_judge.packets), 1)
                self.assertEqual(
                    results[0]["judge"]["profile_revision_id"],
                    str(profile_revision_id),
                )
                self.assertEqual(
                    results[0]["judge"]["input_sha256"],
                    judge_packet_sha256(retry_judge.packets[0]),
                )

    def test_atomic_json_write_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"

            _atomic_write_json(path, {"status": "complete", "count": 1})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "complete", "count": 1},
            )
            self.assertEqual(
                {item.name for item in path.parent.iterdir()},
                {"report.json"},
            )

    def test_answer_checkpoint_path_does_not_overwrite_final_report(self) -> None:
        output = Path("/tmp/stage5-run.json")

        self.assertEqual(
            _answer_checkpoint_path(output),
            Path("/tmp/stage5-run.answers.json"),
        )

    def test_revised_complex_aspects_accept_semantic_and_decimal_variants(self) -> None:
        answers = {
            "complex-01": (
                "AMD reported one customer at 16 percent; Boeing reported U.S. "
                "government contracts at 40 percent. The 24 percentage-point "
                "difference is not directly comparable, and Boeing is cyclical."
            ),
            "complex-02": (
                "American Express's effective tax rate was 24.6% in 2021 and "
                "21.6% in 2022, a decrease of 3.0 percentage points. Boeing's "
                "effective tax rate was 14.7% in 2021 and -0.6% in 2022, a "
                "decrease of 15.3 percentage points. Boeing's change in magnitude "
                "was larger."
            ),
            "complex-07": (
                "Other was 2.95 percent of sales. Fixed Price rose from 1,146.2 "
                "to 1,452.4, offsetting Other so total sales were highest."
            ),
            "complex-08": (
                "The residual was 27.0, and only 1 segment exceeded $50 million."
            ),
        }
        cases = {
            str(case["case_id"]): case
            for case in COMPLEX_CASE_DEFINITIONS
            if case["case_id"] in answers
        }

        for case_id, answer in answers.items():
            case = cases[case_id]
            run = {
                "status": "completed",
                "answer": answer,
                "agent": _agent(),
                "citations": [
                    {"document_id": document_id}
                    for document_id in case["required_citation_document_ids"]
                ],
            }
            with self.subTest(case_id=case_id):
                score = score_complex_case(case, run)
                self.assertTrue(score["strict_correct"], score["aspects"])

    def test_evaluation_parallelism_is_fixed_at_one(self) -> None:
        base = {
            "top_k": 10,
            "parallelism": 1,
            "timeout_seconds": 900.0,
            "poll_seconds": 1.0,
            "strategy": "exact_vector",
            "rerank_mode": "classic",
        }
        _validate_options(argparse.Namespace(**base))
        for parallelism in (0, 2, 3):
            with self.subTest(parallelism=parallelism), self.assertRaises(ValueError):
                _validate_options(
                    argparse.Namespace(**{**base, "parallelism": parallelism})
                )

    def test_loopback_api_validation_rejects_credentials_and_query(self) -> None:
        self.assertEqual(
            _validated_api_base("http://127.0.0.1:8000/api/v1/"),
            "http://127.0.0.1:8000/api/v1",
        )
        for value in (
            "https://127.0.0.1:8000/api/v1",
            "http://user:secret@127.0.0.1:8000/api/v1",
            "http://localhost:8000/api/v1?token=secret",
            "http://localhost.evil:8000/api/v1",
            "http://127.0.0.1:8000/v1",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validated_api_base(value)

    def test_decimal_parser_preserves_parentheses_and_commas(self) -> None:
        self.assertIn(-1234.50, _extract_decimal_values("($1,234.50)"))
        self.assertIn(Decimal("2.95"), _extract_decimal_values("2.95%"))

    def test_scoring_requires_completion_aspects_and_exact_document_citations(self) -> None:
        case = {
            "case_id": "complex-test",
            "required_citation_document_ids": ["doc-a", "doc-b"],
            "forbid_unrelated_citations": True,
            "evaluation_group": "evidence_only",
            "aspects": [
                {
                    "aspect_id": "amount",
                    "answer_variants": ["total", "1,234.50"],
                    "expected_decimal": "1234.50",
                    "numeric_tolerance": "0.01",
                }
            ],
        }
        run = {
            "status": "completed",
            "answer": "The total is $1,234.50.",
            "agent": _agent(),
            "citations": [
                {"document_id": "doc-a"},
                {"document_id": "doc-b"},
            ],
        }
        score = score_complex_case(case, run)
        self.assertTrue(score["strict_correct"])
        self.assertEqual(score["required_document_citation_coverage"], 1.0)

        run["citations"].append({"document_id": "unrelated"})
        score = score_complex_case(case, run)
        self.assertFalse(score["strict_correct"])
        self.assertEqual(score["forbidden_citation_document_ids"], ["unrelated"])

    def test_scoring_maps_runtime_citation_filename_to_corpus_document_id(self) -> None:
        case = {
            "case_id": "complex-filename-map",
            "required_citation_document_ids": ["doc-a"],
            "forbid_unrelated_citations": True,
            "aspects": [
                {
                    "aspect_id": "fact",
                    "answer_variants": ["answer"],
                    "expected_decimal": None,
                }
            ],
        }

        score = score_complex_case(
            case,
            {
                "status": "completed",
                "answer": "The answer is grounded.",
                "agent": _agent(),
                "citations": [
                    {
                        "document_id": "runtime-uuid",
                        "document_original_filename": "DOC-A.pdf",
                    }
                ],
            },
            document_identity_map={"doc-a.pdf": "doc-a"},
        )

        self.assertTrue(score["strict_correct"])
        self.assertEqual(score["cited_document_ids"], ["doc-a"])

    def test_identity_map_accepts_logical_document_filename_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "document_id": "doc-a",
                                "path": "documents/pdf/doc-a.pdf",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            identity_map = load_document_identity_map(root)

        self.assertEqual(identity_map["doc-a.pdf"], "doc-a")

    def test_negative_change_accepts_signed_or_qualified_decrease(self) -> None:
        case = {
            "case_id": "complex-negative",
            "required_citation_document_ids": [],
            "forbid_unrelated_citations": True,
            "aspects": [
                {
                    "aspect_id": "change",
                    "answer_variants": ["decreased", "3.0 percentage points"],
                    "expected_decimal": "-3.0",
                    "numeric_tolerance": "0.01",
                }
            ],
        }
        score = score_complex_case(
            case,
            {
                "status": "completed",
                "answer": "The rate decreased by 3.0 percentage points.",
                "agent": _agent(),
                "citations": [],
            },
        )
        self.assertTrue(score["strict_correct"])

    def test_aspect_answer_match_any_accepts_one_synonym(self) -> None:
        score = score_complex_case(
            {
                "case_id": "complex-synonym",
                "required_citation_document_ids": [],
                "forbid_unrelated_citations": True,
                "aspects": [
                    {
                        "aspect_id": "label",
                        "answer_variants": ["entailment", "entailed"],
                        "answer_match": "any",
                    }
                ],
            },
            {
                "status": "completed",
                "answer": "The statement is entailed.",
                "agent": _agent(),
                "citations": [],
            },
        )
        self.assertTrue(score["strict_correct"])

    def test_protocol_diagnostic_accepts_native_agent_v1_and_v2(self) -> None:
        case = _minimal_case()
        for version in (
            "native_tool_calling_agent_v1",
            "native_tool_calling_agent_v2",
        ):
            with self.subTest(version=version):
                score = score_complex_case(
                    case,
                    {
                        "status": "completed",
                        "answer": "The answer is stated.",
                        "agent": _agent(version),
                        "citations": [],
                    },
                )
                self.assertTrue(score["agent_protocol_ok"])
                self.assertTrue(score["strict_correct"])

        unsupported = score_complex_case(
            case,
            {
                "status": "completed",
                "answer": "The answer is stated.",
                "agent": _agent("native_tool_calling_agent_v0"),
                "citations": [],
            },
        )
        self.assertFalse(unsupported["agent_protocol_ok"])

    def test_summary_separates_domain_inference(self) -> None:
        results = [
            {"score": {"semantic_strict_correct": True, "semantic_at_least_partial": True, "terminal_completed": True, "evaluation_group": "evidence_only", "required_document_citation_coverage": 1.0, "forbidden_citation_document_ids": []}},
            {"score": {"semantic_strict_correct": False, "semantic_at_least_partial": True, "terminal_completed": True, "evaluation_group": "domain_inference", "required_document_citation_coverage": 1.0, "forbidden_citation_document_ids": []}},
        ]
        summary = summarize_results(results)
        self.assertEqual(summary["strict_correct"], 1)
        self.assertEqual(summary["evidence_only_strict_correct"], 1)
        self.assertEqual(summary["domain_inference_cases"], 1)

    def test_summary_uses_only_semantic_flags(self) -> None:
        results = [
            {
                "score": {
                    "semantic_strict_correct": False,
                    "semantic_at_least_partial": False,
                    "strict_correct": True,
                    "at_least_partial": True,
                    "terminal_completed": True,
                    "evaluation_group": "evidence_only",
                }
            },
            {
                "score": {
                    "strict_correct": True,
                    "at_least_partial": True,
                    "terminal_completed": True,
                    "evaluation_group": "evidence_only",
                }
            },
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["strict_correct"], 0)
        self.assertEqual(summary["at_least_partial"], 0)
        self.assertEqual(summary["semantic_strict_correct"], 0)
        self.assertEqual(summary["semantic_at_least_partial"], 0)

    def test_summary_reports_answered_precision_and_runtime_budget_facts(self) -> None:
        result = {
            "score": {
                "semantic_strict_correct": False,
                "semantic_at_least_partial": True,
                "terminal_completed": True,
                "evaluation_group": "evidence_only",
                "required_document_citation_coverage": 1.0,
                "forbidden_citation_document_ids": [],
                "agent_outcome": "answered",
                "answered_precision_ok": True,
            },
            "status": "completed",
            "elapsed_seconds": 4.0,
            "usage": {"totals": {"total_tokens": 12}},
            "timing": {"attempts": {"1": {"diagnostic": {}}}},
            "judge": {
                "calls": {
                    "judge_a": {
                        "transport_attempts": 2,
                        "usage": {"total_tokens": 7},
                    },
                    "judge_b": {
                        "transport_attempts": 1,
                        "usage": {"total_tokens": 5},
                    },
                    "judge_c": None,
                }
            },
        }

        summary = summarize_results([result])

        self.assertEqual(summary["answered_cases"], 1)
        self.assertEqual(summary["answered_precision_cases"], 1)
        self.assertEqual(summary["total_tokens"], 12)
        self.assertEqual(summary["median_elapsed_seconds"], 4.0)
        self.assertEqual(summary["max_elapsed_seconds"], 4.0)
        self.assertEqual(summary["chat_run_retry_cases"], 0)
        self.assertEqual(summary["total_timeout_cases"], 0)
        self.assertEqual(summary["requested_cases"], 1)
        self.assertEqual(summary["attempted_cases"], 1)
        self.assertEqual(summary["not_run_cases"], 0)
        self.assertEqual(summary["judge_calls"], 2)
        self.assertEqual(summary["judge_attempts"], 3)
        self.assertEqual(summary["judge_total_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
