from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.build_document_qa_corpus import (
    CASE_SCHEMA,
    COMPLEX_CASE_DEFINITIONS,
    COMPLEX_CASE_SCHEMA,
    CORPUS_SCHEMA,
    _acceptable_answers,
    _cfqa_acceptable_answers,
    _cfqa_document_relative_path,
    _markdown_row,
    _validate_complex_case,
    validate_corpus,
)


class DocumentQaCorpusTests(unittest.TestCase):
    def test_complex_questions_use_serving_logical_filenames(self) -> None:
        cases = {item["case_id"]: item for item in COMPLEX_CASE_DEFINITIONS}

        self.assertIn(
            "financebench-amd-2022-10k.pdf",
            cases["complex-01"]["question"],
        )
        for case_id in ("complex-01", "complex-02", "complex-03"):
            self.assertIn(
                "financebench-boeing-2022-10k.pdf",
                cases[case_id]["question"],
            )
        for case_id in ("complex-02", "complex-03", "complex-04"):
            self.assertIn(
                "financebench-american-express-2022-10k.pdf",
                cases[case_id]["question"],
            )
        for case_id in ("complex-04", "complex-05"):
            self.assertIn(
                "cfqa-fenghuo-electronics-2022-annual-report.pdf",
                cases[case_id]["question"],
            )

    def test_complex_05_names_the_source_table_rows(self) -> None:
        case = next(
            item for item in COMPLEX_CASE_DEFINITIONS
            if item["case_id"] == "complex-05"
        )

        for label in ("原材料", "管理费用", "研发费用", "合并利润表"):
            self.assertIn(label, case["question"])

    def test_complex_02_uses_boeing_tax_reconciliation_values(self) -> None:
        case = next(
            item for item in COMPLEX_CASE_DEFINITIONS
            if item["case_id"] == "complex-02"
        )
        aspect = next(
            item for item in case["aspects"]
            if item["aspect_id"] == "boeing_effective_tax_rate_change"
        )

        self.assertEqual(
            aspect["answer_variants"],
            ["-0.6%", "14.7%", "decreas"],
        )
        self.assertEqual(aspect["expected_decimal"], "-15.3")
        self.assertEqual(
            aspect["source"][0]["evidence_locator"],
            {"kind": "pdf_page", "page": 77},
        )

    def test_complex_08_residual_matches_the_question_arithmetic(self) -> None:
        case = next(
            item for item in COMPLEX_CASE_DEFINITIONS
            if item["case_id"] == "complex-08"
        )
        aspect = next(
            item for item in case["aspects"]
            if item["aspect_id"] == "other_operating_expense_residual"
        )

        self.assertEqual(aspect["answer_variants"], ["27.0"])
        self.assertEqual(aspect["expected_decimal"], "27.0")
        self.assertIn("166.3 - 94.2 - 45.1", case["notes"])

    def test_complex_absence_cases_remain_rebuildable_offline(self) -> None:
        cases = {item["case_id"]: item for item in COMPLEX_CASE_DEFINITIONS}

        case_11 = cases["complex-11"]
        self.assertTrue(case_11["allow_not_mentioned"])
        self.assertTrue(case_11["requires_complete_scan"])
        self.assertEqual(
            [item["aspect_id"] for item in case_11["aspects"]],
            [
                "contract_15_reverse_engineering",
                "contract_488_reverse_engineering",
            ],
        )
        case_12 = cases["complex-12"]
        self.assertTrue(case_12["allow_not_mentioned"])
        self.assertTrue(case_12["requires_complete_scan"])
        absence = case_12["aspects"][2]
        self.assertEqual(absence["aspect_id"], "contract_82_legal_notice")
        self.assertTrue(absence["allow_not_mentioned"])
        self.assertTrue(absence["requires_complete_scan"])

    def test_complex_absence_definitions_match_pinned_corpus(self) -> None:
        root = Path(__file__).resolve().parents[2]
        pinned_cases = {
            item["case_id"]: item
            for item in (
                json.loads(line)
                for line in (
                    root / "evaluation" / "document-qa-v1" / "complex-cases.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        definitions = {
            item["case_id"]: item for item in COMPLEX_CASE_DEFINITIONS
        }

        for case_id in ("complex-11", "complex-12"):
            with self.subTest(case_id=case_id):
                self.assertEqual(
                    _absence_case_contract(definitions[case_id]),
                    _absence_case_contract(pinned_cases[case_id]),
                )

    def test_complex_absence_source_requires_case_and_aspect_flags(self) -> None:
        absence_case = {
            "schema_version": COMPLEX_CASE_SCHEMA,
            "case_id": "complex-absence",
            "question": "Is this statement mentioned?",
            "language": "en",
            "reasoning_type": "scoped_contract_nli",
            "required_document_ids": ["contract"],
            "required_citation_document_ids": ["contract"],
            "forbid_unrelated_citations": True,
            "aspects": [
                {
                    "aspect_id": "absence",
                    "answer_variants": ["not mentioned"],
                    "answer_match": "all",
                    "expected_decimal": None,
                    "numeric_tolerance": None,
                    "allow_not_mentioned": True,
                    "requires_complete_scan": True,
                    "source": [
                        {
                            "source_case_id": "base-absence",
                            "evidence_locator": {"kind": "base_case_evidence"},
                        }
                    ],
                }
            ],
            "requires_complete_scan": True,
            "allow_not_mentioned": True,
            "source_case_ids": ["base-absence"],
        }
        base_by_id = {
            "base-absence": {
                "document_id": "contract",
                "gold": {"answer": "not_mentioned", "answerable": False},
                "evidence": {"kind": "absence", "spans": []},
            }
        }

        flag_locations = (
            ("case_allow_not_mentioned", (), "allow_not_mentioned"),
            ("case_requires_complete_scan", (), "requires_complete_scan"),
            ("aspect_allow_not_mentioned", ("aspects", 0), "allow_not_mentioned"),
            ("aspect_requires_complete_scan", ("aspects", 0), "requires_complete_scan"),
        )
        for name, path, flag in flag_locations:
            value = json.loads(json.dumps(absence_case))
            target = value
            for item in path:
                target = target[item]
            del target[flag]

            with self.subTest(flag=name), self.assertRaisesRegex(
                RuntimeError,
                "absence|scan",
            ):
                _validate_complex_case(
                    value,
                    base_by_id=base_by_id,
                    document_ids={"contract"},
                    manifest_pages={"contract": None},
                    root=None,
                )

    def test_cfqa_document_path_uses_source_filename_not_logical_id(self) -> None:
        self.assertEqual(
            _cfqa_document_relative_path().as_posix(),
            "documents/pdf/fenghuo-electronics-2022-annual-report.pdf",
        )

    def test_markdown_row_escapes_pipe_backslash_and_newline(self) -> None:
        self.assertEqual(
            _markdown_row(["a|b", "c\\d", "line\nbreak"]),
            "| a\\|b | c\\\\d | line break |",
        )

    def test_answer_aliases_do_not_accept_partial_multi_span_answers(self) -> None:
        self.assertEqual(
            _acceptable_answers(["Total value", "Total loan"], ""),
            ["Total value; Total loan", "Total value, Total loan"],
        )
        self.assertEqual(
            _cfqa_acceptable_answers("成本来自原材料。[13]"),
            ["成本来自原材料。[13]", "成本来自原材料。"],
        )

    def test_minimal_text_corpus_validates_evidence_span(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "documents" / "text" / "contract.txt"
            document.parent.mkdir(parents=True)
            document.write_text("alpha evidence omega", encoding="utf-8")
            document_hash = _sha256(document)
            case = {
                "schema_version": CASE_SCHEMA,
                "case_id": "case-1",
                "document_id": "contract",
                "document_path": "documents/text/contract.txt",
                "source_dataset": "test",
                "language": "en",
                "question": "What is in the middle?",
                "gold": {
                    "answer": "evidence",
                    "acceptable_answers": ["evidence"],
                    "answerable": True,
                },
                "evidence": {
                    "kind": "text_spans",
                    "spans": [
                        {"start": 6, "end": 14, "quote": "evidence"},
                    ],
                },
            }
            cases = root / "cases.jsonl"
            cases.write_text(
                json.dumps(case, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            cases_hash = _sha256(cases)
            material = {
                "cases_sha256": cases_hash,
                "documents": [["contract", document_hash]],
            }
            dataset_hash = hashlib.sha256(
                json.dumps(
                    material,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            manifest = {
                "schema_version": CORPUS_SCHEMA,
                "dataset_id": "test",
                "dataset_sha256": dataset_hash,
                "cases_sha256": cases_hash,
                "case_count": 1,
                "document_count": 1,
                "format_counts": {"txt": 1},
                "documents": [
                    {
                        "document_id": "contract",
                        "path": "documents/text/contract.txt",
                        "format": "txt",
                        "bytes": document.stat().st_size,
                        "sha256": document_hash,
                    }
                ],
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            report = validate_corpus(root)

            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["cases"], 1)

    def test_validation_rejects_tampered_text_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "documents" / "text" / "contract.txt"
            document.parent.mkdir(parents=True)
            document.write_text("evidence", encoding="utf-8")
            document_hash = _sha256(document)
            case = {
                "schema_version": CASE_SCHEMA,
                "case_id": "case-1",
                "document_id": "contract",
                "document_path": "documents/text/contract.txt",
                "source_dataset": "test",
                "language": "en",
                "question": "What is present?",
                "gold": {"answer": "evidence", "answerable": True},
                "evidence": {
                    "kind": "text_spans",
                    "spans": [{"start": 0, "end": 8, "quote": "tampered"}],
                },
            }
            cases = root / "cases.jsonl"
            cases.write_text(json.dumps(case) + "\n", encoding="utf-8")
            cases_hash = _sha256(cases)
            material = {
                "cases_sha256": cases_hash,
                "documents": [["contract", document_hash]],
            }
            manifest = {
                "schema_version": CORPUS_SCHEMA,
                "dataset_id": "test",
                "dataset_sha256": hashlib.sha256(
                    json.dumps(
                        material,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "cases_sha256": cases_hash,
                "case_count": 1,
                "document_count": 1,
                "format_counts": {"txt": 1},
                "documents": [
                    {
                        "document_id": "contract",
                        "path": "documents/text/contract.txt",
                        "format": "txt",
                        "bytes": document.stat().st_size,
                        "sha256": document_hash,
                    }
                ],
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "evidence quote mismatch"):
                validate_corpus(root)

    def test_minimal_text_corpus_accepts_offline_absence_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "documents" / "text" / "contract.txt"
            document.parent.mkdir(parents=True)
            document.write_text("contract text", encoding="utf-8")
            document_hash = _sha256(document)
            case = {
                "schema_version": CASE_SCHEMA,
                "case_id": "case-absence",
                "document_id": "contract",
                "document_path": "documents/text/contract.txt",
                "source_dataset": "test",
                "language": "en",
                "question": "Is the statement mentioned?",
                "gold": {"answer": "not_mentioned", "answerable": False},
                "evidence": {"kind": "absence", "spans": []},
            }
            cases = root / "cases.jsonl"
            cases.write_text(json.dumps(case) + "\n", encoding="utf-8")
            cases_hash = _sha256(cases)
            material = {
                "cases_sha256": cases_hash,
                "documents": [["contract", document_hash]],
            }
            manifest = {
                "schema_version": CORPUS_SCHEMA,
                "dataset_id": "test",
                "dataset_sha256": hashlib.sha256(
                    json.dumps(
                        material,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "cases_sha256": cases_hash,
                "case_count": 1,
                "document_count": 1,
                "format_counts": {"txt": 1},
                "documents": [
                    {
                        "document_id": "contract",
                        "path": "documents/text/contract.txt",
                        "format": "txt",
                        "bytes": document.stat().st_size,
                        "sha256": document_hash,
                    }
                ],
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            report = validate_corpus(root)

            self.assertEqual(report["status"], "valid")


def _absence_case_contract(value: dict[str, object]) -> dict[str, object]:
    aspects = value["aspects"]
    assert isinstance(aspects, list) or isinstance(aspects, tuple)
    return {
        "question": value["question"],
        "allow_not_mentioned": value["allow_not_mentioned"],
        "requires_complete_scan": value["requires_complete_scan"],
        "aspects": [
            {
                "aspect_id": aspect["aspect_id"],
                "allow_not_mentioned": aspect["allow_not_mentioned"],
                "requires_complete_scan": aspect["requires_complete_scan"],
                "source": aspect["source"],
            }
            for aspect in aspects
            if isinstance(aspect, dict)
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
