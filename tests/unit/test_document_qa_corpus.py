from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.build_document_qa_corpus import (
    CASE_SCHEMA,
    CORPUS_SCHEMA,
    _acceptable_answers,
    _cfqa_acceptable_answers,
    _markdown_row,
    validate_corpus,
)


class DocumentQaCorpusTests(unittest.TestCase):
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
