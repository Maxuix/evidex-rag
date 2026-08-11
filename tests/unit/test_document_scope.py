from __future__ import annotations

import unittest
from uuid import UUID

from rag_kb.retrieval.document_scope import (
    RetrievalScopeCandidate,
    resolve_document_scope,
)


DOC_A = UUID("01900000-0000-7000-8000-000000000901")
DOC_B = UUID("01900000-0000-7000-8000-000000000902")
TARGET_A = UUID("01900000-0000-7000-8000-000000000911")
TARGET_B = UUID("01900000-0000-7000-8000-000000000912")


def _candidate(
    document_id: UUID,
    *,
    display_name: str,
    original_filename: str,
    target_id: UUID | None,
    source_status: str = "available",
    build_status: str | None = "ready",
    serving_status: str | None = "serving",
) -> RetrievalScopeCandidate:
    return RetrievalScopeCandidate(
        document_id=document_id,
        document_version_id=UUID("01900000-0000-7000-8000-000000000921"),
        indexed_document_version_id=target_id,
        display_name=display_name,
        original_filename=original_filename,
        source_status=source_status,
        build_status=build_status,
        serving_status=serving_status,
    )


class DocumentScopeTests(unittest.TestCase):
    def test_exact_filename_matching_is_unicode_and_case_insensitive(self) -> None:
        value = resolve_document_scope(
            "请只比较 E\u0301xample_2022.PDF 的数据",
            (
                _candidate(
                    DOC_A,
                    display_name="Example",
                    original_filename="Éxample_2022.pdf",
                    target_id=TARGET_A,
                ),
            ),
        )

        self.assertEqual(value.status, "resolved")
        self.assertEqual(value.document_ids, (DOC_A,))
        self.assertEqual(value.required_names, ("Éxample_2022.pdf",))

    def test_unique_stem_resolves_but_ambiguous_stem_fails_closed(self) -> None:
        unique = resolve_document_scope(
            "Read the Loan-to-Value Ratio document.",
            (
                _candidate(
                    DOC_A,
                    display_name="Loan-to-Value Ratio",
                    original_filename="table.pdf",
                    target_id=TARGET_A,
                ),
            ),
        )
        ambiguous = resolve_document_scope(
            "Use contract.pdf only.",
            (
                _candidate(
                    DOC_A,
                    display_name="Contract A",
                    original_filename="contract.pdf",
                    target_id=TARGET_A,
                ),
                _candidate(
                    DOC_B,
                    display_name="Contract B",
                    original_filename="CONTRACT.PDF",
                    target_id=TARGET_B,
                ),
            ),
        )

        self.assertEqual(unique.document_ids, (DOC_A,))
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertEqual(ambiguous.document_ids, ())
        self.assertEqual(ambiguous.ambiguous_names, ("contract.pdf",))

    def test_unknown_filename_and_unavailable_target_are_not_all_kb(self) -> None:
        value = resolve_document_scope(
            "Use missing-contract.txt only.",
            (
                _candidate(
                    DOC_A,
                    display_name="Missing contract",
                    original_filename="other.pdf",
                    target_id=None,
                    build_status="failed",
                    serving_status="retired",
                ),
            ),
        )

        self.assertEqual(value.status, "unresolved")
        self.assertEqual(value.document_ids, ())
        self.assertIn("missing-contract.txt", value.unresolved_names)

    def test_no_explicit_document_keeps_all_kb_scope(self) -> None:
        value = resolve_document_scope(
            "Summarize the corpus.",
            (
                _candidate(
                    DOC_A,
                    display_name="Annual report",
                    original_filename="report.pdf",
                    target_id=TARGET_A,
                ),
            ),
        )

        self.assertEqual(value.status, "all")
        self.assertEqual(value.document_ids, ())
        self.assertEqual(value.required_names, ())
