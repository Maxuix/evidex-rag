from __future__ import annotations

import unittest
from uuid import UUID

from rag_kb.document_processing.lexical import (
    LEXICAL_ANALYZER_VERSION,
    MAX_DOCUMENT_LEXEMES,
    MAX_QUERY_LEXEMES,
    analyze_document,
    analyze_query,
    build_or_tsquery,
    lexical_manifest_hash,
)


class LexicalAnalyzerTests(unittest.TestCase):
    def test_normalizes_nfc_case_and_compound_identifiers(self) -> None:
        result = analyze_document("Cafe\u0301 POL-7.3/REV:2 policy")

        assert result is not None
        self.assertEqual(
            result.lexemes,
            ("café", "pol_7_3_rev_2", "pol", "7", "3", "rev", "2", "policy"),
        )

    def test_generates_cjk_bigrams_and_keeps_only_single_character_runs(self) -> None:
        self.assertEqual(
            analyze_query("知识库 A 中"),
            ("知识", "识库", "中"),
        )

    def test_removes_bounded_stopwords_and_preserves_first_occurrence(self) -> None:
        self.assertEqual(
            analyze_query("the POLICY policy and 42"),
            ("policy", "42"),
        )

    def test_document_and_query_limits_are_hard(self) -> None:
        document = analyze_document(
            " ".join(f"token{index}" for index in range(MAX_DOCUMENT_LEXEMES + 5))
        )
        assert document is not None
        self.assertEqual(len(document.lexemes), MAX_DOCUMENT_LEXEMES)
        self.assertEqual(
            len(
                analyze_query(
                    " ".join(
                        f"query{index}" for index in range(MAX_QUERY_LEXEMES + 5)
                    )
                )
            ),
            MAX_QUERY_LEXEMES,
        )

    def test_tsquery_is_built_only_from_safe_analyzed_lexemes(self) -> None:
        query = build_or_tsquery(analyze_query("abc'); DROP TABLE x; -- 中文"))

        self.assertEqual(query, "'abc' | 'drop' | 'table' | 'x' | '中文'")
        self.assertNotIn(";", query)
        with self.assertRaises(ValueError):
            build_or_tsquery(("unsafe'value",))

    def test_empty_input_and_zero_chunk_manifest_are_stable(self) -> None:
        self.assertIsNone(analyze_document("the and 的"))
        self.assertIsNone(build_or_tsquery(()))
        self.assertEqual(
            lexical_manifest_hash(LEXICAL_ANALYZER_VERSION, ()),
            lexical_manifest_hash(LEXICAL_ANALYZER_VERSION, ()),
        )

    def test_manifest_hash_sorts_by_chunk_identity(self) -> None:
        first = UUID("01900000-0000-7000-8000-000000000001")
        second = UUID("01900000-0000-7000-8000-000000000002")
        expected = lexical_manifest_hash(
            LEXICAL_ANALYZER_VERSION,
            ((first, "a" * 64), (second, "b" * 64)),
        )
        self.assertEqual(
            expected,
            lexical_manifest_hash(
                LEXICAL_ANALYZER_VERSION,
                ((second, "b" * 64), (first, "a" * 64)),
            ),
        )
        self.assertNotEqual(
            expected,
            lexical_manifest_hash(
                LEXICAL_ANALYZER_VERSION,
                ((first, "c" * 64), (second, "b" * 64)),
            ),
        )


if __name__ == "__main__":
    unittest.main()
