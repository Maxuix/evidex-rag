from __future__ import annotations

import unittest

from tools.evaluate_docling_candidate import (
    LANGCHAIN_EXPORT_MODES,
    MARKERS,
    REQUIRED_CONTENT_MARKERS,
    marker_facts,
    safe_json,
    stable_json_hash,
)


class DoclingCandidateToolTests(unittest.TestCase):
    def test_langchain_evaluation_covers_both_chunker_strategies(self) -> None:
        self.assertEqual(
            LANGCHAIN_EXPORT_MODES,
            (
                "markdown",
                "doc_chunks_hierarchical",
                "doc_chunks_hybrid",
            ),
        )

    def test_xlsx_sheet_name_is_fidelity_not_required_content(self) -> None:
        self.assertEqual(
            REQUIRED_CONTENT_MARKERS["xlsx"],
            ("XLSX-TABLE-902",),
        )
        self.assertIn("XLSX-SHEET-901", MARKERS["xlsx"])

    def test_marker_facts_require_presence_and_source_order(self) -> None:
        self.assertEqual(
            marker_facts("FIRST body SECOND", ("FIRST", "SECOND")),
            {"complete": True, "order_preserved": True},
        )
        self.assertEqual(
            marker_facts("SECOND body FIRST", ("FIRST", "SECOND")),
            {"complete": True, "order_preserved": False},
        )
        self.assertEqual(
            marker_facts("FIRST only", ("FIRST", "SECOND")),
            {"complete": False, "order_preserved": True},
        )

    def test_safe_projection_and_hash_are_deterministic(self) -> None:
        first = {"b": object(), "a": [1, {"z": 2}]}
        second = {"a": [1, {"z": 2}], "b": object()}

        self.assertEqual(safe_json(first), safe_json(second))
        self.assertEqual(
            stable_json_hash(safe_json(first)),
            stable_json_hash(safe_json(second)),
        )


if __name__ == "__main__":
    unittest.main()
