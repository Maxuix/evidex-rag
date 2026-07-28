from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader

from tools.evaluate_multimodal_real import (
    _attachment_metrics,
    _generate_corpus,
    _group_recall,
    _mrr,
    _ndcg,
    _percentile,
    _recall,
    _validated_api_base,
)


class RealEvaluationToolTests(unittest.TestCase):
    def test_generated_corpus_covers_bounded_supported_and_lexical_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = _generate_corpus(Path(directory))

            self.assertEqual(
                set(corpus),
                {
                    "architecture_pdf",
                    "scanned_pdf",
                    "rich_docx",
                    "long_text",
                    "repeated_watermark",
                    "lexical_identifiers",
                    "lexical_hard_negative",
                    "resource_stress",
                },
            )
            self.assertTrue(
                all(
                    path.stat().st_size < 10 * 1024 * 1024
                    for path in corpus.values()
                )
            )
            self.assertTrue(
                all(
                    path.suffix in {".pdf", ".docx", ".txt"}
                    for path in corpus.values()
                )
            )
            self.assertTrue(
                all(
                    len(PdfReader(path).pages) == 1
                    for path in corpus.values()
                    if path.suffix == ".pdf"
                )
            )
            architecture_text = "\n".join(
                page.extract_text() or ""
                for page in PdfReader(corpus["architecture_pdf"]).pages
            )
            scanned_text = "\n".join(
                page.extract_text() or ""
                for page in PdfReader(corpus["scanned_pdf"]).pages
            )
            self.assertIn("ASTER CONTROL PLANE", architecture_text)
            self.assertIn("As shown in Fig. 7", architecture_text)
            self.assertIn("Figure 7", architecture_text)
            self.assertEqual(scanned_text, "")
            lexical_text = corpus["lexical_identifiers"].read_text(
                encoding="utf-8"
            )
            hard_negative = corpus["lexical_hard_negative"].read_text(
                encoding="utf-8"
            )
            self.assertIn("POL-7.3/REV:2", lexical_text)
            self.assertIn("recover_index_target_v2", lexical_text)
            self.assertIn("星河协议XQ-77", lexical_text)
            self.assertNotIn("POL-7.3/REV:2", hard_negative)

    def test_metrics_count_missing_ranks_as_zero(self) -> None:
        cases = [
            {"rank": 1, "recalled": True},
            {"rank": 2, "recalled": True},
            {"rank": None, "recalled": False},
        ]

        self.assertEqual(_recall(cases), 0.666667)
        self.assertEqual(_mrr(cases), 0.5)
        self.assertEqual(_ndcg(cases), 0.543643)

    def test_percentile_uses_linear_interpolation(self) -> None:
        values = [0.1, 0.2, 0.4, 0.8]

        self.assertAlmostEqual(_percentile(values, 0.5), 0.3)
        self.assertAlmostEqual(_percentile(values, 0.95), 0.74)

    def test_composite_group_and_attachment_metrics_are_explicit(self) -> None:
        cases = [
            {
                "group_recalled": True,
                "expects_visual": True,
                "predicted_visual": True,
            },
            {
                "group_recalled": False,
                "expects_visual": False,
                "predicted_visual": True,
            },
            {
                "group_recalled": True,
                "expects_visual": False,
                "predicted_visual": False,
            },
        ]

        self.assertEqual(_group_recall(cases), 0.666667)
        self.assertEqual(
            _attachment_metrics(cases),
            {"precision": 0.5, "accuracy": 0.666667},
        )

    def test_api_base_accepts_only_exact_loopback_scope(self) -> None:
        self.assertEqual(
            _validated_api_base("http://127.0.0.1:58001/api/v1/"),
            "http://127.0.0.1:58001/api/v1",
        )
        for value in (
            "https://127.0.0.1:58001/api/v1",
            "http://127.0.0.1:58001@provider.invalid/api/v1",
            "http://localhost.evil:58001/api/v1",
            "http://localhost:58001/api/v1?secret=value",
            "http://localhost:58001/other",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validated_api_base(value)


if __name__ == "__main__":
    unittest.main()
