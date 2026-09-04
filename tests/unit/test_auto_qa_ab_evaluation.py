from types import SimpleNamespace
import unittest

from tools.evaluate_auto_qa_retrieval import evidence_matches, summarize


def _evidence(
    *,
    filename: str,
    text: str = "",
    location: dict | None = None,
    titles: tuple[str, ...] = (),
    modality: str = "text",
):
    return SimpleNamespace(
        source_metadata={"original_filename": filename},
        source_location=location or {},
        hierarchy={"titles": [{"text": title} for title in titles]},
        text=text,
        modality=modality,
    )


class AutoQaAbEvaluationTest(unittest.TestCase):
    def test_pdf_page_locator_requires_document_and_page(self) -> None:
        case = {
            "group": "direct",
            "document_path": "documents/pdf/report.pdf",
            "evidence": {
                "kind": "pdf_page_alternatives",
                "alternatives": [{"pages": [8]}, {"pages": [146]}],
            },
        }
        self.assertTrue(
            evidence_matches(
                case,
                _evidence(
                    filename="report.pdf",
                    location={"surface_start": 8, "surface_end": 8},
                ),
            )
        )
        self.assertFalse(
            evidence_matches(
                case,
                _evidence(
                    filename="other.pdf",
                    location={"surface_start": 8, "surface_end": 8},
                ),
            )
        )

    def test_text_span_accepts_normalized_substantial_overlap(self) -> None:
        quote = (
            "Each party agrees that it will not modify, reverse engineer, "
            "decompile, create other works from, or disassemble software."
        )
        case = {
            "group": "direct",
            "document_path": "documents/text/nda.txt",
            "evidence": {"kind": "text_spans", "spans": [{"quote": quote}]},
        }
        self.assertTrue(
            evidence_matches(
                case,
                _evidence(
                    filename="nda.txt",
                    text=(
                        "The parties will not modify, reverse engineer, decompile, "
                        "create other works from, or disassemble software."
                    ),
                ),
            )
        )

    def test_markdown_section_matches_heading_or_table(self) -> None:
        case = {
            "group": "direct",
            "document_path": "documents/markdown/table.md",
            "evidence": {
                "kind": "markdown_sections",
                "sections": ["paragraph-1", "table"],
            },
        }
        self.assertTrue(
            evidence_matches(
                case,
                _evidence(filename="table.md", titles=("Narrative", "Paragraph 1")),
            )
        )
        self.assertTrue(
            evidence_matches(
                case,
                _evidence(filename="table.md", text="| a | b |", modality="table"),
            )
        )

    def test_summary_reports_retrieval_and_unanswerable_separately(self) -> None:
        result = summarize(
            [
                {
                    "group": "paraphrase",
                    "elapsed_ms": 10,
                    "hit_1": True,
                    "recall_5": True,
                    "recall_10": True,
                    "reciprocal_rank_10": 1.0,
                    "relevant_representations": ["auto_qa_question"],
                    "matched_question": "question",
                },
                {
                    "group": "unanswerable",
                    "elapsed_ms": 20,
                    "target_document_top_1": False,
                    "target_document_recall_10": True,
                },
            ]
        )
        self.assertEqual(result["retrieval"]["hit_at_1"], 1.0)
        self.assertEqual(result["retrieval"]["matched_question_count"], 1)
        self.assertEqual(result["unanswerable"]["target_document_recall_10"], 1)


if __name__ == "__main__":
    unittest.main()
