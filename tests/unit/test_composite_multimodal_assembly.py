from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from rag_kb.document_processing import (
    assemble_composite_evidence,
    normalize_figure_labels,
)
from rag_kb.domain import (
    ChunkAssetRelationType,
    ErrorCode,
    ParsedAssetDraft,
    ParsedDocument,
    ParsedElement,
    ParserExecutionError,
    ParserLimits,
)


class CompositeMultimodalAssemblyTests(unittest.TestCase):
    def test_figure_labels_normalize_english_chinese_and_panel_suffixes(self) -> None:
        self.assertEqual(
            normalize_figure_labels("See Fig. 7, Figure 7A and 图 8."),
            ("figure:7", "figure:7a", "figure:8"),
        )
        self.assertEqual(normalize_figure_labels("Version 7 has 8 workers"), ())

    def test_caption_and_explicit_references_create_stable_strong_relations(self) -> None:
        parsed = _figure_document()

        first = assemble_composite_evidence(parsed)
        second = assemble_composite_evidence(parsed)

        self.assertEqual(first, second)
        relation_types = [item.relation_type for item in first.relations]
        self.assertIn(ChunkAssetRelationType.CAPTION_OF, relation_types)
        self.assertGreaterEqual(
            relation_types.count(ChunkAssetRelationType.EXPLICIT_FIGURE_REFERENCE),
            2,
        )
        explicit_chunks = {
            item.chunk_unit_key
            for item in first.relations
            if item.relation_type
            is ChunkAssetRelationType.EXPLICIT_FIGURE_REFERENCE
        }
        self.assertGreaterEqual(len(explicit_chunks), 2)
        self.assertTrue(
            all(item.figure_label == "figure:7" for item in first.relations if item.figure_label)
        )
        self.assertTrue(
            all(item.evidence_group_key for item in first.relations)
        )

    def test_relation_expansion_fails_closed_instead_of_truncating(self) -> None:
        with self.assertRaises(ParserExecutionError) as raised:
            assemble_composite_evidence(
                _figure_document(),
                replace(ParserLimits(), max_relations_per_chunk=0),
            )
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)
        self.assertEqual(
            raised.exception.diagnostic["limit_name"], "max_relations_per_chunk"
        )


def _figure_document() -> ParsedDocument:
    checksum = hashlib.sha256(b"image").hexdigest()
    asset = ParsedAssetDraft(
        asset_key="asset-7",
        kind="pdf_image",
        media_type="image/png",
        content=b"image",
        content_sha256=checksum,
        width=640,
        height=480,
        source_location={"page_number": 1},
        processing_metadata={},
    )
    values = (
        ("NarrativeText", "As shown in Fig. 7, the abstract workflow converges.", None, False),
        ("Image", "", "asset-7", False),
        ("FigureCaption", "Figure 7: Composite workflow", None, False),
        ("Title", "Details", None, True),
        ("NarrativeText", "Figure 7 also identifies the validation edge.", None, False),
    )
    elements = tuple(
        ParsedElement(
            ordinal=ordinal,
            text=text,
            token_count=len(text.split()),
            category=category,
            source_location={"page_number": 1},
            hierarchy={},
            is_title=is_title,
            element_key=f"element-{ordinal}",
            asset_key=asset_key,
        )
        for ordinal, (category, text, asset_key, is_title) in enumerate(values)
    )
    return ParsedDocument(
        elements=elements,
        extracted_character_count=sum(len(item.text) for item in elements),
        assets=(asset,),
    )


if __name__ == "__main__":
    unittest.main()
