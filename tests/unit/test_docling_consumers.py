from __future__ import annotations

import hashlib
import unittest
from typing import Any
from uuid import UUID

from PIL import Image
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)
from docling_core.types.doc.common.origin import DocumentOrigin
from docling_core.types.doc.common.reference import ImageRef
from docling_core.types.doc.items.table.table_data import TableCell, TableData

from rag_kb.document_processing.docling import (
    ItemKind,
    assemble_semantic_chunks,
    assemble_structural,
    chunk_assembly_key,
    classify_item,
    composite_evidence,
    docling_item_sequence_hash,
    docling_semantic_units,
    docling_unit_sequence_hash,
    extract_docling_assets,
    item_text,
    iterate_body_items,
    project_source_location,
    relate_assets_to_chunks,
    section_paths,
    text_only_document,
)
from rag_kb.document_processing.docling.assets import (
    ASSET_KIND_PAGE_IMAGE,
    ASSET_KIND_PICTURE,
    ASSET_KIND_TABLE_IMAGE,
)
from rag_kb.document_processing.docling.figures import normalize_figure_labels
from rag_kb.document_processing.semantic_boundaries import build_chunk_plan
from rag_kb.domain import (
    ChunkAssetRelationProvenance,
    ChunkAssetRelationType,
    ContentModality,
    ErrorCode,
    ParserExecutionError,
    ParserLimits,
)


PDF_MIMETYPE = "application/pdf"
DOCX_MIMETYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
VERSION_ID = UUID("01900000-0000-7000-8000-0000000009a1")
PROFILE = "docling_multimodal_local_v2:structural_by_title_token_v3"


def prov(page: int, span: tuple[int, int] = (0, 10)) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page,
        bbox=BoundingBox(l=1, t=2, r=3, b=4, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=span,
    )


def table_data(rows: tuple[tuple[str, ...], ...]) -> TableData:
    cells = [
        TableCell(
            text=value,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=column,
            end_col_offset_idx=column + 1,
            column_header=row == 0,
        )
        for row, values in enumerate(rows)
        for column, value in enumerate(values)
    ]
    return TableData(num_rows=len(rows), num_cols=len(rows[0]), table_cells=cells)


def image(width: int = 120, height: int = 90, colour: int = 200) -> ImageRef:
    return ImageRef.from_pil(
        Image.new("RGB", (width, height), (colour, 10, 10)), dpi=72
    )


def paginated_document(*, page_images: bool = False) -> DoclingDocument:
    """A two-surface document exercising every traversal branch."""

    document = DoclingDocument(name="paginated")
    document.origin = DocumentOrigin(
        mimetype=PDF_MIMETYPE, binary_hash=7, filename="paginated.pdf"
    )
    document.add_page(page_no=1, size=Size(width=600, height=800))
    document.add_page(
        page_no=2,
        size=Size(width=600, height=800),
        image=image(300, 400, colour=5) if page_images else None,
    )
    document.add_title(text="Doc title", prov=prov(1))
    document.add_heading(text="Section A", level=1, prov=prov(1))
    document.add_text(
        label=DocItemLabel.TEXT,
        text="Body paragraph one. As shown in Figure 1, values rise.",
        prov=prov(1),
    )
    caption = document.add_text(
        label=DocItemLabel.CAPTION, text="Figure 1. A chart", prov=prov(1)
    )
    document.add_picture(image=image(), caption=caption, prov=prov(1))
    document.add_text(
        label=DocItemLabel.TEXT, text="Follow-up prose after the figure.", prov=prov(1)
    )
    document.add_heading(text="Section B", level=1, prov=prov(2))
    document.add_table(
        data=table_data((("h1", "h2"), ("a", "b"))), prov=prov(2)
    )
    document.add_text(
        label=DocItemLabel.TEXT, text="Closing paragraph on page two.", prov=prov(2)
    )
    return document


def logical_document() -> DoclingDocument:
    document = DoclingDocument(name="logical")
    document.origin = DocumentOrigin(
        mimetype=DOCX_MIMETYPE, binary_hash=11, filename="logical.docx"
    )
    document.add_heading(text="Overview", level=1)
    document.add_text(label=DocItemLabel.TEXT, text="First logical paragraph.")
    document.add_text(label=DocItemLabel.TEXT, text="Second logical paragraph.")
    return document


class TraversalTests(unittest.TestCase):
    def test_traversal_skips_furniture_and_classifies_every_kind(self) -> None:
        document = paginated_document()
        document.add_text(
            label=DocItemLabel.PAGE_HEADER, text="running header", prov=prov(2)
        )
        kinds = [classify_item(item) for item, _level in iterate_body_items(document)]

        self.assertNotIn(ItemKind.FURNITURE, kinds)
        self.assertEqual(
            kinds,
            [
                ItemKind.TITLE,
                ItemKind.SECTION_HEADER,
                ItemKind.TEXT,
                ItemKind.CAPTION,
                ItemKind.PICTURE,
                ItemKind.TEXT,
                ItemKind.SECTION_HEADER,
                ItemKind.TABLE,
                ItemKind.TEXT,
            ],
        )

    def test_pictures_carry_no_text_and_tables_serialize_through_docling(self) -> None:
        document = paginated_document()
        by_ref = {item.self_ref: item for item, _level in iterate_body_items(document)}

        self.assertEqual(item_text(by_ref["#/pictures/0"], document), "")
        self.assertEqual(
            item_text(by_ref["#/tables/0"], document),
            "| h1   | h2   |\n|------|------|\n| a    | b    |",
        )

    def test_section_paths_follow_heading_levels(self) -> None:
        paths = section_paths(paginated_document())

        self.assertEqual(
            paths["#/texts/2"],
            (
                {"depth": 0, "text": "Doc title"},
                {"depth": 1, "text": "Section A"},
            ),
        )
        self.assertEqual(
            paths["#/tables/0"],
            (
                {"depth": 0, "text": "Doc title"},
                {"depth": 1, "text": "Section B"},
            ),
        )

    def test_oversized_table_html_fails_closed(self) -> None:
        document = DoclingDocument(name="wide")
        document.origin = DocumentOrigin(
            mimetype=PDF_MIMETYPE, binary_hash=1, filename="wide.pdf"
        )
        document.add_page(page_no=1, size=Size(width=600, height=800))
        document.add_table(
            data=table_data((("h",) * 40, *(("x" * 64,) * 40 for _ in range(40)))),
            prov=prov(1),
        )

        with self.assertRaises(ParserExecutionError) as raised:
            assemble_structural(document, ParserLimits(max_table_html_bytes=256))
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)


class StructuralAssemblyTests(unittest.TestCase):
    def test_headings_join_the_content_they_introduce(self) -> None:
        chunks = assemble_structural(paginated_document())

        self.assertTrue(chunks[0].text.startswith("Doc title\n\nSection A\n\nBody"))
        self.assertTrue(chunks[1].text.startswith("Section B\n\n| h1"))
        self.assertNotIn("Doc title", {chunk.text for chunk in chunks})

    def test_visual_and_caption_references_stay_without_spending_tokens(self) -> None:
        chunks = assemble_structural(paginated_document())

        self.assertIn("#/pictures/0", chunks[0].item_refs)
        self.assertIn("#/texts/3", chunks[0].item_refs)
        self.assertNotIn("Figure 1. A chart", chunks[0].text)

    def test_surface_change_and_table_are_hard_boundaries(self) -> None:
        chunks = assemble_structural(paginated_document())
        surfaces = [
            (chunk.source_location["surface_start"], chunk.source_location["surface_end"])
            for chunk in chunks
        ]

        self.assertEqual(surfaces, [(1, 1), (2, 2), (2, 2)])
        self.assertEqual(chunks[1].item_refs, ("#/texts/5", "#/tables/0"))

    def test_logical_documents_get_no_fabricated_surface(self) -> None:
        chunks = assemble_structural(logical_document())

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_location["surface_type"], "logical")
        self.assertNotIn("surface_start", chunks[0].source_location)

    def test_unknown_items_with_text_are_ordinary_content(self) -> None:
        document = logical_document()
        document.add_text(label=DocItemLabel.CHECKBOX_SELECTED, text="Accepted")

        chunks = assemble_structural(document)

        self.assertIn("Accepted", chunks[0].text)

    def test_unknown_items_without_text_never_fail_the_document(self) -> None:
        document = logical_document()
        document.add_text(label=DocItemLabel.CHECKBOX_UNSELECTED, text="")

        chunks = assemble_structural(document)

        self.assertNotIn("#/texts/3", chunks[0].item_refs)

    def test_a_document_without_content_fails_closed(self) -> None:
        document = DoclingDocument(name="empty")
        document.origin = DocumentOrigin(
            mimetype=DOCX_MIMETYPE, binary_hash=3, filename="empty.docx"
        )
        document.add_text(label=DocItemLabel.TEXT, text="")

        with self.assertRaises(ParserExecutionError) as raised:
            assemble_structural(document)
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_OUTPUT_INVALID)

    def test_oversized_item_splits_on_the_frozen_token_budget(self) -> None:
        document = logical_document()
        document.add_text(label=DocItemLabel.TEXT, text=" ".join(["alpha"] * 2000))

        chunks = assemble_structural(document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 800 for chunk in chunks))
        self.assertEqual(chunks[-1].item_refs, ("#/texts/3",))

    def test_repeated_assembly_is_identical(self) -> None:
        first = assemble_structural(paginated_document())
        second = assemble_structural(paginated_document())

        self.assertEqual(
            [(chunk.text, chunk.item_refs, chunk.source_location) for chunk in first],
            [(chunk.text, chunk.item_refs, chunk.source_location) for chunk in second],
        )

    def test_chunk_identity_uses_the_complete_reference_sequence(self) -> None:
        chunk = assemble_structural(paginated_document())[0]
        key = chunk_assembly_key(
            profile="structural_by_title_token_v3",
            source_checksum_sha256="a" * 64,
            assembly_ordinal=0,
            item_refs=chunk.item_refs,
            text=chunk.text,
        )
        shortened = chunk_assembly_key(
            profile="structural_by_title_token_v3",
            source_checksum_sha256="a" * 64,
            assembly_ordinal=0,
            item_refs=chunk.item_refs[:-1],
            text=chunk.text,
        )

        self.assertNotEqual(key, shortened)
        self.assertEqual(
            key,
            chunk_assembly_key(
                profile="structural_by_title_token_v3",
                source_checksum_sha256="a" * 64,
                assembly_ordinal=0,
                item_refs=chunk.item_refs,
                text=chunk.text,
            ),
        )
        self.assertNotEqual(
            key,
            chunk_assembly_key(
                profile="structural_by_title_token_v3",
                source_checksum_sha256="a" * 64,
                assembly_ordinal=1,
                item_refs=chunk.item_refs,
                text=chunk.text,
            ),
        )


class SemanticUnitTests(unittest.TestCase):
    def test_boundaries_come_from_surface_table_and_section(self) -> None:
        units = docling_semantic_units(paginated_document())
        reasons = [unit.hard_boundary_before for unit in units]

        self.assertEqual(reasons[0], "section")
        self.assertIn("page", reasons)
        self.assertIn("table", reasons)

    def test_code_items_form_a_block_boundary(self) -> None:
        document = logical_document()
        document.add_code(text="print('hello world')")
        document.add_text(label=DocItemLabel.TEXT, text="Trailing prose paragraph.")

        units = docling_semantic_units(document)

        self.assertEqual([unit.hard_boundary_before for unit in units][-2:], ["block", "block"])

    def test_visual_references_attach_to_the_neighbouring_unit(self) -> None:
        units = docling_semantic_units(paginated_document())
        references = {reference for unit in units for reference in unit.item_refs}

        self.assertIn("#/pictures/0", references)
        self.assertIn("#/texts/3", references)

    def test_oversized_items_split_with_bounded_fragment_metadata(self) -> None:
        document = logical_document()
        document.add_text(label=DocItemLabel.TEXT, text=" ".join(["beta"] * 400))

        units = docling_semantic_units(document)
        fragments = [
            unit.source_location["fragment"]
            for unit in units
            if "fragment" in unit.source_location
        ]

        self.assertTrue(fragments)
        self.assertEqual(fragments[0]["index"], 0)
        self.assertEqual(fragments[0]["count"], len(fragments))
        self.assertEqual(len(fragments[0]["charspan"]), 2)
        self.assertTrue(all(unit.token_count <= 160 for unit in units))

    def test_a_document_without_analysable_text_fails_closed(self) -> None:
        document = DoclingDocument(name="visual-only")
        document.origin = DocumentOrigin(
            mimetype=PDF_MIMETYPE, binary_hash=5, filename="visual-only.pdf"
        )
        document.add_page(page_no=1, size=Size(width=600, height=800))
        document.add_picture(image=image(), prov=prov(1))

        with self.assertRaises(ParserExecutionError) as raised:
            docling_semantic_units(document)
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_OUTPUT_INVALID)

    def test_the_plan_binds_to_the_docling_unit_projection(self) -> None:
        document = paginated_document()
        units = docling_semantic_units(document)
        sequence_hash = docling_unit_sequence_hash(units)
        plan = build_chunk_plan(
            indexed_document_version_id=VERSION_ID,
            source_checksum_sha256="b" * 64,
            profile_fingerprint="c" * 64,
            units=units,
            vectors=tuple(
                (1.0, 0.0) if index % 2 == 0 else (0.0, 1.0)
                for index in range(len(units))
            ),
            sequence_hash=sequence_hash,
        )
        chunks = assemble_semantic_chunks(document, units, plan)

        self.assertEqual(plan.unit_sequence_hash, sequence_hash)
        self.assertEqual(len(chunks), plan.chunk_count)
        self.assertEqual(
            sorted({reference for chunk in chunks for reference in chunk.item_refs}),
            sorted({reference for unit in units for reference in unit.item_refs}),
        )

    def test_repeated_unit_projection_hashes_identically(self) -> None:
        self.assertEqual(
            docling_unit_sequence_hash(docling_semantic_units(paginated_document())),
            docling_unit_sequence_hash(docling_semantic_units(paginated_document())),
        )


class AssetExtractionTests(unittest.TestCase):
    def test_pictures_become_deterministic_png_assets(self) -> None:
        assets = extract_docling_assets(paginated_document())
        repeated = extract_docling_assets(paginated_document())

        self.assertEqual([asset.kind for asset in assets], [ASSET_KIND_PICTURE])
        self.assertEqual(assets[0].media_type, "image/png")
        self.assertEqual((assets[0].width, assets[0].height), (120, 90))
        self.assertEqual(
            [asset.content_sha256 for asset in assets],
            [asset.content_sha256 for asset in repeated],
        )
        self.assertEqual(
            [asset.asset_key for asset in assets],
            [asset.asset_key for asset in repeated],
        )

    def test_author_captions_stay_on_the_asset(self) -> None:
        asset = extract_docling_assets(paginated_document())[0]

        self.assertEqual(asset.processing_metadata["caption"], "Figure 1. A chart")
        self.assertEqual(asset.processing_metadata["caption_refs"], ["#/texts/3"])
        self.assertEqual(asset.processing_metadata["item_ref"], "#/pictures/0")

    def test_table_images_are_extracted_when_docling_rendered_one(self) -> None:
        document = paginated_document()
        document.tables[0].image = image(200, 150, colour=90)

        kinds = [asset.kind for asset in extract_docling_assets(document)]

        self.assertEqual(kinds, [ASSET_KIND_PICTURE, ASSET_KIND_TABLE_IMAGE])

    def test_page_images_need_an_explicit_or_empty_surface_judgement(self) -> None:
        document = paginated_document(page_images=True)

        default = extract_docling_assets(document)
        requested = extract_docling_assets(
            document, page_image_surfaces=frozenset({2})
        )

        self.assertEqual([asset.kind for asset in default], [ASSET_KIND_PICTURE])
        self.assertEqual(
            [asset.kind for asset in requested],
            [ASSET_KIND_PICTURE, ASSET_KIND_PAGE_IMAGE],
        )
        self.assertEqual(requested[1].source_location["surface_start"], 2)

    def test_a_surface_docling_read_nothing_from_becomes_a_page_image(self) -> None:
        document = paginated_document(page_images=True)
        document.add_page(
            page_no=3, size=Size(width=600, height=800), image=image(300, 400, colour=9)
        )

        kinds = [asset.kind for asset in extract_docling_assets(document)]

        self.assertEqual(kinds, [ASSET_KIND_PICTURE, ASSET_KIND_PAGE_IMAGE])

    def test_logical_documents_never_produce_page_images(self) -> None:
        self.assertEqual(extract_docling_assets(logical_document()), ())

    def test_oversized_images_fail_closed(self) -> None:
        with self.assertRaises(ParserExecutionError) as raised:
            extract_docling_assets(
                paginated_document(), ParserLimits(max_image_pixels=16)
            )
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)

    def test_the_total_asset_budget_is_enforced(self) -> None:
        with self.assertRaises(ParserExecutionError) as raised:
            extract_docling_assets(
                paginated_document(), ParserLimits(max_total_asset_bytes=16)
            )
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)


class DecorativeVisualTests(unittest.TestCase):
    def _document(self, *, repeats: int, size: tuple[int, int]) -> DoclingDocument:
        document = DoclingDocument(name="furniture")
        document.origin = DocumentOrigin(
            mimetype=PDF_MIMETYPE, binary_hash=21, filename="furniture.pdf"
        )
        document.add_page(page_no=1, size=Size(width=600, height=800))
        document.add_text(
            label=DocItemLabel.TEXT, text="Body text beside the mark.", prov=prov(1)
        )
        mark = image(size[0], size[1], colour=64)
        for _ in range(repeats):
            document.add_picture(image=mark, prov=prov(1))
        return document

    def test_visuals_too_small_to_read_are_not_evidence(self) -> None:
        assets = extract_docling_assets(self._document(repeats=1, size=(40, 40)))

        self.assertEqual(assets, ())

    def test_a_small_repeated_mark_is_treated_as_furniture(self) -> None:
        assets = extract_docling_assets(self._document(repeats=3, size=(96, 64)))

        self.assertEqual(assets, ())

    def test_a_repeated_but_large_visual_stays_evidence(self) -> None:
        assets = extract_docling_assets(self._document(repeats=3, size=(420, 460)))

        self.assertEqual(len(assets), 3)

    def test_a_single_ordinary_figure_stays_evidence(self) -> None:
        assets = extract_docling_assets(self._document(repeats=1, size=(96, 64)))

        self.assertEqual(len(assets), 1)


class RelationTests(unittest.TestCase):
    def test_docling_caption_reference_wins_over_surface_heuristics(self) -> None:
        document = paginated_document()
        chunks = assemble_structural(document)
        assets = extract_docling_assets(document)

        relations = relate_assets_to_chunks(document, chunks, assets)

        self.assertEqual(relations[0].chunk_index, 0)
        self.assertEqual(relations[0].relation_type, ChunkAssetRelationType.CAPTION_OF)
        self.assertEqual(
            relations[0].provenance,
            ChunkAssetRelationProvenance.DOCLING_CAPTION_REF_V1,
        )
        self.assertEqual(relations[0].confidence_micros, 1_000_000)
        self.assertEqual(relations[0].figure_label, "figure:1")

    def test_page_images_relate_to_every_chunk_on_their_surface(self) -> None:
        document = paginated_document(page_images=True)
        chunks = assemble_structural(document)
        assets = extract_docling_assets(document, page_image_surfaces=frozenset({2}))

        relations = relate_assets_to_chunks(document, chunks, assets)
        ocr = [
            relation
            for relation in relations
            if relation.relation_type is ChunkAssetRelationType.OCR_OF
        ]

        self.assertEqual([relation.chunk_index for relation in ocr], [1, 2])
        self.assertTrue(
            all(
                relation.provenance
                is ChunkAssetRelationProvenance.DOCLING_PAGE_OCR_V1
                for relation in ocr
            )
        )

    def test_a_table_image_inside_its_chunk_is_a_table_relation(self) -> None:
        document = paginated_document()
        document.tables[0].image = image(200, 150, colour=90)
        chunks = assemble_structural(document)
        assets = extract_docling_assets(document)

        relations = relate_assets_to_chunks(document, chunks, assets)
        table = [
            relation
            for relation in relations
            if relation.relation_type is ChunkAssetRelationType.TABLE_OF
        ]

        self.assertEqual(len(table), 1)
        self.assertEqual(table[0].chunk_index, 1)
        self.assertEqual(
            table[0].provenance,
            ChunkAssetRelationProvenance.DOCLING_TABLE_IDENTITY_V1,
        )

    def test_direct_containment_outranks_shared_parent(self) -> None:
        document = logical_document()
        heading = document.texts[0]
        picture = document.add_picture(image=image(), parent=heading)
        document.add_text(
            label=DocItemLabel.TEXT, text="Trailing prose.", parent=heading
        )
        chunks = assemble_structural(document)
        assets = extract_docling_assets(document)

        relations = relate_assets_to_chunks(document, chunks, assets)

        self.assertIn(picture.self_ref, chunks[0].item_refs)
        self.assertIn(heading.self_ref, chunks[0].item_refs)
        self.assertEqual(
            relations[0].provenance,
            ChunkAssetRelationProvenance.DOCLING_ITEM_REF_V1,
        )

    def test_a_distant_chunk_falls_back_to_surface_co_location(self) -> None:
        document = paginated_document()
        document.add_text(
            label=DocItemLabel.TEXT,
            text="An unrelated paragraph that mentions nothing.",
            prov=prov(1),
        )
        chunks = assemble_structural(document)
        assets = extract_docling_assets(document)

        weak = [
            relation
            for relation in relate_assets_to_chunks(document, chunks, assets)
            if relation.relation_type
            in {
                ChunkAssetRelationType.SAME_PAGE,
                ChunkAssetRelationType.SPATIAL_NEIGHBOR,
            }
        ]

        self.assertTrue(weak)
        self.assertTrue(all(relation.confidence_micros < 1_000_000 for relation in weak))

    def test_at_most_one_relation_per_chunk_and_asset(self) -> None:
        document = paginated_document(page_images=True)
        chunks = assemble_structural(document)
        assets = extract_docling_assets(document, page_image_surfaces=frozenset({2}))

        relations = relate_assets_to_chunks(document, chunks, assets)
        pairs = [(relation.chunk_index, relation.asset_key) for relation in relations]

        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(
            [relation.ordinal for relation in relations], list(range(len(relations)))
        )

    def test_the_per_chunk_relation_budget_is_enforced(self) -> None:
        document = paginated_document(page_images=True)
        chunks = assemble_structural(document)
        assets = extract_docling_assets(document, page_image_surfaces=frozenset({2}))

        with self.assertRaises(ParserExecutionError) as raised:
            relate_assets_to_chunks(
                document, chunks, assets, ParserLimits(max_relations_per_chunk=0)
            )
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_RESOURCE_LIMIT)

    def test_repeated_relation_manifests_match(self) -> None:
        def manifest() -> list[tuple[Any, ...]]:
            document = paginated_document(page_images=True)
            chunks = assemble_structural(document)
            assets = extract_docling_assets(
                document, page_image_surfaces=frozenset({2})
            )
            return [
                (
                    relation.chunk_index,
                    relation.asset_key,
                    relation.relation_type.value,
                    relation.provenance.value,
                    relation.confidence_micros,
                    relation.ordinal,
                )
                for relation in relate_assets_to_chunks(document, chunks, assets)
            ]

        self.assertEqual(manifest(), manifest())


class FigureLabelTests(unittest.TestCase):
    def test_figure_labels_normalize_english_chinese_and_panel_suffixes(self) -> None:
        self.assertEqual(
            normalize_figure_labels("See Fig. 7, Figure 7A and 图 8."),
            ("figure:7", "figure:7a", "figure:8"),
        )
        self.assertEqual(normalize_figure_labels("Version 7 has 8 workers"), ())


class EvidenceProjectionTests(unittest.TestCase):
    def _projection(self, *, page_images: bool = False):
        document = paginated_document(page_images=page_images)
        document.tables[0].image = image(200, 150, colour=90)
        chunks = assemble_structural(document)
        assets = extract_docling_assets(
            document,
            page_image_surfaces=frozenset({2}) if page_images else None,
        )
        relations = relate_assets_to_chunks(document, chunks, assets)
        return (
            composite_evidence(
                document,
                chunks,
                assets,
                relations,
                profile=PROFILE,
                source_checksum_sha256="d" * 64,
            ),
            chunks,
            assets,
        )

    def test_text_only_projection_keeps_chunk_order_and_hashes(self) -> None:
        chunks = assemble_structural(paginated_document())

        processed = text_only_document(chunks, profile=PROFILE)

        self.assertEqual(len(processed.chunks), len(chunks))
        self.assertEqual(
            [draft.ordinal for draft in processed.chunks], list(range(len(chunks)))
        )
        self.assertEqual(
            [draft.text for draft in processed.chunks], [chunk.text for chunk in chunks]
        )
        self.assertEqual(
            processed.chunks[0].content_sha256,
            hashlib.sha256(chunks[0].text.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            processed.extracted_character_count,
            sum(len(chunk.text) for chunk in chunks),
        )

    def test_an_empty_assembly_fails_closed(self) -> None:
        with self.assertRaises(ParserExecutionError) as raised:
            text_only_document((), profile=PROFILE)
        self.assertEqual(raised.exception.code, ErrorCode.PARSER_OUTPUT_INVALID)

    def test_units_interleave_visuals_in_reading_order(self) -> None:
        draft, chunks, _assets = self._projection()

        self.assertEqual(
            [unit.ordinal for unit in draft.units], list(range(len(draft.units)))
        )
        self.assertEqual(len(draft.units), len(chunks) + 1)
        self.assertEqual(
            [unit.modality for unit in draft.units],
            [
                ContentModality.TEXT,
                ContentModality.IMAGE,
                ContentModality.TABLE,
                ContentModality.TEXT,
            ],
        )

    def test_a_table_image_is_an_optional_representation_of_its_chunk(self) -> None:
        draft, _chunks, assets = self._projection()
        table_asset = next(
            asset for asset in assets if asset.kind == ASSET_KIND_TABLE_IMAGE
        )
        table_unit = next(
            unit for unit in draft.units if unit.modality is ContentModality.TABLE
        )

        self.assertEqual(table_unit.asset_key, table_asset.asset_key)
        self.assertEqual(table_unit.required_representations, ("table_text",))
        self.assertNotIn(
            table_asset.asset_key,
            {
                unit.asset_key
                for unit in draft.units
                if unit.modality is ContentModality.IMAGE
            },
        )

    def test_relations_resolve_to_real_visual_units(self) -> None:
        draft, _chunks, _assets = self._projection(page_images=True)
        keys = {unit.unit_key: unit for unit in draft.units}

        self.assertTrue(draft.relations)
        for relation in draft.relations:
            self.assertIn(relation.chunk_unit_key, keys)
            self.assertIn(relation.visual_unit_key, keys)
            self.assertEqual(
                keys[relation.visual_unit_key].asset_key, relation.asset_key
            )
            self.assertTrue(relation.evidence_group_key)
        self.assertEqual(
            [relation.ordinal for relation in draft.relations],
            list(range(len(draft.relations))),
        )

    def test_a_table_relation_closes_on_its_own_chunk(self) -> None:
        draft, _chunks, _assets = self._projection()
        table = next(
            relation
            for relation in draft.relations
            if relation.relation_type is ChunkAssetRelationType.TABLE_OF
        )

        self.assertEqual(table.chunk_unit_key, table.visual_unit_key)

    def test_repeated_projection_is_identical(self) -> None:
        first, _chunks, _assets = self._projection(page_images=True)
        second, _chunks, _assets = self._projection(page_images=True)

        self.assertEqual(
            [(unit.unit_key, unit.ordinal, unit.content) for unit in first.units],
            [(unit.unit_key, unit.ordinal, unit.content) for unit in second.units],
        )
        self.assertEqual(
            docling_item_sequence_hash(paginated_document()),
            docling_item_sequence_hash(paginated_document()),
        )


class LocationProjectionTests(unittest.TestCase):
    def test_a_single_item_keeps_its_original_bounding_box(self) -> None:
        document = paginated_document()

        projection = project_source_location(document, ("#/texts/2",), ParserLimits())

        self.assertEqual(projection["surface_type"], "page")
        self.assertEqual(projection["surface_start"], 1)
        self.assertEqual(projection["bbox"], {"l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0})
        self.assertEqual(projection["coord_origin"], "BOTTOMLEFT")

    def test_several_items_never_get_a_merged_rectangle(self) -> None:
        document = paginated_document()

        projection = project_source_location(
            document, ("#/texts/2", "#/texts/4"), ParserLimits()
        )

        self.assertNotIn("bbox", projection)
        self.assertEqual(projection["item_ref_count"], 2)


if __name__ == "__main__":
    unittest.main()
