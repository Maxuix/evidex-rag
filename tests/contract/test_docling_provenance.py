from __future__ import annotations

import hashlib
import json
import unittest

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.common.origin import DocumentOrigin

from rag_kb.document_processing.docling.provenance import (
    PROVENANCE_VERSION,
    ItemSurface,
    aggregate_provenance,
    surface_kind,
    surface_ordinals,
)
from rag_kb.domain import ErrorCode, ParserExecutionError


BBOX = {"l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}


def surface(ordinal: int, kind: str = "page", *, box: bool = True) -> ItemSurface:
    return ItemSurface(
        kind=kind,
        ordinal=ordinal,
        bbox=dict(BBOX) if box else None,
        coord_origin="BOTTOMLEFT" if box else None,
    )


def document(mimetype: str) -> DoclingDocument:
    value = DoclingDocument(name="probe")
    value.origin = DocumentOrigin(mimetype=mimetype, binary_hash=1, filename="probe")
    return value


class SurfaceKindContractTests(unittest.TestCase):
    def test_each_format_family_maps_to_one_frozen_surface_kind(self) -> None:
        cases = {
            "application/pdf": "page",
            "application/vnd.openxmlformats-officedocument."
            "presentationml.presentation": "slide",
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet": "sheet",
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document": "logical",
            "text/markdown": "logical",
            "text/plain": "logical",
        }
        for mimetype, expected in cases.items():
            with self.subTest(mimetype=mimetype):
                self.assertEqual(surface_kind(document(mimetype)), expected)

    def test_a_document_without_origin_stays_logical(self) -> None:
        self.assertEqual(surface_kind(DoclingDocument(name="probe")), "logical")


class AggregationContractTests(unittest.TestCase):
    def test_one_item_on_one_surface_keeps_its_original_geometry(self) -> None:
        projection = aggregate_provenance(
            ((surface(3),),), ("#/texts/1",), max_metadata_bytes=65_536
        )

        self.assertEqual(
            projection,
            {
                "provenance_version": PROVENANCE_VERSION,
                "surface_type": "page",
                "surface_start": 3,
                "surface_end": 3,
                "bbox": BBOX,
                "coord_origin": "BOTTOMLEFT",
                "item_ref_count": 1,
                "item_refs": ["#/texts/1"],
            },
        )

    def test_several_items_report_a_range_and_never_a_merged_box(self) -> None:
        projection = aggregate_provenance(
            ((surface(1),), (surface(2),), ()),
            ("#/texts/1", "#/texts/2", "#/texts/3"),
            max_metadata_bytes=65_536,
        )

        self.assertEqual(projection["surface_start"], 1)
        self.assertEqual(projection["surface_end"], 2)
        self.assertEqual(projection["item_ref_count"], 3)
        self.assertNotIn("bbox", projection)
        self.assertNotIn("coord_origin", projection)

    def test_one_item_spanning_two_provenance_entries_drops_its_box(self) -> None:
        projection = aggregate_provenance(
            ((surface(1), surface(2)),), ("#/texts/1",), max_metadata_bytes=65_536
        )

        self.assertNotIn("bbox", projection)
        self.assertEqual((projection["surface_start"], projection["surface_end"]), (1, 2))

    def test_items_without_provenance_report_a_logical_surface(self) -> None:
        projection = aggregate_provenance(
            ((), ()), ("#/texts/1", "#/texts/2"), max_metadata_bytes=65_536
        )

        self.assertEqual(projection["surface_type"], "logical")
        self.assertNotIn("surface_start", projection)
        self.assertNotIn("surface_end", projection)

    def test_a_chunk_spanning_two_surface_kinds_fails_closed(self) -> None:
        with self.assertRaises(ParserExecutionError) as raised:
            aggregate_provenance(
                ((surface(1, "page"),), (surface(1, "sheet"),)),
                ("#/texts/1", "#/texts/2"),
                max_metadata_bytes=65_536,
            )

        self.assertEqual(raised.exception.code, ErrorCode.PARSER_OUTPUT_INVALID)
        self.assertEqual(raised.exception.diagnostic["check"], "chunk_surface_kind")

    def test_misaligned_surfaces_and_references_fail_closed(self) -> None:
        with self.assertRaises(ParserExecutionError) as raised:
            aggregate_provenance(
                ((surface(1),),), ("#/texts/1", "#/texts/2"), max_metadata_bytes=65_536
            )

        self.assertEqual(raised.exception.code, ErrorCode.PARSER_OUTPUT_INVALID)

    def test_oversized_reference_lists_are_hashed_rather_than_truncated(self) -> None:
        references = tuple(f"#/texts/{index}" for index in range(400))

        projection = aggregate_provenance(
            tuple((surface(1),) for _ in references),
            references,
            max_metadata_bytes=256,
        )

        self.assertNotIn("item_refs", projection)
        self.assertTrue(projection["item_refs_omitted"])
        self.assertEqual(projection["item_ref_count"], len(references))
        self.assertEqual(
            projection["item_refs_sha256"],
            hashlib.sha256(
                json.dumps(
                    list(references),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )

    def test_a_bounded_projection_stays_inside_the_metadata_budget(self) -> None:
        references = tuple(f"#/texts/{index}" for index in range(400))

        projection = aggregate_provenance(
            tuple((surface(1),) for _ in references),
            references,
            max_metadata_bytes=256,
        )

        encoded = json.dumps(projection, separators=(",", ":")).encode("utf-8")
        self.assertLessEqual(len(encoded), 256)


class MigrationReaderContractTests(unittest.TestCase):
    def test_retired_revision_location_keys_stay_readable(self) -> None:
        self.assertEqual(surface_ordinals({"page_number": 4}), frozenset({4}))
        self.assertEqual(
            surface_ordinals({"page_numbers": [2, 3, 5]}), frozenset({2, 3, 5})
        )
        self.assertEqual(
            surface_ordinals({"page_start": 1, "page_end": 3}), frozenset({1, 3})
        )

    def test_the_new_view_is_read_by_the_same_helper(self) -> None:
        self.assertEqual(
            surface_ordinals(
                {
                    "provenance_version": PROVENANCE_VERSION,
                    "surface_type": "page",
                    "surface_start": 6,
                    "surface_end": 7,
                }
            ),
            frozenset({6, 7}),
        )

    def test_booleans_and_junk_are_never_read_as_ordinals(self) -> None:
        self.assertEqual(
            surface_ordinals(
                {"page_number": True, "page_numbers": ["3", None, False], "surface_start": "9"}
            ),
            frozenset(),
        )


if __name__ == "__main__":
    unittest.main()
