from __future__ import annotations

import unittest
from uuid import UUID

from tools.qualify_musique_mini_corpus import (
    QualificationError,
    _chunk_document_map,
    _document_ids,
    _path_complete,
)


class MusiqueQualificationTests(unittest.TestCase):
    def test_path_complete_accepts_one_hop_and_multihop_paths(self) -> None:
        self.assertTrue(_path_complete((("a",),), {"a"}))
        self.assertTrue(_path_complete((("a", "b", "c"),), {"a", "b", "c"}))
        self.assertFalse(_path_complete((("a", "b"),), {"a"}))

    def test_chunk_document_map_requires_exact_serving_set(self) -> None:
        chunk_id = UUID("00000000-0000-0000-0000-000000000001")
        result = _chunk_document_map(
            ({"index_chunk_id": chunk_id, "original_filename": "a.md"},),
            ({"filename": "a.md", "document_id": "doc-a"},),
        )
        self.assertEqual(result, {chunk_id: "doc-a"})
        with self.assertRaisesRegex(
            QualificationError, "musique_serving_document_set_changed"
        ):
            _chunk_document_map(
                ({"index_chunk_id": chunk_id, "original_filename": "b.md"},),
                ({"filename": "a.md", "document_id": "doc-a"},),
            )

    def test_document_ids_are_ordered_deduplicated_and_ignore_unknown(self) -> None:
        first = UUID("00000000-0000-0000-0000-000000000001")
        second = UUID("00000000-0000-0000-0000-000000000002")
        missing = UUID("00000000-0000-0000-0000-000000000003")
        self.assertEqual(
            _document_ids((first, second, first, missing), {first: "a", second: "b"}),
            ("a", "b"),
        )


if __name__ == "__main__":
    unittest.main()
