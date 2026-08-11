from __future__ import annotations

import unittest
from uuid import UUID

from rag_kb.domain import DuplicateDocumentError


class DuplicateDocumentErrorTests(unittest.TestCase):
    def test_error_keeps_only_the_existing_document_id(self) -> None:
        existing_id = UUID("01900000-0000-7000-8000-000000000099")
        error = DuplicateDocumentError(existing_id)

        self.assertEqual(error.existing_document_id, existing_id)
        self.assertEqual(str(error), "document content already exists")
        self.assertNotIn("/", str(error))
        self.assertNotIn("checksum", str(error))
        self.assertNotIn("filename", str(error))


if __name__ == "__main__":
    unittest.main()
