from __future__ import annotations

import unittest

from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint

from rag_kb.db.compatibility import EXPECTED_APPLICATION_TABLES, EXPECTED_REVISION
from rag_kb.db.models import Base


class CompositeEvidenceSchemaTests(unittest.TestCase):
    def test_current_head_and_inventory_include_composite_relations(self) -> None:
        self.assertEqual(EXPECTED_REVISION, "0001_current_only_baseline")
        self.assertIn("index_chunk_asset_relation", EXPECTED_APPLICATION_TABLES)
        self.assertIn("index_chunk_lexical", EXPECTED_APPLICATION_TABLES)
        self.assertIn("index_lexical_manifest", EXPECTED_APPLICATION_TABLES)
        self.assertEqual(
            set(Base.metadata.tables),
            set(EXPECTED_APPLICATION_TABLES),
        )

    def test_current_rows_require_complete_relation_manifest_facts(self) -> None:
        chunk = Base.metadata.tables["index_chunk"]
        manifest = Base.metadata.tables["index_artifact_manifest"]

        self.assertTrue(chunk.c.embedding_text.nullable)
        self.assertTrue(chunk.c.embedding_text_hash.nullable)
        self.assertFalse(manifest.c.relation_plan.nullable)
        self.assertFalse(manifest.c.relation_count.nullable)
        self.assertFalse(manifest.c.relation_manifest_hash.nullable)
        self.assertIn(
            "ck_index_chunk_index_chunk_embedding_text_pair",
            {
                constraint.name
                for constraint in chunk.constraints
                if isinstance(constraint, CheckConstraint)
            },
        )

    def test_relation_table_has_scoped_foreign_keys_and_stable_edge(self) -> None:
        relation = Base.metadata.tables["index_chunk_asset_relation"]
        foreign_keys = {
            constraint.name
            for constraint in relation.constraints
            if isinstance(constraint, ForeignKeyConstraint)
        }
        uniques = {
            constraint.name
            for constraint in relation.constraints
            if isinstance(constraint, UniqueConstraint)
        }

        self.assertEqual(
            foreign_keys,
            {
                "fk_chunk_asset_relation_same_scope_target",
                "fk_chunk_asset_relation_same_target_chunk",
                "fk_chunk_asset_relation_same_target_visual",
                "fk_chunk_asset_relation_same_target_asset",
            },
        )
        self.assertIn("uq_chunk_asset_relation_stable_edge", uniques)


if __name__ == "__main__":
    unittest.main()
