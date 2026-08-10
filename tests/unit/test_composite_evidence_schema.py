from __future__ import annotations

import unittest

from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint

from rag_kb.db.models import Base
from rag_kb.db.readiness import EXPECTED_REVISION


class CompositeEvidenceSchemaTests(unittest.TestCase):
    def test_current_head_and_inventory_include_composite_relations(self) -> None:
        self.assertEqual(EXPECTED_REVISION, "0008_local_rerank_mode")
        self.assertIn("index_chunk_asset_relation", Base.metadata.tables)
        self.assertIn("index_chunk_lexical", Base.metadata.tables)
        self.assertIn("index_lexical_manifest", Base.metadata.tables)
        self.assertEqual(len(Base.metadata.tables), 29)
        self.assertIn("model_provider", Base.metadata.tables)
        self.assertIn("model_profile_revision", Base.metadata.tables)
        self.assertIn("model_selection", Base.metadata.tables)

        chat_run = Base.metadata.tables["chat_run"]
        self.assertFalse(chat_run.c.workflow_configuration.nullable)
        self.assertFalse(chat_run.c.workflow_state.nullable)
        self.assertEqual(
            {
                constraint.name
                for constraint in chat_run.constraints
                if isinstance(constraint, CheckConstraint)
                and constraint.name is not None
                and "workflow" in constraint.name
            },
            {
                "ck_chat_run_workflow_configuration_v1",
                "ck_chat_run_workflow_state_v1",
            },
        )

    def test_current_rows_require_complete_relation_manifest_facts(self) -> None:
        chunk = Base.metadata.tables["index_chunk"]
        manifest = Base.metadata.tables["index_artifact_manifest"]

        self.assertTrue(chunk.c.embedding_text.nullable)
        self.assertTrue(chunk.c.embedding_text_hash.nullable)
        self.assertTrue(chunk.c.excluded_at.nullable)
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
