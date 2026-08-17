"""Remove the retired self-built entity graph projection.

Revision ID: 0014_remove_legacy_entity_graph
Revises: 0013_graphiti_graph
Create Date: 2026-08-17
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0014_remove_legacy_entity_graph"
down_revision: str | None = "0013_graphiti_graph"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A legacy READY state is invalid once its derived projection is removed.
    # Do not create a Graphiti build here: extraction remains an explicit user action.
    op.execute(
        """
        UPDATE knowledge_base_graph_config
           SET status = 'disabled',
               build_id = uuidv7(),
               active_build_id = NULL,
               chat_profile_revision_id = NULL,
               extractor_version = 'graphiti_v1',
               preflight_extractor_version = NULL,
               last_error_code = NULL,
               updated_at = now()
         WHERE extractor_version <> 'graphiti_v1'
        """
    )
    op.drop_table("graph_relation_assertion")
    op.drop_table("graph_entity_mention")
    op.drop_table("index_graph_chunk")


def downgrade() -> None:
    op.create_table(
        "index_graph_chunk",
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("kb_id", sa.UUID(), nullable=False),
        sa.Column("build_id", sa.UUID(), nullable=False),
        sa.Column("index_chunk_id", sa.UUID(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("extractor_version", sa.String(length=128), nullable=False),
        sa.Column("result_status", sa.String(length=32), nullable=False),
        sa.Column("entity_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("relation_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("result_hash", sa.String(length=64), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "result_status IN ('extracted','empty','skipped_protocol','skipped_resource')",
            name="graph_chunk_result_status_supported",
        ),
        sa.CheckConstraint(
            "entity_count >= 0 AND relation_count >= 0",
            name="graph_chunk_counts_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspace.id"],
            name="fk_index_graph_chunk_workspace_id_workspace", ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["kb_id"], ["knowledge_base.id"],
            name="fk_index_graph_chunk_kb_id_knowledge_base", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "index_chunk_id"],
            ["index_chunk.workspace_id", "index_chunk.kb_id", "index_chunk.id"],
            name="fk_graph_chunk_same_scope_chunk", ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_index_graph_chunk"),
        sa.UniqueConstraint(
            "workspace_id", "kb_id", "build_id", "index_chunk_id",
            name="uq_graph_chunk_build_chunk",
        ),
    )
    op.create_index("ix_graph_chunk_workspace_id", "index_graph_chunk", ["workspace_id"])
    op.create_index("ix_graph_chunk_kb_id", "index_graph_chunk", ["kb_id"])
    op.create_index(
        "ix_graph_chunk_build_status", "index_graph_chunk",
        ["workspace_id", "kb_id", "build_id", "result_status"],
    )

    op.create_table(
        "graph_entity_mention",
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("kb_id", sa.UUID(), nullable=False),
        sa.Column("build_id", sa.UUID(), nullable=False),
        sa.Column("index_chunk_id", sa.UUID(), nullable=False),
        sa.Column("mention_id", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("surface", sa.String(length=512), nullable=False),
        sa.Column("normalized_surface", sa.String(length=512), nullable=False),
        sa.Column("disambiguator", sa.String(length=512), nullable=True),
        sa.Column("disambiguator_support_start", sa.Integer(), nullable=True),
        sa.Column("disambiguator_support_end", sa.Integer(), nullable=True),
        sa.Column("surface_start", sa.Integer(), nullable=False),
        sa.Column("surface_end", sa.Integer(), nullable=False),
        sa.Column("entity_key", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "entity_type IN ('person','organization','location','product','system','document','event','concept')",
            name="graph_mention_entity_type_supported",
        ),
        sa.CheckConstraint(
            "surface_start >= 0 AND surface_end > surface_start",
            name="graph_mention_surface_span_valid",
        ),
        sa.CheckConstraint(
            "(disambiguator_support_start IS NULL AND disambiguator_support_end IS NULL) "
            "OR (disambiguator_support_start >= 0 AND disambiguator_support_end > disambiguator_support_start)",
            name="graph_mention_disambiguator_span_valid",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspace.id"],
            name="fk_graph_entity_mention_workspace_id_workspace", ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["kb_id"], ["knowledge_base.id"],
            name="fk_graph_entity_mention_kb_id_knowledge_base", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "build_id", "index_chunk_id"],
            [
                "index_graph_chunk.workspace_id", "index_graph_chunk.kb_id",
                "index_graph_chunk.build_id", "index_graph_chunk.index_chunk_id",
            ],
            name="fk_graph_mention_chunk", ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_graph_entity_mention"),
        sa.UniqueConstraint(
            "workspace_id", "kb_id", "build_id", "index_chunk_id", "mention_id",
            name="uq_graph_mention_identity",
        ),
    )
    op.create_index("ix_graph_entity_mention_workspace_id", "graph_entity_mention", ["workspace_id"])
    op.create_index("ix_graph_entity_mention_kb_id", "graph_entity_mention", ["kb_id"])
    op.create_index(
        "ix_graph_mention_entity_key", "graph_entity_mention",
        ["workspace_id", "kb_id", "build_id", "entity_key"],
    )
    op.create_index(
        "ix_graph_mention_surface_prefix", "graph_entity_mention",
        ["workspace_id", "kb_id", "build_id", "normalized_surface"],
        postgresql_ops={"normalized_surface": "text_pattern_ops"},
    )

    op.create_table(
        "graph_relation_assertion",
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("kb_id", sa.UUID(), nullable=False),
        sa.Column("build_id", sa.UUID(), nullable=False),
        sa.Column("index_chunk_id", sa.UUID(), nullable=False),
        sa.Column("relation_id", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("subject_mention_id", sa.String(length=64), nullable=False),
        sa.Column("object_mention_id", sa.String(length=64), nullable=False),
        sa.Column("subject_entity_key", sa.String(length=64), nullable=False),
        sa.Column("object_entity_key", sa.String(length=64), nullable=False),
        sa.Column("predicate", sa.String(length=256), nullable=False),
        sa.Column("normalized_predicate", sa.String(length=256), nullable=False),
        sa.Column("support_start", sa.Integer(), nullable=False),
        sa.Column("support_end", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("subject_entity_key <> object_entity_key", name="graph_relation_not_self"),
        sa.CheckConstraint("support_start >= 0 AND support_end > support_start", name="graph_relation_support_span_valid"),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspace.id"],
            name="fk_graph_relation_workspace_id_workspace", ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["kb_id"], ["knowledge_base.id"],
            name="fk_graph_relation_kb_id_knowledge_base", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "build_id", "index_chunk_id", "subject_mention_id"],
            [
                "graph_entity_mention.workspace_id", "graph_entity_mention.kb_id",
                "graph_entity_mention.build_id", "graph_entity_mention.index_chunk_id",
                "graph_entity_mention.mention_id",
            ],
            name="fk_graph_relation_subject_mention", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "build_id", "index_chunk_id", "object_mention_id"],
            [
                "graph_entity_mention.workspace_id", "graph_entity_mention.kb_id",
                "graph_entity_mention.build_id", "graph_entity_mention.index_chunk_id",
                "graph_entity_mention.mention_id",
            ],
            name="fk_graph_relation_object_mention", ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_graph_relation_assertion"),
        sa.UniqueConstraint(
            "workspace_id", "kb_id", "build_id", "index_chunk_id",
            "subject_entity_key", "object_entity_key", "normalized_predicate",
            name="uq_graph_relation_semantic_identity",
        ),
    )
    op.create_index("ix_graph_relation_workspace_id", "graph_relation_assertion", ["workspace_id"])
    op.create_index("ix_graph_relation_kb_id", "graph_relation_assertion", ["kb_id"])
    op.create_index(
        "ix_graph_relation_subject", "graph_relation_assertion",
        ["workspace_id", "kb_id", "build_id", "subject_entity_key"],
    )
    op.create_index(
        "ix_graph_relation_object", "graph_relation_assertion",
        ["workspace_id", "kb_id", "build_id", "object_entity_key"],
    )
