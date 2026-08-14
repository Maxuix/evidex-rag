"""Add current-only entity graph configuration and derived facts.

Revision ID: 0012_entity_graph_rag
Revises: 0011_native_agent_round_limit
Create Date: 2026-08-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0012_entity_graph_rag"
down_revision: str | None = "0011_native_agent_round_limit"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_index_chunk_workspace_kb_id",
        "index_chunk",
        ["workspace_id", "kb_id", "id"],
    )

    op.create_table(
        "knowledge_base_graph_config",
        sa.Column("kb_id", sa.UUID(), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default=sa.text("'disabled'"),
            nullable=False,
        ),
        sa.Column(
            "build_id",
            sa.UUID(),
            server_default=sa.text("uuidv7()"),
            nullable=False,
        ),
        sa.Column("chat_profile_revision_id", sa.UUID(), nullable=True),
        sa.Column(
            "extractor_version",
            sa.String(length=128),
            server_default=sa.text("'entity_graph_v1'"),
            nullable=False,
        ),
        sa.Column("preflight_extractor_version", sa.String(length=128), nullable=True),
        sa.Column("last_error_code", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('disabled','building','ready','failed')",
            name="graph_config_status_supported",
        ),
        sa.CheckConstraint(
            "(status = 'disabled' AND chat_profile_revision_id IS NULL) "
            "OR (status <> 'disabled' AND chat_profile_revision_id IS NOT NULL)",
            name="graph_config_profile_matches_status",
        ),
        sa.CheckConstraint(
            "length(btrim(extractor_version)) > 0",
            name="graph_config_extractor_version_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id"],
            ["knowledge_base.workspace_id", "knowledge_base.id"],
            name="fk_graph_config_same_workspace_kb",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "chat_profile_revision_id"],
            ["model_profile_revision.workspace_id", "model_profile_revision.id"],
            name="fk_graph_config_same_workspace_chat_profile",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("kb_id", name="pk_knowledge_base_graph_config"),
        sa.UniqueConstraint("workspace_id", "kb_id", name="uq_graph_config_workspace_kb"),
        sa.UniqueConstraint("workspace_id", "kb_id", "build_id", name="uq_graph_config_build"),
    )
    op.create_index(
        "ix_knowledge_base_graph_config_workspace_id",
        "knowledge_base_graph_config",
        ["workspace_id"],
    )

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
    op.create_index(
        "ix_graph_chunk_workspace_id", "index_graph_chunk", ["workspace_id"]
    )
    op.create_index("ix_graph_chunk_kb_id", "index_graph_chunk", ["kb_id"])
    op.create_index(
        "ix_graph_chunk_build_status",
        "index_graph_chunk",
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
                "index_graph_chunk.workspace_id",
                "index_graph_chunk.kb_id",
                "index_graph_chunk.build_id",
                "index_graph_chunk.index_chunk_id",
            ],
            name="fk_graph_mention_chunk", ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_graph_entity_mention"),
        sa.UniqueConstraint(
            "workspace_id", "kb_id", "build_id", "index_chunk_id", "mention_id",
            name="uq_graph_mention_identity",
        ),
    )
    op.create_index(
        "ix_graph_entity_mention_workspace_id",
        "graph_entity_mention",
        ["workspace_id"],
    )
    op.create_index(
        "ix_graph_entity_mention_kb_id", "graph_entity_mention", ["kb_id"]
    )
    op.create_index(
        "ix_graph_mention_entity_key",
        "graph_entity_mention",
        ["workspace_id", "kb_id", "build_id", "entity_key"],
    )
    op.create_index(
        "ix_graph_mention_surface_prefix",
        "graph_entity_mention",
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
        sa.CheckConstraint(
            "subject_entity_key <> object_entity_key", name="graph_relation_not_self"
        ),
        sa.CheckConstraint(
            "support_start >= 0 AND support_end > support_start",
            name="graph_relation_support_span_valid",
        ),
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
                "graph_entity_mention.workspace_id",
                "graph_entity_mention.kb_id",
                "graph_entity_mention.build_id",
                "graph_entity_mention.index_chunk_id",
                "graph_entity_mention.mention_id",
            ],
            name="fk_graph_relation_subject_mention", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "build_id", "index_chunk_id", "object_mention_id"],
            [
                "graph_entity_mention.workspace_id",
                "graph_entity_mention.kb_id",
                "graph_entity_mention.build_id",
                "graph_entity_mention.index_chunk_id",
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
    op.create_index(
        "ix_graph_relation_workspace_id", "graph_relation_assertion", ["workspace_id"]
    )
    op.create_index(
        "ix_graph_relation_kb_id", "graph_relation_assertion", ["kb_id"]
    )
    op.create_index(
        "ix_graph_relation_subject",
        "graph_relation_assertion",
        ["workspace_id", "kb_id", "build_id", "subject_entity_key"],
    )
    op.create_index(
        "ix_graph_relation_object",
        "graph_relation_assertion",
        ["workspace_id", "kb_id", "build_id", "object_entity_key"],
    )


def downgrade() -> None:
    for index_name, table_name in (
        ("ix_graph_relation_object", "graph_relation_assertion"),
        ("ix_graph_relation_subject", "graph_relation_assertion"),
        ("ix_graph_relation_kb_id", "graph_relation_assertion"),
        ("ix_graph_relation_workspace_id", "graph_relation_assertion"),
    ):
        op.drop_index(index_name, table_name=table_name)
    op.drop_table("graph_relation_assertion")

    for index_name in (
        "ix_graph_mention_surface_prefix",
        "ix_graph_mention_entity_key",
        "ix_graph_entity_mention_kb_id",
        "ix_graph_entity_mention_workspace_id",
    ):
        op.drop_index(index_name, table_name="graph_entity_mention")
    op.drop_table("graph_entity_mention")

    for index_name in (
        "ix_graph_chunk_build_status",
        "ix_graph_chunk_kb_id",
        "ix_graph_chunk_workspace_id",
    ):
        op.drop_index(index_name, table_name="index_graph_chunk")
    op.drop_table("index_graph_chunk")

    op.drop_index(
        "ix_knowledge_base_graph_config_workspace_id",
        table_name="knowledge_base_graph_config",
    )
    op.drop_table("knowledge_base_graph_config")
    op.drop_constraint(
        "uq_index_chunk_workspace_kb_id", "index_chunk", type_="unique"
    )
