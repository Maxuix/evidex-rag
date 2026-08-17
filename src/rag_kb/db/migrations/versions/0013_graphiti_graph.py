"""Add immutable Graphiti build generations and episode mappings.

Revision ID: 0013_graphiti_graph
Revises: 0012_entity_graph_rag
Create Date: 2026-08-17
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0013_graphiti_graph"
down_revision: str | None = "0012_entity_graph_rag"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_base_graph_config",
        sa.Column("active_build_id", sa.UUID(), nullable=True),
    )
    op.create_table(
        "graphiti_graph_build",
        sa.Column("build_id", sa.UUID(), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("kb_id", sa.UUID(), nullable=False),
        sa.Column("group_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("index_revision_id", sa.UUID(), nullable=False),
        sa.Column("serving_chunk_digest", sa.String(length=64), nullable=False),
        sa.Column("expected_episode_count", sa.Integer(), nullable=False),
        sa.Column("chat_profile_revision_id", sa.UUID(), nullable=False),
        sa.Column("embedding_profile_revision_id", sa.UUID(), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("extractor_version", sa.String(length=128), nullable=False),
        sa.Column("superseded_by", sa.UUID(), nullable=True),
        sa.Column("last_error_code", sa.String(length=128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('building','ready','failed','superseded')",
            name="graphiti_build_status_supported",
        ),
        sa.CheckConstraint(
            "expected_episode_count >= 0 AND embedding_dimension BETWEEN 64 AND 4096",
            name="graphiti_build_counts_valid",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id"],
            ["knowledge_base.workspace_id", "knowledge_base.id"],
            name="fk_graphiti_build_same_workspace_kb",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "chat_profile_revision_id"],
            ["model_profile_revision.workspace_id", "model_profile_revision.id"],
            name="fk_graphiti_build_same_workspace_chat_profile",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "embedding_profile_revision_id"],
            ["model_profile_revision.workspace_id", "model_profile_revision.id"],
            name="fk_graphiti_build_same_workspace_embedding_profile",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("build_id", name="pk_graphiti_graph_build"),
        sa.UniqueConstraint("workspace_id", "kb_id", "build_id", name="uq_graphiti_build_scope"),
        sa.UniqueConstraint("group_id", name="uq_graphiti_build_group"),
    )
    op.create_index(
        "ix_graphiti_build_work",
        "graphiti_graph_build",
        ["workspace_id", "status", "started_at"],
    )
    op.create_foreign_key(
        "fk_graph_config_active_graphiti_build",
        "knowledge_base_graph_config",
        "graphiti_graph_build",
        ["workspace_id", "kb_id", "active_build_id"],
        ["workspace_id", "kb_id", "build_id"],
        ondelete="RESTRICT",
    )
    op.create_table(
        "graphiti_episode_chunk",
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("kb_id", sa.UUID(), nullable=False),
        sa.Column("build_id", sa.UUID(), nullable=False),
        sa.Column("group_id", sa.String(length=255), nullable=False),
        sa.Column("episode_uuid", sa.String(length=64), nullable=False),
        sa.Column("index_revision_id", sa.UUID(), nullable=False),
        sa.Column("index_chunk_id", sa.UUID(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), server_default=sa.text("'completed'"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("status IN ('completed')", name="graphiti_episode_status_supported"),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "build_id"],
            ["graphiti_graph_build.workspace_id", "graphiti_graph_build.kb_id", "graphiti_graph_build.build_id"],
            name="fk_graphiti_episode_build",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "index_chunk_id"],
            ["index_chunk.workspace_id", "index_chunk.kb_id", "index_chunk.id"],
            name="fk_graphiti_episode_chunk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_graphiti_episode_chunk"),
        sa.UniqueConstraint("workspace_id", "kb_id", "build_id", "index_chunk_id", name="uq_graphiti_episode_build_chunk"),
        sa.UniqueConstraint("group_id", "episode_uuid", name="uq_graphiti_episode_group_uuid"),
    )
    op.create_index(
        "ix_graphiti_episode_lookup",
        "graphiti_episode_chunk",
        ["workspace_id", "kb_id", "build_id", "episode_uuid"],
    )


def downgrade() -> None:
    op.drop_index("ix_graphiti_episode_lookup", table_name="graphiti_episode_chunk")
    op.drop_table("graphiti_episode_chunk")
    op.drop_constraint(
        "fk_graph_config_active_graphiti_build",
        "knowledge_base_graph_config",
        type_="foreignkey",
    )
    op.drop_index("ix_graphiti_build_work", table_name="graphiti_graph_build")
    op.drop_table("graphiti_graph_build")
    op.drop_column("knowledge_base_graph_config", "active_build_id")
