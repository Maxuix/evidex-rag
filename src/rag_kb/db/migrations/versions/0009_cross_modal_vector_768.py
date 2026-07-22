"""Add the fixed 768-dimensional cross-modal vector table.

Revision ID: 0009_cross_modal_vector_768
Revises: 0008_multimodal_index_units
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision: str = "0009_cross_modal_vector_768"
down_revision: str | None = "0008_multimodal_index_units"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vector_record_768",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuidv7()"),
            nullable=False,
        ),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("index_chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "embedding_space_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column("representation_kind", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(dim=768), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["kb_id", "index_chunk_id"],
            ["index_chunk.kb_id", "index_chunk.id"],
            name="fk_vector_record_768_same_kb_chunk",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "embedding_space_id"],
            ["embedding_space.workspace_id", "embedding_space.id"],
            name="fk_vector_record_768_same_workspace_embedding",
        ),
        sa.ForeignKeyConstraint(
            ["kb_id"],
            ["knowledge_base.id"],
            name=op.f("fk_vector_record_768_kb_id_knowledge_base"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspace.id"],
            name=op.f("fk_vector_record_768_workspace_id_workspace"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_vector_record_768")),
        sa.UniqueConstraint(
            "index_chunk_id",
            "embedding_space_id",
            "representation_kind",
            name="uq_vector_record_768_chunk_space_representation",
        ),
    )
    op.create_index(
        op.f("ix_vector_record_768_workspace_id"),
        "vector_record_768",
        ["workspace_id"],
    )
    op.create_index(
        op.f("ix_vector_record_768_kb_id"),
        "vector_record_768",
        ["kb_id"],
    )
    op.create_index(
        op.f("ix_vector_record_768_embedding_space_id"),
        "vector_record_768",
        ["embedding_space_id"],
    )
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE vector_record_768 "
        "TO rag_kb_runtime"
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_vector_record_768_embedding_space_id"),
        table_name="vector_record_768",
    )
    op.drop_index(
        op.f("ix_vector_record_768_kb_id"), table_name="vector_record_768"
    )
    op.drop_index(
        op.f("ix_vector_record_768_workspace_id"),
        table_name="vector_record_768",
    )
    op.drop_table("vector_record_768")
