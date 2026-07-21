"""Persist immutable semantic chunk breakpoint plans.

Revision ID: 0006_semantic_chunk_plan
Revises: 0005_answer_policy_defaults
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0006_semantic_chunk_plan"
down_revision: str | None = "0005_answer_policy_defaults"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "index_chunk_plan",
        sa.Column(
            "indexed_document_version_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("source_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("profile_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("unit_sequence_hash", sa.String(length=64), nullable=False),
        sa.Column("unit_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column(
            "boundaries",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("plan_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(boundaries) = 'array'",
            name=op.f("ck_index_chunk_plan_index_chunk_plan_boundaries_array"),
        ),
        sa.CheckConstraint(
            "jsonb_array_length(boundaries) = chunk_count - 1",
            name=op.f("ck_index_chunk_plan_index_chunk_plan_boundary_count"),
        ),
        sa.CheckConstraint(
            "chunk_count > 0",
            name=op.f("ck_index_chunk_plan_index_chunk_plan_chunk_count_positive"),
        ),
        sa.CheckConstraint(
            "unit_count > 0",
            name=op.f("ck_index_chunk_plan_index_chunk_plan_unit_count_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["indexed_document_version_id"],
            ["indexed_document_version.id"],
            name=op.f(
                "fk_index_chunk_plan_indexed_document_version_id_indexed_document_version"
            ),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "indexed_document_version_id",
            name=op.f("pk_index_chunk_plan"),
        ),
    )
    op.execute(
        "GRANT SELECT, INSERT, DELETE ON TABLE index_chunk_plan "
        "TO rag_kb_runtime"
    )
    op.execute("REVOKE UPDATE ON TABLE index_chunk_plan FROM rag_kb_runtime")


def downgrade() -> None:
    op.drop_table("index_chunk_plan")
