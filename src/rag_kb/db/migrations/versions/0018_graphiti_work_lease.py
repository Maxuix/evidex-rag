"""Serialize external Graphiti work per immutable build.

Revision ID: 0018_graphiti_work_lease
Revises: 0017_graph_schema_profiles
Create Date: 2026-08-24
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0018_graphiti_work_lease"
down_revision: str | None = "0017_graph_schema_profiles"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "graphiti_graph_work_lease",
        sa.Column("build_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("lease_token", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claimed_by", sa.String(length=255), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("work_kind", sa.String(length=32), nullable=False),
        sa.Column("index_chunk_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.PrimaryKeyConstraint("build_id", name="pk_graphiti_graph_work_lease"),
        sa.UniqueConstraint("lease_token", name="uq_graphiti_work_lease_token"),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "build_id"],
            [
                "graphiti_graph_build.workspace_id",
                "graphiti_graph_build.kb_id",
                "graphiti_graph_build.build_id",
            ],
            name="fk_graphiti_work_lease_build",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "index_chunk_id"],
            ["index_chunk.workspace_id", "index_chunk.kb_id", "index_chunk.id"],
            name="fk_graphiti_work_lease_chunk",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "work_kind IN ('preflight','chunk','finalize')",
            name="graphiti_work_lease_kind_supported",
        ),
    )
    op.create_index(
        "ix_graphiti_work_lease_expiry",
        "graphiti_graph_work_lease",
        ["lease_expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_graphiti_work_lease_expiry",
        table_name="graphiti_graph_work_lease",
    )
    op.drop_table("graphiti_graph_work_lease")
