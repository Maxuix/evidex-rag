"""Add durable PDF parsing progress and continuation facts.

Revision ID: 0006_pdf_parsing_progress
Revises: 0005_content_management
Create Date: 2026-08-09
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0006_pdf_parsing_progress"
down_revision: str | None = "0005_content_management"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "indexing_job",
        sa.Column(
            "progress",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "indexing_job",
        sa.Column(
            "continuation_pending",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    op.add_column(
        "indexing_job",
        sa.Column(
            "continuation_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        op.f("ck_indexing_job_indexing_job_continuation_count_nonnegative"),
        "indexing_job",
        "continuation_count >= 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_indexing_job_indexing_job_continuation_count_nonnegative"),
        "indexing_job",
        type_="check",
    )
    op.drop_column("indexing_job", "continuation_count")
    op.drop_column("indexing_job", "continuation_pending")
    op.drop_column("indexing_job", "progress")
