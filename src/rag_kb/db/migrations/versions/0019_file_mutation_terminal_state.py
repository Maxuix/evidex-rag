"""Persist terminal outcomes for source-file mutations.

Revision ID: 0019_file_mutation_terminal_state
Revises: 0018_graphiti_work_lease
Create Date: 2026-08-24
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0019_file_mutation_terminal_state"
down_revision: str | None = "0018_graphiti_work_lease"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "content_mutation",
        sa.Column("failure_code", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "content_mutation",
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.drop_constraint(
        "ck_content_mutation_content_mutation_status_supported",
        "content_mutation",
        type_="check",
    )
    op.create_check_constraint(
        "ck_content_mutation_content_mutation_status_supported",
        "content_mutation",
        "status IN ('pending', 'completed', 'failed')",
    )
    op.create_check_constraint(
        "ck_content_mutation_content_mutation_failure_facts_match_status",
        "content_mutation",
        "(status = 'failed' AND failure_code IS NOT NULL AND failed_at IS NOT NULL) OR "
        "(status <> 'failed' AND failure_code IS NULL AND failed_at IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_content_mutation_content_mutation_failure_facts_match_status",
        "content_mutation",
        type_="check",
    )
    op.drop_constraint(
        "ck_content_mutation_content_mutation_status_supported",
        "content_mutation",
        type_="check",
    )
    op.create_check_constraint(
        "ck_content_mutation_content_mutation_status_supported",
        "content_mutation",
        "status IN ('pending', 'completed')",
    )
    op.drop_column("content_mutation", "failed_at")
    op.drop_column("content_mutation", "failure_code")
