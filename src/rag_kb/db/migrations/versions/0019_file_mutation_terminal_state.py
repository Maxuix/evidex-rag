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


def _drop_previous_status_constraint() -> None:
    """Drop the pre-0019 status check across legacy naming variants."""

    op.execute(
        """
        DO $$
        DECLARE
            constraint_name text;
        BEGIN
            SELECT conname
              INTO constraint_name
              FROM pg_constraint
             WHERE conrelid = 'content_mutation'::regclass
               AND contype = 'c'
               AND pg_get_constraintdef(oid) LIKE '%status%'
               AND pg_get_constraintdef(oid) LIKE '%pending%'
               AND pg_get_constraintdef(oid) LIKE '%completed%'
               AND pg_get_constraintdef(oid) NOT LIKE '%failed%'
             ORDER BY conname
             LIMIT 1;

            IF constraint_name IS NOT NULL THEN
                EXECUTE format(
                    'ALTER TABLE content_mutation DROP CONSTRAINT %I',
                    constraint_name
                );
            END IF;
        END $$;
        """
    )


def upgrade() -> None:
    # Alembic creates version_num as VARCHAR(32), while this revision ID is
    # longer. Widen the bookkeeping column before Alembic records the head.
    op.alter_column(
        "alembic_version",
        "version_num",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
    )
    op.add_column(
        "content_mutation",
        sa.Column("failure_code", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "content_mutation",
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
    )
    _drop_previous_status_constraint()
    op.create_check_constraint(
        "content_mutation_status_supported",
        "content_mutation",
        "status IN ('pending', 'completed', 'failed')",
    )
    op.create_check_constraint(
        "content_mutation_failure_facts_match_status",
        "content_mutation",
        "(status = 'failed' AND failure_code IS NOT NULL AND failed_at IS NOT NULL) OR "
        "(status <> 'failed' AND failure_code IS NULL AND failed_at IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_content_mutation_content_mutation_failure_facts_match_status"),
        "content_mutation",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_content_mutation_content_mutation_status_supported"),
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
