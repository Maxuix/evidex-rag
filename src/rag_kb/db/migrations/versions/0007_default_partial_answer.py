"""Default existing and new knowledge bases to partial answers.

Revision ID: 0007_default_partial_answer
Revises: 0006_pdf_parsing_progress
Create Date: 2026-08-10
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0007_default_partial_answer"
down_revision: str | None = "0006_pdf_parsing_progress"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_PARTIAL_DEFAULT = sa.text(
    "'{\"answer_style\": \"concise\", "
    "\"insufficiency_policy\": \"partial_answer\"}'::jsonb"
)
_REFUSE_DEFAULT = sa.text(
    "'{\"answer_style\": \"concise\", "
    "\"insufficiency_policy\": \"refuse\"}'::jsonb"
)


def upgrade() -> None:
    op.alter_column(
        "knowledge_base",
        "answer_policy_defaults",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=False,
        server_default=_PARTIAL_DEFAULT,
    )
    op.execute(
        sa.text(
            """
            UPDATE knowledge_base
               SET answer_policy_defaults = jsonb_set(
                       answer_policy_defaults,
                       '{insufficiency_policy}',
                       '"partial_answer"'::jsonb,
                       true
                   ),
                   updated_at = now()
             WHERE answer_policy_defaults ->> 'insufficiency_policy' = 'refuse'
            """
        )
    )


def downgrade() -> None:
    # Row values remain user-managed. Reverting them would overwrite knowledge bases
    # that explicitly selected partial answers before this migration.
    op.alter_column(
        "knowledge_base",
        "answer_policy_defaults",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=False,
        server_default=_REFUSE_DEFAULT,
    )
