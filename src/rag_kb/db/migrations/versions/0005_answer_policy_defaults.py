"""Persist knowledge-base answer policy defaults.

Revision ID: 0005_answer_policy_defaults
Revises: 0004_chat_request_hash
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0005_answer_policy_defaults"
down_revision: str | None = "0004_chat_request_hash"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_DEFAULTS = sa.text(
    "'{\"answer_style\": \"concise\", "
    "\"insufficiency_policy\": \"refuse\"}'::jsonb"
)


def upgrade() -> None:
    op.add_column(
        "knowledge_base",
        sa.Column(
            "answer_policy_defaults",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=_DEFAULTS,
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("knowledge_base", "answer_policy_defaults")
