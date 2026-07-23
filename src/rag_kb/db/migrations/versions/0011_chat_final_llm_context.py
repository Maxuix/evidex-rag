"""Persist the final model-input context for completed ChatRuns.

Revision ID: 0011_chat_final_llm_context
Revises: 0010_composite_evidence_v2
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0011_chat_final_llm_context"
down_revision: str | None = "0010_composite_evidence_v2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_run",
        sa.Column(
            "final_llm_context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("chat_run", "final_llm_context")
