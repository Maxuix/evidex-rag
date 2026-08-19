"""Remove the unused final model context snapshot.

Revision ID: 0015_remove_retired_chat_state
Revises: 0014_remove_legacy_entity_graph
Create Date: 2026-08-19
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0015_remove_retired_chat_state"
down_revision: str | None = "0014_remove_legacy_entity_graph"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("chat_run", "final_llm_context")


def downgrade() -> None:
    op.add_column(
        "chat_run",
        sa.Column(
            "final_llm_context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
