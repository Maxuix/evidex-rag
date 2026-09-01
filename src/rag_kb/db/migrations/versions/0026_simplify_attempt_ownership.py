"""Remove persisted worker names from Chat and indexing attempt ownership.

Revision ID: 0026_simplify_attempt_ownership
Revises: 0025_remove_dynamic_identity
Create Date: 2026-09-01
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0026_simplify_attempt_ownership"
down_revision: str | None = "0025_remove_dynamic_identity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _require_idle_work() -> None:
    facts = op.get_bind().execute(
        sa.text(
            """
            SELECT
              NOT EXISTS (
                SELECT 1 FROM chat_run
                 WHERE status = 'running'
                    OR claimed_at IS NOT NULL
                    OR heartbeat_at IS NOT NULL
              ) AS chat_idle,
              NOT EXISTS (
                SELECT 1 FROM indexing_job
                 WHERE status = 'running'
                    OR claimed_at IS NOT NULL
                    OR heartbeat_at IS NOT NULL
              ) AS indexing_idle,
              NOT EXISTS (
                SELECT 1 FROM graphiti_graph_work_lease
              ) AS graph_idle
            """
        )
    ).mappings().one()
    active = [
        name.removesuffix("_idle")
        for name, idle in facts.items()
        if not idle
    ]
    if active:
        raise RuntimeError(
            "attempt ownership migration requires idle work: " + ", ".join(active)
        )


def upgrade() -> None:
    _require_idle_work()
    op.drop_column("chat_run", "claimed_by")
    op.drop_column("indexing_job", "claimed_by")


def downgrade() -> None:
    _require_idle_work()
    op.add_column("chat_run", sa.Column("claimed_by", sa.String(255)))
    op.add_column("indexing_job", sa.Column("claimed_by", sa.String(255)))
