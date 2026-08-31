"""Stop enforcing the retired deadline-reserve field in chat snapshots.

Revision ID: 0024_remove_agent_deadline_reserve
Revises: 0023_agent_trace_diagnostics
Create Date: 2026-08-31
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0024_remove_agent_deadline_reserve"
down_revision: str | None = "0023_agent_trace_diagnostics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_CURRENT_AGENT_CONFIGURATION_DEFAULT = (
    "jsonb_build_object("
    "'version', 'native_tool_calling_agent_v3', "
    "'budget', jsonb_build_object("
    "'max_model_rounds', 8, 'max_graph_calls', 2, "
    "'max_total_tokens', 150000, 'max_evidence_items', 64, "
    "'max_retrieval_calls', 16))"
)


def upgrade() -> None:
    # Existing ChatRuns are deliberately left untouched.  The application
    # reader accepts the historical key and ignores it.
    op.drop_constraint(op.f("ck_chat_run_agent_configuration_v3"), "chat_run")
    op.drop_constraint(op.f("ck_chat_run_agent_trace_v3"), "chat_run")
    op.alter_column(
        "chat_run",
        "agent_configuration",
        server_default=sa.text(_CURRENT_AGENT_CONFIGURATION_DEFAULT),
    )


def downgrade() -> None:
    raise RuntimeError(
        "Migration 0024_remove_agent_deadline_reserve does not support "
        "downgrade; restore a matching backup if an older schema is required."
    )
