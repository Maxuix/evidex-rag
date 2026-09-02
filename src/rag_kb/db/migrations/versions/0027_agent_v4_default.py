"""Default new ChatRuns onto native_tool_calling_agent_v4.

Revision ID: 0027_agent_v4_default
Revises: 0026_simplify_attempt_ownership
Create Date: 2026-09-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0027_agent_v4_default"
down_revision: str | None = "0026_simplify_attempt_ownership"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_V4_AGENT_CONFIGURATION_DEFAULT = (
    "jsonb_build_object("
    "'version', 'native_tool_calling_agent_v4', "
    "'budget', jsonb_build_object("
    "'max_model_rounds', 8, 'max_graph_calls', 2, "
    "'max_total_tokens', 150000, 'max_evidence_items', 64, "
    "'max_retrieval_calls', 16))"
)


def upgrade() -> None:
    # Existing ChatRuns keep their stored configuration; only the column default
    # for new rows changes.
    op.alter_column(
        "chat_run",
        "agent_configuration",
        server_default=sa.text(_V4_AGENT_CONFIGURATION_DEFAULT),
    )


def downgrade() -> None:
    raise RuntimeError(
        "Migration 0027_agent_v4_default does not support "
        "downgrade; restore a matching backup if an older schema is required."
    )
