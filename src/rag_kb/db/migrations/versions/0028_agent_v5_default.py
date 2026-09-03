"""Default new ChatRuns onto native_tool_calling_agent_v5.

Revision ID: 0028_agent_v5_default
Revises: 0027_agent_v4_default
Create Date: 2026-09-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0028_agent_v5_default"
down_revision: str | None = "0027_agent_v4_default"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_V5_AGENT_CONFIGURATION_DEFAULT = (
    "jsonb_build_object("
    "'version', 'native_tool_calling_agent_v5', "
    "'budget', jsonb_build_object("
    "'max_total_tokens', 400000))"
)


def upgrade() -> None:
    # Existing ChatRuns keep their stored configuration; only the column default
    # for new rows changes.
    op.alter_column(
        "chat_run",
        "agent_configuration",
        server_default=sa.text(_V5_AGENT_CONFIGURATION_DEFAULT),
    )


def downgrade() -> None:
    raise RuntimeError(
        "Migration 0028_agent_v5_default does not support "
        "downgrade; restore a matching backup if an older schema is required."
    )
