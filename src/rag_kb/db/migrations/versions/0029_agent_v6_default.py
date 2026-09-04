"""Default new ChatRuns onto native_tool_calling_agent_v6.

Revision ID: 0029_agent_v6_default
Revises: 0028_agent_v5_default
Create Date: 2026-09-04
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0029_agent_v6_default"
down_revision: str | None = "0028_agent_v5_default"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_V6_AGENT_CONFIGURATION_DEFAULT = (
    "jsonb_build_object("
    "'version', 'native_tool_calling_agent_v6', "
    "'budget', jsonb_build_object("
    "'max_total_tokens', 400000))"
)


def upgrade() -> None:
    # Existing ChatRuns keep their stored configuration; only new rows use v6.
    op.alter_column(
        "chat_run",
        "agent_configuration",
        server_default=sa.text(_V6_AGENT_CONFIGURATION_DEFAULT),
    )


def downgrade() -> None:
    raise RuntimeError(
        "Migration 0029_agent_v6_default does not support "
        "downgrade; restore a matching backup if an older schema is required."
    )
