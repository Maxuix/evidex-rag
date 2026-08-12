"""Add frozen native-agent budget and bounded terminal trace.

Revision ID: 0009_native_tool_calling_agent
Revises: 0008_local_rerank_mode
Create Date: 2026-08-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0009_native_tool_calling_agent"
down_revision: str | None = "0008_local_rerank_mode"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_CONFIG_DEFAULT = sa.text(
    "jsonb_build_object("
    "'version', 'native_tool_calling_agent_v1', "
    "'budget', jsonb_build_object("
    "'model_rounds', 8, 'retrieval_calls', 6, "
    "'calculation_calls', 4, 'evidence_refs', 20))"
)


def upgrade() -> None:
    op.add_column(
        "chat_run",
        sa.Column(
            "agent_configuration",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=_CONFIG_DEFAULT,
        ),
    )
    op.add_column(
        "chat_run",
        sa.Column(
            "agent_trace",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.create_check_constraint(
        "ck_chat_run_agent_configuration_v1",
        "chat_run",
        "jsonb_typeof(agent_configuration) = 'object' "
        "AND agent_configuration->>'version' = 'native_tool_calling_agent_v1' "
        "AND jsonb_typeof(agent_configuration->'budget') = 'object' "
        "AND pg_column_size(agent_configuration) <= 4096",
    )
    op.create_check_constraint(
        "ck_chat_run_agent_trace_v1",
        "chat_run",
        "agent_trace IS NULL OR ("
        "jsonb_typeof(agent_trace) = 'object' "
        "AND agent_trace->>'version' = 'native_tool_calling_agent_v1' "
        "AND jsonb_typeof(agent_trace->'events') = 'array' "
        "AND jsonb_array_length(agent_trace->'events') <= 32 "
        "AND pg_column_size(agent_trace) <= 65536)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_chat_run_agent_trace_v1", "chat_run", type_="check")
    op.drop_constraint("ck_chat_run_agent_configuration_v1", "chat_run", type_="check")
    op.drop_column("chat_run", "agent_trace")
    op.drop_column("chat_run", "agent_configuration")
