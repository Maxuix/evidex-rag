"""Allow the clarify outcome in ChatRun agent traces.

Revision ID: 0021_clarify_answer_outcome
Revises: 0020_grant_graphiti_work_lease_runtime
Create Date: 2026-08-28
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "0021_clarify_answer_outcome"
down_revision: str | None = "0020_grant_graphiti_work_lease_runtime"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TRACE_CONSTRAINT_PREFIX = (
    "agent_trace IS NULL OR (("
    "jsonb_typeof(agent_trace) = 'object' "
    "AND (agent_trace - ARRAY['version', 'events', 'budget', 'usage', "
    "'outcome']::text[]) = '{}'::jsonb "
    "AND agent_trace->>'version' = 'native_tool_calling_agent_v3' "
    "AND jsonb_typeof(agent_trace->'events') = 'array' "
    "AND jsonb_array_length(agent_trace->'events') <= 32 "
    "AND jsonb_typeof(agent_trace->'budget') = 'object' "
    "AND (agent_trace->'budget') ? 'max_model_rounds' "
    "AND (agent_trace->'budget') ? 'max_graph_calls' "
    "AND ((agent_trace->'budget') - "
    "ARRAY['max_model_rounds', 'max_graph_calls']::text[]) = '{}'::jsonb "
    "AND agent_trace->'budget' = agent_configuration->'budget' "
    "AND jsonb_typeof(agent_trace->'usage') = 'object' "
)
_TRACE_CONSTRAINT_SUFFIX = "AND pg_column_size(agent_trace) <= 65536) IS TRUE)"


def upgrade() -> None:
    op.drop_constraint(op.f("ck_chat_run_agent_trace_v3"), "chat_run")
    op.create_check_constraint(
        op.f("ck_chat_run_agent_trace_v3"),
        "chat_run",
        _TRACE_CONSTRAINT_PREFIX
        + "AND agent_trace->>'outcome' IN ('answered', 'partial', 'refused', 'clarify') "
        + _TRACE_CONSTRAINT_SUFFIX,
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_chat_run_agent_trace_v3"), "chat_run")
    op.create_check_constraint(
        op.f("ck_chat_run_agent_trace_v3"),
        "chat_run",
        _TRACE_CONSTRAINT_PREFIX
        + "AND agent_trace->>'outcome' IN ('answered', 'partial', 'refused') "
        + _TRACE_CONSTRAINT_SUFFIX,
    )
