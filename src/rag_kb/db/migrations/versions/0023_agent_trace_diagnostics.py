"""Allow backwards-compatible agent trace diagnostics.

Revision ID: 0023_agent_trace_diagnostics
Revises: 0022_agent_resource_budget
Create Date: 2026-08-30
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "0023_agent_trace_diagnostics"
down_revision: str | None = "0022_agent_resource_budget"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_TRACE_CONSTRAINT = (
    "agent_trace IS NULL OR ((jsonb_typeof(agent_trace) = 'object' "
    "AND (agent_trace - ARRAY['version', 'events', 'budget', 'usage', "
    "'diagnostics', 'outcome']::text[]) = '{}'::jsonb "
    "AND agent_trace->>'version' = 'native_tool_calling_agent_v3' "
    "AND jsonb_typeof(agent_trace->'events') = 'array' "
    "AND jsonb_array_length(agent_trace->'events') <= 32 "
    "AND jsonb_typeof(agent_trace->'budget') = 'object' "
    "AND (agent_trace->'budget') ? 'max_model_rounds' "
    "AND (agent_trace->'budget') ? 'max_graph_calls' "
    "AND (agent_trace->'budget') ? 'max_total_tokens' "
    "AND (agent_trace->'budget') ? 'max_evidence_items' "
    "AND (agent_trace->'budget') ? 'max_retrieval_calls' "
    "AND (agent_trace->'budget') ? 'soft_deadline_reserve_seconds' "
    "AND ((agent_trace->'budget') - "
    "ARRAY['max_model_rounds', 'max_graph_calls', 'max_total_tokens', "
    "'max_evidence_items', 'max_retrieval_calls', "
    "'soft_deadline_reserve_seconds']::text[]) = '{}'::jsonb "
    "AND agent_trace->'budget' = agent_configuration->'budget' "
    "AND jsonb_typeof(agent_trace->'usage') = 'object' "
    "AND (NOT agent_trace ? 'diagnostics' OR "
    "jsonb_typeof(agent_trace->'diagnostics') = 'object') "
    "AND agent_trace->>'outcome' IN "
    "('answered', 'partial', 'refused', 'clarify') "
    "AND pg_column_size(agent_trace) <= 65536) IS TRUE)"
)

_PREVIOUS_TRACE_CONSTRAINT = _TRACE_CONSTRAINT.replace(
    "'diagnostics', ", ""
).replace(
    "AND (NOT agent_trace ? 'diagnostics' OR "
    "jsonb_typeof(agent_trace->'diagnostics') = 'object') ",
    "",
)


def upgrade() -> None:
    op.drop_constraint(op.f("ck_chat_run_agent_trace_v3"), "chat_run")
    op.create_check_constraint(
        op.f("ck_chat_run_agent_trace_v3"),
        "chat_run",
        _TRACE_CONSTRAINT,
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_chat_run_agent_trace_v3"), "chat_run")
    op.execute(
        "UPDATE chat_run SET agent_trace = agent_trace - 'diagnostics' "
        "WHERE agent_trace ? 'diagnostics'"
    )
    op.create_check_constraint(
        op.f("ck_chat_run_agent_trace_v3"),
        "chat_run",
        _PREVIOUS_TRACE_CONSTRAINT,
    )
