"""Add cumulative resource budgets to the ChatRun agent configuration.

Revision ID: 0022_agent_resource_budget
Revises: 0021_clarify_answer_outcome
Create Date: 2026-08-28
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0022_agent_resource_budget"
down_revision: str | None = "0021_clarify_answer_outcome"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_V2_DEFAULT = sa.text(
    "jsonb_build_object("
    "'version', 'native_tool_calling_agent_v3', "
    "'budget', jsonb_build_object("
    "'max_model_rounds', 8, 'max_graph_calls', 2, "
    "'max_total_tokens', 150000, 'max_evidence_items', 64, "
    "'max_retrieval_calls', 16, "
    "'soft_deadline_reserve_seconds', 60))"
)
_V1_DEFAULT = sa.text(
    "jsonb_build_object("
    "'version', 'native_tool_calling_agent_v3', "
    "'budget', jsonb_build_object("
    "'max_model_rounds', 8, 'max_graph_calls', 2))"
)

_NEW_BUDGET_DEFAULTS = (
    "jsonb_build_object("
    "'max_total_tokens', 150000, 'max_evidence_items', 64, "
    "'max_retrieval_calls', 16, 'soft_deadline_reserve_seconds', 60)"
)
_NEW_BUDGET_KEYS = (
    "ARRAY['max_total_tokens', 'max_evidence_items', "
    "'max_retrieval_calls', 'soft_deadline_reserve_seconds']::text[]"
)

_CONFIGURATION_CONSTRAINT_V2 = (
    "(jsonb_typeof(agent_configuration) = 'object' "
    "AND (agent_configuration - ARRAY['version', 'budget']::text[]) = '{}'::jsonb "
    "AND agent_configuration->>'version' = 'native_tool_calling_agent_v3' "
    "AND jsonb_typeof(agent_configuration->'budget') = 'object' "
    "AND (agent_configuration->'budget') ? 'max_model_rounds' "
    "AND (agent_configuration->'budget') ? 'max_graph_calls' "
    "AND (agent_configuration->'budget') ? 'max_total_tokens' "
    "AND (agent_configuration->'budget') ? 'max_evidence_items' "
    "AND (agent_configuration->'budget') ? 'max_retrieval_calls' "
    "AND (agent_configuration->'budget') ? 'soft_deadline_reserve_seconds' "
    "AND ((agent_configuration->'budget') - "
    "ARRAY['max_model_rounds', 'max_graph_calls', 'max_total_tokens', "
    "'max_evidence_items', 'max_retrieval_calls', "
    "'soft_deadline_reserve_seconds']::text[]) = '{}'::jsonb "
    "AND CASE WHEN "
    "jsonb_typeof(agent_configuration->'budget'->'max_model_rounds') "
    "= 'number' AND (agent_configuration->'budget'->>'max_model_rounds') "
    "~ '^(0|[1-9][0-9]*)$' THEN "
    "(agent_configuration->'budget'->>'max_model_rounds')::integer "
    "BETWEEN 1 AND 12 ELSE FALSE END "
    "AND CASE WHEN "
    "jsonb_typeof(agent_configuration->'budget'->'max_graph_calls') "
    "= 'number' AND (agent_configuration->'budget'->>'max_graph_calls') "
    "~ '^(0|[1-9][0-9]*)$' THEN "
    "(agent_configuration->'budget'->>'max_graph_calls')::integer "
    "BETWEEN 1 AND 2 ELSE FALSE END "
    "AND CASE WHEN "
    "jsonb_typeof(agent_configuration->'budget'->'max_total_tokens') "
    "= 'number' AND (agent_configuration->'budget'->>'max_total_tokens') "
    "~ '^(0|[1-9][0-9]*)$' THEN "
    "(agent_configuration->'budget'->>'max_total_tokens')::integer "
    "BETWEEN 1000 AND 10000000 ELSE FALSE END "
    "AND CASE WHEN "
    "jsonb_typeof(agent_configuration->'budget'->'max_evidence_items') "
    "= 'number' AND (agent_configuration->'budget'->>'max_evidence_items') "
    "~ '^(0|[1-9][0-9]*)$' THEN "
    "(agent_configuration->'budget'->>'max_evidence_items')::integer "
    "BETWEEN 1 AND 512 ELSE FALSE END "
    "AND CASE WHEN "
    "jsonb_typeof(agent_configuration->'budget'->'max_retrieval_calls') "
    "= 'number' AND (agent_configuration->'budget'->>'max_retrieval_calls') "
    "~ '^(0|[1-9][0-9]*)$' THEN "
    "(agent_configuration->'budget'->>'max_retrieval_calls')::integer "
    "BETWEEN 1 AND 64 ELSE FALSE END "
    "AND CASE WHEN "
    "jsonb_typeof(agent_configuration->'budget'->'soft_deadline_reserve_seconds') "
    "= 'number' THEN "
    "(agent_configuration->'budget'->>'soft_deadline_reserve_seconds')::float "
    "BETWEEN 0 AND 600 ELSE FALSE END "
    "AND pg_column_size(agent_configuration) <= 4096) IS TRUE"
)

_TRACE_CONSTRAINT_TEMPLATE = (
    "agent_trace IS NULL OR (("
    "jsonb_typeof(agent_trace) = 'object' "
    "AND (agent_trace - ARRAY['version', 'events', 'budget', 'usage', "
    "'outcome']::text[]) = '{{}}'::jsonb "
    "AND agent_trace->>'version' = 'native_tool_calling_agent_v3' "
    "AND jsonb_typeof(agent_trace->'events') = 'array' "
    "AND jsonb_array_length(agent_trace->'events') <= 32 "
    "AND jsonb_typeof(agent_trace->'budget') = 'object' "
    "AND (agent_trace->'budget') ? 'max_model_rounds' "
    "AND (agent_trace->'budget') ? 'max_graph_calls' "
    "{budget_keys}"
    "AND ((agent_trace->'budget') - "
    "ARRAY[{budget_key_list}]::text[]) = '{{}}'::jsonb "
    "AND agent_trace->'budget' = agent_configuration->'budget' "
    "AND jsonb_typeof(agent_trace->'usage') = 'object' "
    "AND agent_trace->>'outcome' IN "
    "('answered', 'partial', 'refused', 'clarify') "
    "AND pg_column_size(agent_trace) <= 65536) IS TRUE)"
)


def _trace_constraint(*, extended: bool) -> str:
    if extended:
        keys = (
            "AND (agent_trace->'budget') ? 'max_total_tokens' "
            "AND (agent_trace->'budget') ? 'max_evidence_items' "
            "AND (agent_trace->'budget') ? 'max_retrieval_calls' "
            "AND (agent_trace->'budget') ? 'soft_deadline_reserve_seconds' "
        )
        key_list = (
            "'max_model_rounds', 'max_graph_calls', 'max_total_tokens', "
            "'max_evidence_items', 'max_retrieval_calls', "
            "'soft_deadline_reserve_seconds'"
        )
    else:
        keys = ""
        key_list = "'max_model_rounds', 'max_graph_calls'"
    return _TRACE_CONSTRAINT_TEMPLATE.format(
        budget_keys=keys, budget_key_list=key_list
    )


_CONFIGURATION_CONSTRAINT_V1 = (
    "(jsonb_typeof(agent_configuration) = 'object' "
    "AND (agent_configuration - ARRAY['version', 'budget']::text[]) = '{}'::jsonb "
    "AND agent_configuration->>'version' = 'native_tool_calling_agent_v3' "
    "AND jsonb_typeof(agent_configuration->'budget') = 'object' "
    "AND (agent_configuration->'budget') ? 'max_model_rounds' "
    "AND (agent_configuration->'budget') ? 'max_graph_calls' "
    "AND ((agent_configuration->'budget') - "
    "ARRAY['max_model_rounds', 'max_graph_calls']::text[]) = '{}'::jsonb "
    "AND CASE WHEN "
    "jsonb_typeof(agent_configuration->'budget'->'max_model_rounds') "
    "= 'number' AND (agent_configuration->'budget'->>'max_model_rounds') "
    "~ '^(0|[1-9][0-9]*)$' THEN "
    "(agent_configuration->'budget'->>'max_model_rounds')::integer "
    "BETWEEN 1 AND 12 ELSE FALSE END "
    "AND CASE WHEN "
    "jsonb_typeof(agent_configuration->'budget'->'max_graph_calls') "
    "= 'number' AND (agent_configuration->'budget'->>'max_graph_calls') "
    "~ '^(0|[1-9][0-9]*)$' THEN "
    "(agent_configuration->'budget'->>'max_graph_calls')::integer "
    "BETWEEN 1 AND 2 ELSE FALSE END "
    "AND pg_column_size(agent_configuration) <= 4096) IS TRUE"
)


def upgrade() -> None:
    op.drop_constraint(
        op.f("ck_chat_run_agent_configuration_v3"), "chat_run"
    )
    op.drop_constraint(op.f("ck_chat_run_agent_trace_v3"), "chat_run")
    op.execute(
        """
        DO $$
        DECLARE
          run_row RECORD;
        BEGIN
          FOR run_row IN
            SELECT id, agent_configuration, agent_trace
              FROM chat_run
          LOOP
            IF (
              jsonb_typeof(run_row.agent_configuration) <> 'object'
              OR run_row.agent_configuration->>'version'
                 <> 'native_tool_calling_agent_v3'
              OR (run_row.agent_configuration - ARRAY['version', 'budget']::text[])
                 <> '{}'::jsonb
              OR jsonb_typeof(run_row.agent_configuration->'budget') <> 'object'
              OR ((run_row.agent_configuration->'budget')
                  - ARRAY['max_model_rounds', 'max_graph_calls']::text[])
                 <> '{}'::jsonb
              OR NOT (run_row.agent_configuration->'budget') ? 'max_model_rounds'
              OR NOT (run_row.agent_configuration->'budget') ? 'max_graph_calls'
            ) THEN
              RAISE EXCEPTION 'unknown agent configuration shape in chat_run %',
                run_row.id;
            END IF;
            IF run_row.agent_trace IS NOT NULL AND (
              run_row.agent_trace->'budget'
              <> run_row.agent_configuration->'budget'
            ) THEN
              RAISE EXCEPTION 'agent trace budget mismatch in chat_run %',
                run_row.id;
            END IF;
          END LOOP;
        END $$
        """
    )
    op.execute(
        f"""
        UPDATE chat_run
           SET agent_configuration = jsonb_build_object(
                 'version', 'native_tool_calling_agent_v3',
                 'budget',
                 (agent_configuration->'budget') || {_NEW_BUDGET_DEFAULTS}
               ),
               agent_trace = CASE
                 WHEN agent_trace IS NULL THEN NULL
                 ELSE jsonb_set(
                   agent_trace,
                   '{{budget}}',
                   (agent_configuration->'budget') || {_NEW_BUDGET_DEFAULTS}
                 )
               END
        """
    )
    op.alter_column(
        "chat_run",
        "agent_configuration",
        server_default=_V2_DEFAULT,
    )
    op.create_check_constraint(
        op.f("ck_chat_run_agent_configuration_v3"),
        "chat_run",
        _CONFIGURATION_CONSTRAINT_V2,
    )
    op.create_check_constraint(
        op.f("ck_chat_run_agent_trace_v3"),
        "chat_run",
        _trace_constraint(extended=True),
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_chat_run_agent_trace_v3"), "chat_run")
    op.drop_constraint(
        op.f("ck_chat_run_agent_configuration_v3"), "chat_run"
    )
    op.execute(
        f"""
        UPDATE chat_run
           SET agent_configuration = jsonb_build_object(
                 'version', 'native_tool_calling_agent_v3',
                 'budget',
                 (agent_configuration->'budget') - {_NEW_BUDGET_KEYS}
               ),
               agent_trace = CASE
                 WHEN agent_trace IS NULL THEN NULL
                 ELSE jsonb_set(
                   agent_trace,
                   '{{budget}}',
                   (agent_trace->'budget') - {_NEW_BUDGET_KEYS}
                 )
               END
        """
    )
    op.alter_column(
        "chat_run",
        "agent_configuration",
        server_default=_V1_DEFAULT,
    )
    op.create_check_constraint(
        op.f("ck_chat_run_agent_configuration_v3"),
        "chat_run",
        _CONFIGURATION_CONSTRAINT_V1,
    )
    op.create_check_constraint(
        op.f("ck_chat_run_agent_trace_v3"),
        "chat_run",
        _trace_constraint(extended=False),
    )
