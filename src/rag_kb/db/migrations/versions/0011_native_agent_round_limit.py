"""Keep only the native Agent model-round loop guard.

Revision ID: 0011_native_agent_round_limit
Revises: 0010_drop_legacy_workflow
Create Date: 2026-08-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0011_native_agent_round_limit"
down_revision: str | None = "0010_drop_legacy_workflow"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_V1_DEFAULT = sa.text(
    "jsonb_build_object("
    "'version', 'native_tool_calling_agent_v1', "
    "'budget', jsonb_build_object("
    "'model_rounds', 8, 'retrieval_calls', 6, "
    "'calculation_calls', 4, 'evidence_refs', 20))"
)
_V2_DEFAULT = sa.text(
    "jsonb_build_object("
    "'version', 'native_tool_calling_agent_v2', "
    "'budget', jsonb_build_object('max_model_rounds', 8))"
)


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1 FROM pg_constraint
             WHERE conrelid = 'public.chat_run'::regclass
               AND conname = 'ck_chat_run_ck_chat_run_agent_trace_v1'
          ) THEN
            ALTER TABLE chat_run
              DROP CONSTRAINT ck_chat_run_ck_chat_run_agent_trace_v1;
          ELSE
            ALTER TABLE chat_run DROP CONSTRAINT ck_chat_run_agent_trace_v1;
          END IF;
        END $$
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1 FROM pg_constraint
             WHERE conrelid = 'public.chat_run'::regclass
               AND conname = 'ck_chat_run_ck_chat_run_agent_configuration_v1'
          ) THEN
            ALTER TABLE chat_run
              DROP CONSTRAINT ck_chat_run_ck_chat_run_agent_configuration_v1;
          ELSE
            ALTER TABLE chat_run DROP CONSTRAINT ck_chat_run_agent_configuration_v1;
          END IF;
        END $$
        """
    )
    op.execute(
        """
        UPDATE chat_run
           SET agent_configuration = jsonb_build_object(
                 'version', 'native_tool_calling_agent_v2',
                 'budget', jsonb_build_object(
                   'max_model_rounds', COALESCE(
                     (agent_configuration->'budget'->>'model_rounds')::integer,
                     8
                   )
                 )
               ),
               agent_trace = CASE
                 WHEN agent_trace IS NULL THEN NULL
                 ELSE jsonb_build_object(
                   'version', 'native_tool_calling_agent_v2',
                   'events', agent_trace->'events',
                   'budget', jsonb_build_object('max_model_rounds', COALESCE(
                     (agent_configuration->'budget'->>'model_rounds')::integer,
                     8
                   )),
                   'usage', agent_trace->'usage',
                   'outcome', agent_trace->'outcome'
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
        op.f("ck_chat_run_agent_configuration_v2"),
        "chat_run",
        "(jsonb_typeof(agent_configuration) = 'object' "
        "AND (agent_configuration - ARRAY['version', 'budget']::text[]) = '{}'::jsonb "
        "AND agent_configuration->>'version' = 'native_tool_calling_agent_v2' "
        "AND jsonb_typeof(agent_configuration->'budget') = 'object' "
        "AND (agent_configuration->'budget') ? 'max_model_rounds' "
        "AND ((agent_configuration->'budget') - 'max_model_rounds'::text) "
        "= '{}'::jsonb "
        "AND CASE WHEN "
        "jsonb_typeof(agent_configuration->'budget'->'max_model_rounds') "
        "= 'number' AND (agent_configuration->'budget'->>'max_model_rounds') "
        "~ '^(0|[1-9][0-9]*)$' THEN "
        "(agent_configuration->'budget'->>'max_model_rounds')::integer "
        "BETWEEN 1 AND 12 ELSE FALSE END "
        "AND pg_column_size(agent_configuration) <= 4096) IS TRUE",
    )
    op.create_check_constraint(
        op.f("ck_chat_run_agent_trace_v2"),
        "chat_run",
        "agent_trace IS NULL OR (("
        "jsonb_typeof(agent_trace) = 'object' "
        "AND (agent_trace - ARRAY['version', 'events', 'budget', 'usage', "
        "'outcome']::text[]) = '{}'::jsonb "
        "AND agent_trace->>'version' = 'native_tool_calling_agent_v2' "
        "AND jsonb_typeof(agent_trace->'events') = 'array' "
        "AND jsonb_array_length(agent_trace->'events') <= 32 "
        "AND jsonb_typeof(agent_trace->'budget') = 'object' "
        "AND (agent_trace->'budget') ? 'max_model_rounds' "
        "AND ((agent_trace->'budget') - 'max_model_rounds'::text) = '{}'::jsonb "
        "AND agent_trace->'budget' = agent_configuration->'budget' "
        "AND jsonb_typeof(agent_trace->'usage') = 'object' "
        "AND agent_trace->>'outcome' IN ('answered', 'partial', 'refused') "
        "AND pg_column_size(agent_trace) <= 65536) IS TRUE)",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_chat_run_agent_trace_v2"), "chat_run", type_="check"
    )
    op.drop_constraint(
        op.f("ck_chat_run_agent_configuration_v2"), "chat_run", type_="check"
    )
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1
              FROM chat_run
             WHERE agent_trace IS NOT NULL AND (
                 COALESCE((agent_trace->'usage'->>'model_rounds')::integer, 0) > 12
                 OR COALESCE((agent_trace->'usage'->>'retrieval_calls')::integer, 0) > 12
                 OR COALESCE((agent_trace->'usage'->>'calculation_calls')::integer, 0) > 4
                 OR COALESCE((agent_trace->'usage'->>'evidence_refs')::integer, 0) > 100
                 OR COALESCE((agent_trace->'usage'->>'model_rounds')::integer, 0) >
                    GREATEST(
                      COALESCE((agent_configuration->'budget'->>'max_model_rounds')::integer, 8),
                      2
                    )
               )
          ) THEN
            RAISE EXCEPTION 'native Agent v2 usage cannot be represented by v1 budgets';
          END IF;
        END $$
        """
    )
    op.execute(
        """
        UPDATE chat_run
           SET agent_configuration = jsonb_build_object(
                 'version', 'native_tool_calling_agent_v1',
                 'budget', jsonb_build_object(
                   'model_rounds', GREATEST(
                     COALESCE((agent_configuration->'budget'->>'max_model_rounds')::integer, 8),
                     COALESCE((agent_trace->'usage'->>'model_rounds')::integer, 0),
                     2
                   ),
                   'retrieval_calls', GREATEST(
                     6,
                     COALESCE((agent_trace->'usage'->>'retrieval_calls')::integer, 0)
                   ),
                   'calculation_calls', 4,
                   'evidence_refs', GREATEST(
                     20,
                     COALESCE((agent_trace->'usage'->>'evidence_refs')::integer, 0)
                   )
                 )
               ),
               agent_trace = CASE
                 WHEN agent_trace IS NULL THEN NULL
                 ELSE jsonb_build_object(
                   'version', 'native_tool_calling_agent_v1',
                   'events', agent_trace->'events',
                   'budget', jsonb_build_object(
                     'model_rounds', GREATEST(
                       COALESCE((agent_configuration->'budget'->>'max_model_rounds')::integer, 8),
                       COALESCE((agent_trace->'usage'->>'model_rounds')::integer, 0),
                       2
                     ),
                     'retrieval_calls', GREATEST(
                       6,
                       COALESCE((agent_trace->'usage'->>'retrieval_calls')::integer, 0)
                     ),
                     'calculation_calls', 4,
                     'evidence_refs', GREATEST(
                       20,
                       COALESCE((agent_trace->'usage'->>'evidence_refs')::integer, 0)
                     )
                   ),
                   'usage', agent_trace->'usage',
                   'outcome', agent_trace->'outcome'
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
