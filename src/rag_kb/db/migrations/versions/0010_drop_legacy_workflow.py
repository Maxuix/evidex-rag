"""Drop retired workflow diagnostics from ChatRun.

Revision ID: 0010_drop_legacy_workflow
Revises: 0009_native_tool_calling_agent
Create Date: 2026-08-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0010_drop_legacy_workflow"
down_revision: str | None = "0009_native_tool_calling_agent"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_CONFIG_DEFAULT = sa.text(
    "jsonb_build_object("
    "'version', 'chat_workflow_v1', "
    "'requested_mode', 'simple', "
    "'budget', jsonb_build_object("
    "'decision_rounds', 4, 'retrieval_calls', 6, "
    "'parallel_queries', 3, 'verifier_continuations', 1, "
    "'no_progress_rounds', 1))"
)
_STATE_DEFAULT = sa.text(
    "jsonb_build_object("
    "'version', 'chat_workflow_v1', "
    "'resolved_mode', 'simple', "
    "'route_status', 'not_applicable', "
    "'route_reason_codes', '[]'::jsonb, "
    "'research_result', 'null'::jsonb, "
    "'search_trace', 'null'::jsonb)"
)


def upgrade() -> None:
    op.drop_constraint(
        "ck_chat_run_workflow_state_v1", "chat_run", type_="check"
    )
    op.drop_constraint(
        "ck_chat_run_workflow_configuration_v1", "chat_run", type_="check"
    )
    op.drop_column("chat_run", "workflow_state")
    op.drop_column("chat_run", "workflow_configuration")


def downgrade() -> None:
    op.add_column(
        "chat_run",
        sa.Column(
            "workflow_configuration",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=_CONFIG_DEFAULT,
        ),
    )
    op.add_column(
        "chat_run",
        sa.Column(
            "workflow_state",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=_STATE_DEFAULT,
        ),
    )
    op.create_check_constraint(
        "ck_chat_run_workflow_configuration_v1",
        "chat_run",
        "jsonb_typeof(workflow_configuration) = 'object' "
        "AND workflow_configuration->>'version' = 'chat_workflow_v1' "
        "AND workflow_configuration->>'requested_mode' IN ('simple','agent','auto') "
        "AND jsonb_typeof(workflow_configuration->'budget') = 'object' "
        "AND pg_column_size(workflow_configuration) <= 65536",
    )
    op.create_check_constraint(
        "ck_chat_run_workflow_state_v1",
        "chat_run",
        "jsonb_typeof(workflow_state) = 'object' "
        "AND workflow_state->>'version' = 'chat_workflow_v1' "
        "AND workflow_state->>'resolved_mode' IN ('pending','simple','agent') "
        "AND workflow_state->>'route_status' IN "
        "('not_applicable','pending','resolved','fallback') "
        "AND jsonb_typeof(workflow_state->'route_reason_codes') = 'array' "
        "AND pg_column_size(workflow_state) <= 65536",
    )
