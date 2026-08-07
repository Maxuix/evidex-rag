"""Add bounded workflow snapshots to ChatRun.

Revision ID: 0002_selectable_retrieval
Revises: 0001_current_only_baseline
Create Date: 2026-08-06
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0002_selectable_retrieval"
down_revision: Union[str, Sequence[str], None] = "0001_current_only_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_SIMPLE_CONFIGURATION = sa.text(
    "jsonb_build_object("
    "'version', 'chat_workflow_v1', "
    "'requested_mode', 'simple', "
    "'budget', jsonb_build_object("
    "'decision_rounds', 4, 'retrieval_calls', 6, "
    "'parallel_queries', 3, 'verifier_continuations', 1, "
    "'no_progress_rounds', 1))"
)
_SIMPLE_STATE = sa.text(
    "jsonb_build_object("
    "'version', 'chat_workflow_v1', "
    "'resolved_mode', 'simple', "
    "'route_status', 'not_applicable', "
    "'route_reason_codes', '[]'::jsonb, "
    "'research_result', 'null'::jsonb, "
    "'search_trace', 'null'::jsonb)"
)


def upgrade() -> None:
    op.add_column(
        "chat_run",
        sa.Column(
            "workflow_configuration",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=_SIMPLE_CONFIGURATION,
            nullable=False,
        ),
    )
    op.add_column(
        "chat_run",
        sa.Column(
            "workflow_state",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=_SIMPLE_STATE,
            nullable=False,
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


def downgrade() -> None:
    op.drop_constraint(
        "ck_chat_run_workflow_state_v1", "chat_run", type_="check"
    )
    op.drop_constraint(
        "ck_chat_run_workflow_configuration_v1", "chat_run", type_="check"
    )
    op.drop_column("chat_run", "workflow_state")
    op.drop_column("chat_run", "workflow_configuration")
