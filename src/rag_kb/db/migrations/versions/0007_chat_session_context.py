"""Persist bounded Session context and contextualized queries.

Revision ID: 0007_chat_session_context
Revises: 0006_semantic_chunk_plan
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0007_chat_session_context"
down_revision: str | None = "0006_semantic_chunk_plan"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1 FROM chat_run
            WHERE status IN ('queued', 'running')
            GROUP BY session_id HAVING count(*) > 1
          ) THEN
            RAISE EXCEPTION USING
              ERRCODE = 'check_violation',
              MESSAGE = 'cannot enforce one nonterminal ChatRun per Session: duplicate queued/running runs exist';
          END IF;
        END $$
        """
    )
    op.add_column(
        "chat_run",
        sa.Column(
            "conversation_context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "chat_run",
        sa.Column(
            "contextualized_query",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.execute(
        """
        UPDATE chat_run
        SET conversation_context = jsonb_build_object(
          'version', 'session_context_v1',
          'strategy', 'recent_completed_turns_v1',
          'turns', '[]'::jsonb,
          'token_budget', 4000,
          'token_count', 0,
          'candidate_turn_count', 0,
          'truncated', false,
          'content_hash', 'sha256:45f76aa530878a50f94da86cc77ac1584f58c20c82988739d888b7bbb637652c'
        )
        """
    )
    op.alter_column("chat_run", "conversation_context", nullable=False)
    op.create_index(
        "uq_chat_run_session_nonterminal",
        "chat_run",
        ["session_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("uq_chat_run_session_nonterminal", table_name="chat_run")
    op.drop_column("chat_run", "contextualized_query")
    op.drop_column("chat_run", "conversation_context")
