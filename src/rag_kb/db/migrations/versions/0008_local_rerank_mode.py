"""Replace the retrieval rerank boolean with an explicit mode.

Revision ID: 0008_local_rerank_mode
Revises: 0007_default_partial_answer
Create Date: 2026-08-11
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0008_local_rerank_mode"
down_revision: str | None = "0007_default_partial_answer"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_CURRENT_DEFAULT = sa.text(
    "'{\"strategy\": \"exact_vector\", \"top_k\": 10, "
    "\"rerank_mode\": \"classic\"}'::jsonb"
)
_LEGACY_DEFAULT = sa.text(
    "'{\"strategy\": \"exact_vector\", \"top_k\": 10, "
    "\"rerank\": true}'::jsonb"
)


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE knowledge_base
               SET retrieval_defaults =
                       (retrieval_defaults - 'rerank')
                       || jsonb_build_object(
                           'rerank_mode',
                           CASE
                               WHEN retrieval_defaults -> 'rerank' = 'false'::jsonb
                                   THEN 'none'
                               ELSE 'classic'
                           END
                       ),
                   updated_at = now()
             WHERE NOT (retrieval_defaults ? 'rerank_mode')
            """
        )
    )
    op.alter_column(
        "knowledge_base",
        "retrieval_defaults",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=False,
        server_default=_CURRENT_DEFAULT,
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE knowledge_base
               SET retrieval_defaults =
                       (retrieval_defaults - 'rerank_mode')
                       || jsonb_build_object(
                           'rerank',
                           COALESCE(
                               retrieval_defaults ->> 'rerank_mode',
                               'classic'
                           ) <> 'none'
                       ),
                   updated_at = now()
             WHERE retrieval_defaults ? 'rerank_mode'
            """
        )
    )
    op.alter_column(
        "knowledge_base",
        "retrieval_defaults",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=False,
        server_default=_LEGACY_DEFAULT,
    )
