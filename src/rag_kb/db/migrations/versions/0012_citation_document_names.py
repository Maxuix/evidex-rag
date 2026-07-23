"""Persist document-name snapshots for chat citations.

Revision ID: 0012_citation_document_names
Revises: 0011_chat_final_llm_context
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0012_citation_document_names"
down_revision: str | None = "0011_chat_final_llm_context"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "citation",
        sa.Column("document_display_name_snapshot", sa.String(512), nullable=True),
    )
    op.add_column(
        "citation",
        sa.Column(
            "document_original_filename_snapshot",
            sa.String(1024),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("citation", "document_original_filename_snapshot")
    op.drop_column("citation", "document_display_name_snapshot")
