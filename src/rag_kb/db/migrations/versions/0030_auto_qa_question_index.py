"""Add Auto-QA question index configuration and question rows.

Revision ID: 0030_auto_qa_question_index
Revises: 0029_agent_v6_default
Create Date: 2026-09-04
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0030_auto_qa_question_index"
down_revision: str | Sequence[str] | None = "0029_agent_v6_default"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "index_revision",
        sa.Column(
            "auto_qa_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{\"enabled\": false}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "index_revision",
        sa.Column(
            "auto_qa_model_profile_revision_id",
            sa.UUID(),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_index_revision_auto_qa_model_same_workspace",
        "index_revision",
        "model_profile_revision",
        ["workspace_id", "auto_qa_model_profile_revision_id"],
        ["workspace_id", "id"],
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "ck_index_revision_index_revision_auto_qa_config_consistent",
        "index_revision",
        "jsonb_typeof(auto_qa_config) = 'object' "
        "AND auto_qa_config ? 'enabled' "
        "AND ("
        "((auto_qa_config->>'enabled') = 'false' "
        "AND auto_qa_model_profile_revision_id IS NULL) "
        "OR ((auto_qa_config->>'enabled') = 'true' "
        "AND auto_qa_model_profile_revision_id IS NOT NULL)"
        ") "
        "AND ("
        "NOT (auto_qa_config ? 'questions_per_chunk') "
        "OR (auto_qa_config->>'questions_per_chunk') = '5'"
        ")",
    )
    op.create_table(
        "index_chunk_question",
        sa.Column("index_chunk_id", sa.UUID(), nullable=False),
        sa.Column("ordinal", sa.SmallInteger(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "ordinal BETWEEN 0 AND 4",
            name="ck_index_chunk_question_index_chunk_question_ordinal_range",
        ),
        sa.CheckConstraint(
            "length(btrim(question)) > 0",
            name="ck_index_chunk_question_index_chunk_question_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["index_chunk_id"],
            ["index_chunk.id"],
            name=op.f("fk_index_chunk_question_index_chunk_id_index_chunk"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "index_chunk_id",
            "ordinal",
            name=op.f("pk_index_chunk_question"),
        ),
        sa.UniqueConstraint(
            "index_chunk_id",
            "question",
            name="uq_index_chunk_question_text",
        ),
    )
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE index_chunk_question "
        "TO rag_kb_runtime"
    )


def downgrade() -> None:
    op.execute(
        "REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLE index_chunk_question "
        "FROM rag_kb_runtime"
    )
    op.drop_table("index_chunk_question")
    op.drop_constraint(
        "ck_index_revision_index_revision_auto_qa_config_consistent",
        "index_revision",
        type_="check",
    )
    op.drop_constraint(
        "fk_index_revision_auto_qa_model_same_workspace",
        "index_revision",
        type_="foreignkey",
    )
    op.drop_column("index_revision", "auto_qa_model_profile_revision_id")
    op.drop_column("index_revision", "auto_qa_config")
