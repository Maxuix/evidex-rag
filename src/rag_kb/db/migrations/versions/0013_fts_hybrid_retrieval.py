"""Add versioned PostgreSQL FTS derived data and completeness manifests.

Revision ID: 0013_fts_hybrid_retrieval
Revises: 0012_citation_document_names
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0013_fts_hybrid_retrieval"
down_revision: str | None = "0012_citation_document_names"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "index_chunk_lexical",
        sa.Column("index_chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analyzer_version", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "indexed_document_version_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("lexical_text", sa.Text(), nullable=False),
        sa.Column("lexical_text_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "lexical_tsv",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple'::regconfig, lexical_text)",
                persisted=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "indexed_document_version_id"],
            [
                "indexed_document_version.workspace_id",
                "indexed_document_version.kb_id",
                "indexed_document_version.id",
            ],
            name="fk_chunk_lexical_same_scope_target",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["indexed_document_version_id", "index_chunk_id"],
            ["index_chunk.indexed_document_version_id", "index_chunk.id"],
            name="fk_chunk_lexical_same_target_chunk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "index_chunk_id",
            "analyzer_version",
            name="pk_index_chunk_lexical",
        ),
    )
    op.create_index(
        "ix_index_chunk_lexical_scope",
        "index_chunk_lexical",
        [
            "workspace_id",
            "kb_id",
            "analyzer_version",
            "indexed_document_version_id",
        ],
    )
    op.create_index(
        "ix_index_chunk_lexical_tsv",
        "index_chunk_lexical",
        ["lexical_tsv"],
        postgresql_using="gin",
    )
    op.create_table(
        "index_lexical_manifest",
        sa.Column(
            "indexed_document_version_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("analyzer_version", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("lexical_chunk_count", sa.Integer(), nullable=False),
        sa.Column("lexical_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "lexical_chunk_count >= 0",
            name="ck_index_lexical_manifest_lexical_manifest_chunk_count_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "indexed_document_version_id"],
            [
                "indexed_document_version.workspace_id",
                "indexed_document_version.kb_id",
                "indexed_document_version.id",
            ],
            name="fk_lexical_manifest_same_scope_target",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "indexed_document_version_id",
            "analyzer_version",
            name="pk_index_lexical_manifest",
        ),
    )
    op.execute(
        "GRANT SELECT, INSERT, DELETE ON TABLE "
        "index_chunk_lexical, index_lexical_manifest TO rag_kb_runtime"
    )


def downgrade() -> None:
    op.drop_table("index_lexical_manifest")
    op.drop_index(
        "ix_index_chunk_lexical_tsv",
        table_name="index_chunk_lexical",
        postgresql_using="gin",
    )
    op.drop_index(
        "ix_index_chunk_lexical_scope",
        table_name="index_chunk_lexical",
    )
    op.drop_table("index_chunk_lexical")
