"""add durable local source-file cleanup records

Revision ID: 0003_local_files
Revises: 0002_content_lifecycle
Create Date: 2026-07-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003_local_files"
down_revision: Union[str, Sequence[str], None] = "0002_content_lifecycle"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "source_file_cleanup",
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("document_version_id", sa.UUID(), nullable=False),
        sa.Column("storage_uri", sa.Text(), nullable=False),
        sa.Column("reason", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column(
            "attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_error_code", sa.String(length=128), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name=op.f("ck_source_file_cleanup_source_file_cleanup_attempt_nonnegative"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'completed', 'failed')",
            name=op.f("ck_source_file_cleanup_source_file_cleanup_status_supported"),
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_version.id"],
            name=op.f(
                "fk_source_file_cleanup_document_version_id_document_version"
            ),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspace.id"],
            name=op.f("fk_source_file_cleanup_workspace_id_workspace"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_source_file_cleanup")),
        sa.UniqueConstraint(
            "document_version_id",
            name=op.f("uq_source_file_cleanup_document_version_id"),
        ),
    )
    op.create_index(
        "ix_source_file_cleanup_due",
        "source_file_cleanup",
        ["workspace_id", "status", "next_attempt_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_source_file_cleanup_workspace_id"),
        "source_file_cleanup",
        ["workspace_id"],
        unique=False,
    )
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE source_file_cleanup "
        "TO rag_kb_runtime"
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_source_file_cleanup_workspace_id"),
        table_name="source_file_cleanup",
    )
    op.drop_index("ix_source_file_cleanup_due", table_name="source_file_cleanup")
    op.drop_table("source_file_cleanup")
