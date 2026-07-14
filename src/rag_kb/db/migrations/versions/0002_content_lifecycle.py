"""add content lifecycle idempotency and immutability guards

Revision ID: 0002_content_lifecycle
Revises: 0001_p0_foundation
Create Date: 2026-07-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002_content_lifecycle"
down_revision: Union[str, Sequence[str], None] = "0001_p0_foundation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "content_mutation",
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("principal_id", sa.String(length=255), nullable=False),
        sa.Column("client_id", sa.String(length=255), nullable=False),
        sa.Column("endpoint", sa.String(length=255), nullable=False),
        sa.Column("idempotency_key", sa.UUID(), nullable=False),
        sa.Column("request_hash", sa.String(length=71), nullable=False),
        sa.Column("operation", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("kb_id", sa.UUID(), nullable=True),
        sa.Column("document_id", sa.UUID(), nullable=True),
        sa.Column("document_version_id", sa.UUID(), nullable=True),
        sa.Column("source_change_id", sa.UUID(), nullable=True),
        sa.Column("indexed_document_version_id", sa.UUID(), nullable=True),
        sa.Column("index_revision_id", sa.UUID(), nullable=True),
        sa.Column("job_id", sa.UUID(), nullable=True),
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
            "kb_id IS NOT NULL OR document_id IS NOT NULL",
            name=op.f("ck_content_mutation_content_mutation_has_result"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'completed')",
            name=op.f("ck_content_mutation_content_mutation_status_supported"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["document.id"],
            name=op.f("fk_content_mutation_document_id_document"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"], ["document_version.id"],
            name=op.f("fk_content_mutation_document_version_id_document_version"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["indexed_document_version_id"], ["indexed_document_version.id"],
            name=op.f("fk_content_mutation_indexed_document_version_id_indexed_document_version"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["index_revision_id"], ["index_revision.id"],
            name=op.f("fk_content_mutation_index_revision_id_index_revision"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["indexing_job.id"],
            name=op.f("fk_content_mutation_job_id_indexing_job"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["kb_id"], ["knowledge_base.id"],
            name=op.f("fk_content_mutation_kb_id_knowledge_base"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_change_id"], ["source_change.id"],
            name=op.f("fk_content_mutation_source_change_id_source_change"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspace.id"],
            name=op.f("fk_content_mutation_workspace_id_workspace"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_content_mutation")),
        sa.UniqueConstraint(
            "principal_id", "client_id", "endpoint", "idempotency_key",
            name="uq_content_mutation_idempotency_scope",
        ),
    )
    op.create_index(
        op.f("ix_content_mutation_workspace_id"),
        "content_mutation",
        ["workspace_id"],
        unique=False,
    )
    op.execute(
        """
        CREATE FUNCTION enforce_document_version_source_immutability()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'document versions are immutable'
                    USING ERRCODE = '23514';
            END IF;
            IF NEW.workspace_id IS DISTINCT FROM OLD.workspace_id
               OR NEW.kb_id IS DISTINCT FROM OLD.kb_id
               OR NEW.document_id IS DISTINCT FROM OLD.document_id
               OR NEW.version_number IS DISTINCT FROM OLD.version_number
               OR NEW.checksum_sha256 IS DISTINCT FROM OLD.checksum_sha256
               OR NEW.storage_uri IS DISTINCT FROM OLD.storage_uri
               OR NEW.original_filename IS DISTINCT FROM OLD.original_filename
               OR NEW.media_type IS DISTINCT FROM OLD.media_type
               OR NEW.size_bytes IS DISTINCT FROM OLD.size_bytes
               OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
                RAISE EXCEPTION 'document version source fields are immutable'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER ck_document_version_source_immutable
        BEFORE UPDATE OR DELETE ON document_version
        FOR EACH ROW EXECUTE FUNCTION enforce_document_version_source_immutability()
        """
    )
    op.execute(
        """
        CREATE FUNCTION enforce_source_change_immutability()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION 'source changes are immutable'
                USING ERRCODE = '23514';
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER ck_source_change_immutable
        BEFORE UPDATE OR DELETE ON source_change
        FOR EACH ROW EXECUTE FUNCTION enforce_source_change_immutability()
        """
    )
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE content_mutation "
        "TO rag_kb_runtime"
    )
    op.execute(
        "REVOKE ALL ON FUNCTION enforce_document_version_source_immutability() "
        "FROM PUBLIC"
    )
    op.execute(
        "REVOKE ALL ON FUNCTION enforce_source_change_immutability() FROM PUBLIC"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER ck_source_change_immutable ON source_change")
    op.execute("DROP FUNCTION enforce_source_change_immutability()")
    op.execute("DROP TRIGGER ck_document_version_source_immutable ON document_version")
    op.execute("DROP FUNCTION enforce_document_version_source_immutability()")
    op.drop_index(
        op.f("ix_content_mutation_workspace_id"),
        table_name="content_mutation",
    )
    op.drop_table("content_mutation")
