"""Add user-facing knowledge-base and chunk lifecycle controls.

Revision ID: 0005_content_management
Revises: 0004_flexible_embedding_spaces
Create Date: 2026-08-09
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0005_content_management"
down_revision: Union[str, Sequence[str], None] = "0004_flexible_embedding_spaces"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "knowledge_base",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.drop_constraint(
        "uq_knowledge_base_workspace_name",
        "knowledge_base",
        type_="unique",
    )
    op.create_index(
        "uq_knowledge_base_workspace_active_name",
        "knowledge_base",
        ["workspace_id", "name"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "ix_knowledge_base_workspace_deleted_at",
        "knowledge_base",
        ["workspace_id", "deleted_at"],
    )

    op.add_column(
        "index_chunk",
        sa.Column("excluded_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_provisioned_kb_active_revision()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            target_kb_id uuid;
        BEGIN
            IF TG_TABLE_NAME = 'knowledge_base' THEN
                target_kb_id := NEW.id;
            ELSIF TG_OP = 'DELETE' THEN
                target_kb_id := OLD.kb_id;
            ELSE
                target_kb_id := NEW.kb_id;
            END IF;

            IF EXISTS (
                SELECT 1
                  FROM knowledge_base kb
                 WHERE kb.id = target_kb_id
                   AND kb.provisioned_at IS NOT NULL
                   AND kb.deleted_at IS NULL
                   AND (
                       kb.active_index_revision_id IS NULL
                       OR NOT EXISTS (
                           SELECT 1
                             FROM index_revision revision
                            WHERE revision.kb_id = kb.id
                              AND revision.id = kb.active_index_revision_id
                              AND revision.status = 'active'
                       )
                   )
            ) THEN
                RAISE EXCEPTION
                    'provisioned knowledge base % must select its active revision',
                    target_kb_id
                    USING ERRCODE = '23514';
            END IF;
            RETURN NULL;
        END;
        $$
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                  FROM knowledge_base
                 GROUP BY workspace_id, name
                HAVING count(*) > 1
            ) THEN
                RAISE EXCEPTION
                    'cannot downgrade: deleted knowledge bases reuse an active name';
            END IF;
        END $$
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_provisioned_kb_active_revision()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            target_kb_id uuid;
        BEGIN
            IF TG_TABLE_NAME = 'knowledge_base' THEN
                target_kb_id := NEW.id;
            ELSIF TG_OP = 'DELETE' THEN
                target_kb_id := OLD.kb_id;
            ELSE
                target_kb_id := NEW.kb_id;
            END IF;

            IF EXISTS (
                SELECT 1
                  FROM knowledge_base kb
                 WHERE kb.id = target_kb_id
                   AND kb.provisioned_at IS NOT NULL
                   AND (
                       kb.active_index_revision_id IS NULL
                       OR NOT EXISTS (
                           SELECT 1
                             FROM index_revision revision
                            WHERE revision.kb_id = kb.id
                              AND revision.id = kb.active_index_revision_id
                              AND revision.status = 'active'
                       )
                   )
            ) THEN
                RAISE EXCEPTION
                    'provisioned knowledge base % must select its active revision',
                    target_kb_id
                    USING ERRCODE = '23514';
            END IF;
            RETURN NULL;
        END;
        $$
        """
    )
    op.drop_column("index_chunk", "excluded_at")
    op.drop_index(
        "ix_knowledge_base_workspace_deleted_at",
        table_name="knowledge_base",
    )
    op.drop_index(
        "uq_knowledge_base_workspace_active_name",
        table_name="knowledge_base",
    )
    op.create_unique_constraint(
        "uq_knowledge_base_workspace_name",
        "knowledge_base",
        ["workspace_id", "name"],
    )
    op.drop_column("knowledge_base", "deleted_at")
