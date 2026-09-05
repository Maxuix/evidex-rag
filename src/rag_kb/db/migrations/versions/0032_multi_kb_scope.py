"""Preserve chat history while adding frozen multi-knowledge-base scope."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0032_multi_kb_scope"
down_revision = "0031_auto_qa_grounding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("knowledge_base", sa.Column("description", sa.Text(), nullable=False, server_default=""))
    op.create_table(
        "chat_session_kb",
        sa.Column("session_id", sa.UUID(), primary_key=True),
        sa.Column("kb_id", sa.UUID(), primary_key=True),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id", "session_id"], ["chat_session.workspace_id", "chat_session.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id", "kb_id"], ["knowledge_base.workspace_id", "knowledge_base.id"], ondelete="CASCADE"),
    )
    op.create_table(
        "chat_run_kb",
        sa.Column("run_id", sa.UUID(), sa.ForeignKey("chat_run.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("kb_id", sa.UUID(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("index_revision_id", sa.UUID(), nullable=True),
        sa.Column("graph_build_id", sa.UUID(), nullable=True),
        sa.Column("retrieval_strategy", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
    )
    op.execute("INSERT INTO chat_session_kb SELECT id, kb_id, workspace_id FROM chat_session")
    op.execute("""
        INSERT INTO chat_run_kb (run_id, kb_id, name, description, index_revision_id, retrieval_strategy, status)
        SELECT r.id, r.kb_id, k.name, k.description, r.index_revision_id, r.retrieval_strategy,
               CASE WHEN k.deleted_at IS NOT NULL THEN 'deleted'
                    WHEN r.index_revision_id IS NULL THEN 'index_unavailable' ELSE 'ready' END
        FROM chat_run r JOIN knowledge_base k ON k.id = r.kb_id
    """)
    for name, kind in (("knowledge_base_id_snapshot", sa.UUID()), ("knowledge_base_name_snapshot", sa.String(255)), ("index_revision_id_snapshot", sa.UUID())):
        op.add_column("citation", sa.Column(name, kind, nullable=True))
    op.execute("""
        UPDATE citation c SET knowledge_base_id_snapshot=r.kb_id,
            knowledge_base_name_snapshot=k.name, index_revision_id_snapshot=r.index_revision_id
        FROM chat_message m JOIN chat_run r ON r.id=m.chat_run_id
             JOIN knowledge_base k ON k.id=r.kb_id
        WHERE c.assistant_message_id=m.id
    """)
    # Abort the transaction rather than switching to an incompletely backfilled scope.
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM chat_session s WHERE NOT EXISTS
            (SELECT 1 FROM chat_session_kb k WHERE k.session_id=s.id))
        OR EXISTS (SELECT 1 FROM chat_run r WHERE NOT EXISTS
            (SELECT 1 FROM chat_run_kb k WHERE k.run_id=r.id))
        OR EXISTS (SELECT 1 FROM citation WHERE knowledge_base_id_snapshot IS NULL)
        THEN RAISE EXCEPTION 'multi-KB history backfill incomplete'; END IF;
    END $$""")
    op.drop_constraint("fk_chat_session_same_workspace_kb", "chat_session", type_="foreignkey")
    op.create_foreign_key("fk_chat_session_workspace_id_workspace", "chat_session", "workspace", ["workspace_id"], ["id"], ondelete="RESTRICT")
    op.alter_column("chat_session", "kb_id", nullable=True)
    for column, table in (("kb_id", "knowledge_base"), ("index_revision_id", "index_revision")):
        name = f"fk_chat_run_{column}_{table}"
        op.drop_constraint(name, "chat_run", type_="foreignkey")
        op.alter_column("chat_run", column, nullable=True)
        op.create_foreign_key(name, "chat_run", table, [column], ["id"], ondelete="SET NULL")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE chat_session_kb, chat_run_kb TO rag_kb_runtime")


def downgrade() -> None:
    raise RuntimeError("Multi-KB history cannot be reduced to single-KB columns without data loss; restore a verified backup instead.")
