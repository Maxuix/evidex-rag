"""Preserve legacy questions while recording validated support for new questions."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0031_auto_qa_grounding"
down_revision = "0030_auto_qa_question_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("index_chunk", sa.Column("auto_qa_question_count", sa.SmallInteger(), nullable=True))
    op.create_check_constraint("ck_index_chunk_index_chunk_auto_qa_count", "index_chunk", "auto_qa_question_count BETWEEN 0 AND 5")
    op.add_column("index_chunk_question", sa.Column("support", postgresql.JSONB(none_as_null=True), nullable=True))


def downgrade() -> None:
    op.drop_column("index_chunk_question", "support")
    op.drop_constraint("ck_index_chunk_index_chunk_auto_qa_count", "index_chunk", type_="check")
    op.drop_column("index_chunk", "auto_qa_question_count")
