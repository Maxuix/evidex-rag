"""align ChatRun request hash with the canonical SHA-256 representation

Revision ID: 0004_chat_request_hash
Revises: 0003_local_files
Create Date: 2026-07-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0004_chat_request_hash"
down_revision: Union[str, Sequence[str], None] = "0003_local_files"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "chat_run",
        "request_hash",
        existing_type=sa.String(length=64),
        type_=sa.String(length=71),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "chat_run",
        "request_hash",
        existing_type=sa.String(length=71),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
