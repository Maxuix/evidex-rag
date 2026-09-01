"""Remove request identity from the single-user local data model.

Revision ID: 0025_remove_dynamic_identity
Revises: 0024_remove_agent_deadline_reserve
Create Date: 2026-09-01
"""

from __future__ import annotations

from collections.abc import Sequence
import os
import re
from uuid import UUID

from alembic import op
import sqlalchemy as sa


revision: str = "0025_remove_dynamic_identity"
down_revision: str | None = "0024_remove_agent_deadline_reserve"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IDENTITY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _configured_identity() -> tuple[UUID, str, str]:
    workspace = UUID(os.environ["RAG_KB__IDENTITY__WORKSPACE_ID"])
    principal = os.environ["RAG_KB__IDENTITY__PRINCIPAL_ID"]
    client = os.environ["RAG_KB__IDENTITY__CLIENT_ID"]
    if not _IDENTITY_PATTERN.fullmatch(principal):
        raise RuntimeError("configured principal identity is invalid")
    if not _IDENTITY_PATTERN.fullmatch(client):
        raise RuntimeError("configured client identity is invalid")
    return workspace, principal, client


def _preflight() -> None:
    workspace, principal, client = _configured_identity()
    bind = op.get_bind()
    facts = bind.execute(
        sa.text(
            """
            SELECT
              NOT EXISTS (
                SELECT 1 FROM chat_session
                 WHERE workspace_id <> :workspace OR principal_id <> :principal
              ) AS sessions_match,
              NOT EXISTS (
                SELECT 1 FROM chat_run
                 WHERE workspace_id <> :workspace
                    OR principal_id <> :principal OR client_id <> :client
              ) AS runs_match,
              NOT EXISTS (
                SELECT 1 FROM content_mutation
                 WHERE workspace_id <> :workspace
                    OR principal_id <> :principal OR client_id <> :client
              ) AS mutations_match,
              NOT EXISTS (
                SELECT endpoint, idempotency_key FROM chat_run
                 GROUP BY endpoint, idempotency_key HAVING count(*) > 1
              ) AS runs_unique,
              NOT EXISTS (
                SELECT endpoint, idempotency_key FROM content_mutation
                 GROUP BY endpoint, idempotency_key HAVING count(*) > 1
              ) AS mutations_unique
            """
        ),
        {"workspace": workspace, "principal": principal, "client": client},
    ).mappings().one()
    failed = [name for name, value in facts.items() if not value]
    if failed:
        raise RuntimeError(
            "dynamic identity removal preflight failed: " + ", ".join(failed)
        )


def upgrade() -> None:
    _preflight()

    op.drop_constraint("uq_chat_run_idempotency_scope", "chat_run", type_="unique")
    op.create_unique_constraint(
        "uq_chat_run_idempotency_scope",
        "chat_run",
        ["endpoint", "idempotency_key"],
    )
    op.drop_column("chat_run", "client_id")
    op.drop_column("chat_run", "principal_id")

    op.drop_column("chat_session", "principal_id")

    op.drop_constraint(
        "uq_content_mutation_idempotency_scope",
        "content_mutation",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_content_mutation_idempotency_scope",
        "content_mutation",
        ["endpoint", "idempotency_key"],
    )
    op.drop_column("content_mutation", "client_id")
    op.drop_column("content_mutation", "principal_id")


def downgrade() -> None:
    _, principal, client = _configured_identity()
    bind = op.get_bind()

    op.add_column("chat_session", sa.Column("principal_id", sa.String(255)))
    bind.execute(
        sa.text("UPDATE chat_session SET principal_id = :principal"),
        {"principal": principal},
    )
    op.alter_column("chat_session", "principal_id", nullable=False)

    op.add_column("chat_run", sa.Column("principal_id", sa.String(255)))
    op.add_column("chat_run", sa.Column("client_id", sa.String(255)))
    bind.execute(
        sa.text(
            "UPDATE chat_run SET principal_id = :principal, client_id = :client"
        ),
        {"principal": principal, "client": client},
    )
    op.alter_column("chat_run", "principal_id", nullable=False)
    op.alter_column("chat_run", "client_id", nullable=False)
    op.drop_constraint("uq_chat_run_idempotency_scope", "chat_run", type_="unique")
    op.create_unique_constraint(
        "uq_chat_run_idempotency_scope",
        "chat_run",
        ["principal_id", "client_id", "endpoint", "idempotency_key"],
    )

    op.add_column("content_mutation", sa.Column("principal_id", sa.String(255)))
    op.add_column("content_mutation", sa.Column("client_id", sa.String(255)))
    bind.execute(
        sa.text(
            "UPDATE content_mutation "
            "SET principal_id = :principal, client_id = :client"
        ),
        {"principal": principal, "client": client},
    )
    op.alter_column("content_mutation", "principal_id", nullable=False)
    op.alter_column("content_mutation", "client_id", nullable=False)
    op.drop_constraint(
        "uq_content_mutation_idempotency_scope",
        "content_mutation",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_content_mutation_idempotency_scope",
        "content_mutation",
        ["principal_id", "client_id", "endpoint", "idempotency_key"],
    )
