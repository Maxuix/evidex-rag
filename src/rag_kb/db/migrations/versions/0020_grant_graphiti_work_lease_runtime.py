"""Grant the runtime role access to Graphiti work leases.

Revision ID: 0020_grant_graphiti_work_lease_runtime
Revises: 0019_file_mutation_terminal_state
Create Date: 2026-08-26
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "0020_grant_graphiti_work_lease_runtime"
down_revision: str | None = "0019_file_mutation_terminal_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The baseline default privilege applies to normal migrations, but an
    # already-restored database can have the migration table owner differ from
    # ``rag_kb_migration``.  Make the work-lease privilege explicit so the
    # runtime Worker can reconcile an interrupted Graphiti lease.
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE graphiti_graph_work_lease "
        "TO rag_kb_runtime"
    )


def downgrade() -> None:
    op.execute(
        "REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLE graphiti_graph_work_lease "
        "FROM rag_kb_runtime"
    )
