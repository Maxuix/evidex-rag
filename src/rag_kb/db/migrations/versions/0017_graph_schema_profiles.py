"""Freeze Graph schema profile identities on configs and builds.

Revision ID: 0017_graph_schema_profiles
Revises: 0016_first_class_graph_tool
Create Date: 2026-08-24
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0017_graph_schema_profiles"
down_revision: str | None = "0016_first_class_graph_tool"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SOFTWARE_PROFILE_KEY = "software_knowledge_v1"
SOFTWARE_PROFILE_DIGEST = (
    "6cae93809f060d21f0c85ba04cde955abdc5259fd93e1b1757fb7445d51eaf38"
)
GENERIC_PROFILE_KEY = "generic_open_domain_v1"
GENERIC_PROFILE_DIGEST = (
    "3b351f4e2c601226f922d12b60d4c9f98a4770f4ec04e94b08f5a3f0d021eaf0"
)


def upgrade() -> None:
    op.add_column(
        "knowledge_base_graph_config",
        sa.Column("schema_profile_key", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "knowledge_base_graph_config",
        sa.Column("schema_profile_digest", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "graphiti_graph_build",
        sa.Column("schema_profile_key", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "graphiti_graph_build",
        sa.Column("schema_profile_digest", sa.String(length=64), nullable=True),
    )

    op.execute(
        sa.text(
            """
            UPDATE knowledge_base_graph_config
               SET schema_profile_key = :key,
                   schema_profile_digest = :digest
            """
        ).bindparams(key=SOFTWARE_PROFILE_KEY, digest=SOFTWARE_PROFILE_DIGEST)
    )
    op.execute(
        sa.text(
            """
            UPDATE graphiti_graph_build
               SET schema_profile_key = :key,
                   schema_profile_digest = :digest
            """
        ).bindparams(key=SOFTWARE_PROFILE_KEY, digest=SOFTWARE_PROFILE_DIGEST)
    )

    op.alter_column(
        "knowledge_base_graph_config",
        "schema_profile_key",
        nullable=False,
        server_default=sa.text(f"'{GENERIC_PROFILE_KEY}'"),
    )
    op.alter_column(
        "knowledge_base_graph_config",
        "schema_profile_digest",
        nullable=False,
        server_default=sa.text(f"'{GENERIC_PROFILE_DIGEST}'"),
    )
    op.alter_column("graphiti_graph_build", "schema_profile_key", nullable=False)
    op.alter_column("graphiti_graph_build", "schema_profile_digest", nullable=False)

    op.create_check_constraint(
        "graph_config_schema_profile_key_nonempty",
        "knowledge_base_graph_config",
        "length(btrim(schema_profile_key)) > 0",
    )
    op.create_check_constraint(
        "graph_config_schema_profile_digest_valid",
        "knowledge_base_graph_config",
        "schema_profile_digest ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "graphiti_build_schema_profile_key_nonempty",
        "graphiti_graph_build",
        "length(btrim(schema_profile_key)) > 0",
    )
    op.create_check_constraint(
        "graphiti_build_schema_profile_digest_valid",
        "graphiti_graph_build",
        "schema_profile_digest ~ '^[0-9a-f]{64}$'",
    )


def downgrade() -> None:
    op.drop_constraint(
        "graphiti_build_schema_profile_digest_valid",
        "graphiti_graph_build",
        type_="check",
    )
    op.drop_constraint(
        "graphiti_build_schema_profile_key_nonempty",
        "graphiti_graph_build",
        type_="check",
    )
    op.drop_constraint(
        "graph_config_schema_profile_digest_valid",
        "knowledge_base_graph_config",
        type_="check",
    )
    op.drop_constraint(
        "graph_config_schema_profile_key_nonempty",
        "knowledge_base_graph_config",
        type_="check",
    )
    op.drop_column("graphiti_graph_build", "schema_profile_digest")
    op.drop_column("graphiti_graph_build", "schema_profile_key")
    op.drop_column("knowledge_base_graph_config", "schema_profile_digest")
    op.drop_column("knowledge_base_graph_config", "schema_profile_key")
