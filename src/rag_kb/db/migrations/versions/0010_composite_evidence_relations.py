"""Add composite chunk text and normalized visual relations.

Revision ID: 0010_composite_evidence_v2
Revises: 0009_cross_modal_vector_768
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0010_composite_evidence_v2"
down_revision: str | None = "0009_cross_modal_vector_768"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_indexed_version_workspace_kb_id",
        "indexed_document_version",
        ["workspace_id", "kb_id", "id"],
    )
    op.add_column("index_chunk", sa.Column("embedding_text", sa.Text(), nullable=True))
    op.add_column(
        "index_chunk",
        sa.Column("embedding_text_hash", sa.String(length=64), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_index_chunk_index_chunk_embedding_text_pair"),
        "index_chunk",
        "(embedding_text IS NULL) = (embedding_text_hash IS NULL)",
    )

    op.add_column(
        "index_artifact_manifest",
        sa.Column(
            "relation_plan",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "index_artifact_manifest", sa.Column("relation_count", sa.Integer(), nullable=True)
    )
    op.add_column(
        "index_artifact_manifest",
        sa.Column("relation_manifest_hash", sa.String(length=64), nullable=True),
    )
    op.create_check_constraint(
        op.f(
            "ck_index_artifact_manifest_artifact_manifest_relation_plan_consistent"
        ),
        "index_artifact_manifest",
        "(relation_plan IS NULL AND relation_count IS NULL AND relation_manifest_hash IS NULL) "
        "OR (jsonb_typeof(relation_plan) = 'array' AND relation_count >= 0 "
        "AND jsonb_array_length(relation_plan) = relation_count "
        "AND relation_manifest_hash IS NOT NULL)",
    )

    op.create_table(
        "index_chunk_asset_relation",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuidv7()"),
            nullable=False,
        ),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "indexed_document_version_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("visual_unit_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relation_type", sa.String(length=64), nullable=False),
        sa.Column("confidence_micros", sa.Integer(), nullable=False),
        sa.Column("figure_label", sa.String(length=128), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("provenance", sa.String(length=128), nullable=False),
        sa.Column("evidence_group_key", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "relation_type IN ("
            "'explicit_figure_reference', 'caption_of', 'inline_figure', "
            "'ocr_of', 'table_of', 'spatial_neighbor', 'same_page')",
            name=op.f(
                "ck_index_chunk_asset_relation_chunk_asset_relation_type_supported"
            ),
        ),
        sa.CheckConstraint(
            "confidence_micros BETWEEN 0 AND 1000000",
            name=op.f(
                "ck_index_chunk_asset_relation_chunk_asset_relation_confidence_micros"
            ),
        ),
        sa.CheckConstraint(
            "ordinal >= 0",
            name=op.f(
                "ck_index_chunk_asset_relation_chunk_asset_relation_ordinal_nonnegative"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "kb_id", "indexed_document_version_id"],
            [
                "indexed_document_version.workspace_id",
                "indexed_document_version.kb_id",
                "indexed_document_version.id",
            ],
            name="fk_chunk_asset_relation_same_scope_target",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["indexed_document_version_id", "chunk_id"],
            ["index_chunk.indexed_document_version_id", "index_chunk.id"],
            name="fk_chunk_asset_relation_same_target_chunk",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["indexed_document_version_id", "visual_unit_id"],
            ["index_chunk.indexed_document_version_id", "index_chunk.id"],
            name="fk_chunk_asset_relation_same_target_visual",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["indexed_document_version_id", "asset_id"],
            ["index_asset.indexed_document_version_id", "index_asset.id"],
            name="fk_chunk_asset_relation_same_target_asset",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_index_chunk_asset_relation"),
        sa.UniqueConstraint(
            "indexed_document_version_id",
            "chunk_id",
            "asset_id",
            "relation_type",
            name="uq_chunk_asset_relation_stable_edge",
        ),
    )
    op.create_index(
        "ix_chunk_asset_relation_chunk",
        "index_chunk_asset_relation",
        ["workspace_id", "kb_id", "indexed_document_version_id", "chunk_id"],
    )
    op.create_index(
        "ix_chunk_asset_relation_asset",
        "index_chunk_asset_relation",
        ["workspace_id", "kb_id", "indexed_document_version_id", "asset_id"],
    )
    op.execute(
        "GRANT SELECT, INSERT, DELETE ON TABLE index_chunk_asset_relation "
        "TO rag_kb_runtime"
    )


def downgrade() -> None:
    op.drop_index(
        "ix_chunk_asset_relation_asset", table_name="index_chunk_asset_relation"
    )
    op.drop_index(
        "ix_chunk_asset_relation_chunk", table_name="index_chunk_asset_relation"
    )
    op.drop_table("index_chunk_asset_relation")
    op.drop_constraint(
        op.f(
            "ck_index_artifact_manifest_artifact_manifest_relation_plan_consistent"
        ),
        "index_artifact_manifest",
        type_="check",
    )
    op.drop_column("index_artifact_manifest", "relation_manifest_hash")
    op.drop_column("index_artifact_manifest", "relation_count")
    op.drop_column("index_artifact_manifest", "relation_plan")
    op.drop_constraint(
        op.f("ck_index_chunk_index_chunk_embedding_text_pair"),
        "index_chunk",
        type_="check",
    )
    op.drop_column("index_chunk", "embedding_text_hash")
    op.drop_column("index_chunk", "embedding_text")
    op.drop_constraint(
        "uq_indexed_version_workspace_kb_id",
        "indexed_document_version",
        type_="unique",
    )
