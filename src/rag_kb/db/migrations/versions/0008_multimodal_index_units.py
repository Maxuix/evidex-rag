"""Add multimodal evidence units, assets, manifests, and revision space roles.

Revision ID: 0008_multimodal_index_units
Revises: 0007_chat_session_context
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0008_multimodal_index_units"
down_revision: str | None = "0007_chat_session_context"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_index_revision_workspace_id", "index_revision", ["workspace_id", "id"]
    )
    op.add_column(
        "index_revision",
        sa.Column(
            "enrichment_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "index_revision",
        sa.Column(
            "representation_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.create_table(
        "index_revision_embedding_space",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("index_revision_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(length=64), nullable=False),
        sa.Column("embedding_space_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("required", sa.Boolean(), nullable=False),
        sa.Column("retrieval_weight_micros", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role IN ('text_retrieval', 'semantic_analysis', 'cross_modal_retrieval')",
            name=op.f("ck_index_revision_embedding_space_revision_space_role_supported"),
        ),
        sa.CheckConstraint(
            "retrieval_weight_micros IS NULL OR retrieval_weight_micros > 0",
            name=op.f("ck_index_revision_embedding_space_revision_space_weight_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "embedding_space_id"],
            ["embedding_space.workspace_id", "embedding_space.id"],
            name="fk_revision_space_same_workspace_embedding",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "index_revision_id"],
            ["index_revision.workspace_id", "index_revision.id"],
            name="fk_revision_space_same_workspace_revision",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspace.id"],
            name=op.f("fk_index_revision_embedding_space_workspace_id_workspace"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "index_revision_id", "role", name=op.f("pk_index_revision_embedding_space")
        ),
    )
    op.create_index(
        op.f("ix_index_revision_embedding_space_workspace_id"),
        "index_revision_embedding_space",
        ["workspace_id"],
    )
    op.create_index(
        op.f("ix_index_revision_embedding_space_embedding_space_id"),
        "index_revision_embedding_space",
        ["embedding_space_id"],
    )
    op.execute(
        """
        INSERT INTO index_revision_embedding_space (
          workspace_id, index_revision_id, role, embedding_space_id, required,
          retrieval_weight_micros
        )
        SELECT workspace_id, id, 'text_retrieval', embedding_space_id, true, 1000000
        FROM index_revision
        """
    )
    op.execute(
        """
        INSERT INTO index_revision_embedding_space (
          workspace_id, index_revision_id, role, embedding_space_id, required,
          retrieval_weight_micros
        )
        SELECT workspace_id, id, 'semantic_analysis', embedding_space_id, true, NULL
        FROM index_revision
        WHERE chunking_config ->> 'profile' = 'semantic_breakpoint_v1'
        """
    )

    op.create_table(
        "index_asset",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("indexed_document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset_key", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("storage_uri", sa.Text(), nullable=False),
        sa.Column("media_type", sa.String(length=255), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("source_location", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "processing_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("height IS NULL OR height > 0", name=op.f("ck_index_asset_index_asset_height_positive")),
        sa.CheckConstraint("width IS NULL OR width > 0", name=op.f("ck_index_asset_index_asset_width_positive")),
        sa.ForeignKeyConstraint(["document_id", "document_version_id"], ["document_version.document_id", "document_version.id"], name="fk_index_asset_same_document_version"),
        sa.ForeignKeyConstraint(["indexed_document_version_id"], ["indexed_document_version.id"], name=op.f("fk_index_asset_indexed_document_version_id_indexed_document_version"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["kb_id", "document_id"], ["document.kb_id", "document.id"], name="fk_index_asset_same_kb_document"),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_base.id"], name=op.f("fk_index_asset_kb_id_knowledge_base"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], name=op.f("fk_index_asset_workspace_id_workspace"), ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_index_asset")),
        sa.UniqueConstraint("indexed_document_version_id", "asset_key", name=op.f("uq_index_asset_indexed_document_version_id_asset_key")),
        sa.UniqueConstraint("indexed_document_version_id", "id", name="uq_index_asset_target_id"),
    )
    op.create_index(op.f("ix_index_asset_workspace_id"), "index_asset", ["workspace_id"])
    op.create_index(op.f("ix_index_asset_kb_id"), "index_asset", ["kb_id"])
    op.create_index(op.f("ix_index_asset_indexed_document_version_id"), "index_asset", ["indexed_document_version_id"])

    op.create_table(
        "index_artifact_manifest",
        sa.Column("indexed_document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("profile_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("element_sequence_hash", sa.String(length=64), nullable=False),
        sa.Column("asset_manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("unit_plan", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("representation_matrix", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("unit_count", sa.Integer(), nullable=False),
        sa.Column("asset_count", sa.Integer(), nullable=False),
        sa.Column("representation_count", sa.Integer(), nullable=False),
        sa.Column("manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("asset_count >= 0", name=op.f("ck_index_artifact_manifest_artifact_manifest_assets_nonnegative")),
        sa.CheckConstraint("jsonb_typeof(representation_matrix) = 'array'", name=op.f("ck_index_artifact_manifest_artifact_manifest_representation_matrix_array")),
        sa.CheckConstraint("representation_count >= 0", name=op.f("ck_index_artifact_manifest_artifact_manifest_representations_nonnegative")),
        sa.CheckConstraint("jsonb_typeof(unit_plan) = 'array'", name=op.f("ck_index_artifact_manifest_artifact_manifest_unit_plan_array")),
        sa.CheckConstraint("unit_count >= 0", name=op.f("ck_index_artifact_manifest_artifact_manifest_units_nonnegative")),
        sa.ForeignKeyConstraint(["indexed_document_version_id"], ["indexed_document_version.id"], name=op.f("fk_index_artifact_manifest_indexed_document_version_id_indexed_document_version"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("indexed_document_version_id", name=op.f("pk_index_artifact_manifest")),
    )

    op.add_column("index_chunk", sa.Column("unit_key", sa.String(length=255), nullable=True))
    op.add_column("index_chunk", sa.Column("modality", sa.String(length=32), nullable=True))
    op.add_column("index_chunk", sa.Column("index_asset_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("index_chunk", sa.Column("evidence_group_key", sa.String(length=255), nullable=True))
    op.add_column("index_chunk", sa.Column("relations", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False))
    op.execute("UPDATE index_chunk SET unit_key = 'legacy-text:' || ordinal::text, modality = 'text'")
    op.alter_column("index_chunk", "unit_key", nullable=False)
    op.alter_column("index_chunk", "modality", nullable=False)
    op.create_check_constraint("ck_index_chunk_index_chunk_modality_supported", "index_chunk", "modality IN ('text', 'image', 'table')")
    op.create_unique_constraint("uq_index_chunk_indexed_document_version_id_unit_key", "index_chunk", ["indexed_document_version_id", "unit_key"])
    op.create_unique_constraint("uq_index_chunk_target_id", "index_chunk", ["indexed_document_version_id", "id"])
    op.create_foreign_key("fk_index_chunk_same_target_asset", "index_chunk", "index_asset", ["indexed_document_version_id", "index_asset_id"], ["indexed_document_version_id", "id"])

    op.add_column("vector_record_1024", sa.Column("representation_kind", sa.String(length=64), server_default=sa.text("'text'"), nullable=False))
    op.drop_constraint("uq_vector_record_1024_index_chunk_id", "vector_record_1024", type_="unique")
    op.create_unique_constraint("uq_vector_record_1024_chunk_space_representation", "vector_record_1024", ["index_chunk_id", "embedding_space_id", "representation_kind"])

    op.add_column("citation", sa.Column("modality", sa.String(length=32), server_default=sa.text("'text'"), nullable=False))
    op.add_column("citation", sa.Column("asset_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("citation", sa.Column("matched_representations", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False))

    for table in (
        "index_revision_embedding_space",
        "index_asset",
        "index_artifact_manifest",
    ):
        op.execute(f"GRANT SELECT, INSERT, DELETE ON TABLE {table} TO rag_kb_runtime")
    op.execute("GRANT UPDATE ON TABLE index_asset TO rag_kb_runtime")


def downgrade() -> None:
    op.drop_column("citation", "matched_representations")
    op.drop_column("citation", "asset_snapshot")
    op.drop_column("citation", "modality")
    op.drop_constraint("uq_vector_record_1024_chunk_space_representation", "vector_record_1024", type_="unique")
    op.create_unique_constraint("uq_vector_record_1024_index_chunk_id", "vector_record_1024", ["index_chunk_id"])
    op.drop_column("vector_record_1024", "representation_kind")
    op.drop_constraint("fk_index_chunk_same_target_asset", "index_chunk", type_="foreignkey")
    op.drop_constraint("uq_index_chunk_target_id", "index_chunk", type_="unique")
    op.drop_constraint("uq_index_chunk_indexed_document_version_id_unit_key", "index_chunk", type_="unique")
    op.drop_constraint("ck_index_chunk_index_chunk_modality_supported", "index_chunk", type_="check")
    op.drop_column("index_chunk", "relations")
    op.drop_column("index_chunk", "evidence_group_key")
    op.drop_column("index_chunk", "index_asset_id")
    op.drop_column("index_chunk", "modality")
    op.drop_column("index_chunk", "unit_key")
    op.drop_table("index_artifact_manifest")
    op.drop_table("index_asset")
    op.drop_index(op.f("ix_index_revision_embedding_space_embedding_space_id"), table_name="index_revision_embedding_space")
    op.drop_index(op.f("ix_index_revision_embedding_space_workspace_id"), table_name="index_revision_embedding_space")
    op.drop_table("index_revision_embedding_space")
    op.drop_column("index_revision", "representation_config")
    op.drop_column("index_revision", "enrichment_config")
    op.drop_constraint("uq_index_revision_workspace_id", "index_revision", type_="unique")
