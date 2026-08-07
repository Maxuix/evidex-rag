"""Add flexible embedding dimensions and one variable-dimension vector table.

Revision ID: 0004_flexible_embedding_spaces
Revises: 0003_model_settings
Create Date: 2026-08-07
"""

from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0004_flexible_embedding_spaces"
down_revision: Union[str, Sequence[str], None] = "0003_model_settings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_profile_revision",
        sa.Column(
            "validation_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.create_check_constraint(
        "model_profile_revision_validation_snapshot_object",
        "model_profile_revision",
        "validation_snapshot IS NULL OR "
        "(jsonb_typeof(validation_snapshot) = 'object' "
        "AND pg_column_size(validation_snapshot) <= 65536)",
    )
    op.execute(
        """
        UPDATE model_profile_revision revision
           SET validation_snapshot = jsonb_build_object(
               'schema_version', 'embedding_validation_v1',
               'provider_supported_dimensions', NULL,
               'verified_dimensions', jsonb_build_array(
                   (revision.configuration ->> 'dimension')::integer
               ),
               'provider_default_dimension', NULL,
               'recommended_dimension', NULL,
               'selected_dimension',
                   (revision.configuration ->> 'dimension')::integer,
               'selection_source', 'legacy_explicit',
               'dimension_request_mode', 'explicit',
               'input_capabilities', CASE profile.kind
                   WHEN 'multimodal_embedding'
                       THEN '["text_document","text_query","image"]'::jsonb
                   ELSE '["text_document","text_query"]'::jsonb
               END,
               'shared_text_image_space_confirmed', false,
               'distance_metric', 'cosine',
               'vector_data_type', 'float32',
               'normalization', 'l2'
           )
          FROM model_profile profile
         WHERE profile.id = revision.profile_id
           AND profile.kind IN ('text_embedding', 'multimodal_embedding')
           AND revision.validation_status = 'valid'
           AND jsonb_typeof(revision.configuration -> 'dimension') = 'number'
        """
    )

    op.drop_constraint(
        op.f("ck_embedding_space_embedding_space_dimension_positive"),
        "embedding_space",
        type_="check",
    )
    op.create_check_constraint(
        "embedding_space_dimension_supported",
        "embedding_space",
        "dimension BETWEEN 64 AND 4096",
    )
    op.create_unique_constraint(
        "uq_embedding_space_workspace_id_dimension",
        "embedding_space",
        ["workspace_id", "id", "dimension"],
    )

    op.create_table(
        "vector_record",
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("kb_id", sa.UUID(), nullable=False),
        sa.Column("index_chunk_id", sa.UUID(), nullable=False),
        sa.Column("embedding_space_id", sa.UUID(), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("representation_kind", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "embedding_dimension BETWEEN 64 AND 4096",
            name=op.f("ck_vector_record_vector_record_dimension_supported"),
        ),
        sa.CheckConstraint(
            "vector_dims(embedding) = embedding_dimension",
            name=op.f("ck_vector_record_vector_record_dimension_matches_value"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "embedding_space_id", "embedding_dimension"],
            ["embedding_space.workspace_id", "embedding_space.id", "embedding_space.dimension"],
            name="fk_vector_record_same_workspace_embedding_dimension",
        ),
        sa.ForeignKeyConstraint(
            ["kb_id", "index_chunk_id"],
            ["index_chunk.kb_id", "index_chunk.id"],
            name="fk_vector_record_same_kb_chunk",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspace.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["kb_id"], ["knowledge_base.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "index_chunk_id",
            "embedding_space_id",
            "representation_kind",
            name="uq_vector_record_chunk_space_representation",
        ),
    )
    op.create_index(
        "ix_vector_record_space_representation",
        "vector_record",
        ["embedding_space_id", "representation_kind"],
    )
    op.create_index(op.f("ix_vector_record_kb_id"), "vector_record", ["kb_id"])
    op.create_index(
        op.f("ix_vector_record_workspace_id"), "vector_record", ["workspace_id"]
    )

    op.execute(
        """
        INSERT INTO vector_record (
            id, workspace_id, kb_id, index_chunk_id, embedding_space_id,
            embedding_dimension, representation_kind, embedding, created_at
        )
        SELECT id, workspace_id, kb_id, index_chunk_id, embedding_space_id,
               1024, representation_kind, embedding, created_at
          FROM vector_record_1024
        UNION ALL
        SELECT id, workspace_id, kb_id, index_chunk_id, embedding_space_id,
               768, representation_kind, embedding, created_at
          FROM vector_record_768
        """
    )
    op.execute(
        """
        DO $$
        DECLARE
            expected_count bigint;
            migrated_count bigint;
            invalid_count bigint;
        BEGIN
            SELECT (SELECT count(*) FROM vector_record_1024)
                 + (SELECT count(*) FROM vector_record_768)
              INTO expected_count;
            SELECT count(*) INTO migrated_count FROM vector_record;
            IF migrated_count <> expected_count THEN
                RAISE EXCEPTION 'vector migration count mismatch: expected %, observed %',
                    expected_count, migrated_count;
            END IF;
            SELECT count(*) INTO invalid_count
              FROM vector_record vector
              JOIN embedding_space space
                ON space.workspace_id = vector.workspace_id
               AND space.id = vector.embedding_space_id
             WHERE vector_dims(vector.embedding) <> vector.embedding_dimension
                OR vector.embedding_dimension <> space.dimension;
            IF invalid_count <> 0 THEN
                RAISE EXCEPTION 'vector migration dimension mismatch: % invalid rows',
                    invalid_count;
            END IF;
        END $$
        """
    )
    op.drop_table("vector_record_768")
    op.drop_table("vector_record_1024")


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM vector_record
                 WHERE embedding_dimension NOT IN (768, 1024)
            ) THEN
                RAISE EXCEPTION
                    'cannot downgrade: vector_record contains dimensions other than 768/1024';
            END IF;
            IF EXISTS (
                SELECT 1
                  FROM model_profile_revision revision
                  JOIN model_profile profile ON profile.id = revision.profile_id
                 WHERE profile.kind IN ('text_embedding', 'multimodal_embedding')
                   AND (
                       jsonb_typeof(revision.configuration -> 'dimension')
                           IS DISTINCT FROM 'number'
                       OR revision.configuration ->> 'dimension'
                           NOT IN ('768', '1024')
                       OR NOT revision.configuration @> jsonb_build_object(
                           'distance_metric', 'cosine',
                           'vector_data_type', 'float32',
                           'normalization', 'l2'
                       )
                   )
            ) THEN
                RAISE EXCEPTION
                    'cannot downgrade: model profiles contain flexible embedding facts';
            END IF;
            IF EXISTS (
                SELECT 1 FROM embedding_space
                 WHERE dimension NOT IN (768, 1024)
                    OR normalization <> 'l2'
            ) THEN
                RAISE EXCEPTION
                    'cannot downgrade: embedding spaces require the flexible schema';
            END IF;
        END $$
        """
    )
    _create_legacy_vector_table("vector_record_1024", 1024, legacy=False)
    _create_legacy_vector_table("vector_record_768", 768, legacy=True)
    op.execute(
        """
        INSERT INTO vector_record_1024 (
            id, workspace_id, kb_id, index_chunk_id, embedding_space_id,
            representation_kind, embedding, created_at
        )
        SELECT id, workspace_id, kb_id, index_chunk_id, embedding_space_id,
               representation_kind, embedding::vector(1024), created_at
          FROM vector_record WHERE embedding_dimension = 1024
        """
    )
    op.execute(
        """
        INSERT INTO vector_record_768 (
            id, workspace_id, kb_id, index_chunk_id, embedding_space_id,
            representation_kind, embedding, created_at
        )
        SELECT id, workspace_id, kb_id, index_chunk_id, embedding_space_id,
               representation_kind, embedding::vector(768), created_at
          FROM vector_record WHERE embedding_dimension = 768
        """
    )
    op.drop_table("vector_record")

    op.drop_constraint(
        "uq_embedding_space_workspace_id_dimension",
        "embedding_space",
        type_="unique",
    )
    op.drop_constraint(
        op.f("ck_embedding_space_embedding_space_dimension_supported"),
        "embedding_space",
        type_="check",
    )
    op.create_check_constraint(
        "embedding_space_dimension_positive", "embedding_space", "dimension > 0"
    )
    op.drop_constraint(
        op.f(
            "ck_model_profile_revision_model_profile_revision_validation_snapshot_object"
        ),
        "model_profile_revision",
        type_="check",
    )
    op.drop_column("model_profile_revision", "validation_snapshot")


def _create_legacy_vector_table(name: str, dimension: int, *, legacy: bool) -> None:
    suffix = "_768" if legacy else ""
    unique_name = (
        "uq_vector_record_768_chunk_space_representation"
        if legacy
        else "uq_vector_record_1024_chunk_space_representation"
    )
    op.create_table(
        name,
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("kb_id", sa.UUID(), nullable=False),
        sa.Column("index_chunk_id", sa.UUID(), nullable=False),
        sa.Column("embedding_space_id", sa.UUID(), nullable=False),
        sa.Column("representation_kind", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(dimension), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["kb_id", "index_chunk_id"],
            ["index_chunk.kb_id", "index_chunk.id"],
            name=f"fk_vector_record{suffix}_same_kb_chunk",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "embedding_space_id"],
            ["embedding_space.workspace_id", "embedding_space.id"],
            name=f"fk_vector_record{suffix}_same_workspace_embedding",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspace.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["kb_id"], ["knowledge_base.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "index_chunk_id",
            "embedding_space_id",
            "representation_kind",
            name=unique_name,
        ),
    )
    op.create_index(op.f(f"ix_{name}_workspace_id"), name, ["workspace_id"])
    op.create_index(op.f(f"ix_{name}_kb_id"), name, ["kb_id"])
    op.create_index(
        op.f(f"ix_{name}_embedding_space_id"), name, ["embedding_space_id"]
    )
