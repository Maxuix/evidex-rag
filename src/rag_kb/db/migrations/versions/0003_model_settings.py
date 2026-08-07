"""Add user-managed model providers, profiles, revisions, and defaults.

Revision ID: 0003_model_settings
Revises: 0002_selectable_retrieval
Create Date: 2026-08-07
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0003_model_settings"
down_revision: Union[str, Sequence[str], None] = "0002_selectable_retrieval"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_provider",
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "id", name="uq_model_provider_workspace_id"),
        sa.UniqueConstraint("workspace_id", "name", name="uq_model_provider_workspace_name"),
    )
    op.create_index(op.f("ix_model_provider_workspace_id"), "model_provider", ["workspace_id"])

    op.create_table(
        "model_provider_revision",
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("provider_id", sa.UUID(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("protocol", sa.String(length=64), nullable=False),
        sa.Column("base_url", sa.Text(), nullable=False),
        sa.Column("secret_reference", sa.String(length=64), nullable=False),
        sa.Column("timeout_seconds", sa.Float(), nullable=False),
        sa.Column("max_retries", sa.Integer(), nullable=False),
        sa.Column("max_concurrency", sa.Integer(), nullable=False),
        sa.Column("configuration_fingerprint", sa.String(length=80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("revision > 0", name=op.f("ck_model_provider_revision_model_provider_revision_positive")),
        sa.CheckConstraint("timeout_seconds > 0 AND max_retries >= 0 AND max_concurrency > 0", name=op.f("ck_model_provider_revision_model_provider_revision_limits_valid")),
        sa.CheckConstraint("protocol IN ('openai_compatible','tongyi_multimodal')", name=op.f("ck_model_provider_revision_model_provider_revision_protocol_supported")),
        sa.ForeignKeyConstraint(["workspace_id", "provider_id"], ["model_provider.workspace_id", "model_provider.id"], name="fk_model_provider_revision_same_workspace_provider", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_id", "revision", name="uq_model_provider_revision_number"),
        sa.UniqueConstraint("workspace_id", "id", name="uq_model_provider_revision_workspace_id"),
    )
    op.create_index(op.f("ix_model_provider_revision_provider_id"), "model_provider_revision", ["provider_id"])
    op.create_index(op.f("ix_model_provider_revision_workspace_id"), "model_provider_revision", ["workspace_id"])

    op.create_table(
        "model_profile",
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("provider_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("kind IN ('chat','text_embedding','multimodal_embedding')", name=op.f("ck_model_profile_model_profile_kind_supported")),
        sa.ForeignKeyConstraint(["workspace_id", "provider_id"], ["model_provider.workspace_id", "model_provider.id"], name="fk_model_profile_same_workspace_provider", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "id", name="uq_model_profile_workspace_id"),
        sa.UniqueConstraint("workspace_id", "name", name="uq_model_profile_workspace_name"),
    )
    op.create_index(op.f("ix_model_profile_provider_id"), "model_profile", ["provider_id"])
    op.create_index(op.f("ix_model_profile_workspace_id"), "model_profile", ["workspace_id"])

    op.create_table(
        "model_profile_revision",
        sa.Column("id", sa.UUID(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("profile_id", sa.UUID(), nullable=False),
        sa.Column("provider_revision_id", sa.UUID(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("configuration", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("configuration_fingerprint", sa.String(length=80), nullable=False),
        sa.Column("capability_fingerprint", sa.String(length=80), nullable=False),
        sa.Column("compatibility_fingerprint", sa.String(length=80), nullable=True),
        sa.Column("validation_status", sa.String(length=32), nullable=False),
        sa.Column("validation_error_code", sa.String(length=128), nullable=True),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("jsonb_typeof(configuration) = 'object' AND pg_column_size(configuration) <= 65536", name=op.f("ck_model_profile_revision_model_profile_revision_configuration_object")),
        sa.CheckConstraint("revision > 0", name=op.f("ck_model_profile_revision_model_profile_revision_positive")),
        sa.CheckConstraint("validation_status IN ('unverified','valid','invalid')", name=op.f("ck_model_profile_revision_model_profile_revision_validation_status_supported")),
        sa.ForeignKeyConstraint(["workspace_id", "profile_id"], ["model_profile.workspace_id", "model_profile.id"], name="fk_model_profile_revision_same_workspace_profile", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id", "provider_revision_id"], ["model_provider_revision.workspace_id", "model_provider_revision.id"], name="fk_model_profile_revision_same_workspace_provider_revision", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("profile_id", "revision", name="uq_model_profile_revision_number"),
        sa.UniqueConstraint("workspace_id", "id", name="uq_model_profile_revision_workspace_id"),
    )
    op.create_index(op.f("ix_model_profile_revision_profile_id"), "model_profile_revision", ["profile_id"])
    op.create_index(op.f("ix_model_profile_revision_provider_revision_id"), "model_profile_revision", ["provider_revision_id"])
    op.create_index(op.f("ix_model_profile_revision_workspace_id"), "model_profile_revision", ["workspace_id"])

    op.add_column(
        "embedding_space",
        sa.Column("model_profile_revision_id", sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        "fk_embedding_space_model_profile_revision",
        "embedding_space",
        "model_profile_revision",
        ["model_profile_revision_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        op.f("ix_embedding_space_model_profile_revision_id"),
        "embedding_space",
        ["model_profile_revision_id"],
    )

    op.create_table(
        "model_selection",
        sa.Column("workspace_id", sa.UUID(), nullable=False),
        sa.Column("chat_profile_revision_id", sa.UUID(), nullable=True),
        sa.Column("text_embedding_profile_revision_id", sa.UUID(), nullable=True),
        sa.Column("multimodal_embedding_profile_revision_id", sa.UUID(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id", "chat_profile_revision_id"], ["model_profile_revision.workspace_id", "model_profile_revision.id"], name="fk_model_selection_same_workspace_chat_profile", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["workspace_id", "multimodal_embedding_profile_revision_id"], ["model_profile_revision.workspace_id", "model_profile_revision.id"], name="fk_model_selection_same_workspace_multimodal_embedding_profile", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["workspace_id", "text_embedding_profile_revision_id"], ["model_profile_revision.workspace_id", "model_profile_revision.id"], name="fk_model_selection_same_workspace_text_embedding_profile", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("workspace_id"),
    )


def downgrade() -> None:
    op.drop_table("model_selection")
    op.drop_index(op.f("ix_embedding_space_model_profile_revision_id"), table_name="embedding_space")
    op.drop_constraint("fk_embedding_space_model_profile_revision", "embedding_space", type_="foreignkey")
    op.drop_column("embedding_space", "model_profile_revision_id")
    op.drop_index(op.f("ix_model_profile_revision_workspace_id"), table_name="model_profile_revision")
    op.drop_index(op.f("ix_model_profile_revision_provider_revision_id"), table_name="model_profile_revision")
    op.drop_index(op.f("ix_model_profile_revision_profile_id"), table_name="model_profile_revision")
    op.drop_table("model_profile_revision")
    op.drop_index(op.f("ix_model_profile_workspace_id"), table_name="model_profile")
    op.drop_index(op.f("ix_model_profile_provider_id"), table_name="model_profile")
    op.drop_table("model_profile")
    op.drop_index(op.f("ix_model_provider_revision_workspace_id"), table_name="model_provider_revision")
    op.drop_index(op.f("ix_model_provider_revision_provider_id"), table_name="model_provider_revision")
    op.drop_table("model_provider_revision")
    op.drop_index(op.f("ix_model_provider_workspace_id"), table_name="model_provider")
    op.drop_table("model_provider")
