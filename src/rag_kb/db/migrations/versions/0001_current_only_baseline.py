"""Create the one current-only application schema from an empty database.

Revision ID: 0001_current_only_baseline
Revises:
Create Date: 2026-07-29
"""

from typing import Sequence, Union

from alembic import op
import pgvector.sqlalchemy.vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0001_current_only_baseline'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table('workspace',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_workspace')),
    sa.UniqueConstraint('name', name=op.f('uq_workspace_name'))
    )
    op.create_table('embedding_space',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('provider_identity', sa.String(length=255), nullable=False),
    sa.Column('endpoint_identity', sa.String(length=255), nullable=False),
    sa.Column('requested_model', sa.String(length=255), nullable=False),
    sa.Column('resolved_model', sa.String(length=255), nullable=False),
    sa.Column('model_version', sa.String(length=255), nullable=False),
    sa.Column('deployment_revision', sa.String(length=255), nullable=True),
    sa.Column('dimension', sa.Integer(), nullable=False),
    sa.Column('distance_metric', sa.String(length=32), nullable=False),
    sa.Column('vector_data_type', sa.String(length=32), nullable=False),
    sa.Column('normalization', sa.String(length=32), nullable=False),
    sa.Column('configuration_fingerprint', sa.String(length=80), nullable=False),
    sa.Column('tokenizer_fingerprint', sa.String(length=80), nullable=True),
    sa.Column('compatibility_fingerprint', sa.String(length=80), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('dimension > 0', name=op.f('ck_embedding_space_embedding_space_dimension_positive')),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_embedding_space_workspace_id_workspace'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_embedding_space')),
    sa.UniqueConstraint('compatibility_fingerprint', name=op.f('uq_embedding_space_compatibility_fingerprint')),
    sa.UniqueConstraint('workspace_id', 'id', name='uq_embedding_space_workspace_id')
    )
    op.create_index(op.f('ix_embedding_space_workspace_id'), 'embedding_space', ['workspace_id'], unique=False)
    op.create_table('knowledge_base',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('source_change_seq', sa.BigInteger(), server_default=sa.text('0'), nullable=False),
    sa.Column('active_index_revision_id', sa.UUID(), nullable=True),
    sa.Column('provisioned_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('retrieval_defaults', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('answer_policy_defaults', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{\"answer_style\": \"concise\", \"insufficiency_policy\": \"refuse\"}'::jsonb"), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('source_change_seq >= 0', name=op.f('ck_knowledge_base_knowledge_base_source_change_seq_nonnegative')),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_knowledge_base_workspace_id_workspace'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_knowledge_base')),
    sa.UniqueConstraint('workspace_id', 'id', name='uq_knowledge_base_workspace_id'),
    sa.UniqueConstraint('workspace_id', 'name', name='uq_knowledge_base_workspace_name')
    )
    op.create_index(op.f('ix_knowledge_base_workspace_id'), 'knowledge_base', ['workspace_id'], unique=False)
    op.create_table('chat_session',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('kb_id', sa.UUID(), nullable=False),
    sa.Column('principal_id', sa.String(length=255), nullable=False),
    sa.Column('title', sa.String(length=512), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['workspace_id', 'kb_id'], ['knowledge_base.workspace_id', 'knowledge_base.id'], name='fk_chat_session_same_workspace_kb'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_chat_session')),
    sa.UniqueConstraint('workspace_id', 'id', name='uq_chat_session_workspace_id')
    )
    op.create_table('document',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('kb_id', sa.UUID(), nullable=False),
    sa.Column('current_version_id', sa.UUID(), nullable=True),
    sa.Column('display_name', sa.String(length=512), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_document_kb_id_knowledge_base'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_document_workspace_id_workspace'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_document')),
    sa.UniqueConstraint('kb_id', 'id', name='uq_document_kb_id')
    )
    op.create_index(op.f('ix_document_kb_id'), 'document', ['kb_id'], unique=False)
    op.create_index(op.f('ix_document_workspace_id'), 'document', ['workspace_id'], unique=False)
    op.create_table('index_revision',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('kb_id', sa.UUID(), nullable=False),
    sa.Column('embedding_space_id', sa.UUID(), nullable=False),
    sa.Column('status', sa.Enum('building', 'ready', 'active', 'retired', 'failed', name='index_revision_status'), nullable=False),
    sa.Column('source_snapshot_seq', sa.BigInteger(), nullable=False),
    sa.Column('parser_config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('chunking_config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('enrichment_config', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('representation_config', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('error_code', sa.String(length=128), nullable=True),
    sa.Column('error_detail', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('source_snapshot_seq >= 0', name=op.f('ck_index_revision_index_revision_snapshot_nonnegative')),
    sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_index_revision_kb_id_knowledge_base'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['workspace_id', 'embedding_space_id'], ['embedding_space.workspace_id', 'embedding_space.id'], name='fk_index_revision_same_workspace_embedding'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_index_revision_workspace_id_workspace'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_index_revision')),
    sa.UniqueConstraint('kb_id', 'id', name='uq_index_revision_kb_id_id'),
    sa.UniqueConstraint('workspace_id', 'id', name='uq_index_revision_workspace_id')
    )
    op.create_index(op.f('ix_index_revision_kb_id'), 'index_revision', ['kb_id'], unique=False)
    op.create_index(op.f('ix_index_revision_workspace_id'), 'index_revision', ['workspace_id'], unique=False)
    op.create_index('uq_one_active_revision_per_kb', 'index_revision', ['kb_id'], unique=True, postgresql_where=sa.text("status = 'active'"))
    op.create_table('chat_message',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('session_id', sa.UUID(), nullable=False),
    sa.Column('chat_run_id', sa.UUID(), nullable=True),
    sa.Column('role', sa.Enum('user', 'assistant', name='chat_message_role'), nullable=False),
    sa.Column('assistant_status', sa.Enum('generating', 'completed', 'failed', name='assistant_message_status'), nullable=True),
    sa.Column('client_request_id', sa.UUID(), nullable=True),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("(role = 'user' AND assistant_status IS NULL) OR (role = 'assistant' AND assistant_status IS NOT NULL)", name=op.f('ck_chat_message_chat_message_status_matches_role')),
    sa.ForeignKeyConstraint(['workspace_id', 'session_id'], ['chat_session.workspace_id', 'chat_session.id'], name='fk_chat_message_same_workspace_session'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_chat_message')),
    sa.UniqueConstraint('session_id', 'client_request_id', name=op.f('uq_chat_message_session_id_client_request_id'))
    )
    op.create_index('uq_assistant_message_chat_run', 'chat_message', ['chat_run_id'], unique=True, postgresql_where=sa.text("role = 'assistant' AND chat_run_id IS NOT NULL"))
    op.create_table('document_version',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('kb_id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('version_number', sa.Integer(), nullable=False),
    sa.Column('source_status', sa.Enum('available', 'unavailable', 'deleted', name='document_source_status'), nullable=False),
    sa.Column('checksum_sha256', sa.String(length=64), nullable=False),
    sa.Column('storage_uri', sa.Text(), nullable=False),
    sa.Column('original_filename', sa.String(length=1024), nullable=False),
    sa.Column('media_type', sa.String(length=255), nullable=False),
    sa.Column('size_bytes', sa.BigInteger(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('size_bytes >= 0', name=op.f('ck_document_version_document_version_size_nonnegative')),
    sa.CheckConstraint('version_number > 0', name=op.f('ck_document_version_document_version_number_positive')),
    sa.ForeignKeyConstraint(['kb_id', 'document_id'], ['document.kb_id', 'document.id'], name='fk_document_version_same_kb'),
    sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_document_version_kb_id_knowledge_base'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_document_version_workspace_id_workspace'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_document_version')),
    sa.UniqueConstraint('document_id', 'id', name='uq_document_version_document_id'),
    sa.UniqueConstraint('document_id', 'version_number', name=op.f('uq_document_version_document_id_version_number')),
    sa.UniqueConstraint('kb_id', 'id', name='uq_document_version_kb_id')
    )
    op.create_index(op.f('ix_document_version_kb_id'), 'document_version', ['kb_id'], unique=False)
    op.create_index(op.f('ix_document_version_workspace_id'), 'document_version', ['workspace_id'], unique=False)
    op.create_table('chat_run',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('kb_id', sa.UUID(), nullable=False),
    sa.Column('session_id', sa.UUID(), nullable=False),
    sa.Column('user_message_id', sa.UUID(), nullable=False),
    sa.Column('index_revision_id', sa.UUID(), nullable=False),
    sa.Column('status', sa.Enum('queued', 'running', 'completed', 'failed', 'cancelled', name='chat_run_status'), nullable=False),
    sa.Column('principal_id', sa.String(length=255), nullable=False),
    sa.Column('client_id', sa.String(length=255), nullable=False),
    sa.Column('endpoint', sa.String(length=255), nullable=False),
    sa.Column('idempotency_key', sa.UUID(), nullable=False),
    sa.Column('request_hash', sa.String(length=71), nullable=False),
    sa.Column('requested_policy', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('effective_policy', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('retrieval_strategy', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('model_configuration', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('conversation_context', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('contextualized_query', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('attempt', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('claimed_by', sa.String(length=255), nullable=True),
    sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('next_attempt_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_code', sa.String(length=128), nullable=True),
    sa.Column('error_detail', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('error_retryable', sa.Boolean(), nullable=True),
    sa.Column('usage', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('timing', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('final_llm_context', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint('attempt >= 0', name=op.f('ck_chat_run_chat_run_attempt_nonnegative')),
    sa.ForeignKeyConstraint(['index_revision_id'], ['index_revision.id'], name=op.f('fk_chat_run_index_revision_id_index_revision'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_chat_run_kb_id_knowledge_base'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_message_id'], ['chat_message.id'], name=op.f('fk_chat_run_user_message_id_chat_message'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['workspace_id', 'session_id'], ['chat_session.workspace_id', 'chat_session.id'], name='fk_chat_run_same_workspace_session'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_chat_run')),
    sa.UniqueConstraint('principal_id', 'client_id', 'endpoint', 'idempotency_key', name='uq_chat_run_idempotency_scope')
    )
    op.create_index('ix_chat_run_claim', 'chat_run', ['status', 'next_attempt_at', 'created_at'], unique=False)
    op.create_index(op.f('ix_chat_run_kb_id'), 'chat_run', ['kb_id'], unique=False)
    op.create_index('uq_chat_run_session_nonterminal', 'chat_run', ['session_id'], unique=True, postgresql_where=sa.text("status IN ('queued', 'running')"))
    op.create_table('indexed_document_version',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('kb_id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('document_version_id', sa.UUID(), nullable=False),
    sa.Column('index_revision_id', sa.UUID(), nullable=False),
    sa.Column('source_change_seq', sa.BigInteger(), nullable=False),
    sa.Column('build_status', sa.Enum('queued', 'processing', 'ready', 'failed', name='index_build_status'), nullable=False),
    sa.Column('serving_status', sa.Enum('candidate', 'serving', 'retired', name='index_serving_status'), nullable=False),
    sa.Column('error_code', sa.String(length=128), nullable=True),
    sa.Column('error_detail', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("serving_status <> 'serving' OR build_status = 'ready'", name='ck_serving_requires_ready'),
    sa.CheckConstraint('source_change_seq > 0', name=op.f('ck_indexed_document_version_indexed_version_seq_positive')),
    sa.ForeignKeyConstraint(['document_id', 'document_version_id'], ['document_version.document_id', 'document_version.id'], name='fk_indexed_version_same_document_version'),
    sa.ForeignKeyConstraint(['kb_id', 'document_id'], ['document.kb_id', 'document.id'], name='fk_indexed_version_same_kb_document'),
    sa.ForeignKeyConstraint(['kb_id', 'document_version_id'], ['document_version.kb_id', 'document_version.id'], name='fk_indexed_version_same_kb_version'),
    sa.ForeignKeyConstraint(['kb_id', 'index_revision_id'], ['index_revision.kb_id', 'index_revision.id'], name='fk_indexed_version_same_kb_revision'),
    sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_indexed_document_version_kb_id_knowledge_base'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_indexed_document_version_workspace_id_workspace'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_indexed_document_version')),
    sa.UniqueConstraint('document_version_id', 'index_revision_id', name=op.f('uq_indexed_document_version_document_version_id_index_revision_id')),
    sa.UniqueConstraint('workspace_id', 'kb_id', 'id', name='uq_indexed_version_workspace_kb_id')
    )
    op.create_index(op.f('ix_indexed_document_version_kb_id'), 'indexed_document_version', ['kb_id'], unique=False)
    op.create_index(op.f('ix_indexed_document_version_workspace_id'), 'indexed_document_version', ['workspace_id'], unique=False)
    op.create_index('uq_one_serving_version_per_document_revision', 'indexed_document_version', ['document_id', 'index_revision_id'], unique=True, postgresql_where=sa.text("build_status = 'ready' AND serving_status = 'serving'"))
    op.create_table('source_change',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('kb_id', sa.UUID(), nullable=False),
    sa.Column('source_change_seq', sa.BigInteger(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('document_version_id', sa.UUID(), nullable=True),
    sa.Column('change_kind', sa.Enum('upsert', 'delete', name='source_change_kind'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("(change_kind = 'delete' AND document_version_id IS NULL) OR (change_kind = 'upsert' AND document_version_id IS NOT NULL)", name=op.f('ck_source_change_source_change_version_matches_kind')),
    sa.CheckConstraint('source_change_seq > 0', name=op.f('ck_source_change_source_change_seq_positive')),
    sa.ForeignKeyConstraint(['kb_id', 'document_id'], ['document.kb_id', 'document.id'], name='fk_source_change_same_kb_document'),
    sa.ForeignKeyConstraint(['kb_id', 'document_version_id'], ['document_version.kb_id', 'document_version.id'], name='fk_source_change_same_kb_version'),
    sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_source_change_kb_id_knowledge_base'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_source_change_workspace_id_workspace'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_source_change')),
    sa.UniqueConstraint('kb_id', 'source_change_seq', name='uq_source_change_sequence')
    )
    op.create_index(op.f('ix_source_change_workspace_id'), 'source_change', ['workspace_id'], unique=False)
    op.create_table(
        'source_file_cleanup',
        sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('document_version_id', sa.UUID(), nullable=False),
        sa.Column('storage_uri', sa.Text(), nullable=False),
        sa.Column('reason', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=32), server_default=sa.text("'pending'"), nullable=False),
        sa.Column('attempt_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('next_attempt_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_error_code', sa.String(length=128), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('attempt_count >= 0', name=op.f('ck_source_file_cleanup_source_file_cleanup_attempt_nonnegative')),
        sa.CheckConstraint("status IN ('pending', 'completed', 'failed')", name=op.f('ck_source_file_cleanup_source_file_cleanup_status_supported')),
        sa.ForeignKeyConstraint(['document_version_id'], ['document_version.id'], name=op.f('fk_source_file_cleanup_document_version_id_document_version'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_source_file_cleanup_workspace_id_workspace'), ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_source_file_cleanup')),
        sa.UniqueConstraint('document_version_id', name=op.f('uq_source_file_cleanup_document_version_id')),
    )
    op.create_index('ix_source_file_cleanup_due', 'source_file_cleanup', ['workspace_id', 'status', 'next_attempt_at'], unique=False)
    op.create_index(op.f('ix_source_file_cleanup_workspace_id'), 'source_file_cleanup', ['workspace_id'], unique=False)
    op.create_table(
        'index_revision_embedding_space',
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('index_revision_id', sa.UUID(), nullable=False),
        sa.Column('role', sa.String(length=64), nullable=False),
        sa.Column('embedding_space_id', sa.UUID(), nullable=False),
        sa.Column('required', sa.Boolean(), nullable=False),
        sa.Column('retrieval_weight_micros', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("role IN ('text_retrieval', 'semantic_analysis', 'cross_modal_retrieval')", name=op.f('ck_index_revision_embedding_space_revision_space_role_supported')),
        sa.CheckConstraint('retrieval_weight_micros IS NULL OR retrieval_weight_micros > 0', name=op.f('ck_index_revision_embedding_space_revision_space_weight_positive')),
        sa.ForeignKeyConstraint(['workspace_id', 'embedding_space_id'], ['embedding_space.workspace_id', 'embedding_space.id'], name='fk_revision_space_same_workspace_embedding'),
        sa.ForeignKeyConstraint(['workspace_id', 'index_revision_id'], ['index_revision.workspace_id', 'index_revision.id'], name='fk_revision_space_same_workspace_revision', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_index_revision_embedding_space_workspace_id_workspace'), ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('index_revision_id', 'role', name=op.f('pk_index_revision_embedding_space')),
    )
    op.create_index(op.f('ix_index_revision_embedding_space_workspace_id'), 'index_revision_embedding_space', ['workspace_id'])
    op.create_index(op.f('ix_index_revision_embedding_space_embedding_space_id'), 'index_revision_embedding_space', ['embedding_space_id'])
    op.create_table(
        'index_chunk_plan',
        sa.Column('indexed_document_version_id', sa.UUID(), nullable=False),
        sa.Column('source_checksum_sha256', sa.String(length=64), nullable=False),
        sa.Column('profile_fingerprint', sa.String(length=64), nullable=False),
        sa.Column('unit_sequence_hash', sa.String(length=64), nullable=False),
        sa.Column('unit_count', sa.Integer(), nullable=False),
        sa.Column('chunk_count', sa.Integer(), nullable=False),
        sa.Column('boundaries', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('plan_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("jsonb_typeof(boundaries) = 'array'", name=op.f('ck_index_chunk_plan_index_chunk_plan_boundaries_array')),
        sa.CheckConstraint('jsonb_array_length(boundaries) = chunk_count - 1', name=op.f('ck_index_chunk_plan_index_chunk_plan_boundary_count')),
        sa.CheckConstraint('chunk_count > 0', name=op.f('ck_index_chunk_plan_index_chunk_plan_chunk_count_positive')),
        sa.CheckConstraint('unit_count > 0', name=op.f('ck_index_chunk_plan_index_chunk_plan_unit_count_positive')),
        sa.ForeignKeyConstraint(['indexed_document_version_id'], ['indexed_document_version.id'], name=op.f('fk_index_chunk_plan_indexed_document_version_id_indexed_document_version'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('indexed_document_version_id', name=op.f('pk_index_chunk_plan')),
    )
    op.create_table(
        'index_asset',
        sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('kb_id', sa.UUID(), nullable=False),
        sa.Column('document_id', sa.UUID(), nullable=False),
        sa.Column('document_version_id', sa.UUID(), nullable=False),
        sa.Column('indexed_document_version_id', sa.UUID(), nullable=False),
        sa.Column('asset_key', sa.String(length=128), nullable=False),
        sa.Column('kind', sa.String(length=64), nullable=False),
        sa.Column('storage_uri', sa.Text(), nullable=False),
        sa.Column('media_type', sa.String(length=255), nullable=False),
        sa.Column('checksum_sha256', sa.String(length=64), nullable=False),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('source_location', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('processing_metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('height IS NULL OR height > 0', name=op.f('ck_index_asset_index_asset_height_positive')),
        sa.CheckConstraint('width IS NULL OR width > 0', name=op.f('ck_index_asset_index_asset_width_positive')),
        sa.ForeignKeyConstraint(['document_id', 'document_version_id'], ['document_version.document_id', 'document_version.id'], name='fk_index_asset_same_document_version'),
        sa.ForeignKeyConstraint(['indexed_document_version_id'], ['indexed_document_version.id'], name=op.f('fk_index_asset_indexed_document_version_id_indexed_document_version'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['kb_id', 'document_id'], ['document.kb_id', 'document.id'], name='fk_index_asset_same_kb_document'),
        sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_index_asset_kb_id_knowledge_base'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_index_asset_workspace_id_workspace'), ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_index_asset')),
        sa.UniqueConstraint('indexed_document_version_id', 'asset_key', name=op.f('uq_index_asset_indexed_document_version_id_asset_key')),
        sa.UniqueConstraint('indexed_document_version_id', 'id', name='uq_index_asset_target_id'),
    )
    op.create_index(op.f('ix_index_asset_workspace_id'), 'index_asset', ['workspace_id'])
    op.create_index(op.f('ix_index_asset_kb_id'), 'index_asset', ['kb_id'])
    op.create_index(op.f('ix_index_asset_indexed_document_version_id'), 'index_asset', ['indexed_document_version_id'])
    op.create_table(
        'index_artifact_manifest',
        sa.Column('indexed_document_version_id', sa.UUID(), nullable=False),
        sa.Column('source_checksum_sha256', sa.String(length=64), nullable=False),
        sa.Column('profile_fingerprint', sa.String(length=64), nullable=False),
        sa.Column('element_sequence_hash', sa.String(length=64), nullable=False),
        sa.Column('asset_manifest_hash', sa.String(length=64), nullable=False),
        sa.Column('unit_plan', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('representation_matrix', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('unit_count', sa.Integer(), nullable=False),
        sa.Column('asset_count', sa.Integer(), nullable=False),
        sa.Column('representation_count', sa.Integer(), nullable=False),
        sa.Column('relation_plan', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('relation_count', sa.Integer(), nullable=False),
        sa.Column('relation_manifest_hash', sa.String(length=64), nullable=False),
        sa.Column('manifest_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('asset_count >= 0', name=op.f('ck_index_artifact_manifest_artifact_manifest_assets_nonnegative')),
        sa.CheckConstraint("jsonb_typeof(representation_matrix) = 'array'", name=op.f('ck_index_artifact_manifest_artifact_manifest_representation_matrix_array')),
        sa.CheckConstraint('representation_count >= 0', name=op.f('ck_index_artifact_manifest_artifact_manifest_representations_nonnegative')),
        sa.CheckConstraint("jsonb_typeof(unit_plan) = 'array'", name=op.f('ck_index_artifact_manifest_artifact_manifest_unit_plan_array')),
        sa.CheckConstraint('unit_count >= 0', name=op.f('ck_index_artifact_manifest_artifact_manifest_units_nonnegative')),
        sa.CheckConstraint("jsonb_typeof(relation_plan) = 'array' AND relation_count >= 0 AND jsonb_array_length(relation_plan) = relation_count", name=op.f('ck_index_artifact_manifest_artifact_manifest_relation_plan_consistent')),
        sa.ForeignKeyConstraint(['indexed_document_version_id'], ['indexed_document_version.id'], name=op.f('fk_index_artifact_manifest_indexed_document_version_id_indexed_document_version'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('indexed_document_version_id', name=op.f('pk_index_artifact_manifest')),
    )
    op.create_table('index_chunk',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('kb_id', sa.UUID(), nullable=False),
    sa.Column('indexed_document_version_id', sa.UUID(), nullable=False),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('unit_key', sa.String(length=255), nullable=False),
    sa.Column('modality', sa.String(length=32), nullable=False),
    sa.Column('index_asset_id', sa.UUID(), nullable=True),
    sa.Column('evidence_group_key', sa.String(length=255), nullable=True),
    sa.Column('relations', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('embedding_text', sa.Text(), nullable=True),
    sa.Column('embedding_text_hash', sa.String(length=64), nullable=True),
    sa.Column('token_count', sa.Integer(), nullable=False),
    sa.Column('source_location', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('hierarchy', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('source_metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('ordinal >= 0', name=op.f('ck_index_chunk_index_chunk_ordinal_nonnegative')),
    sa.CheckConstraint("length(btrim(unit_key)) > 0", name=op.f('ck_index_chunk_index_chunk_unit_key_nonempty')),
    sa.CheckConstraint('token_count >= 0', name=op.f('ck_index_chunk_index_chunk_tokens_nonnegative')),
    sa.CheckConstraint("modality IN ('text', 'image', 'table')", name=op.f('ck_index_chunk_index_chunk_modality_supported')),
    sa.CheckConstraint("(embedding_text IS NULL) = (embedding_text_hash IS NULL)", name=op.f('ck_index_chunk_index_chunk_embedding_text_pair')),
    sa.ForeignKeyConstraint(['indexed_document_version_id', 'index_asset_id'], ['index_asset.indexed_document_version_id', 'index_asset.id'], name='fk_index_chunk_same_target_asset'),
    sa.ForeignKeyConstraint(['indexed_document_version_id'], ['indexed_document_version.id'], name=op.f('fk_index_chunk_indexed_document_version_id_indexed_document_version'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_index_chunk_kb_id_knowledge_base'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_index_chunk_workspace_id_workspace'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_index_chunk')),
    sa.UniqueConstraint('indexed_document_version_id', 'ordinal', name=op.f('uq_index_chunk_indexed_document_version_id_ordinal')),
    sa.UniqueConstraint('indexed_document_version_id', 'unit_key', name=op.f('uq_index_chunk_indexed_document_version_id_unit_key')),
    sa.UniqueConstraint('indexed_document_version_id', 'id', name='uq_index_chunk_target_id'),
    sa.UniqueConstraint('kb_id', 'id', name='uq_index_chunk_kb_id')
    )
    op.create_index(op.f('ix_index_chunk_indexed_document_version_id'), 'index_chunk', ['indexed_document_version_id'], unique=False)
    op.create_index(op.f('ix_index_chunk_kb_id'), 'index_chunk', ['kb_id'], unique=False)
    op.create_index(op.f('ix_index_chunk_workspace_id'), 'index_chunk', ['workspace_id'], unique=False)
    op.create_table('indexing_job',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('kb_id', sa.UUID(), nullable=False),
    sa.Column('indexed_document_version_id', sa.UUID(), nullable=False),
    sa.Column('status', sa.Enum('queued', 'running', 'completed', 'failed', 'cancelled', name='job_status'), nullable=False),
    sa.Column('phase', sa.String(length=64), nullable=False),
    sa.Column('attempt', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('claimed_by', sa.String(length=255), nullable=True),
    sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('next_attempt_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_code', sa.String(length=128), nullable=True),
    sa.Column('error_detail', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('attempt >= 0', name=op.f('ck_indexing_job_indexing_job_attempt_nonnegative')),
    sa.ForeignKeyConstraint(['indexed_document_version_id'], ['indexed_document_version.id'], name=op.f('fk_indexing_job_indexed_document_version_id_indexed_document_version'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_indexing_job_kb_id_knowledge_base'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_indexing_job_workspace_id_workspace'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_indexing_job')),
    sa.UniqueConstraint('indexed_document_version_id', name=op.f('uq_indexing_job_indexed_document_version_id'))
    )
    op.create_index('ix_indexing_job_claim', 'indexing_job', ['status', 'next_attempt_at', 'created_at'], unique=False)
    op.create_index(op.f('ix_indexing_job_kb_id'), 'indexing_job', ['kb_id'], unique=False)
    op.create_index(op.f('ix_indexing_job_workspace_id'), 'indexing_job', ['workspace_id'], unique=False)
    op.create_table(
        'content_mutation',
        sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('principal_id', sa.String(length=255), nullable=False),
        sa.Column('client_id', sa.String(length=255), nullable=False),
        sa.Column('endpoint', sa.String(length=255), nullable=False),
        sa.Column('idempotency_key', sa.UUID(), nullable=False),
        sa.Column('request_hash', sa.String(length=71), nullable=False),
        sa.Column('operation', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('kb_id', sa.UUID(), nullable=True),
        sa.Column('document_id', sa.UUID(), nullable=True),
        sa.Column('document_version_id', sa.UUID(), nullable=True),
        sa.Column('source_change_id', sa.UUID(), nullable=True),
        sa.Column('indexed_document_version_id', sa.UUID(), nullable=True),
        sa.Column('index_revision_id', sa.UUID(), nullable=True),
        sa.Column('job_id', sa.UUID(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('kb_id IS NOT NULL OR document_id IS NOT NULL', name=op.f('ck_content_mutation_content_mutation_has_result')),
        sa.CheckConstraint("status IN ('pending', 'completed')", name=op.f('ck_content_mutation_content_mutation_status_supported')),
        sa.ForeignKeyConstraint(['document_id'], ['document.id'], name=op.f('fk_content_mutation_document_id_document'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['document_version_id'], ['document_version.id'], name=op.f('fk_content_mutation_document_version_id_document_version'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['indexed_document_version_id'], ['indexed_document_version.id'], name=op.f('fk_content_mutation_indexed_document_version_id_indexed_document_version'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['index_revision_id'], ['index_revision.id'], name=op.f('fk_content_mutation_index_revision_id_index_revision'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['job_id'], ['indexing_job.id'], name=op.f('fk_content_mutation_job_id_indexing_job'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_content_mutation_kb_id_knowledge_base'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['source_change_id'], ['source_change.id'], name=op.f('fk_content_mutation_source_change_id_source_change'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_content_mutation_workspace_id_workspace'), ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_content_mutation')),
        sa.UniqueConstraint('principal_id', 'client_id', 'endpoint', 'idempotency_key', name='uq_content_mutation_idempotency_scope'),
    )
    op.create_index(op.f('ix_content_mutation_workspace_id'), 'content_mutation', ['workspace_id'], unique=False)
    op.create_table('citation',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('assistant_message_id', sa.UUID(), nullable=False),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('index_chunk_id', sa.UUID(), nullable=True),
    sa.Column('document_id_snapshot', sa.UUID(), nullable=False),
    sa.Column('document_version_id_snapshot', sa.UUID(), nullable=False),
    sa.Column('document_display_name_snapshot', sa.String(length=512), nullable=False),
    sa.Column('document_original_filename_snapshot', sa.String(length=1024), nullable=False),
    sa.Column('quoted_text', sa.Text(), nullable=False),
    sa.Column('source_location', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('modality', sa.String(length=32), server_default=sa.text("'text'"), nullable=False),
    sa.Column('asset_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('matched_representations', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('score', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('ordinal >= 0', name=op.f('ck_citation_citation_ordinal_nonnegative')),
    sa.ForeignKeyConstraint(['assistant_message_id'], ['chat_message.id'], name=op.f('fk_citation_assistant_message_id_chat_message'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['index_chunk_id'], ['index_chunk.id'], name=op.f('fk_citation_index_chunk_id_index_chunk'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_citation_workspace_id_workspace'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_citation')),
    sa.UniqueConstraint('assistant_message_id', 'ordinal', name=op.f('uq_citation_assistant_message_id_ordinal'))
    )
    op.create_index(op.f('ix_citation_assistant_message_id'), 'citation', ['assistant_message_id'], unique=False)
    op.create_index(op.f('ix_citation_workspace_id'), 'citation', ['workspace_id'], unique=False)
    op.create_table('vector_record_1024',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('workspace_id', sa.UUID(), nullable=False),
    sa.Column('kb_id', sa.UUID(), nullable=False),
    sa.Column('index_chunk_id', sa.UUID(), nullable=False),
    sa.Column('embedding_space_id', sa.UUID(), nullable=False),
    sa.Column('representation_kind', sa.String(length=64), server_default=sa.text("'text'"), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1024), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['kb_id', 'index_chunk_id'], ['index_chunk.kb_id', 'index_chunk.id'], name='fk_vector_record_same_kb_chunk'),
    sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_vector_record_1024_kb_id_knowledge_base'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['workspace_id', 'embedding_space_id'], ['embedding_space.workspace_id', 'embedding_space.id'], name='fk_vector_record_same_workspace_embedding'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_vector_record_1024_workspace_id_workspace'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_vector_record_1024')),
    sa.UniqueConstraint('index_chunk_id', 'embedding_space_id', 'representation_kind', name='uq_vector_record_1024_chunk_space_representation')
    )
    op.create_index(op.f('ix_vector_record_1024_embedding_space_id'), 'vector_record_1024', ['embedding_space_id'], unique=False)
    op.create_index(op.f('ix_vector_record_1024_kb_id'), 'vector_record_1024', ['kb_id'], unique=False)
    op.create_index(op.f('ix_vector_record_1024_workspace_id'), 'vector_record_1024', ['workspace_id'], unique=False)
    op.create_table(
        'vector_record_768',
        sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('kb_id', sa.UUID(), nullable=False),
        sa.Column('index_chunk_id', sa.UUID(), nullable=False),
        sa.Column('embedding_space_id', sa.UUID(), nullable=False),
        sa.Column('representation_kind', sa.String(length=64), nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=768), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['kb_id', 'index_chunk_id'], ['index_chunk.kb_id', 'index_chunk.id'], name='fk_vector_record_768_same_kb_chunk'),
        sa.ForeignKeyConstraint(['workspace_id', 'embedding_space_id'], ['embedding_space.workspace_id', 'embedding_space.id'], name='fk_vector_record_768_same_workspace_embedding'),
        sa.ForeignKeyConstraint(['kb_id'], ['knowledge_base.id'], name=op.f('fk_vector_record_768_kb_id_knowledge_base'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspace.id'], name=op.f('fk_vector_record_768_workspace_id_workspace'), ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_vector_record_768')),
        sa.UniqueConstraint('index_chunk_id', 'embedding_space_id', 'representation_kind', name='uq_vector_record_768_chunk_space_representation'),
    )
    op.create_index(op.f('ix_vector_record_768_workspace_id'), 'vector_record_768', ['workspace_id'])
    op.create_index(op.f('ix_vector_record_768_kb_id'), 'vector_record_768', ['kb_id'])
    op.create_index(op.f('ix_vector_record_768_embedding_space_id'), 'vector_record_768', ['embedding_space_id'])
    op.create_table(
        'index_chunk_asset_relation',
        sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('kb_id', sa.UUID(), nullable=False),
        sa.Column('indexed_document_version_id', sa.UUID(), nullable=False),
        sa.Column('chunk_id', sa.UUID(), nullable=False),
        sa.Column('visual_unit_id', sa.UUID(), nullable=False),
        sa.Column('asset_id', sa.UUID(), nullable=False),
        sa.Column('relation_type', sa.String(length=64), nullable=False),
        sa.Column('confidence_micros', sa.Integer(), nullable=False),
        sa.Column('figure_label', sa.String(length=128), nullable=True),
        sa.Column('ordinal', sa.Integer(), nullable=False),
        sa.Column('provenance', sa.String(length=128), nullable=False),
        sa.Column('evidence_group_key', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("relation_type IN ('explicit_figure_reference', 'caption_of', 'inline_figure', 'ocr_of', 'table_of', 'spatial_neighbor', 'same_page')", name=op.f('ck_index_chunk_asset_relation_chunk_asset_relation_type_supported')),
        sa.CheckConstraint('confidence_micros BETWEEN 0 AND 1000000', name=op.f('ck_index_chunk_asset_relation_chunk_asset_relation_confidence_micros')),
        sa.CheckConstraint('ordinal >= 0', name=op.f('ck_index_chunk_asset_relation_chunk_asset_relation_ordinal_nonnegative')),
        sa.ForeignKeyConstraint(['workspace_id', 'kb_id', 'indexed_document_version_id'], ['indexed_document_version.workspace_id', 'indexed_document_version.kb_id', 'indexed_document_version.id'], name='fk_chunk_asset_relation_same_scope_target', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['indexed_document_version_id', 'chunk_id'], ['index_chunk.indexed_document_version_id', 'index_chunk.id'], name='fk_chunk_asset_relation_same_target_chunk', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['indexed_document_version_id', 'visual_unit_id'], ['index_chunk.indexed_document_version_id', 'index_chunk.id'], name='fk_chunk_asset_relation_same_target_visual', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['indexed_document_version_id', 'asset_id'], ['index_asset.indexed_document_version_id', 'index_asset.id'], name='fk_chunk_asset_relation_same_target_asset', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name='pk_index_chunk_asset_relation'),
        sa.UniqueConstraint('indexed_document_version_id', 'chunk_id', 'asset_id', 'relation_type', name='uq_chunk_asset_relation_stable_edge'),
    )
    op.create_index('ix_chunk_asset_relation_chunk', 'index_chunk_asset_relation', ['workspace_id', 'kb_id', 'indexed_document_version_id', 'chunk_id'])
    op.create_index('ix_chunk_asset_relation_asset', 'index_chunk_asset_relation', ['workspace_id', 'kb_id', 'indexed_document_version_id', 'asset_id'])
    op.create_table(
        'index_chunk_lexical',
        sa.Column('index_chunk_id', sa.UUID(), nullable=False),
        sa.Column('analyzer_version', sa.String(length=64), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('kb_id', sa.UUID(), nullable=False),
        sa.Column('indexed_document_version_id', sa.UUID(), nullable=False),
        sa.Column('lexical_text', sa.Text(), nullable=False),
        sa.Column('lexical_text_hash', sa.String(length=64), nullable=False),
        sa.Column('lexical_tsv', postgresql.TSVECTOR(), sa.Computed("to_tsvector('simple'::regconfig, lexical_text)", persisted=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['workspace_id', 'kb_id', 'indexed_document_version_id'], ['indexed_document_version.workspace_id', 'indexed_document_version.kb_id', 'indexed_document_version.id'], name='fk_chunk_lexical_same_scope_target', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['indexed_document_version_id', 'index_chunk_id'], ['index_chunk.indexed_document_version_id', 'index_chunk.id'], name='fk_chunk_lexical_same_target_chunk', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('index_chunk_id', 'analyzer_version', name='pk_index_chunk_lexical'),
    )
    op.create_index('ix_index_chunk_lexical_scope', 'index_chunk_lexical', ['workspace_id', 'kb_id', 'analyzer_version', 'indexed_document_version_id'])
    op.create_index('ix_index_chunk_lexical_tsv', 'index_chunk_lexical', ['lexical_tsv'], postgresql_using='gin')
    op.create_table(
        'index_lexical_manifest',
        sa.Column('indexed_document_version_id', sa.UUID(), nullable=False),
        sa.Column('analyzer_version', sa.String(length=64), nullable=False),
        sa.Column('workspace_id', sa.UUID(), nullable=False),
        sa.Column('kb_id', sa.UUID(), nullable=False),
        sa.Column('lexical_chunk_count', sa.Integer(), nullable=False),
        sa.Column('lexical_manifest_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('lexical_chunk_count >= 0', name='ck_index_lexical_manifest_lexical_manifest_chunk_count_nonnegative'),
        sa.ForeignKeyConstraint(['workspace_id', 'kb_id', 'indexed_document_version_id'], ['indexed_document_version.workspace_id', 'indexed_document_version.kb_id', 'indexed_document_version.id'], name='fk_lexical_manifest_same_scope_target', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('indexed_document_version_id', 'analyzer_version', name='pk_index_lexical_manifest'),
    )
    op.create_foreign_key(
        'fk_active_revision_same_kb',
        'knowledge_base',
        'index_revision',
        ['id', 'active_index_revision_id'],
        ['kb_id', 'id'],
        deferrable=True,
        initially='DEFERRED',
    )
    op.create_foreign_key(
        'fk_document_current_version',
        'document',
        'document_version',
        ['id', 'current_version_id'],
        ['document_id', 'id'],
        deferrable=True,
        initially='DEFERRED',
    )
    op.create_foreign_key(
        'fk_chat_message_run',
        'chat_message',
        'chat_run',
        ['chat_run_id'],
        ['id'],
    )
    op.execute(
        """
        CREATE FUNCTION enforce_provisioned_kb_active_revision()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            target_kb_id uuid;
        BEGIN
            IF TG_TABLE_NAME = 'knowledge_base' THEN
                target_kb_id := NEW.id;
            ELSIF TG_OP = 'DELETE' THEN
                target_kb_id := OLD.kb_id;
            ELSE
                target_kb_id := NEW.kb_id;
            END IF;

            IF EXISTS (
                SELECT 1
                  FROM knowledge_base kb
                 WHERE kb.id = target_kb_id
                   AND kb.provisioned_at IS NOT NULL
                   AND (
                       kb.active_index_revision_id IS NULL
                       OR NOT EXISTS (
                           SELECT 1
                             FROM index_revision revision
                            WHERE revision.kb_id = kb.id
                              AND revision.id = kb.active_index_revision_id
                              AND revision.status = 'active'
                       )
                   )
            ) THEN
                RAISE EXCEPTION
                    'provisioned knowledge base % must select its active revision',
                    target_kb_id
                    USING ERRCODE = '23514';
            END IF;
            RETURN NULL;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER ck_provisioned_kb_active_revision
        AFTER INSERT OR UPDATE ON knowledge_base
        DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW EXECUTE FUNCTION enforce_provisioned_kb_active_revision()
        """
    )
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER ck_revision_selected_as_active
        AFTER INSERT OR UPDATE OR DELETE ON index_revision
        DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW EXECUTE FUNCTION enforce_provisioned_kb_active_revision()
        """
    )
    op.execute(
        """
        CREATE FUNCTION enforce_document_version_source_immutability()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'document versions are immutable'
                    USING ERRCODE = '23514';
            END IF;
            IF NEW.workspace_id IS DISTINCT FROM OLD.workspace_id
               OR NEW.kb_id IS DISTINCT FROM OLD.kb_id
               OR NEW.document_id IS DISTINCT FROM OLD.document_id
               OR NEW.version_number IS DISTINCT FROM OLD.version_number
               OR NEW.checksum_sha256 IS DISTINCT FROM OLD.checksum_sha256
               OR NEW.storage_uri IS DISTINCT FROM OLD.storage_uri
               OR NEW.original_filename IS DISTINCT FROM OLD.original_filename
               OR NEW.media_type IS DISTINCT FROM OLD.media_type
               OR NEW.size_bytes IS DISTINCT FROM OLD.size_bytes
               OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
                RAISE EXCEPTION 'document version source fields are immutable'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER ck_document_version_source_immutable
        BEFORE UPDATE OR DELETE ON document_version
        FOR EACH ROW EXECUTE FUNCTION enforce_document_version_source_immutability()
        """
    )
    op.execute(
        """
        CREATE FUNCTION enforce_source_change_immutability()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION 'source changes are immutable'
                USING ERRCODE = '23514';
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER ck_source_change_immutable
        BEFORE UPDATE OR DELETE ON source_change
        FOR EACH ROW EXECUTE FUNCTION enforce_source_change_immutability()
        """
    )
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
        "TO rag_kb_runtime"
    )
    op.execute(
        "REVOKE INSERT, UPDATE, DELETE ON TABLE alembic_version FROM rag_kb_runtime"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES FOR ROLE rag_kb_migration IN SCHEMA public "
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag_kb_runtime"
    )
    op.execute("REVOKE UPDATE ON TABLE index_chunk_plan FROM rag_kb_runtime")
    op.execute(
        "REVOKE ALL ON FUNCTION enforce_provisioned_kb_active_revision() FROM PUBLIC"
    )
    op.execute(
        "REVOKE ALL ON FUNCTION enforce_document_version_source_immutability() "
        "FROM PUBLIC"
    )
    op.execute(
        "REVOKE ALL ON FUNCTION enforce_source_change_immutability() FROM PUBLIC"
    )


def downgrade() -> None:
    op.execute(
        "ALTER DEFAULT PRIVILEGES FOR ROLE rag_kb_migration IN SCHEMA public "
        "REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM rag_kb_runtime"
    )
    op.execute(
        "REVOKE SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
        "FROM rag_kb_runtime"
    )
    op.execute("DROP TRIGGER ck_source_change_immutable ON source_change")
    op.execute("DROP FUNCTION enforce_source_change_immutability()")
    op.execute("DROP TRIGGER ck_document_version_source_immutable ON document_version")
    op.execute("DROP FUNCTION enforce_document_version_source_immutability()")
    op.execute("DROP TRIGGER ck_revision_selected_as_active ON index_revision")
    op.execute("DROP TRIGGER ck_provisioned_kb_active_revision ON knowledge_base")
    op.execute("DROP FUNCTION enforce_provisioned_kb_active_revision()")
    op.drop_constraint('fk_chat_message_run', 'chat_message', type_='foreignkey')
    op.drop_constraint('fk_document_current_version', 'document', type_='foreignkey')
    op.drop_constraint('fk_active_revision_same_kb', 'knowledge_base', type_='foreignkey')
    op.drop_table('index_lexical_manifest')
    op.drop_index('ix_index_chunk_lexical_tsv', table_name='index_chunk_lexical', postgresql_using='gin')
    op.drop_index('ix_index_chunk_lexical_scope', table_name='index_chunk_lexical')
    op.drop_table('index_chunk_lexical')
    op.drop_index('ix_chunk_asset_relation_asset', table_name='index_chunk_asset_relation')
    op.drop_index('ix_chunk_asset_relation_chunk', table_name='index_chunk_asset_relation')
    op.drop_table('index_chunk_asset_relation')
    op.drop_index(op.f('ix_vector_record_768_embedding_space_id'), table_name='vector_record_768')
    op.drop_index(op.f('ix_vector_record_768_kb_id'), table_name='vector_record_768')
    op.drop_index(op.f('ix_vector_record_768_workspace_id'), table_name='vector_record_768')
    op.drop_table('vector_record_768')
    op.drop_index(op.f('ix_vector_record_1024_workspace_id'), table_name='vector_record_1024')
    op.drop_index(op.f('ix_vector_record_1024_kb_id'), table_name='vector_record_1024')
    op.drop_index(op.f('ix_vector_record_1024_embedding_space_id'), table_name='vector_record_1024')
    op.drop_table('vector_record_1024')
    op.drop_index(op.f('ix_citation_workspace_id'), table_name='citation')
    op.drop_index(op.f('ix_citation_assistant_message_id'), table_name='citation')
    op.drop_table('citation')
    op.drop_index(op.f('ix_content_mutation_workspace_id'), table_name='content_mutation')
    op.drop_table('content_mutation')
    op.drop_index(op.f('ix_indexing_job_workspace_id'), table_name='indexing_job')
    op.drop_index(op.f('ix_indexing_job_kb_id'), table_name='indexing_job')
    op.drop_index('ix_indexing_job_claim', table_name='indexing_job')
    op.drop_table('indexing_job')
    op.drop_index(op.f('ix_index_chunk_workspace_id'), table_name='index_chunk')
    op.drop_index(op.f('ix_index_chunk_kb_id'), table_name='index_chunk')
    op.drop_index(op.f('ix_index_chunk_indexed_document_version_id'), table_name='index_chunk')
    op.drop_table('index_chunk')
    op.drop_table('index_artifact_manifest')
    op.drop_index(op.f('ix_index_asset_indexed_document_version_id'), table_name='index_asset')
    op.drop_index(op.f('ix_index_asset_kb_id'), table_name='index_asset')
    op.drop_index(op.f('ix_index_asset_workspace_id'), table_name='index_asset')
    op.drop_table('index_asset')
    op.drop_table('index_chunk_plan')
    op.drop_index(op.f('ix_source_file_cleanup_workspace_id'), table_name='source_file_cleanup')
    op.drop_index('ix_source_file_cleanup_due', table_name='source_file_cleanup')
    op.drop_table('source_file_cleanup')
    op.drop_index(op.f('ix_source_change_workspace_id'), table_name='source_change')
    op.drop_table('source_change')
    op.drop_index('uq_one_serving_version_per_document_revision', table_name='indexed_document_version', postgresql_where=sa.text("build_status = 'ready' AND serving_status = 'serving'"))
    op.drop_index(op.f('ix_indexed_document_version_workspace_id'), table_name='indexed_document_version')
    op.drop_index(op.f('ix_indexed_document_version_kb_id'), table_name='indexed_document_version')
    op.drop_table('indexed_document_version')
    op.drop_index(op.f('ix_chat_run_kb_id'), table_name='chat_run')
    op.drop_index('uq_chat_run_session_nonterminal', table_name='chat_run', postgresql_where=sa.text("status IN ('queued', 'running')"))
    op.drop_index('ix_chat_run_claim', table_name='chat_run')
    op.drop_table('chat_run')
    op.drop_index(op.f('ix_document_version_workspace_id'), table_name='document_version')
    op.drop_index(op.f('ix_document_version_kb_id'), table_name='document_version')
    op.drop_table('document_version')
    op.drop_index('uq_assistant_message_chat_run', table_name='chat_message', postgresql_where=sa.text("role = 'assistant' AND chat_run_id IS NOT NULL"))
    op.drop_table('chat_message')
    op.drop_index('uq_one_active_revision_per_kb', table_name='index_revision', postgresql_where=sa.text("status = 'active'"))
    op.drop_index(op.f('ix_index_revision_workspace_id'), table_name='index_revision')
    op.drop_index(op.f('ix_index_revision_kb_id'), table_name='index_revision')
    op.drop_index(op.f('ix_index_revision_embedding_space_embedding_space_id'), table_name='index_revision_embedding_space')
    op.drop_index(op.f('ix_index_revision_embedding_space_workspace_id'), table_name='index_revision_embedding_space')
    op.drop_table('index_revision_embedding_space')
    op.drop_table('index_revision')
    op.drop_index(op.f('ix_document_workspace_id'), table_name='document')
    op.drop_index(op.f('ix_document_kb_id'), table_name='document')
    op.drop_table('document')
    op.drop_table('chat_session')
    op.drop_index(op.f('ix_knowledge_base_workspace_id'), table_name='knowledge_base')
    op.drop_table('knowledge_base')
    op.drop_index(op.f('ix_embedding_space_workspace_id'), table_name='embedding_space')
    op.drop_table('embedding_space')
    op.drop_table('workspace')
    for enum_name in (
        'chat_run_status',
        'assistant_message_status',
        'chat_message_role',
        'job_status',
        'index_serving_status',
        'index_build_status',
        'document_source_status',
        'index_revision_status',
        'source_change_kind',
    ):
        op.execute(f'DROP TYPE IF EXISTS {enum_name}')
