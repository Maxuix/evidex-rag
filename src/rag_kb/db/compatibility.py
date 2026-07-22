"""Read-only validation of the migration-created PostgreSQL capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


EXPECTED_REVISION = "0008_multimodal_index_units"
EXPECTED_POSTGRES_MAJOR = 18
EXPECTED_PGVECTOR_VERSION = "0.8.2"
EXPECTED_VECTOR_TABLE = "vector_record_1024"
EXPECTED_VECTOR_COLUMN = "embedding"
EXPECTED_VECTOR_TYPE = "vector(1024)"
EXPECTED_APPLICATION_TABLES = frozenset(
    {
        "chat_message",
        "chat_run",
        "chat_session",
        "citation",
        "content_mutation",
        "document",
        "document_version",
        "embedding_space",
        "eval_case",
        "eval_dataset",
        "eval_result",
        "eval_run",
        "index_chunk",
        "index_chunk_plan",
        "index_asset",
        "index_artifact_manifest",
        "index_revision",
        "index_revision_embedding_space",
        "indexed_document_version",
        "indexing_job",
        "knowledge_base",
        "source_change",
        "source_file_cleanup",
        "vector_record_1024",
        "workspace",
    }
)


class DatabaseCompatibilityError(RuntimeError):
    """The existing database is incompatible with the frozen application schema."""


@dataclass(frozen=True)
class DatabaseCompatibility:
    postgres_major: int
    pgvector_version: str
    migration_revision: str
    vector_type: str
    application_table_count: int


async def validate_database_compatibility(
    connection: AsyncConnection,
) -> DatabaseCompatibility:
    """Inspect catalogs using SELECT statements only; never provision or repair."""

    server_version_num = int(
        await connection.scalar(text("SELECT current_setting('server_version_num')"))
    )
    postgres_major = server_version_num // 10000
    if postgres_major != EXPECTED_POSTGRES_MAJOR:
        raise DatabaseCompatibilityError(
            f"expected PostgreSQL {EXPECTED_POSTGRES_MAJOR}, found {postgres_major}"
        )

    pgvector_version = await connection.scalar(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    )
    if pgvector_version != EXPECTED_PGVECTOR_VERSION:
        raise DatabaseCompatibilityError(
            f"expected pgvector {EXPECTED_PGVECTOR_VERSION}, found {pgvector_version}"
        )

    migration_revision = await connection.scalar(
        text("SELECT version_num FROM alembic_version")
    )
    if migration_revision != EXPECTED_REVISION:
        raise DatabaseCompatibilityError(
            f"expected migration {EXPECTED_REVISION}, found {migration_revision}"
        )

    table_rows = await connection.execute(
        text(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"
        )
    )
    actual_tables = frozenset(table_rows.scalars())
    if actual_tables != EXPECTED_APPLICATION_TABLES:
        missing = sorted(EXPECTED_APPLICATION_TABLES - actual_tables)
        unexpected = sorted(actual_tables - EXPECTED_APPLICATION_TABLES)
        raise DatabaseCompatibilityError(
            f"application tables differ; missing={missing}, unexpected={unexpected}"
        )

    vector_type = await connection.scalar(
        text(
            "SELECT format_type(attribute.atttypid, attribute.atttypmod) "
            "FROM pg_attribute attribute "
            "JOIN pg_class relation ON relation.oid = attribute.attrelid "
            "JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace "
            "WHERE namespace.nspname = 'public' "
            "AND relation.relname = :table_name "
            "AND attribute.attname = :column_name "
            "AND NOT attribute.attisdropped"
        ),
        {
            "table_name": EXPECTED_VECTOR_TABLE,
            "column_name": EXPECTED_VECTOR_COLUMN,
        },
    )
    if vector_type != EXPECTED_VECTOR_TYPE:
        raise DatabaseCompatibilityError(
            f"expected {EXPECTED_VECTOR_TYPE}, found {vector_type}"
        )

    has_cosine_operator = await connection.scalar(
        text(
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_operator operator "
            "JOIN pg_type left_type ON left_type.oid = operator.oprleft "
            "JOIN pg_type right_type ON right_type.oid = operator.oprright "
            "WHERE operator.oprname = '<=>' "
            "AND left_type.typname = 'vector' "
            "AND right_type.typname = 'vector')"
        )
    )
    if not has_cosine_operator:
        raise DatabaseCompatibilityError("pgvector cosine operator <=> is unavailable")

    has_hnsw_index = await connection.scalar(
        text(
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_indexes "
            "WHERE schemaname = 'public' "
            "AND tablename = :table_name "
            "AND indexdef ILIKE '% USING hnsw %')"
        ),
        {"table_name": EXPECTED_VECTOR_TABLE},
    )
    if has_hnsw_index:
        raise DatabaseCompatibilityError("HNSW must remain disabled in P1A")

    return DatabaseCompatibility(
        postgres_major=postgres_major,
        pgvector_version=pgvector_version,
        migration_revision=migration_revision,
        vector_type=vector_type,
        application_table_count=len(actual_tables),
    )
