#!/bin/sh
set -eu

: "${POSTGRES_DB:?POSTGRES_DB is required}"
: "${RAG_KB_MIGRATION_PASSWORD:?RAG_KB_MIGRATION_PASSWORD is required}"
: "${RAG_KB_RUNTIME_PASSWORD:?RAG_KB_RUNTIME_PASSWORD is required}"

psql \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" \
  --set ON_ERROR_STOP=1 \
  --set "database_name=$POSTGRES_DB" \
  --set "migration_password=$RAG_KB_MIGRATION_PASSWORD" \
  --set "runtime_password=$RAG_KB_RUNTIME_PASSWORD" \
  --file /opt/rag-kb/bootstrap-roles.sql
