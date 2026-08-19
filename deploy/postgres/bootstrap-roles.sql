\set ON_ERROR_STOP on

-- Passwords are supplied as psql variables by local tooling; no credential is
-- stored in this file. Roles are fixed because Alembic grants are immutable.
SELECT format(
    'ALTER ROLE %I PASSWORD %L',
    :'admin_user',
    :'admin_password'
)
\gexec

SELECT format(
    'CREATE ROLE rag_kb_migration LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION',
    :'migration_password'
)
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_kb_migration')
\gexec

SELECT format(
    'CREATE ROLE rag_kb_runtime LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION',
    :'runtime_password'
)
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_kb_runtime')
\gexec

SELECT format(
    'ALTER ROLE rag_kb_migration PASSWORD %L',
    :'migration_password'
)
\gexec

SELECT format(
    'ALTER ROLE rag_kb_runtime PASSWORD %L',
    :'runtime_password'
)
\gexec

ALTER DATABASE :"database_name" OWNER TO rag_kb_migration;
ALTER SCHEMA public OWNER TO rag_kb_migration;
CREATE EXTENSION IF NOT EXISTS vector;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE CONNECT, TEMPORARY ON DATABASE :"database_name" FROM PUBLIC;
GRANT CONNECT ON DATABASE :"database_name" TO rag_kb_runtime;
GRANT USAGE ON SCHEMA public TO rag_kb_runtime;
