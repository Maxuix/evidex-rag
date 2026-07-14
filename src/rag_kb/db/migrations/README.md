# Database Migrations

Alembic runs only with `RAG_KB__DATABASE__MIGRATION_DSN`. API and Worker startup
never invokes this directory. The migration login owns schema DDL; the runtime
login receives only explicitly granted DML and read access.

The PostgreSQL roles and the cluster-level pgvector extension must first be
bootstrapped by an administrator with `deploy/postgres/bootstrap-roles.sql`.
Passwords are passed as `psql` variables and are never stored in source control.
Alembic owns application tables, constraints, triggers, enums, and grants; it
does not require superuser privileges.
