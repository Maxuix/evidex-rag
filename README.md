# Enterprise Knowledge Base and RAG

This repository is building a local-first enterprise knowledge base with
evidence-grounded retrieval and answering. The current milestone is the P1A
Core Vertical Slice. Knowledge-base/document lifecycle, bounded text upload,
local source-file consistency, and isolated parsing are implemented; indexing
execution through answering is still under construction.

Authoritative project documents:

- [Architecture design](docs/Enterprise-knowledge-base-design.md)
- [Implementation roadmap](docs/implementation-plans/00-implementation-roadmap.md)
- [Execution tracker](docs/implementation-plans/EXECUTION-TRACKER.md)
- [Module boundaries](docs/architecture/module-boundaries.md)
- [Configuration and composition roots](docs/architecture/configuration.md)
- [P0/P1A database schema](docs/architecture/database-schema.md)
- [Async data access and transactions](docs/architecture/async-data-access.md)
- [Public API and error conventions](docs/architecture/api-conventions.md)
- [Identity and security boundaries](docs/architecture/identity-security.md)
- [Local runtime and diagnostics](docs/architecture/local-runtime.md)
- [Local source-file consistency](docs/architecture/local-file-consistency.md)
- [File admission and parser isolation](docs/architecture/file-admission-parser-isolation.md)

## Current Layout

```text
apps/                 API, single Worker, and test-frontend entrypoint areas
src/rag_kb/           Domain, application, contracts, and infrastructure modules
evaluation/           Versioned evaluation inputs and reports
tests/                Unit, contract, integration, and end-to-end suites
deploy/               Local deployment assets (implemented in S02-W07)
tools/                Reproducible project and baseline checks
verification/         Stage 01 compatibility and provider evidence
```

`apps/api` and `apps/worker` are composition roots for separately runnable
processes. Business logic belongs under `src/rag_kb` and follows the dependency
rules enforced by `architecture.toml`.

Configuration is loaded explicitly from `RAG_KB__<GROUP>__<FIELD>` environment
variables. [`.env.example`](.env.example) documents the complete development
surface with non-working secret placeholders.

## Python Environment

Python `3.12.13` and the application package set are frozen. With `uv` installed:

```bash
uv venv --python 3.12.13
uv pip install --python .venv/bin/python --require-hashes -r requirements.lock
```

The root `requirements.lock` is the application install source. Its package
versions and hashes originate from the passing Stage 01 compatibility baseline.
Pydantic is declared directly because application DTOs will import it; it was
already present and tested in the frozen Stage 01 resolution.

## Foundation Checks

```bash
PYTHONPATH=src:. .venv/bin/python tools/check_architecture.py
.venv/bin/python tools/check_application_lock.py
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/unit -v
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/contract -v
PYTHONPATH=src:. .venv/bin/python tools/check_openapi_compatibility.py
.venv/bin/python tools/check_compose_contract.py
PYTHONPATH=src:. .venv/bin/python tools/run_db_integration.py
.venv/bin/python tools/run_compose_smoke.py
```

## Local Compose Runtime

Copy the checked example, replace its application placeholders as needed, and
provide the three Compose database passwords in the shell. `RAG_KB_ENV_FILE`
selects the application settings file without injecting that control variable
into the application process.

```bash
cp .env.example .env
export RAG_KB_ENV_FILE=.env
export POSTGRES_ADMIN_PASSWORD='replace-with-a-local-password'
export RAG_KB_MIGRATION_PASSWORD='replace-with-a-different-local-password'
export RAG_KB_RUNTIME_PASSWORD='replace-with-another-local-password'

docker compose up -d --wait postgres storage-init
docker compose --profile tools run --rm migrate
docker compose up -d --wait api worker frontend
```

The shell is then available at `http://127.0.0.1:3000`, API documentation at
`http://127.0.0.1:8000/api/v1/docs`, and diagnostics at `/health/live` and
`/health/ready`. Stop processes while retaining data with `docker compose down`.
See the [local runtime guide](docs/architecture/local-runtime.md) for port
overrides, logs, checks, and the explicitly destructive clean reset.

## Current Boundary

The Compose profile is a local validation runtime. Knowledge-base management,
document reads/deletion, bounded `.txt`/`.md` upload, restart-safe local source
storage, and isolated transient parsing exist. Indexing-job execution, durable
chunks/vectors, retrieval, chat, and business frontend screens are not yet
available. It has one
fixed development identity and provides no enterprise authentication or
authorization, durable audit system, formal backup/recovery, high availability,
production hardening, or hostile multi-tenant isolation guarantee.
