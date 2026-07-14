# Enterprise Knowledge Base and RAG

This repository is building a local-first enterprise knowledge base with
evidence-grounded retrieval and answering. The current milestone is P0
Foundation; application business capabilities are not implemented yet.

Authoritative project documents:

- [Architecture design](docs/Enterprise-knowledge-base-design.md)
- [Implementation roadmap](docs/implementation-plans/00-implementation-roadmap.md)
- [Execution tracker](docs/implementation-plans/EXECUTION-TRACKER.md)
- [Module boundaries](docs/architecture/module-boundaries.md)
- [Configuration and composition roots](docs/architecture/configuration.md)
- [P0/P1A database schema](docs/architecture/database-schema.md)

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
PYTHONPATH=src:. .venv/bin/python tools/run_db_integration.py
```

## Current Boundary

This skeleton does not yet provide upload, indexing, retrieval, chat, a frontend,
database migrations, or a runnable Compose environment. Those capabilities are
introduced only by their ordered work items in the execution tracker. Until the
local-release stage is complete, this repository makes no claim of enterprise
authentication, high availability, formal backup, production hardening, or
multi-tenant isolation.
