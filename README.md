# Enterprise Knowledge Base and RAG

This repository is building a local-first enterprise knowledge base with
evidence-grounded retrieval and answering. The current milestone is the P1A
Core Vertical Slice. Knowledge-base/document lifecycle, bounded text upload,
local source-file consistency, isolated parsing, retry-safe indexing execution,
conditional current-version promotion, PostgreSQL Worker scheduling, indexing
status/retry, exact retrieval, durable evidence-grounded answering, and a local
public-API observation frontend are implemented.

Authoritative project documents:

- [P1A local development release](docs/release/README.md)
- [P1A local development guide](docs/release/local-development-guide.md)
- [P1A capability matrix](docs/release/capability-matrix.md)
- [P1A known limitations](docs/release/known-limitations.md)
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
- [End-to-end integration boundary](docs/architecture/end-to-end-integration.md)
- [Quality and security regression](docs/architecture/quality-security-regression.md)
- [Local operations and recovery exercises](docs/architecture/operations-recovery-exercises.md)
- [Local source-file consistency](docs/architecture/local-file-consistency.md)
- [File admission and parser isolation](docs/architecture/file-admission-parser-isolation.md)
- [Indexing pipeline](docs/architecture/indexing-pipeline.md)
- [Current-version promotion and deletion](docs/architecture/promotion-deletion.md)
- [Single-Worker scheduling and recovery](docs/architecture/worker-scheduling.md)
- [Indexing operations and local maintenance](docs/architecture/indexing-operations.md)
- [Test frontend observation boundary](docs/architecture/test-frontend.md)

## Current Layout

```text
apps/                 API, Worker, maintenance, and test-frontend entrypoints
src/rag_kb/           Domain, application, contracts, and infrastructure modules
evaluation/           Versioned evaluation inputs and reports
tests/                Unit, contract, integration, and end-to-end suites
deploy/               Local deployment assets (implemented in S02-W07)
tools/                Reproducible project and baseline checks
verification/         Stage 01 compatibility and provider evidence
start-local.sh        One-command local startup, migration, and health wait
```

`apps/api`, `apps/worker`, and the one-shot `apps/maintenance` tool are
composition roots. Business logic belongs under `src/rag_kb` and follows the
dependency rules enforced by `architecture.toml`.

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
.venv/bin/python tools/check_frontend_lock.py
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/unit -v
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/contract -v
PYTHONPATH=src:. .venv/bin/python tools/check_openapi_compatibility.py
.venv/bin/python tools/check_frontend_api_contract.py
.venv/bin/python tools/check_compose_contract.py
PYTHONPATH=src:. .venv/bin/python tools/run_db_integration.py
.venv/bin/python tools/run_compose_smoke.py
.venv/bin/python tools/run_e2e_integration.py
.venv/bin/python tools/run_quality_security_regression.py \
  --report /tmp/s06-w03-quality-security-report.json
PYTHONPATH=src:. .venv/bin/python tools/run_operations_recovery.py \
  --report /tmp/s06-w04-report-v1.0.json
PYTHONPATH=src:. .venv/bin/python tools/check_release_package.py
PYTHONPATH=src:. .venv/bin/python tools/run_p1a_release_validation.py \
  --report /tmp/p1a-local-release.json
```

## Local Compose Runtime

Copy the checked example once and replace its model-provider placeholders:

```bash
cp .env.example .env
```

Then start or resume the complete local stack with one command:

```bash
./start-local.sh
```

The script safely creates or reuses one local database password for all three
database roles in the ignored, mode-`0600` `.env.local`, runs the explicit
migration, starts all long-lived services, waits for health, and prints the
local URLs. Existing containers are imported without printing their
credentials; an existing volume without recoverable credentials is never
silently re-keyed.

Run one idempotent bounded cleanup pass with:

```bash
docker compose --profile tools run --rm maintenance
```

The observation frontend is then available at `http://127.0.0.1:3000`, API documentation at
`http://127.0.0.1:8000/api/v1/docs`, and diagnostics at `/health/live` and
`/health/ready`. Stop processes while retaining data with `docker compose down`.
See the [local runtime guide](docs/architecture/local-runtime.md) for port
overrides, logs, checks, and the explicitly destructive clean reset.
For a clean-checkout walkthrough, synthetic sample import, closed-loop demo,
evaluation, troubleshooting, and release limits, use the
[P1A local development guide](docs/release/local-development-guide.md).

## Current Boundary

The Compose profile is a local validation runtime. Knowledge-base management,
document reads/deletion, bounded `.txt`/`.md` upload, restart-safe local source
storage, isolated transient parsing, an internal retry-safe command for durable
Chunk/pgvector creation, and conditional promotion to one current serving
version exist. The Worker polls PostgreSQL with bounded claims, independent
heartbeats, deadlines, retries, and stale recovery. Exact retrieval, durable
chat/status/terminal SSE, and the Documents, Chat, and Retrieval Debug frontend
views are public. The runtime has one fixed development identity and provides no enterprise authentication or
authorization, durable audit system, formal backup/recovery, high availability,
production hardening, or hostile multi-tenant isolation guarantee.
