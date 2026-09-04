# Local Knowledge Base and RAG

A local Docker Compose application for importing documents, building pgvector
and versioned PostgreSQL FTS derived indexes, inspecting exact or hybrid
retrieval, and chatting with source citations.

## Start

Run:

```bash
cp .env.example .env.local  # first setup only; replace database placeholders
PYTHONPATH=src:. .venv/bin/python tools/local_runtime.py doctor
./start-local.sh
```

Open:

- User Chat: <http://127.0.0.1:3000>
- API docs: <http://127.0.0.1:8000/api/v1/docs>

The personal runtime has one identity: primary checkout, `.env.local`, Compose
project `rag`, and the four loopback ports recorded in that manifest. Linked
worktrees are for code/test work and cannot rebuild or migrate the personal
stack. The doctor reports stale env/override files and competing Compose
projects without printing values. The starter never guesses project/ports,
extracts credentials from containers, or rewrites the manifest; it reconciles
the declared local database roles, applies migrations, builds revision-labelled
images, starts the services, and waits for health. Existing volumes are retained.

Retired `.env`, per-worktree overrides, and migration backups are not runtime
inputs. The doctor reports them as stale if they reappear. Local image builds
use the configured HTTPS TUNA and Hugging Face mirror endpoints while retaining
Debian's official security repository. `RAG_KB_BUILD_DEBIAN_MIRROR`,
`RAG_KB_BUILD_PYPI_INDEX_URL`, and `RAG_KB_BUILD_HF_ENDPOINT` can override the
Python/model build-time download endpoints. `RAG_KB_BUILD_NPM_REGISTRY`
overrides the frontend package mirror. Frozen model revisions, lock files, and
SHA-256 manifests still determine the accepted dependency/model bytes.
When a prior local application image exists, its frozen model bundles seed the
next build through a read-only build context and are verified again; clean
machines use the network path and populate the persistent BuildKit cache.
If Docker TLS egress is unavailable but the repository's host frontend
dependencies pass `npm ls --all`, the starter builds Vite on the host and sends
only the generated `dist` as a read-only build context. Clean machines retain
the locked Docker `npm ci` path.

The environment has no model provider fallback. After startup, use the
bottom-right model settings in Web Chat to add, validate, and select Chat and
Embedding models.

Chat model integration uses the LangChain adapter, and the single evidence-only
Chat path is a bounded native tool-calling Agent loop. PostgreSQL
`ChatRun` remains the durable execution state, while the existing pgvector and
embedding adapters remain responsible for indexing and retrieval. These
implementations have no runtime rollback switches; strict configuration rejects
the retired backend keys. See the runtime and configuration sections in the
[current architecture](docs/architecture.md) for the supported setup.

The checked-in local development profile explicitly enables best-effort live
Agent progress with the historically named
`RAG_KB__CHAT_DELIVERY__PREVIEW_ENABLED=true`; the Settings default remains
disabled for deployments that do not opt in. Progress is content-safe,
ephemeral, and non-replayed; the authoritative answer still comes only from the
terminal ChatRun. Enabling it uses one additional PostgreSQL connection in each
of the API and Worker processes.

Direct retrieval defaults to exact vector. Chat defaults to the `auto` mode,
whose semantic lane is also exact vector; its public mode choices are `text`,
`auto`, and `graph`. To enable keyword search in Text/Auto and hybrid backfill
for Graph, set `RAG_KB__RETRIEVAL__HYBRID_ENABLED=true`; normal indexing creates
the required lexical rows and completeness manifest for every target. Direct
retrieval requests remain exact unless they explicitly select
`strategy=hybrid`.

## Host Checks

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/basic -v
```

All Python, backend, and evaluator tests run directly in the repository `.venv`.
Testing never builds, pulls, or tags Docker images and never creates, recreates,
restarts, stops, or removes containers. Docker is a runtime/deployment concern,
not a test prerequisite. No quality, security, load, recovery,
compatibility, or release matrix is part of the normal workflow.
The suite is intentionally layered: basic tests protect import and architecture
boundaries, unit/contract tests protect behavior and public schemas, and
database integration tests own migration and catalog invariants. Redundant
source-shape, ORM inventory, retired-setting, and example-file snapshot tests
are not kept as parallel gates.

Database integration tests also run from the host `.venv`. They use two
different roles against a disposable test database:

- `RAG_KB_TEST_MIGRATION_DSN` is used for Alembic, schema cleanup, and
  `TRUNCATE ... CASCADE`.
- `RAG_KB_TEST_RUNTIME_SQLALCHEMY_DSN` is the
  `postgresql+asyncpg://` DSN used by application integration tests.

When the database integration suite is explicitly requested, the repository
runner provisions a unique, loopback-only temporary PostgreSQL container with
the database name `rag_kb_test`, applies migrations, runs the suite with those
DSNs, and removes the container in a `finally` cleanup:

```bash
PYTHONPATH=src:. .venv/bin/python tools/run_database_tests.py
```

The runner never reads `.env.local`, never uses the canonical Compose project
`rag`, never publishes beyond `127.0.0.1`, and never reuses the formal
`rag_kb` database. Direct test execution is also supported when the required
DSNs are supplied by an already-running disposable database. If they are
missing, database test modules fail immediately with a non-zero error that
points to this runner; they are never silently skipped. The runner also treats
any reported skipped database test as a failure. A normal unit/contract test
request still does not start Docker; the temporary database runner is an
explicit database-test operation.

## Host-Python Evaluation

Evaluator dry-runs are offline and do not need the personal stack or an
evaluation runtime:

```bash
PYTHONPATH=src:. .venv/bin/python tools/evaluate_agent_complex_qa.py --dry-run
```

This is the one maintained regression entry point. It validates the frozen
`evaluation/document-qa-v1` corpus offline and, when separately authorized,
measures retrieval, citation and answer quality through an identity-bound host
test runtime. Completed campaign runners and corpora are archived under
`archive/evaluations/` and are not current commands.

All evaluator modes, including real runs, execute in the host `.venv`. A
mode that accesses an API, database, Graph, or Provider may connect only to
user-provided, already-running, disposable host-side test dependencies with a
matching identity. If those dependencies are absent or incompatible, the run
stops as unverified. A second Compose project is not a test prerequisite and
must not be created for evaluation.

Real model acceptance remains separately authorized and must use the repository
Provider/model requirements. Docker lifecycle belongs to the formal `rag`
runtime and the user's actual end-to-end operation; it is not part of test or
evaluator setup.

## Runtime Operations (Not Tests)

The following commands inspect or change the personal runtime. They are not
part of code validation and must not be inferred from a request to test:

```bash
docker compose --env-file .env.local --project-name rag ps
docker compose --env-file .env.local --project-name rag logs --no-color api worker
PYTHONPATH=src:. .venv/bin/python tools/smoke_local.py
docker compose --env-file .env.local --project-name rag down
PYTHONPATH=src:. .venv/bin/python tools/reset_local.py \
  --project-name rag \
  --inspect-only \
  --confirm DESTROY_RAG_KB_LOCAL_DATA
```

Application-safe JSONL logs persist across container recreation in
`.runtime/logs` and rotate automatically. The diagnostics command exports the
latest 72 hours of allowlisted events plus health, Compose and Git state to a
private zip below `.runtime/diagnostics`; it never reads `.env` or includes raw
Docker output, request/model content, exception messages or source lines.
Search `.runtime/logs` with a response `X-Trace-ID`, Run/Job ID, or exception
fingerprint to follow one issue across API and Worker events.

After inspecting the exact project-owned volumes, repeat the reset command
without `--inspect-only`. It permanently removes the canonical Compose project's
PostgreSQL and source-data volumes while preserving its inference-model cache.

See the [current architecture](docs/architecture.md) for configuration, runtime
boundaries, validation, and troubleshooting commands.

## Project Documentation

- [Single current architecture](docs/architecture.md)
- [Current plan](.agent/PLAN.md)
- [Current TODO](.agent/TODO.md)
- [Current progress tracker](.agent/TRACKER.md)
- [Actual history log](.agent/LOG.md)
- [Executed test reports](docs/test/)
- [Concept-only roadmap](docs/roadmap/01-0819-project-roadmap.md)
- [Historical archive](archive/README.md)

`.agent/` is a lightweight handoff area: plans are only for multi-stage or risky
work, and routine local work does not need repeated approval. `AGENTS.md` holds
the concise maintenance and safety rules; `docs/architecture.md` remains the
single maintained architecture record.
