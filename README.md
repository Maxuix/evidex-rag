# Local Knowledge Base and RAG

A local Docker Compose application for importing documents, building pgvector
and versioned PostgreSQL FTS derived indexes, inspecting exact or hybrid
retrieval, and chatting with source citations.

## Start

Copy `.env.example` to `.env`, replace the database placeholders, then run:

```bash
./start-local.sh
```

Open:

- User Chat: <http://127.0.0.1:3000>
- API docs: <http://127.0.0.1:8000/api/v1/docs>

The starter manages the local database password in ignored `.env.local`, runs
migrations, rebuilds the API/Worker and frontend images with Docker layer
cache, starts all services, and waits for health. Existing database and file
volumes are retained. Local image builds default to HTTPS TUNA mirrors for PyPI
and the Debian main repository while retaining Debian's official security
repository; build-only mirror URLs remain explicitly overridable.

The default environment has no legacy model provider. After startup, use the
bottom-right model settings in Web Chat to add, validate, and select Chat and
Embedding models. The commented `MODEL_PROVIDER` example is only an optional
fallback for historical runs and Embedding Spaces.

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

Exact-vector retrieval remains the default. To evaluate the optional hybrid
path, explicitly set `RAG_KB__RETRIEVAL__HYBRID_ENABLED=true`; normal indexing
creates the required lexical rows and completeness manifest for every target.
Existing requests remain exact unless they select `strategy=hybrid` or Chat
`retrieval.mode=hybrid`.

## Basic Check

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/basic -v
PYTHONPATH=src:. .venv/bin/python tools/smoke_local.py
```

These are the default project checks. No quality, security, load, recovery,
compatibility, or release matrix is part of the normal workflow.
The suite is intentionally layered: basic tests protect import and architecture
boundaries, unit/contract tests protect behavior and public schemas, and
database integration tests own migration and catalog invariants. Redundant
source-shape, ORM inventory, retired-setting, and example-file snapshot tests
are not kept as parallel gates.

## Useful Commands

```bash
docker compose --env-file .env.local ps
docker compose --env-file .env.local logs --no-color api worker
PYTHONPATH=src:. .venv/bin/python tools/collect_diagnostics.py
docker compose --env-file .env.local down
PYTHONPATH=src:. .venv/bin/python tools/reset_local.py \
  --env-file .env.local \
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
without `--inspect-only`. It permanently removes the selected Compose project's
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

The `.agent/` task system separates intent, concrete actions, progress, and
actual history. `docs/architecture.md` is the only maintained architecture
document; reviews, executed test reports, and concept-only roadmaps use their
dedicated folders. Test reports record results and do not replace the validation
rules in `AGENTS.md`. See that file for lifecycle, naming, authorization, and
validation rules.
