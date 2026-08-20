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
still use the configured HTTPS TUNA mirrors while retaining Debian's official
security repository.

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

Database integration tests use a dedicated runner instead of the personal
database:

```bash
python3 tools/run_database_tests.py
python3 tools/run_database_tests.py \
  tests.integration.db.test_schema.DatabaseSchemaTests.test_runtime_role_is_dml_only_and_readiness_is_read_only \
  -v
```

Each invocation uses the already-present pinned PostgreSQL image without
pulling, starts a uniquely named container on a Docker-chosen loopback port,
uses passwordless `trust` authentication only inside that temporary instance,
migrates its empty tmpfs database, runs the requested `unittest` targets, and
removes the container. It finds the primary worktree's Python 3.12 virtual
environment when the current linked worktree has no `.venv`.

## Isolated Evaluation

Evaluator dry-runs are offline and do not need the personal stack or an
evaluation runtime:

```bash
PYTHONPATH=src:. .venv/bin/python tools/evaluate_adaptive_graph_route.py --dry-run
PYTHONPATH=src:. .venv/bin/python tools/run_adaptive_graph_r4.py --dry-run
PYTHONPATH=src:. .venv/bin/python tools/run_adaptive_graph_r7_stage_a.py --dry-run
PYTHONPATH=src:. .venv/bin/python tools/evaluate_agent_complex_qa.py --dry-run
PYTHONPATH=src:. .venv/bin/python tools/evaluate_multimodal_real.py --dry-run
```

Any mode that can access an API, database, Graph, or Provider requires the
private runtime created for the fixed `rag-eval` Compose project. Preview is
read-only:

```bash
PYTHONPATH=src:. .venv/bin/python tools/evaluation_runtime.py preview
```

Create and destroy require their exact confirmation values shown by preview.
Creation first proves that the existing local `rag` app/frontend images match
the current checkout's corresponding build inputs, then adds eval-only tags;
it fails instead of rebuilding, pulling, or downloading when they differ. It
restores only the verified frozen evaluation backup into eval-owned volumes.
The Adaptive Graph identity comes from the checksummed completed R7 artifact
in that same backup and must exactly match the restored database's serving
KB/index/build and answer/judge profiles.
Destruction validates the project, per-create owner labels, known container/
volume/network set, and private runtime directory before removing those
objects. Neither command targets the personal `rag` volumes. Provider and Judge
execution remains a separate, per-run authorization decision.

## Useful Commands

```bash
docker compose --env-file .env.local --project-name rag ps
docker compose --env-file .env.local --project-name rag logs --no-color api worker
PYTHONPATH=src:. .venv/bin/python tools/collect_diagnostics.py
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
