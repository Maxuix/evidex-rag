# Local Knowledge Base and RAG

A local Docker Compose application for importing documents, building pgvector
and versioned PostgreSQL FTS derived indexes, inspecting exact or hybrid
retrieval, and chatting with source citations.

## Start

Configure the model endpoints in `.env` once, then run:

```bash
./start-local.sh
```

Open:

- User Chat: <http://127.0.0.1:3000>
- Diagnostic UI: <http://127.0.0.1:3001>
- API docs: <http://127.0.0.1:8000/api/v1/docs>

The starter manages the local database password in ignored `.env.local`, runs
migrations, rebuilds the API/Worker and both frontend images with Docker layer
cache, starts all services, and waits for health. Existing database and file
volumes are retained. Local image builds default to HTTPS TUNA mirrors for PyPI
and the Debian main repository while retaining Debian's official security
repository; build-only mirror URLs remain explicitly overridable.

Chat model integration defaults to LangChain and the fixed evidence-only chat
workflow defaults to a checkpoint-free LangGraph `StateGraph`. PostgreSQL
`ChatRun` remains the durable execution state, while the existing pgvector and
embedding adapters remain responsible for indexing and retrieval. These
implementations have no runtime rollback switches; strict configuration rejects
the retired backend keys. See the
[local development guide](docs/release/local-development-guide.md) for the
supported settings.

The checked-in local development profile explicitly enables Chat answer
preview with `RAG_KB__CHAT_DELIVERY__PREVIEW_ENABLED=true`; the Settings
fallback remains disabled for deployments that do not opt in. Preview is
ephemeral, unvalidated plain text sent before the authoritative terminal
answer. It is not replayed or persisted, may be lost, and uses one additional
PostgreSQL connection in each of the API and Worker processes. The UI always
labels it as unvalidated and replaces it with the committed answer.

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

## Useful Commands

```bash
docker compose --env-file .env.local ps
docker compose --env-file .env.local logs --no-color api worker
docker compose --env-file .env.local down
PYTHONPATH=src:. .venv/bin/python tools/reset_local.py \
  --env-file .env.local \
  --project-name rag \
  --inspect-only \
  --confirm DESTROY_RAG_KB_LOCAL_DATA
```

After inspecting the exact project-owned volumes, repeat the reset command
without `--inspect-only`. It permanently removes the selected Compose project's
PostgreSQL and source-data volumes while preserving its inference-model cache.

See the [local development guide](docs/release/local-development-guide.md) for
configuration, sample import, and troubleshooting.

## Project Documentation

- [Current architecture and local-first boundaries](docs/Enterprise-knowledge-base-design.md)
- [Current execution tracker](docs/implementation-plans/EXECUTION-TRACKER.md)
- [Implementation-plan workflow and template](docs/implementation-plans/README.md)

The architecture document is a high-level map of the current implementation.
The tracker and a short dated plan are used only for active multi-stage, risky,
schema-changing, or destructive work; focused changes can proceed without that
process overhead. See `AGENTS.md` for the local-first maintenance rules.
