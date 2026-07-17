# Local Knowledge Base and RAG

A local Docker Compose application for importing text documents, building a
pgvector index, inspecting retrieval, and chatting with source citations.

## Start

Configure the model endpoints in `.env` once, then run:

```bash
./start-local.sh
```

Open:

- Frontend: <http://127.0.0.1:3000>
- API docs: <http://127.0.0.1:8000/api/v1/docs>

The starter manages the local database password in ignored `.env.local`, runs
migrations, starts PostgreSQL, API, Worker, and frontend, and waits for health.

Chat model integration defaults to LangChain and the fixed evidence-only chat
workflow defaults to a checkpoint-free LangGraph `StateGraph`. PostgreSQL
`ChatRun` remains the durable execution state, while the existing pgvector and
embedding adapters remain responsible for indexing and retrieval. See the
[local development guide](docs/release/local-development-guide.md) for the two
configuration-only rollback switches.

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
  --env-file .env \
  --confirm DESTROY_RAG_KB_LOCAL_DATA
```

The reset command permanently removes local application data.

See the [local development guide](docs/release/local-development-guide.md) for
configuration, sample import, and troubleshooting.

## Project Documentation

- [Current complete architecture](docs/Enterprise-knowledge-base-design.md)
- [Current execution tracker](docs/implementation-plans/EXECUTION-TRACKER.md)
- [Implementation-plan workflow and template](docs/implementation-plans/README.md)

The architecture document records current implementation facts, the tracker
records the active plan and phase, and a dated implementation plan must be
created before each new work item. Keep all three synchronized with changes in
the same work.
