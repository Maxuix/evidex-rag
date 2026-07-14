# Test Layout

- `unit/`: isolated domain, service, and tooling behavior.
- `contract/`: public repository, adapter, workflow, and API contracts.
- `integration/`: checks using real infrastructure such as PostgreSQL/pgvector.
- `e2e/`: externally observable application flows.

The current unit suite covers structure, dependency boundaries, configuration,
composition, and the async-only persistence contract. Database integration tests
run the real migrations and exercise schema, role, lifecycle idempotency,
immutable versions, atomic SourceChange allocation, concurrency, workspace-bound
repositories, and Unit of Work behavior against the pinned PostgreSQL/pgvector
image. Contract tests drive the ASGI application directly and freeze Problem
Details, cursor, content DTO, idempotency, lifespan, and OpenAPI behavior.
End-to-end tests are populated by later eligible work items.
