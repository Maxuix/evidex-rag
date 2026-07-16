# Test Layout

- `unit/`: isolated domain, service, and tooling behavior.
- `contract/`: public repository, adapter, workflow, and API contracts.
- `integration/`: checks using real infrastructure such as PostgreSQL/pgvector.
- `e2e/`: externally observable application flows.

The current unit suite covers structure, dependency boundaries, configuration,
composition, and the async-only persistence contract. Database integration tests
run the real migrations and exercise schema, role, lifecycle idempotency,
immutable versions, atomic SourceChange allocation, concurrency, workspace-bound
repositories, Unit of Work behavior, retry-safe Chunk/Vector upserts, partial
failure replay, fixed embedding compatibility, and non-serving completeness
gates against the pinned PostgreSQL/pgvector image. Contract tests drive the ASGI
application directly and freeze Problem Details, cursor, content DTO,
idempotency, lifespan, and OpenAPI behavior.

`apps/web-test/src/*.test.tsx` and `src/api/*.test.ts` cover the Stage 06 frontend
with Vitest, jsdom, and Testing Library. Python tooling tests freeze the frontend
dependency lock/licenses, static server, consumed OpenAPI subset, deterministic
smoke environment, and independent Compose image boundary.
Full externally observable upload-to-citation browser flows begin in `S06-W02`.
