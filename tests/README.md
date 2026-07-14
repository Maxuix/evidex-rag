# Test Layout

- `unit/`: isolated domain, service, and tooling behavior.
- `contract/`: public repository, adapter, workflow, and API contracts.
- `integration/`: checks using real infrastructure such as PostgreSQL/pgvector.
- `e2e/`: externally observable application flows.

The current unit suite covers structure, dependency boundaries, configuration,
composition, and the async-only persistence contract. Database integration tests
run the real migration and exercise schema, role, concurrency, repository, and
Unit of Work behavior against the pinned PostgreSQL/pgvector image. Contract and
end-to-end suites are populated by later eligible work items.
