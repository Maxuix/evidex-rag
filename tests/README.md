# Test Layout

- `unit/`: isolated domain, service, and tooling behavior.
- `contract/`: public repository, adapter, workflow, and API contracts.
- `integration/`: checks using real infrastructure such as PostgreSQL/pgvector.
- `e2e/`: externally observable application flows.

`S02-W01` adds only standard-library unit checks for the repository structure and
dependency-boundary tool. Later work items populate the other suites when their
prerequisites are implemented.
