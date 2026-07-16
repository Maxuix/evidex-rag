# Order API Operations Runbook

The service name is `order-api`, the deployment unit is `ORD-SVC-02`, and
the supported contract is `/api/v1`.

Check `/health/live` to confirm that the process is running and
`/health/ready` to confirm that the database is available.

Error `ORD-4091` means that an idempotency key was reused with a different
request hash. Error `ORD-5032` means that the configured model provider was
temporarily unavailable.
