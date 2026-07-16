# Stage 06 Operations and Recovery Evidence

`s06-w04-report-v1.0.json` is produced by the isolated
`tools/run_operations_recovery.py` suite. The report binds the Compose override,
dedicated network-only Provider, reset tool, runner, and fixtures by SHA-256.

Reproduce it without touching the ordinary Compose project:

```bash
PYTHONPATH=src:. .venv/bin/python tools/run_operations_recovery.py \
  --report /tmp/s06-w04-report-v1.0.json
```

Dynamic timestamps, duration, randomized ports/project name, and per-run
passwords are intentionally not stable report inputs. Scenario names, numeric
outcomes, limitations, and file hashes must agree. The runner reports success
only after its final `down --volumes --remove-orphans` succeeds.

This evidence proves bounded local process recovery and destructive reset of a
disposable project. It is not backup, host recovery, RPO/RTO, high availability,
formal retention, compliance, or production-readiness evidence.
