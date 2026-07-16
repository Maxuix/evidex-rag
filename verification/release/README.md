# P1A Local Release Evidence

`p1a-local-release-v1.0.json` is the immutable Stage 06 completion report.
It predates the later `start-local.sh` usability amendment. That small
amendment was verified with focused unit tests and a real
`docker compose down` followed by successful one-command recovery of all four
healthy services; the complete release matrix was intentionally not rerun.

```bash
PYTHONPATH=src:. .venv/bin/python tools/run_p1a_release_validation.py \
  --report /tmp/p1a-local-release.json
```

The runner first validates the release package, then executes the complete
quality/security matrix and the isolated operations/recovery suite. The nested
runners use deterministic providers, scrub inherited project/model settings,
create uniquely named disposable Compose projects, and remove their volumes.

The checked report binds release documents, tools, locks, manifests, provider
declarations, and prior versioned evidence by SHA-256. Dynamic timestamps and
durations may differ on reproduction. No database password, provider key,
request/document content, raw model response, or ephemeral project identifier is
recorded.

This is a loopback-only local-development release. It is not evidence of
production security, backup/restore, high availability, RPO/RTO, compliance,
host-failure recovery, capacity, or hostile multi-tenant isolation.
