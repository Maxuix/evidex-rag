# End-to-End Verification Artifacts

`s06-w02-report-v1.0.json` records the isolated Stage 06 public API integration
run. Reproduce it with:

```bash
.venv/bin/python tools/run_e2e_integration.py \
  --report /tmp/s06-w02-e2e-report.json
```

Compare the generated report with the checked artifact after verifying the
input hashes. Timestamps and duration are observational; scenario names,
isolation properties, public-boundary claims, limits, and input hashes are the
stable acceptance facts.
