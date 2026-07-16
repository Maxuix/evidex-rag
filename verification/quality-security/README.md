# Quality and Security Verification Artifacts

`s06-w03-report-v1.0.json` records the complete Stage 06 quality/security
matrix. Reproduce it with:

```bash
.venv/bin/python tools/run_quality_security_regression.py \
  --report /tmp/s06-w03-quality-security-report.json
```

The runner executes all 18 checks rather than trusting the checked report. A
fresh report has different timestamps and durations; compare its matrix,
coverage, security assertions, golden metrics, limitations, and input hashes.
