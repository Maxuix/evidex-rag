# Compatibility Verification Artifacts

The canonical Python/PostgreSQL probe artifacts verify the Stage 01 component
set; they are not the Stage 02 application project structure. The frontend JSON
is a separately scoped Stage 06 extension and does not change the canonical
Stage 01 report.

## Inputs and outputs

- `requirements.in`: exact direct probe inputs.
- `requirements.lock`: Python 3.12.13 / manylinux aarch64 resolution with distribution hashes.
- `python-packages.txt`: exact installed set from the passing container.
- `container-images-v1.0.json`: current OCI index and platform-manifest digests.
- `verification-report-v1.0.json`: canonical passing probe result for the current baseline.
- `frontend-build-v1.0.json`: separate Stage 06 npm, build, image, contract, and smoke evidence.

## Reproduction outline

1. Pull and run the database using the pinned reference in `container-images-v1.0.json`.
2. Set `PGPASSWORD` to the same ephemeral password supplied to that disposable database container. Do not commit or log a real credential.
3. Start the pinned Python image with this repository mounted read-only.
4. Install with `python -m pip install --require-hashes -r requirements.lock`.
5. Run `python -m pip check`.
6. Run `python tools/verify_compatibility.py`.

The probe recreates only a disposable table named `compatibility_vector_probe`; it creates the `vector` extension and therefore must use the disposable migration/admin test identity, not a future runtime DML-only role.

The Stage 06 frontend report is not emitted by `verify_compatibility.py`. Recreate
its deterministic lock check with `tools/check_frontend_lock.py`, then run the
pinned Node container command in `apps/web-test/README.md`, the frontend API and
Compose contract checks, and `tools/run_compose_smoke.py`. npm registry audit and
signature results require network access and are recorded with the report.
