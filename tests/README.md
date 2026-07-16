# Basic Tests

The normal project check is intentionally small:

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/basic -v
```

For a running local stack:

```bash
PYTHONPATH=src:. .venv/bin/python tools/smoke_local.py
```

Core unit, API contract, and database behavior tests remain for targeted defect
work. They are not part of the default workflow. Do not add broad matrices or
versioned reports unless Maxui explicitly requests them.
