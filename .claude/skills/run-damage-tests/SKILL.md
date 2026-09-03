---
name: run-damage-tests
description: >-
  Run the iRent Car Damage Evaluation API checks: the pytest suite and the ruff
  linter. Use this whenever the user asks to test, verify, validate, lint, or
  check the FastAPI app, or before staging a commit. Also covers starting the
  local dev server for a manual smoke test.
---

# Running the iRent damage API checks

This repo has a project virtualenv at `.venv/`. Always run Python through it —
the machine's system `python`/`py` is a broken Store shim / old conda 3.8 and
must not be used.

Interpreter: `./.venv/Scripts/python.exe` (run from the project root).

## Test suite

```bash
./.venv/Scripts/python.exe -m pytest -q
```

- `pytest.ini` sets `pythonpath = .`, so run from the project root, not `tests/`.
- The suite runs entirely against `MockEvaluator` — no model weights or `torch`
  needed. Expect ~12 passing tests.
- A `StarletteDeprecationWarning` about `httpx` from `fastapi.testclient` is
  known and harmless; it is not a failure.

## Linter

```bash
./.venv/Scripts/python.exe -m ruff check .
```

Config is `ruff.toml` (rules `E,F,I,UP,B,PLC`, line length 100). A clean run
prints `All checks passed!`.

## Manual smoke test (optional)

Only when the user wants to hit real HTTP endpoints:

```bash
./.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then in another shell:

```bash
curl -s http://127.0.0.1:8000/api/v1/health
curl -s -F "file=@some-car.jpg" http://127.0.0.1:8000/api/v1/damage/evaluate
```

Stop the server when done.

## Reporting

Report the real outcome. If tests fail, show the failing test names and the
relevant traceback lines. Do not report success unless both `pytest` and
`ruff check .` pass.
