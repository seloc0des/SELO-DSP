# SELO AI Backend Test Suite

This suite is designed to run on your Ubuntu server install without needing live external services.

## Groups
- Smoke/API (default): no external deps, uses fakes/mocks.
- Persona/Reflection API: DI patched to fakes.
- Search: network and subprocess patched.
- Router unit: in-process only.
- Optional Postgres repo tests: can be added with a real DB profile later.

## One-time setup
- Ensure Python v3.10+ and pip v23+ on the server.
- From `selo-ai/backend/`:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `pip install -r requirements-dev.txt`

## Running
- From `selo-ai/backend/`:
  - All tests: `pytest -q`
  - Specific file: `pytest -q tests/test_smoke_api.py`
  - With logs: `pytest -q -s`

## Environment knobs
- `BRAVE_SEARCH_API_KEY` is patched in tests; not required to be set.
- No Postgres or Ollama required for default tests.

## Notes
- Lifespan and heavy DI are disabled via test fixtures to keep tests fast and isolated (see `tests/conftest.py`).
