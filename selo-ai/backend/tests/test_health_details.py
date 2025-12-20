import pytest


@pytest.mark.anyio
async def test_health_details_defaults_ok(client):
    resp = await client.get("/health/details")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["probes"]["llm"]["requested"] is False
    assert data["probes"]["db"]["requested"] is False


@pytest.mark.anyio
async def test_health_details_llm_probe(client, monkeypatch):
    # Patch DI to return a fake router
    class _FakeRouter:
        def __init__(self):
            self.calls = []
        async def route(self, **kwargs):
            self.calls.append(kwargs)
            return {"ok": True}

    from backend.main import health_details
    import main as main_mod

    async def fake_get_llm_router():
        return _FakeRouter()

    # monkeypatch the dependency resolver inside main module
    monkeypatch.setattr(main_mod, "get_llm_router", fake_get_llm_router, raising=False)

    resp = await client.get("/health/details", params={"probe_llm": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")  # should be ok with our fake
    assert data["probes"]["llm"]["requested"] is True
    assert data["probes"]["llm"]["ok"] is True


@pytest.mark.anyio
async def test_health_details_db_probe(client, monkeypatch):
    # Provide a fake async engine with async connect context manager
    class _FakeResult:
        def scalar(self):
            return 1
    class _FakeConn:
        async def execute(self, _):
            return _FakeResult()
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
    class _FakeEngine:
        def __init__(self):
            pass
        def connect(self):
            # return async context manager
            return _FakeConn()

    import main as main_mod

    # main.health_details prefers .db.session.engine if available; override import resolution by setting attribute
    monkeypatch.setattr(main_mod, "create_async_engine", lambda *a, **k: _FakeEngine(), raising=False)

    resp = await client.get("/health/details", params={"probe_db": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["probes"]["db"]["requested"] is True
    assert data["probes"]["db"]["ok"] is True
