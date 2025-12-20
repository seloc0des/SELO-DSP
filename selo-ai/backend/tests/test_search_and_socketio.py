import json
import types
import pytest

from typing import Any

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "selo-ai" / "backend"))

from backend.main import app, get_socketio_app  # type: ignore


@pytest.mark.anyio
async def test_search_endpoint_with_mocks(client, monkeypatch):
    class _Resp:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {
                "web": {
                    "results": [
                        {"title": "t1", "url": "http://example.com/1", "description": "d1"},
                        {"title": "t2", "url": "http://example.com/2", "description": "d2"},
                    ]
                }
            }
            self.text = "ok"
        def json(self):
            return self._payload

    def fake_requests_get(url, *args, **kwargs):
        return _Resp()

    class _Popen:
        def __init__(self, *a, **k):
            self.returncode = 0
        def communicate(self):
            return ("summarized", "")

    # Patch network and subprocess calls
    monkeypatch.setattr("main.requests.get", fake_requests_get)
    monkeypatch.setattr("main.subprocess.Popen", _Popen)

    # Also ensure Brave key presence to bypass 500
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test")

    resp = await client.post("/search", params={"query": "weather boston"})
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    assert isinstance(data["result"], str)


def test_get_socketio_app_returns_asgi_or_app():
    asgi = get_socketio_app()
    # When no socketio server set, it should return the FastAPI app itself
    # which is callable with scope, receive, send in ASGI context. Here we just check it's not None.
    assert asgi is not None
