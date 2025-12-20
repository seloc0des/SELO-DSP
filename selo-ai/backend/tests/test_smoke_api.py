import pytest


@pytest.mark.anyio
async def test_health_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert "time" in data


@pytest.mark.anyio
async def test_chat_basic_flow(client):
    payload = {
        "session_id": "test-session",
        "prompt": "Say hello",
        "model": "mistral:latest",
    }
    resp = await client.post("/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data and isinstance(data["response"], str)
    assert "turn_id" in data and isinstance(data["turn_id"], str)
    assert "history" in data and isinstance(data["history"], list)
