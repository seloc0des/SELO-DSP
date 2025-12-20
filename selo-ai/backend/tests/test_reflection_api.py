import os
import pytest


class _FakeProcessor:
    def __init__(self):
        self.reflection_repo = self
        self.generated = []
        self._items = {}

    async def generate_reflection(self, **kwargs):
        rid = f"r-{len(self.generated)+1}"
        item = {
            "reflection_id": rid,
            "reflection_type": kwargs.get("reflection_type", "message"),
            "user_profile_id": kwargs.get("user_profile_id"),
            "result": "ok",
        }
        self.generated.append(item)
        self._items[rid] = item
        return item

    # repo
    async def list_reflections(self, user_profile_id: str, reflection_type=None, limit=10, offset=0):
        out = [v for v in self._items.values() if v["user_profile_id"] == user_profile_id]
        if reflection_type:
            out = [v for v in out if v["reflection_type"] == reflection_type]
        return out[offset:offset+limit]

    async def get_reflection(self, reflection_id: str):
        return self._items.get(reflection_id)

    async def delete_reflection(self, reflection_id: str):
        return self._items.pop(reflection_id, None) is not None


class _FakeRouter:
    async def route(self, **kwargs):
        return {"ok": True}


@pytest.mark.anyio
async def test_reflection_generate_requires_system_key(client, monkeypatch):
    from api import reflection as refl

    fake_proc = _FakeProcessor()
    fake_router = _FakeRouter()

    async def fake_get_proc():
        return fake_proc

    async def fake_get_router():
        return fake_router

    monkeypatch.setattr(refl, "get_reflection_processor", fake_get_proc)
    monkeypatch.setattr(refl, "get_llm_router", fake_get_router)

    # No system key set -> should 403 for user-triggered
    os.environ.pop("SELO_SYSTEM_API_KEY", None)
    payload = {
        "reflection_type": "message",
        "user_profile_id": "u1",
        "memory_ids": [],
        "trigger_source": "user"
    }
    r = await client.post("/reflection/generate", json=payload)
    assert r.status_code == 403

    # Set system key and pass it in header
    os.environ["SELO_SYSTEM_API_KEY"] = "secret"
    r = await client.post("/reflection/generate", json=payload, headers={"api-key": "secret"})
    assert r.status_code == 200
    data = r.json()
    assert data["reflection_id"].startswith("r-")


@pytest.mark.anyio
async def test_reflection_list_get_delete(client, monkeypatch):
    from api import reflection as refl

    fake_proc = _FakeProcessor()

    # seed one reflection via internal state
    fake_proc._items["r-1"] = {
        "reflection_id": "r-1",
        "reflection_type": "message",
        "user_profile_id": "u1",
        "result": "ok",
    }

    async def fake_get_proc():
        return fake_proc

    monkeypatch.setattr(refl, "get_reflection_processor", fake_get_proc)

    r = await client.get("/reflection/list", params={"user_profile_id": "u1"})
    assert r.status_code == 200
    assert r.json()["count"] == 1

    r = await client.get("/reflection/r-1")
    assert r.status_code == 200
    assert r.json()["reflection_id"] == "r-1"

    r = await client.delete("/reflection/r-1")
    assert r.status_code == 200
    r = await client.get("/reflection/r-1")
    assert r.status_code == 404
