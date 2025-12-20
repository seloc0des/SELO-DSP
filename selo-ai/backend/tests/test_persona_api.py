import pytest


class _FakePersonaIntegration:
    def __init__(self):
        self.persona_repo = self
        self.persona_engine = self
        self._personas = []
        self._traits = {}

    async def initialize(self):
        return True

    async def close(self):
        return True

    async def ensure_default_persona(self, user_id: str, name: str):
        pid = "p-1"
        self._personas.append({"id": pid, "user_id": user_id, "name": name, "is_default": True})
        return {"success": True, "persona_id": pid, "created": True, "name": name}

    async def get_system_prompt(self, persona_id: str):
        return {"success": True, "system_prompt": f"You are {persona_id}"}

    # repo methods
    async def get_personas_for_user(self, user_id: str, is_active=None):
        class P:
            def __init__(self, d):
                self._d = d
            def to_dict(self):
                return self._d
        return [P(p) for p in self._personas if p["user_id"] == user_id]

    async def get_traits_for_persona(self, persona_id: str, category=None):
        class T:
            def __init__(self, d):
                self._d = d
            def to_dict(self):
                return self._d
        traits = self._traits.get(persona_id, [])
        if category:
            traits = [t for t in traits if t.get("category") == category]
        return [T(t) for t in traits]

    # engine method
    async def add_trait(self, persona_id: str, trait_data: dict):
        self._traits.setdefault(persona_id, []).append(trait_data)
        return trait_data


class _FakeRouter:
    def __init__(self):
        self.calls = []
    async def route(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


@pytest.mark.anyio
async def test_persona_endpoints_smoke(client, monkeypatch):
    from api import dependencies

    fake_integration = _FakePersonaIntegration()
    fake_router = _FakeRouter()

    async def fake_get_pi():
        return fake_integration

    async def fake_get_router():
        return fake_router

    monkeypatch.setattr(dependencies, "get_persona_integration", fake_get_pi)
    monkeypatch.setattr(dependencies, "get_llm_router", fake_get_router)

    # ensure-default
    payload = {"user_id": "u1", "name": "SELO", "description": ""}
    r = await client.post("/persona/ensure-default", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["persona_id"] == "p-1"

    # system-prompt
    r = await client.get("/persona/system-prompt/p-1")
    assert r.status_code == 200
    assert r.json()["data"]["system_prompt"].startswith("You are")

    # user personas
    r = await client.get("/persona/user/u1")
    assert r.status_code == 200
    d = r.json()
    assert d["data"]["count"] >= 1

    # add trait
    trait = {
        "category": "style",
        "name": "concise",
        "value": 0.9,
        "description": "be concise",
        "confidence": 0.8,
        "stability": 0.5
    }
    r = await client.post("/persona/traits/p-1", json=trait)
    assert r.status_code == 200

    # get traits
    r = await client.get("/persona/traits/p-1")
    assert r.status_code == 200
    assert r.json()["data"]["count"] >= 1
