import pytest


class FakeSio:
    def __init__(self):
        self.events = []
    async def emit(self, event, data, room=None, namespace=None):
        self.events.append({
            "event": event,
            "data": data,
            "room": room,
            "namespace": namespace,
        })


@pytest.mark.anyio
async def test_emit_reflection_event_targeted():
    # Lazy import to avoid side effects
    from backend.socketio.namespaces.reflection import ReflectionNamespace

    ns = ReflectionNamespace(reflection_processor=None)
    fake_sio = FakeSio()
    ns.sio_server = fake_sio
    # Two clients connected, only one matches user_id
    ns.connected_clients = {
        "sid-a": {"user_id": "user-1"},
        "sid-b": {"user_id": "user-2"},
    }

    await ns.emit_reflection_event("reflection_generated", {"x": 1}, user_id="user-2")

    # Should emit once to room=sid-b in reflection namespace
    assert len(fake_sio.events) == 1
    ev = fake_sio.events[0]
    assert ev["event"] == "reflection_generated"
    assert ev["data"] == {"x": 1}
    assert ev["room"] == "sid-b"
    assert ev["namespace"] == "/reflection"


@pytest.mark.anyio
async def test_emit_reflection_event_broadcast():
    from backend.socketio.namespaces.reflection import ReflectionNamespace

    ns = ReflectionNamespace(reflection_processor=None)
    fake_sio = FakeSio()
    ns.sio_server = fake_sio
    ns.connected_clients = {
        "sid-a": {"user_id": "user-1"},
        "sid-b": {"user_id": "user-2"},
    }

    await ns.emit_reflection_event("reflection_updated", {"y": 2})

    # Broadcast is a single emit with no room in this implementation
    assert len(fake_sio.events) == 1
    ev = fake_sio.events[0]
    assert ev["event"] == "reflection_updated"
    assert ev["data"] == {"y": 2}
    assert ev["room"] is None
    assert ev["namespace"] == "/reflection"
