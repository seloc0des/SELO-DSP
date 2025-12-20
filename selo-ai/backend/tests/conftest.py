import asyncio
import os
import pytest
from typing import AsyncGenerator
from httpx import AsyncClient

# Ensure backend package import works when running from repo root
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.main import app  # type: ignore


@pytest.fixture(scope="session")
def anyio_backend():
    # Use asyncio for httpx AsyncClient
    return "asyncio"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def _patch_services_for_tests() -> AsyncGenerator[None, None]:
    """
    Patch app.state.services with lightweight fakes to avoid external dependencies
    (DB, Ollama, Brave API) during API smoke tests.
    """

    class _FakeUser:
        def __init__(self):
            self.id = "test-user"

    class _FakeConversation:
        def __init__(self, session_id: str):
            import uuid
            self.id = uuid.uuid4()
            self.session_id = session_id

    class _FakeConversationRepo:
        def __init__(self):
            self._msgs = {}

        async def get_or_create_conversation(self, session_id: str, user_id: str):
            conv = _FakeConversation(session_id)
            return conv

        async def add_message(self, conversation_id: str, role: str, content: str, **kwargs):
            self._msgs.setdefault(conversation_id, []).append({
                "role": role,
                "content": content
            })
            return {"id": "msg", "role": role, "content": content}

        async def get_conversation_history(self, session_id: str, limit: int = 50):
            # Return a tiny history with the last user prompt only for simplicity
            return [{"role": "user", "content": "Hello"}]

    class _FakeUserRepo:
        async def get_or_create_default_user(self):
            return _FakeUser()

    class _FakeLLMRouter:
        async def route(self, task_type: str, prompt: str):
            return {"completion": "Hello from fake LLM"}

    class _FakeReflectionProcessor:
        async def generate_reflection(self, **kwargs):
            return {"content": "Inner reflection result"}

    # Save originals if any
    original_services = getattr(app.state, "services", None)

    # Disable lifespan to prevent startup DI/imports during tests
    try:
        app.router.lifespan_context = None  # type: ignore[attr-defined]
    except Exception:
        pass

    app.state.services = {
        "llm_router": _FakeLLMRouter(),
        "conversation_repo": _FakeConversationRepo(),
        "user_repo": _FakeUserRepo(),
        "reflection_processor": _FakeReflectionProcessor(),
        "scheduler_config": {},
    }

    try:
        yield
    finally:
        # Restore original if existed
        if original_services is not None:
            app.state.services = original_services


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
