import pytest
import asyncio

from typing import Dict, Any

# Import router and craft simple fake controllers
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "selo-ai" / "backend"))

from llm.router import LLMRouter  # type: ignore


class _FakeLLM:
    def __init__(self, name: str):
        self.default_model = name
        self.completions: list[Dict[str, Any]] = []
        self.embeddings: list[Dict[str, Any]] = []

    async def complete(self, **kwargs):
        self.completions.append(kwargs)
        return {"completion": f"done:{self.default_model}"}

    async def get_embedding(self, **kwargs):
        self.embeddings.append(kwargs)
        return {"embedding": [0.0, 1.0, 2.0]}


@pytest.mark.anyio
async def test_routing_chat_vs_analytical():
    conv = _FakeLLM("conv-model")
    anal = _FakeLLM("anal-model")
    router = LLMRouter(conv, anal)

    r1 = await router.route(task_type="chat", prompt="hello")
    r2 = await router.route(task_type="reflection", prompt="think")

    assert r1["llm_role"] == "conversational"
    assert r2["llm_role"] == "analytical"

    # usage log captured
    log = router.get_usage_log()
    assert any(e["task_type"] == "chat" and e["llm_role"] == "conversational" for e in log)
    assert any(e["task_type"] == "reflection" and e["llm_role"] == "analytical" for e in log)


@pytest.mark.anyio
async def test_routing_embedding_calls_get_embedding():
    conv = _FakeLLM("conv-model")
    anal = _FakeLLM("anal-model")
    router = LLMRouter(conv, anal)

    out = await router.route(task_type="embedding", text="abc")
    assert "embedding" in out
    assert out["task_type"] == "embedding"
    assert out["llm_role"] == "analytical"  # falls to analytical by default for non-chat
