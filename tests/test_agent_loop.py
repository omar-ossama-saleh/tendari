"""Agent engine: tool execution, error path, iteration cap, context budgeting.

Driven by a fake provider + fake session so the loop is exercised with no DB and
no API keys.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from pydantic import BaseModel

from app.agent.engine import _truncate_to_budget, run_agent, safe_execute
from app.agent.providers.base import LLMResponse, ToolCall, Usage
from app.agent.providers.mock import MockProvider
from app.agent.registry import ToolContext, ToolError, ToolRegistry, ToolSpec
from app.config import settings
from app.models import Conversation, Message, ToolCall as ToolCallRow, UsageRecord, Workspace


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeSession:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.usage: list[UsageRecord] = []
        self.tool_calls: list[ToolCallRow] = []

    def add(self, obj: Any) -> None:
        if isinstance(obj, Message):
            if obj.id is None:
                obj.id = uuid.uuid4()
            self.messages.append(obj)
        elif isinstance(obj, ToolCallRow):
            if obj.id is None:
                obj.id = uuid.uuid4()
            self.tool_calls.append(obj)
        elif isinstance(obj, UsageRecord):
            self.usage.append(obj)

    async def flush(self) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def scalars(self, *_args: Any, **_kwargs: Any):
        # _load_history is called once, before any loop messages are added.
        return list(self.messages)

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False


class ScriptedProvider:
    """Returns scripted turns; once exhausted, always returns a plain answer."""

    name = "scripted"

    def __init__(self, turns: list[LLMResponse], fallback_text: str = "Done.") -> None:
        self._turns = turns
        self._fallback = fallback_text
        self.calls = 0

    async def chat(self, *, system, messages, tools, stream=False, emit=None) -> LLMResponse:
        i = self.calls
        self.calls += 1
        if i < len(self._turns):
            return self._turns[i]
        return LLMResponse(text=self._fallback, tool_calls=[], usage=_usage())


def _usage() -> Usage:
    return Usage(model=settings.chat_model, prompt_tokens=10, completion_tokens=5)


def _tool_turn(name: str, args: dict | None = None) -> LLMResponse:
    return LLMResponse(
        text=None,
        tool_calls=[ToolCall(id=f"c_{uuid.uuid4().hex[:6]}", name=name, arguments=args or {"query": "x"})],
        usage=_usage(),
    )


class DummyArgs(BaseModel):
    query: str = "x"


def _registry(handler) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(name="dummy", description="d", args_model=DummyArgs, handler=handler))
    return reg


def _ctx() -> ToolContext:
    ws = Workspace(id=uuid.uuid4(), name="Acme", api_key_hash="x")
    conv = Conversation(id=uuid.uuid4(), workspace_id=ws.id)
    return ToolContext(workspace=ws, conversation=conv, session_factory=lambda: None)


async def _run(provider, registry) -> Any:
    ctx = _ctx()
    return await run_agent(
        session=FakeSession(),
        session_factory=FakeSession,  # record_usage opens its own (fake) session
        workspace=ctx.workspace,
        conversation=ctx.conversation,
        user_text="hello",
        registry=registry,
        provider=provider,
    )


# --------------------------------------------------------------------------- #
# safe_execute
# --------------------------------------------------------------------------- #
async def test_safe_execute_success() -> None:
    async def ok(args, ctx):
        return {"value": 42}

    res = await safe_execute(_registry(ok), ToolCall("c1", "dummy", {"query": "x"}), _ctx())
    assert res.status == "success"
    assert res.data == {"value": 42}
    assert res.latency_ms >= 0


async def test_safe_execute_tool_error_is_structured() -> None:
    async def boom(args, ctx):
        raise ToolError("not allowed")

    res = await safe_execute(_registry(boom), ToolCall("c1", "dummy", {"query": "x"}), _ctx())
    assert res.status == "error"
    assert "not allowed" in res.error


async def test_safe_execute_unexpected_exception_is_caught() -> None:
    async def crash(args, ctx):
        raise RuntimeError("kaboom")

    res = await safe_execute(_registry(crash), ToolCall("c1", "dummy", {"query": "x"}), _ctx())
    assert res.status == "error"
    assert "kaboom" not in res.error  # internal detail not leaked to the model


async def test_safe_execute_unknown_tool() -> None:
    async def ok(args, ctx):
        return {}

    res = await safe_execute(_registry(ok), ToolCall("c1", "nope", {}), _ctx())
    assert res.status == "error"
    assert "Unknown tool" in res.error


# --------------------------------------------------------------------------- #
# loop behavior
# --------------------------------------------------------------------------- #
async def test_loop_single_tool_then_answer() -> None:
    async def ok(args, ctx):
        return {"results": []}

    provider = ScriptedProvider([_tool_turn("dummy"), LLMResponse(text="Final answer.", usage=_usage())])
    result = await _run(provider, _registry(ok))
    assert result.text == "Final answer."
    assert result.tool_calls == [{"tool": "dummy", "status": "success"}]
    assert result.prompt_tokens > 0 and result.cost_usd > 0


async def test_loop_tool_error_then_answer() -> None:
    async def boom(args, ctx):
        raise ToolError("denied")

    provider = ScriptedProvider([_tool_turn("dummy"), LLMResponse(text="Recovered.", usage=_usage())])
    result = await _run(provider, _registry(boom))
    assert result.text == "Recovered."
    assert result.tool_calls == [{"tool": "dummy", "status": "error"}]


async def test_loop_hits_iteration_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_agent_iterations", 3)

    async def ok(args, ctx):
        return {"results": []}

    class AlwaysTool:
        name = "always"

        async def chat(self, *, system, messages, tools, stream=False, emit=None):
            return _tool_turn("dummy")

    result = await _run(AlwaysTool(), _registry(ok))
    assert "step limit" in result.text.lower()
    assert len(result.tool_calls) == 3  # one tool call per capped iteration


async def test_loop_provider_error_degrades_gracefully() -> None:
    async def ok(args, ctx):
        return {}

    class FailingProvider:
        name = "failing"

        async def chat(self, *, system, messages, tools, stream=False, emit=None):
            raise RuntimeError("upstream 429")

    # Must NOT raise; returns a graceful fallback instead of 500ing.
    result = await _run(FailingProvider(), _registry(ok))
    assert "trouble" in result.text.lower()
    assert result.final_message_id is not None


async def test_loop_parallel_tool_calls() -> None:
    async def ok(args, ctx):
        return {"ran": True}

    multi = LLMResponse(
        text=None,
        tool_calls=[
            ToolCall("a", "dummy", {"query": "1"}),
            ToolCall("b", "dummy", {"query": "2"}),
        ],
        usage=_usage(),
    )
    provider = ScriptedProvider([multi, LLMResponse(text="Both done.", usage=_usage())])
    result = await _run(provider, _registry(ok))
    assert result.text == "Both done."
    assert len(result.tool_calls) == 2


# --------------------------------------------------------------------------- #
# context budgeting
# --------------------------------------------------------------------------- #
def test_truncate_keeps_recent_and_starts_with_user() -> None:
    messages = [
        {"role": "user", "content": "x" * 400},       # old, ~100 tokens
        {"role": "assistant", "content": "y" * 400},
        {"role": "user", "content": "recent question"},
    ]
    kept = _truncate_to_budget(messages, budget=20)
    assert kept[0]["role"] == "user"
    assert kept[-1]["content"] == "recent question"


def test_truncate_drops_leading_tool_message() -> None:
    messages = [
        {"role": "tool", "tool_call_id": "x", "name": "t", "content": "{}"},
        {"role": "user", "content": "hi"},
    ]
    kept = _truncate_to_budget(messages, budget=10_000)
    assert kept[0]["role"] == "user"


# --------------------------------------------------------------------------- #
# mock provider
# --------------------------------------------------------------------------- #
async def test_mock_provider_searches_then_answers() -> None:
    provider = MockProvider()
    tools = [{"name": "search_help_docs", "description": "d", "input_schema": {}}]

    first = await provider.chat(
        system="s", messages=[{"role": "user", "content": "return window?"}], tools=tools
    )
    assert first.tool_calls and first.tool_calls[0].name == "search_help_docs"

    tool_result = {
        "role": "tool",
        "tool_call_id": first.tool_calls[0].id,
        "name": "search_help_docs",
        "content": '{"status":"success","data":{"results":[{"doc_title":"Return Policy","content":"30 day window","score":0.9}]}}',
    }
    second = await provider.chat(
        system="s",
        messages=[{"role": "user", "content": "return window?"}, tool_result],
        tools=tools,
    )
    assert not second.tool_calls
    assert "Return Policy" in second.text
