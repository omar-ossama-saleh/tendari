"""Provider abstraction — the contract that hides Anthropic vs OpenAI vs mock.

The engine speaks ONE normalized message format and one tool-schema format; each
concrete provider adapter translates to/from its SDK. Swap providers by config.

Normalized message shapes (list passed to ``Provider.chat``):
  {"role": "user", "content": str}
  {"role": "assistant", "content": str | None,
   "tool_calls": [{"id": str, "name": str, "arguments": dict}]}   # tool_calls optional
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}  # content = JSON string

Tool schema shape (list passed as ``tools``):
  {"name": str, "description": str, "input_schema": <JSON Schema dict>}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

# An emit callback: (event_name, data) -> awaitable. Used for SSE streaming.
EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    model: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None


class Provider(ABC):
    """A chat-completions provider with tool-calling support."""

    name: str

    @abstractmethod
    async def chat(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
        emit: EmitFn | None = None,
    ) -> LLMResponse:
        """Run one model turn. When ``stream`` and ``emit`` are set, the provider
        emits ``token`` events as text arrives; it must still return the full
        ``LLMResponse`` at the end."""
        raise NotImplementedError
