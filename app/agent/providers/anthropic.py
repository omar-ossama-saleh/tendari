"""Anthropic provider adapter (Claude). Translates the normalized format to the
Messages API and back. Non-streaming for M2; token streaming is added in M3."""

from __future__ import annotations

from typing import Any

from app.agent.providers.base import EmitFn, LLMResponse, ToolCall, Usage
from app.config import settings


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.chat_model

    @staticmethod
    def _to_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        i = 0
        while i < len(messages):
            m = messages[i]
            role = m["role"]
            if role == "tool":
                # Merge consecutive tool results into ONE user message (Anthropic
                # expects all tool_results for a turn grouped together).
                blocks = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tm = messages[i]
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tm["tool_call_id"],
                            "content": tm["content"],
                        }
                    )
                    i += 1
                out.append({"role": "user", "content": blocks})
                continue
            if role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                content: list[dict[str, Any]] = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["arguments"],
                        }
                    )
                # Anthropic rejects an empty assistant turn; use a placeholder
                # text block if there's neither text nor a tool_use.
                if not content:
                    content = [{"type": "text", "text": "(no content)"}]
                out.append({"role": "assistant", "content": content})
            i += 1
        return out

    @staticmethod
    def _to_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]

    async def chat(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
        emit: EmitFn | None = None,
    ) -> LLMResponse:
        resp = await self._client.messages.create(
            model=self._model,
            system=system,
            messages=self._to_messages(messages),
            tools=self._to_tools(tools) or [],
            max_tokens=settings.max_output_tokens,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        text = "".join(text_parts) or None
        usage = Usage(
            model=resp.model,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
        if stream and emit is not None and text:
            for word in text.split(" "):
                await emit("token", {"text": word + " "})
        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage)
