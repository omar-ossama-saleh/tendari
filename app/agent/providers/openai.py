"""OpenAI provider adapter. Translates the normalized format to Chat Completions
tool-calling and back. Non-streaming for M2; token streaming is added in M3."""

from __future__ import annotations

import json
from typing import Any

from app.agent.providers.base import EmitFn, LLMResponse, ToolCall, Usage
from app.config import settings


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.chat_model

    @staticmethod
    def _to_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            role = m["role"]
            if role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content")}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in m["tool_calls"]
                    ]
                out.append(msg)
            elif role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["tool_call_id"],
                        "content": m["content"],
                    }
                )
        return out

    @staticmethod
    def _to_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
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
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_messages(system, messages),
            # Current canonical Chat Completions param (newer models reject max_tokens).
            "max_completion_tokens": settings.max_output_tokens,
        }
        if tools:
            kwargs["tools"] = self._to_tools(tools)

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message

        tool_calls: list[ToolCall] = []
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        text = choice.content or None
        usage = Usage(
            model=resp.model,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )
        if stream and emit is not None and text:
            for word in text.split(" "):
                await emit("token", {"text": word + " "})
        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage)
