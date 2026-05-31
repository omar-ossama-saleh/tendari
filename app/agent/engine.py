"""The agent engine — the core decision loop. Domain-agnostic.

Flow (one user message):
  persist user message
  → build message list (system prompt + history, truncated to a token budget)
  → loop, capped at MAX_AGENT_ITERATIONS:
        call provider(messages, tools)         record usage every call
        no tool calls?  → persist + return the answer
        tool calls?     → authorize + execute them CONCURRENTLY, one failure
                          becomes a structured error fed back to the model,
                          never a 500; append results; loop again
  → hit the cap → return a graceful fallback

Design decisions (see handoff §6 — don't change without understanding why):
  1. Iteration cap bounds tool loops / runaway spend.
  2. Tool failure → structured error back to the model, not an exception.
  3. Tools run in parallel via asyncio.gather, each with its OWN db session.
  4. authorize() runs before execute() — the model can never reach another
     workspace's data, because it only supplies tool args, not a workspace id.
  5. provider.chat() hides Anthropic/OpenAI/mock differences.
  6. Oldest history is truncated to a token budget before each call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from time import monotonic
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.agent.providers.base import EmitFn, Provider, ToolCall
from app.agent.registry import ToolContext, ToolError, ToolRegistry
from app.agent.prompts import build_system_prompt
from app.config import settings
from app.models import Conversation, Message, ToolCall as ToolCallRow, Workspace
from app.observability.usage import record_usage

logger = logging.getLogger("tendari.engine")

_ENDPOINT = "conversations.messages"
_FALLBACK = (
    "I'm sorry — I wasn't able to complete that within my step limit. "
    "Let me hand this to a human teammate, or feel free to rephrase."
)
_PROVIDER_ERROR_FALLBACK = (
    "I'm having trouble reaching my AI service right now. "
    "Please try again in a moment."
)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class ExecResult:
    status: str  # success | error
    latency_ms: int
    data: Any = None
    error: str | None = None

    def model_payload(self) -> dict[str, Any]:
        """What the model sees as the tool result (no internal latency)."""
        if self.status == "success":
            return {"status": "success", "data": self.data}
        return {"status": "error", "error": self.error}


@dataclass
class AgentResult:
    final_message_id: uuid.UUID | None
    text: str
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: Decimal = Decimal("0")


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
async def _persist_message(
    session: AsyncSession, conversation_id: uuid.UUID, role: str, content: str | None
) -> Message:
    # Explicit timestamp: Postgres now() is constant within a transaction, so we
    # set wall-clock time per row to keep intra-request message ordering stable.
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        created_at=datetime.now(timezone.utc),
    )
    session.add(message)
    await session.flush()
    return message


async def _persist_tool_call(
    session: AsyncSession, message_id: uuid.UUID, call: ToolCall, result: ExecResult
) -> None:
    session.add(
        ToolCallRow(
            message_id=message_id,
            tool_name=call.name,
            arguments=call.arguments,
            result=result.data,
            status=result.status,
            error=result.error,
            latency_ms=result.latency_ms,
            created_at=datetime.now(timezone.utc),
        )
    )
    await session.flush()


async def _load_history(session: AsyncSession, conversation_id: uuid.UUID) -> list[Message]:
    rows = await session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
        .options(selectinload(Message.tool_calls))
    )
    return list(rows)


def _history_to_messages(history: list[Message]) -> list[dict[str, Any]]:
    """Rebuild the normalized provider message list from persisted history."""
    out: list[dict[str, Any]] = []
    for msg in history:
        if msg.role == "user":
            out.append({"role": "user", "content": msg.content or ""})
        elif msg.role == "assistant":
            if msg.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": str(tc.id),
                                "name": tc.tool_name,
                                "arguments": tc.arguments,
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    payload = (
                        {"status": "success", "data": tc.result}
                        if tc.status == "success"
                        else {"status": "error", "error": tc.error}
                    )
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(tc.id),
                            "name": tc.tool_name,
                            "content": json.dumps(payload),
                        }
                    )
            else:
                out.append({"role": "assistant", "content": msg.content or ""})
        # 'tool'-role rows aren't persisted directly; results live in tool_calls.
    return out


# --------------------------------------------------------------------------- #
# Context budgeting
# --------------------------------------------------------------------------- #
def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _message_tokens(message: dict[str, Any]) -> int:
    total = _estimate_tokens(str(message.get("content") or ""))
    for tc in message.get("tool_calls", []):
        total += _estimate_tokens(json.dumps(tc.get("arguments", {})))
    return total


def _truncate_to_budget(messages: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Keep the most recent messages within a token budget, preserving validity.

    Drops oldest turns first, then trims any leading messages that aren't a
    `user` message so the result is a valid request (no orphan tool_result, and
    a user-first sequence).
    """
    kept: list[dict[str, Any]] = []
    total = 0
    for message in reversed(messages):
        cost = _message_tokens(message)
        if kept and total + cost > budget:
            break
        kept.insert(0, message)
        total += cost
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


# --------------------------------------------------------------------------- #
# Tool execution
# --------------------------------------------------------------------------- #
async def safe_execute(registry: ToolRegistry, call: ToolCall, ctx: ToolContext) -> ExecResult:
    """Authorize then execute a tool. Any failure becomes a structured error."""
    t0 = monotonic()
    try:
        # SECURITY: scope/arg check FIRST, before any side effect.
        args = await registry.authorize(call.name, call.arguments, ctx)
        data = await registry.execute(call.name, args, ctx)
        return ExecResult("success", _ms(t0), data=data)
    except ToolError as exc:
        return ExecResult("error", _ms(t0), error=str(exc))
    except Exception:  # never let a tool bug 500 the whole request
        logger.exception("Unexpected error in tool %s", call.name)
        return ExecResult("error", _ms(t0), error="The tool failed unexpectedly.")


def _ms(t0: float) -> int:
    return int((monotonic() - t0) * 1000)


def _done_payload(message_id: uuid.UUID, prompt: int, completion: int, cost: Decimal) -> dict[str, Any]:
    return {
        "message_id": str(message_id),
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cost_usd": float(cost),
        },
    }


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
async def run_agent(
    *,
    session: AsyncSession,
    session_factory: async_sessionmaker,
    workspace: Workspace,
    conversation: Conversation,
    user_text: str,
    registry: ToolRegistry,
    provider: Provider,
    stream: bool = False,
    emit: EmitFn | None = None,
) -> AgentResult:
    # Defense-in-depth: the router resolves the conversation scoped to the
    # workspace, but assert the invariant here too (the engine trusts ctx).
    if conversation.workspace_id != workspace.id:
        raise RuntimeError("conversation/workspace mismatch")

    async def _emit(event: str, data: dict[str, Any]) -> None:
        if emit is not None:
            await emit(event, data)

    await _persist_message(session, conversation.id, "user", user_text)

    system = build_system_prompt(workspace)
    tools = registry.schemas()
    history = await _load_history(session, conversation.id)
    messages = _truncate_to_budget(
        _history_to_messages(history), settings.context_token_budget
    )

    ctx = ToolContext(workspace=workspace, conversation=conversation, session_factory=session_factory)

    tool_summary: list[dict[str, str]] = []
    prompt_tokens = completion_tokens = 0
    cost_total = Decimal("0")

    for _step in range(settings.max_agent_iterations):
        t0 = monotonic()
        try:
            response = await provider.chat(
                system=system, messages=messages, tools=tools, stream=stream, emit=emit
            )
        except Exception:
            # A provider hiccup (timeout/429/5xx) must degrade gracefully, not
            # 500 the customer. Persist a friendly message and return what we've
            # accumulated; usage from earlier iterations is already durably saved.
            logger.exception("Provider call failed; returning graceful fallback")
            final = await _persist_message(
                session, conversation.id, "assistant", _PROVIDER_ERROR_FALLBACK
            )
            await _emit("done", _done_payload(final.id, prompt_tokens, completion_tokens, cost_total))
            return AgentResult(
                final_message_id=final.id,
                text=_PROVIDER_ERROR_FALLBACK,
                tool_calls=tool_summary,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_total,
            )
        latency = _ms(t0)

        if response.usage is not None:
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            # Autonomous commit — cost audit survives a later request rollback.
            cost_total += await record_usage(
                session_factory,
                workspace_id=workspace.id,
                conversation_id=conversation.id,
                usage=response.usage,
                latency_ms=latency,
                endpoint=_ENDPOINT,
            )

        # No tool calls → the model is answering. Done.
        if not response.tool_calls:
            text = response.text or ""
            final = await _persist_message(session, conversation.id, "assistant", text)
            await _emit("done", _done_payload(final.id, prompt_tokens, completion_tokens, cost_total))
            return AgentResult(
                final_message_id=final.id,
                text=text,
                tool_calls=tool_summary,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_total,
            )

        # Persist the assistant tool-calling turn (results attached after exec).
        assistant = await _persist_message(
            session, conversation.id, "assistant", response.text
        )
        messages.append(
            {
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in response.tool_calls
                ],
            }
        )

        # Run the requested tools CONCURRENTLY. Each opens its own session.
        async def _run_one(call: ToolCall) -> tuple[ToolCall, ExecResult]:
            await _emit("tool_call_start", {"tool": call.name, "args": call.arguments})
            result = await safe_execute(registry, call, ctx)
            await _emit("tool_call_result", {"tool": call.name, "status": result.status})
            return call, result

        executed = await asyncio.gather(*[_run_one(c) for c in response.tool_calls])

        # Persist results + feed them back (sequential: shared request session).
        for call, result in executed:
            await _persist_tool_call(session, assistant.id, call, result)
            tool_summary.append({"tool": call.name, "status": result.status})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(result.model_payload()),
                }
            )

    # Iteration cap reached — fail gracefully.
    final = await _persist_message(session, conversation.id, "assistant", _FALLBACK)
    await _emit("done", _done_payload(final.id, prompt_tokens, completion_tokens, cost_total))
    return AgentResult(
        final_message_id=final.id,
        text=_FALLBACK,
        tool_calls=tool_summary,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_total,
    )
