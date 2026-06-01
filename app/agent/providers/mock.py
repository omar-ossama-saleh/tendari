"""Deterministic mock provider — lets the agent run end-to-end with no API keys.

It simulates a tool-using support agent: it routes order-status questions to
``lookup_order`` and other questions to ``search_help_docs``; once tool results
are present it composes an answer (order status, or a cited help-doc passage).
Good enough to demo and to unit-test the engine loop; real, fluent answers come
from the Anthropic/OpenAI adapters.

This module intentionally knows a little about the vertical tool names + result
shapes — it is a DEMO SIMULATOR standing in for a real model, not production
logic, so the coupling is deliberate and confined here.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.agent.providers.base import EmitFn, LLMResponse, ToolCall, Usage
from app.config import settings

_ORDER_NUMBER_RE = re.compile(r"#?\b(\d{3,})\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_ORDER_HINTS = ("order", "where", "track", "shipping", "delivery", "package")
_ESCALATE_HINTS = ("escalate", "human", "real person", "agent", "manager", "supervisor", "speak to someone")
_TICKET_HINTS = ("ticket", "file a", "open a complaint", "log a", "raise a")
_REFUND_HINTS = ("refund", "money back", "return", "damaged", "broken", "defective", "reimburse")


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class MockProvider:
    name = "mock"

    def _usage(self, system: str, messages: list[dict], completion: str) -> Usage:
        prompt = system + "".join(str(m.get("content") or "") for m in messages)
        return Usage(
            model=settings.chat_model,
            prompt_tokens=_estimate_tokens(prompt),
            completion_tokens=_estimate_tokens(completion),
        )

    @staticmethod
    def _latest_user_text(messages: list[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return str(m.get("content") or "")
        return ""

    @staticmethod
    def _current_turn(messages: list[dict]) -> list[dict]:
        """Messages after the latest user message — i.e. THIS turn's activity.

        Without this the mock would treat a prior turn's tool result as the
        current one and never call a tool again in a multi-turn conversation.
        """
        last_user = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1
        )
        return messages[last_user + 1 :]

    @staticmethod
    def _tool_payloads(messages: list[dict]) -> list[dict]:
        payloads: list[dict] = []
        for m in messages:
            if m.get("role") != "tool":
                continue
            try:
                payload = json.loads(m.get("content") or "{}")
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "success" and isinstance(payload.get("data"), dict):
                payloads.append(payload["data"])
        return payloads

    def _choose_first_tool(self, user_text: str, tool_names: set[str]) -> ToolCall | None:
        lowered = user_text.lower()

        if "escalate_to_human" in tool_names and any(h in lowered for h in _ESCALATE_HINTS):
            return ToolCall("mock_escalate", "escalate_to_human", {"reason": user_text[:300]})

        if "create_ticket" in tool_names and any(h in lowered for h in _TICKET_HINTS):
            subject = user_text.strip()[:60] or "Support request"
            return ToolCall(
                "mock_ticket", "create_ticket",
                {"subject": subject, "body": user_text, "priority": "normal"},
            )

        order_match = _ORDER_NUMBER_RE.search(user_text)
        if (
            "initiate_refund" in tool_names
            and order_match
            and any(h in lowered for h in _REFUND_HINTS)
        ):
            return ToolCall(
                "mock_refund", "initiate_refund",
                {"order_number": order_match.group(1), "reason": user_text[:300]},
            )

        email_match = _EMAIL_RE.search(user_text)
        if "send_email" in tool_names and "email" in lowered and email_match:
            return ToolCall(
                "mock_email", "send_email",
                {"to": email_match.group(0), "subject": "Following up on your request", "body": user_text},
            )

        if (
            "lookup_order" in tool_names
            and order_match
            and any(hint in lowered for hint in _ORDER_HINTS)
        ):
            return ToolCall("mock_lookup_order", "lookup_order", {"order_number": order_match.group(1)})

        if "search_help_docs" in tool_names:
            return ToolCall("mock_search", "search_help_docs", {"query": user_text})
        return None

    def _compose_answer(self, messages: list[dict]) -> str:
        for data in self._tool_payloads(messages):
            if data.get("status") == "pending_approval" and data.get("action_id"):
                return (
                    f"I've submitted a refund request for review (reference "
                    f"{data['action_id']}). It hasn't been processed yet — a teammate "
                    "will review and approve it."
                )
            if data.get("ticket_id"):
                return (
                    f"I've opened ticket {data['ticket_id']} ({data.get('priority', 'normal')} "
                    "priority). Our team will follow up with you."
                )
            if data.get("escalated"):
                return "I've escalated this to a human teammate who will reach out shortly."
            if "delivery" in data:
                msg = {
                    "sent": "I've sent that email.",
                    "logged": "I've queued that email (no email provider is configured, so it was logged).",
                    "skipped_duplicate": "That email was already sent, so I didn't send a duplicate.",
                }
                return msg.get(data["delivery"], "Email handled.")
            if data.get("orders"):
                return self._order_answer(data["orders"][0])
            if data.get("results"):
                top = data["results"][0]
                snippet = " ".join((top.get("content") or "").split())[:300]
                return f"Based on our help docs: {snippet} (Source: {top.get('doc_title', 'a help doc')})"
            if data.get("found") is False:
                return data.get("message") or "I couldn't find that."
        return (
            "I couldn't find that in our help docs. If you'd like, I can escalate "
            "this to a human teammate."
        )

    @staticmethod
    def _order_answer(order: dict) -> str:
        parts = [f"Order #{order.get('order_number')} is currently '{order.get('status')}'."]
        if order.get("shipping_status"):
            parts.append(f"Shipping status: {order['shipping_status']}.")
        if order.get("tracking_number"):
            parts.append(f"Tracking number: {order['tracking_number']}.")
        return " ".join(parts)

    async def chat(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool = False,
        emit: EmitFn | None = None,
    ) -> LLMResponse:
        tool_names = {t["name"] for t in tools}
        current_turn = self._current_turn(messages)
        has_tool_results = any(m.get("role") == "tool" for m in current_turn)

        # If we haven't called a tool THIS turn yet, route to the best one.
        if not has_tool_results:
            call = self._choose_first_tool(self._latest_user_text(messages), tool_names)
            if call is not None:
                return LLMResponse(
                    text=None,
                    tool_calls=[call],
                    usage=self._usage(system, messages, f"[{call.name}]"),
                )

        # Otherwise answer from THIS turn's tool results.
        answer = self._compose_answer(current_turn)
        if stream and emit is not None:
            for word in answer.split(" "):
                await emit("token", {"text": word + " "})
        return LLMResponse(
            text=answer, tool_calls=[], usage=self._usage(system, messages, answer)
        )
