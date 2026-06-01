"""Tool arg validation + mock routing (DB-free; handlers verified live)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent.providers.mock import MockProvider
from app.tools.email import SendEmailArgs
from app.tools.escalate import EscalateArgs
from app.tools.help_docs import SearchHelpDocsArgs
from app.tools.orders import LookupOrderArgs
from app.tools.tickets import CreateTicketArgs

_TOOLS = [
    {"name": name, "description": "d", "input_schema": {}}
    for name in ("search_help_docs", "lookup_order", "create_ticket", "send_email", "escalate_to_human")
]


async def _first_tool(text: str):
    resp = await MockProvider().chat(
        system="s", messages=[{"role": "user", "content": text}], tools=_TOOLS
    )
    return resp.tool_calls[0]


async def _answer_for(tool_name: str, data: dict) -> str:
    import json

    tool_msg = {
        "role": "tool",
        "tool_call_id": "x",
        "name": tool_name,
        "content": json.dumps({"status": "success", "data": data}),
    }
    resp = await MockProvider().chat(
        system="s", messages=[{"role": "user", "content": "q"}, tool_msg], tools=_TOOLS
    )
    return resp.text


def test_lookup_order_requires_an_arg() -> None:
    with pytest.raises(ValidationError):
        LookupOrderArgs()
    assert LookupOrderArgs(order_number="1002").order_number == "1002"
    assert LookupOrderArgs(email="a@b.com").email == "a@b.com"


def test_search_help_docs_requires_query() -> None:
    with pytest.raises(ValidationError):
        SearchHelpDocsArgs(query="")
    assert SearchHelpDocsArgs(query="returns").query == "returns"


async def test_mock_routes_order_question_to_lookup_order() -> None:
    provider = MockProvider()
    resp = await provider.chat(
        system="s",
        messages=[{"role": "user", "content": "Where is my order #1002?"}],
        tools=_TOOLS,
    )
    assert resp.tool_calls[0].name == "lookup_order"
    assert resp.tool_calls[0].arguments == {"order_number": "1002"}


async def test_mock_routes_general_question_to_search() -> None:
    provider = MockProvider()
    resp = await provider.chat(
        system="s",
        messages=[{"role": "user", "content": "What is your return window?"}],
        tools=_TOOLS,
    )
    assert resp.tool_calls[0].name == "search_help_docs"


def test_write_tool_arg_validation() -> None:
    with pytest.raises(ValidationError):
        CreateTicketArgs(subject="")  # subject required
    assert CreateTicketArgs(subject="Broken item").priority == "normal"
    with pytest.raises(ValidationError):
        CreateTicketArgs(subject="x", priority="urgent")  # not in literal

    with pytest.raises(ValidationError):
        EscalateArgs(reason="")
    with pytest.raises(ValidationError):
        SendEmailArgs(to="not-an-email", subject="s", body="b")
    assert SendEmailArgs(to="a@b.com", subject="s", body="b").to == "a@b.com"


async def test_mock_routes_escalation() -> None:
    call = await _first_tool("I want to escalate this to a human, please.")
    assert call.name == "escalate_to_human"


async def test_mock_routes_ticket() -> None:
    call = await _first_tool("Please open a ticket about my broken tent.")
    assert call.name == "create_ticket"
    assert call.arguments["priority"] == "normal"


async def test_mock_routes_email() -> None:
    call = await _first_tool("Can you email me a summary at dana@example.com?")
    assert call.name == "send_email"
    assert call.arguments["to"] == "dana@example.com"


async def test_mock_composes_write_tool_answers() -> None:
    assert "ticket abc-123" in (await _answer_for("create_ticket", {"ticket_id": "abc-123", "priority": "high"}))
    assert "escalated" in (await _answer_for("escalate_to_human", {"escalated": True}))
    assert "logged" in (await _answer_for("send_email", {"delivery": "logged", "to": "a@b.com"}))
    assert "already sent" in (await _answer_for("send_email", {"delivery": "skipped_duplicate", "to": "a@b.com"}))


async def test_mock_picks_new_tool_on_a_later_turn() -> None:
    # A prior turn already used a tool; the NEW turn must route fresh, not replay.
    history = [
        {"role": "user", "content": "Where is order #1002?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "lookup_order", "arguments": {"order_number": "1002"}}]},
        {"role": "tool", "tool_call_id": "1", "name": "lookup_order", "content": '{"status":"success","data":{"orders":[{"order_number":"1002","status":"delivered"}]}}'},
        {"role": "assistant", "content": "Order #1002 is delivered."},
        {"role": "user", "content": "Please escalate this to a human."},  # new turn
    ]
    resp = await MockProvider().chat(system="s", messages=history, tools=_TOOLS)
    assert resp.tool_calls and resp.tool_calls[0].name == "escalate_to_human"


async def test_mock_composes_order_status_answer() -> None:
    provider = MockProvider()
    tool_msg = {
        "role": "tool",
        "tool_call_id": "x",
        "name": "lookup_order",
        "content": '{"status":"success","data":{"found":true,"orders":[{"order_number":"1002","status":"delivered","shipping_status":"delivered","tracking_number":"TRK1002"}]}}',
    }
    resp = await provider.chat(
        system="s",
        messages=[{"role": "user", "content": "where is #1002?"}, tool_msg],
        tools=_TOOLS,
    )
    assert not resp.tool_calls
    assert "1002" in resp.text
    assert "delivered" in resp.text
    assert "TRK1002" in resp.text
