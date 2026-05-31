"""Tool arg validation + mock routing (DB-free; handlers verified live)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent.providers.mock import MockProvider
from app.tools.help_docs import SearchHelpDocsArgs
from app.tools.orders import LookupOrderArgs

_TOOLS = [
    {"name": "search_help_docs", "description": "d", "input_schema": {}},
    {"name": "lookup_order", "description": "d", "input_schema": {}},
]


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
