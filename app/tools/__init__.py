"""Vertical (e-commerce) tool layer. Registers tools into the agent registry.

Swapping verticals means changing this package — not the engine. More tools are
added in later milestones (lookup_order, create_ticket, send_email,
initiate_refund, escalate_to_human).
"""

from __future__ import annotations

from functools import lru_cache

from app.agent.registry import ToolRegistry
from app.tools.email import SEND_EMAIL
from app.tools.escalate import ESCALATE_TO_HUMAN
from app.tools.help_docs import SEARCH_HELP_DOCS
from app.tools.orders import LOOKUP_ORDER
from app.tools.refunds import INITIATE_REFUND
from app.tools.tickets import CREATE_TICKET


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SEARCH_HELP_DOCS)
    registry.register(LOOKUP_ORDER)
    registry.register(CREATE_TICKET)
    registry.register(SEND_EMAIL)
    registry.register(ESCALATE_TO_HUMAN)
    registry.register(INITIATE_REFUND)
    return registry


@lru_cache
def get_registry() -> ToolRegistry:
    """Process-wide registry (tool specs are stateless)."""
    return build_registry()
