"""Vertical (e-commerce) tool layer. Registers tools into the agent registry.

Swapping verticals means changing this package — not the engine. More tools are
added in later milestones (lookup_order, create_ticket, send_email,
initiate_refund, escalate_to_human).
"""

from __future__ import annotations

from functools import lru_cache

from app.agent.registry import ToolRegistry
from app.tools.help_docs import SEARCH_HELP_DOCS
from app.tools.orders import LOOKUP_ORDER


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SEARCH_HELP_DOCS)
    registry.register(LOOKUP_ORDER)
    return registry


@lru_cache
def get_registry() -> ToolRegistry:
    """Process-wide registry (tool specs are stateless)."""
    return build_registry()
