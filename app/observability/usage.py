"""Per-LLM-call usage + cost recording.

Token prices change constantly, so they are CONFIG, never constants in logic:
the defaults below are overridable per-model via LLM_PRICING_JSON. Prices are
USD per 1,000 tokens. Update them to current provider pricing when you deploy.
"""

from __future__ import annotations

import logging
import uuid
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.providers.base import Usage
from app.config import settings
from app.models import UsageRecord

logger = logging.getLogger("tendari.usage")

# Default USD price per 1,000 tokens. Verify/adjust against current pricing.
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input_per_1k": 0.001, "output_per_1k": 0.005},
    "claude-sonnet-4-6": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "claude-opus-4-8": {"input_per_1k": 0.015, "output_per_1k": 0.075},
    "gpt-4o": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
    "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
}

_CENTS = Decimal("0.000001")


def _pricing_for(model: str) -> dict[str, float] | None:
    overrides = settings.pricing_overrides
    if model in overrides:
        return overrides[model]
    return _DEFAULT_PRICING.get(model)


def compute_cost_usd(usage: Usage) -> Decimal:
    """Cost for one call from token counts × per-model price. 0 for unknown models."""
    pricing = _pricing_for(usage.model)
    if pricing is None:
        logger.warning("No pricing for model %r; recording cost 0.", usage.model)
        return Decimal("0")
    cost = (
        Decimal(str(pricing.get("input_per_1k", 0))) * Decimal(usage.prompt_tokens) / 1000
        + Decimal(str(pricing.get("output_per_1k", 0))) * Decimal(usage.completion_tokens) / 1000
    )
    return cost.quantize(_CENTS, rounding=ROUND_HALF_UP)


async def record_usage(
    session_factory: async_sessionmaker,
    *,
    workspace_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    usage: Usage,
    latency_ms: int | None,
    endpoint: str | None,
) -> Decimal:
    """Record one LLM call's usage on its OWN committed session and return cost.

    Cost accounting is a durable audit: real spend already happened, so it must
    survive even if the surrounding request later rolls back (e.g. a provider
    error on a subsequent loop iteration). Hence an autonomous transaction rather
    than the request session.
    """
    cost = compute_cost_usd(usage)
    async with session_factory() as session:
        session.add(
            UsageRecord(
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                model=usage.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                endpoint=endpoint,
            )
        )
        await session.commit()
    return cost
