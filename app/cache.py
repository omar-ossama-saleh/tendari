"""Redis-backed idempotency claims (reuses the Redis already in the stack).

Used by side-effecting tools (e.g. send_email) to guarantee an action fires at
most once: claim a key with SET NX; if the claim fails, the action already
happened. On a failed side effect, release the key so a retry can proceed.
"""

from __future__ import annotations

import contextlib
from functools import lru_cache

import redis.asyncio as aioredis

from app.config import settings

_PREFIX = "tendari:idem:"
_DEFAULT_TTL_S = 24 * 60 * 60


@lru_cache
def get_redis() -> "aioredis.Redis":
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def claim_once(key: str, ttl_s: int = _DEFAULT_TTL_S) -> bool:
    """Return True if this is the first claim of ``key`` (caller may proceed)."""
    return bool(await get_redis().set(f"{_PREFIX}{key}", "1", nx=True, ex=ttl_s))


async def release(key: str) -> None:
    """Release a previously claimed key (e.g. after a failed side effect)."""
    with contextlib.suppress(Exception):
        await get_redis().delete(f"{_PREFIX}{key}")
