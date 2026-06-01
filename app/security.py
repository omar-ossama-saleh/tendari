"""API-key hashing and generation.

API keys are high-entropy random tokens, so a fast cryptographic hash
(SHA-256) is the right primitive — unlike low-entropy passwords, they do not
need a slow KDF. We persist only the hash; the raw key is shown exactly once
(at generation / seed time) and never stored.
"""

from __future__ import annotations

import hashlib
import secrets

_API_KEY_PREFIX = "tendari_sk"


def hash_api_key(raw_key: str) -> str:
    """Return the hex SHA-256 of an API key. Deterministic — used for lookup."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    """Generate a new opaque API key. Return value is the only time the raw
    key exists — the caller must persist only ``hash_api_key(key)``."""
    return f"{_API_KEY_PREFIX}_{secrets.token_urlsafe(32)}"
