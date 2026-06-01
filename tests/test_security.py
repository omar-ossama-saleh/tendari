"""API-key hashing/generation invariants."""

from __future__ import annotations

from app.security import generate_api_key, hash_api_key


def test_hash_is_deterministic() -> None:
    assert hash_api_key("abc") == hash_api_key("abc")


def test_hash_differs_per_key() -> None:
    assert hash_api_key("abc") != hash_api_key("abd")


def test_hash_is_hex_sha256() -> None:
    digest = hash_api_key("anything")
    assert len(digest) == 64
    int(digest, 16)  # raises if not hex


def test_generated_keys_are_unique_and_prefixed() -> None:
    a, b = generate_api_key(), generate_api_key()
    assert a != b
    assert a.startswith("tendari_sk_")
