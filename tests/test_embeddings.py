"""Offline embedder: determinism, dimension, normalization, similarity ordering."""

from __future__ import annotations

import math

import pytest

from app.config import settings
from app.rag import embeddings
from app.rag.embeddings import _offline_embed, embed_texts, is_zero_vector


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalized


def test_offline_embed_is_deterministic() -> None:
    assert _offline_embed("refund policy", 256) == _offline_embed("refund policy", 256)


def test_offline_embed_has_correct_dim_and_norm() -> None:
    vec = _offline_embed("hello world", 1536)
    assert len(vec) == 1536
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-6)


def test_offline_embed_empty_is_zero_vector() -> None:
    vec = _offline_embed("", 64)
    assert vec == [0.0] * 64
    assert is_zero_vector(vec)


def test_offline_embed_handles_unicode_words() -> None:
    # \w matches unicode letters, so accented/non-Latin text still embeds.
    assert not is_zero_vector(_offline_embed("café résumé", 256))


def test_offline_embed_punctuation_only_is_zero() -> None:
    assert is_zero_vector(_offline_embed("!!! ??? ---", 64))


def test_related_texts_are_more_similar_than_unrelated() -> None:
    q = _offline_embed("what is your refund return window", 1536)
    related = _offline_embed("our refund return window is 30 days", 1536)
    unrelated = _offline_embed("bright orange elephant balloon festival", 1536)
    assert _cosine(q, related) > _cosine(q, unrelated)


async def test_embed_texts_uses_offline_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", None)
    assert embeddings.using_openai() is False
    vectors = await embed_texts(["alpha", "beta"])
    assert len(vectors) == 2
    assert all(len(v) == settings.embedding_dim for v in vectors)


async def test_embed_texts_empty_input() -> None:
    assert await embed_texts([]) == []
