"""Embeddings: OpenAI when a key is configured, deterministic offline otherwise.

Anthropic has no embeddings API, so embeddings always go through OpenAI in real
use. With no key (tests / offline demo) we fall back to a deterministic
feature-hashing embedder so the whole RAG path still works end-to-end without
external calls — at reduced retrieval quality.

The SAME path is used for both ingestion and querying within a deployment
(decided by key presence), so vectors are always comparable. Mixing an
OpenAI-embedded corpus with an offline-embedded query (or vice versa) would be
meaningless; don't add a key to a corpus that was ingested offline without
re-ingesting.
"""

from __future__ import annotations

import hashlib
import math
import re

from app.config import settings

# \w (unicode) so non-Latin text still produces a non-zero vector.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_OPENAI_BATCH = 100


def using_openai() -> bool:
    return bool(settings.openai_api_key)


def is_zero_vector(vec: list[float]) -> bool:
    return not any(vec)


def _offline_embed(text: str, dim: int) -> list[float]:
    """Deterministic L2-normalized feature-hashing embedding of `text`."""
    vec = [0.0] * dim
    for token in _TOKEN_RE.findall(text.lower()):
        digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 (not security)
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


async def _openai_embed(texts: list[str]) -> list[list[float]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    # text-embedding-3-* support an explicit output dimension; pin it so the
    # returned vectors always match the DB column (and EMBEDDING_DIM).
    kwargs: dict = {"model": settings.embedding_model, "input": None}
    if settings.embedding_model.startswith("text-embedding-3"):
        kwargs["dimensions"] = settings.embedding_dim

    out: list[list[float]] = []
    for start in range(0, len(texts), _OPENAI_BATCH):
        kwargs["input"] = texts[start : start + _OPENAI_BATCH]
        resp = await client.embeddings.create(**kwargs)
        out.extend(item.embedding for item in resp.data)
    return out


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns vectors of length ``settings.embedding_dim``."""
    if not texts:
        return []
    vectors = (
        await _openai_embed(texts)
        if using_openai()
        else [_offline_embed(t, settings.embedding_dim) for t in texts]
    )
    for vec in vectors:
        if len(vec) != settings.embedding_dim:
            raise RuntimeError(
                f"Embedding length {len(vec)} != configured EMBEDDING_DIM "
                f"{settings.embedding_dim}; check EMBEDDING_MODEL/EMBEDDING_DIM."
            )
    return vectors


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]
