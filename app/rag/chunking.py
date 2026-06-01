"""Text chunking for RAG.

Chunking strategy (an owned decision): pack natural text units — paragraphs,
falling back to sentences, falling back to word windows for pathological cases —
greedily up to a target token budget, then carry a small overlap into the next
chunk so a fact split across a boundary is still retrievable from one chunk.

Token counts are *estimated* (≈4 chars/token) rather than computed with a model
tokenizer: it keeps ingestion dependency-free and the estimate only needs to be
good enough to size chunks, not to bill. Actual billing uses provider-reported
usage elsewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WHITESPACE_RUN = re.compile(r"[ \t]+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token)."""
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    content: str
    token_count: int


def _normalize(text: str) -> str:
    # Collapse intra-line whitespace, trim trailing spaces, keep paragraph breaks.
    lines = [_WHITESPACE_RUN.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(lines).strip()


def _segment(text: str, target_tokens: int) -> list[str]:
    """Break text into units no larger than ``target_tokens`` (approx)."""
    units: list[str] = []
    for para in _PARAGRAPH_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if estimate_tokens(para) <= target_tokens:
            units.append(para)
            continue
        # Paragraph too big → sentences.
        for sent in _SENTENCE_SPLIT.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if estimate_tokens(sent) <= target_tokens:
                units.append(sent)
            else:
                units.extend(_split_by_words(sent, target_tokens))
    return units


def _split_by_words(text: str, target_tokens: int) -> list[str]:
    """Hard-split an oversized unit on word boundaries, each window <= target."""
    words = text.split()
    out: list[str] = []
    buf: list[str] = []
    for word in words:
        # Flush BEFORE crossing the budget, keeping the crossing word for next.
        if buf and estimate_tokens(" ".join([*buf, word])) > target_tokens:
            out.append(" ".join(buf))
            buf = []
        buf.append(word)
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_text(text: str, target_tokens: int, overlap_tokens: int) -> list[Chunk]:
    """Chunk ``text`` into overlapping ~``target_tokens`` pieces."""
    text = _normalize(text)
    if not text:
        return []

    # Overlap must be strictly smaller than target or packing can't make progress.
    overlap_tokens = max(0, min(overlap_tokens, target_tokens // 2))
    units = _segment(text, target_tokens)

    contents: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def carry_overlap() -> tuple[list[str], int]:
        carry: list[str] = []
        carry_tokens = 0
        for unit in reversed(buf):
            t = estimate_tokens(unit)
            if carry and carry_tokens + t > overlap_tokens:
                break
            carry.insert(0, unit)
            carry_tokens += t
        return carry, carry_tokens

    for unit in units:
        t = estimate_tokens(unit)
        if buf and buf_tokens + t > target_tokens:
            contents.append(" ".join(buf).strip())
            if overlap_tokens:
                buf, buf_tokens = carry_overlap()
                # Don't let the overlap carry push the next chunk over target.
                if buf_tokens + t > target_tokens:
                    buf, buf_tokens = [], 0
            else:
                buf, buf_tokens = [], 0
        buf.append(unit)
        buf_tokens += t

    if buf:
        contents.append(" ".join(buf).strip())

    return [
        Chunk(chunk_index=i, content=c, token_count=estimate_tokens(c))
        for i, c in enumerate(contents)
        if c
    ]
