"""Chunker behavior: sizing, overlap, indices, edge cases."""

from __future__ import annotations

from app.rag.chunking import chunk_text, estimate_tokens


def _words(s: str) -> set[str]:
    return set(s.lower().split())


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   \n\n  ", 100, 10) == []


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_text("Our return window is 30 days.", 100, 10)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert "30 days" in chunks[0].content


def test_long_text_splits_with_sequential_indices() -> None:
    text = " ".join(f"Sentence number {i} about returns and refunds." for i in range(40))
    chunks = chunk_text(text, target_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Chunks stay within the target budget (small slack for join/rounding).
    for c in chunks:
        assert c.token_count <= 20 + 4


def test_overlap_shares_content_between_adjacent_chunks() -> None:
    text = ". ".join(f"clause {i} keyword{i}" for i in range(30)) + "."
    chunks = chunk_text(text, target_tokens=15, overlap_tokens=8)
    assert len(chunks) > 1
    # Each adjacent pair shares at least one token due to the overlap carry.
    assert any(_words(chunks[i].content) & _words(chunks[i + 1].content)
               for i in range(len(chunks) - 1))


def test_oversized_sentence_is_hard_split() -> None:
    giant = "word " * 400  # one long run, no sentence breaks
    chunks = chunk_text(giant, target_tokens=20, overlap_tokens=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 20 + 4


def test_estimate_tokens_is_positive() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") >= 1
