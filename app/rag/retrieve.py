"""Workspace-scoped top-k retrieval over chunk embeddings (pgvector cosine)."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Chunk, Document
from app.rag.embeddings import embed_query, is_zero_vector

_VECTOR_DIM_RE = re.compile(r"\((\d+)\)")
_MAX_TOP_K = 50


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: uuid.UUID
    doc_title: str
    chunk_index: int
    content: str
    score: float  # cosine similarity in [-1, 1]; higher is closer


async def assert_vector_dim_matches(session: AsyncSession) -> None:
    """Fail loudly if EMBEDDING_DIM diverges from the live chunks.embedding column.

    Guards the documented footgun: the migration pins the vector dimension, so
    changing EMBEDDING_DIM without a new migration would silently break inserts.
    """
    fmt = await session.scalar(
        text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        )
    )
    db_dim = None
    if fmt:
        match = _VECTOR_DIM_RE.search(fmt)
        if match:
            db_dim = int(match.group(1))
    if db_dim is not None and db_dim != settings.embedding_dim:
        raise RuntimeError(
            f"EMBEDDING_DIM={settings.embedding_dim} does not match the live "
            f"chunks.embedding dimension ({db_dim}). Create a new migration for "
            f"the new embedding model instead of changing EMBEDDING_DIM alone."
        )


async def retrieve_chunks(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    query: str,
    k: int | None = None,
) -> list[RetrievedChunk]:
    """Return the ``k`` most similar ready-document chunks for this workspace."""
    # Distinguish "not provided" (None) from an explicit k; clamp to a sane range.
    k = settings.retrieval_top_k if k is None else max(1, min(k, _MAX_TOP_K))
    if not query.strip():
        return []

    query_vec = await embed_query(query)
    # A zero-vector query (no embeddable tokens) yields NaN cosine distances.
    if is_zero_vector(query_vec):
        return []
    distance = Chunk.embedding.cosine_distance(query_vec).label("distance")

    stmt = (
        select(
            Chunk.document_id,
            Chunk.chunk_index,
            Chunk.content,
            Document.title,
            distance,
        )
        .join(Document, Document.id == Chunk.document_id)
        # SECURITY: every retrieval is hard-scoped to the caller's workspace.
        .where(
            Chunk.workspace_id == workspace_id,
            Chunk.embedding.is_not(None),
            Document.status == "ready",
        )
        .order_by(distance)
        .limit(k)
    )

    rows = (await session.execute(stmt)).all()
    return [
        RetrievedChunk(
            document_id=row.document_id,
            doc_title=row.title,
            chunk_index=row.chunk_index,
            content=row.content,
            score=1.0 - float(row.distance),
        )
        for row in rows
    ]
