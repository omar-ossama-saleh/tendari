"""Document ingestion: resolve source text → chunk → embed → store chunks.

The heavy work (embedding) runs in a Celery worker. The async core
``ingest_document`` is pure-ish (takes a session) so it's unit-testable; the
Celery task wraps it in a fresh async engine because the sync worker runs each
task in its own event loop (reusing the module engine across loops is unsafe).
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings
from app.models import Chunk, Document
from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_texts, is_zero_vector
from app.rag.loaders import IngestError, fetch_url
from app.rag.retrieve import assert_vector_dim_matches
from app.tasks import celery_app

logger = logging.getLogger("tendari.ingest")


async def _resolve_text(source_type: str, raw_text: str | None, source_ref: str | None) -> str:
    if source_type == "url":
        if not source_ref:
            raise IngestError("source_ref (URL) is required for url ingestion.")
        return await asyncio.to_thread(fetch_url, source_ref)
    if raw_text is None or not raw_text.strip():
        raise IngestError("No text content to ingest.")
    return raw_text


async def ingest_document(
    session: AsyncSession,
    document_id: uuid.UUID,
    *,
    raw_text: str | None,
    source_type: str,
    source_ref: str | None,
) -> int:
    """Ingest one document; returns the number of chunks created.

    Idempotent: re-ingesting replaces any existing chunks for the document.
    On failure the document is marked ``failed`` with the error recorded.
    """
    document = await session.get(Document, document_id)
    if document is None:
        logger.warning("Document %s vanished before ingestion", document_id)
        return 0

    try:
        await assert_vector_dim_matches(session)
        document.status = "processing"
        await session.flush()

        content = await _resolve_text(source_type, raw_text, source_ref)
        chunks = chunk_text(content, settings.chunk_target_tokens, settings.chunk_overlap_tokens)
        if not chunks:
            raise IngestError("Document produced no chunks after normalization.")

        embeddings = await embed_texts([c.content for c in chunks])

        # Replace existing chunks (idempotent re-ingest).
        await session.execute(delete(Chunk).where(Chunk.document_id == document.id))
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            session.add(
                Chunk(
                    document_id=document.id,
                    workspace_id=document.workspace_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    # A zero vector would yield NaN cosine distance; store NULL so
                    # retrieval's `embedding IS NOT NULL` filter skips it.
                    embedding=None if is_zero_vector(embedding) else embedding,
                    token_count=chunk.token_count,
                    meta={},
                )
            )

        document.status = "ready"
        document.error = None
        await session.commit()
        logger.info("Ingested document %s (%d chunks)", document_id, len(chunks))
        return len(chunks)

    except Exception as exc:
        await session.rollback()
        # Record the failure on the document in a fresh unit of work.
        failed = await session.get(Document, document_id)
        if failed is not None:
            failed.status = "failed"
            failed.error = str(exc)[:1000]
            await session.commit()
        logger.exception("Ingestion failed for document %s", document_id)
        raise


async def _run_ingest(
    document_id: uuid.UUID,
    raw_text: str | None,
    source_type: str,
    source_ref: str | None,
) -> int:
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as session:
            return await ingest_document(
                session,
                document_id,
                raw_text=raw_text,
                source_type=source_type,
                source_ref=source_ref,
            )
    finally:
        await engine.dispose()


@celery_app.task(name="ingest_document", bind=True, max_retries=0)
def ingest_document_task(
    self,  # noqa: ANN001
    document_id: str,
    raw_text: str | None,
    source_type: str,
    source_ref: str | None,
) -> int:
    """Celery entrypoint. Runs the async ingest core in a fresh event loop."""
    return asyncio.run(
        _run_ingest(uuid.UUID(document_id), raw_text, source_type, source_ref)
    )
