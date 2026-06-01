"""Documents (RAG) endpoints: upload/ingest, list, get, delete.

A single POST endpoint accepts either ``multipart/form-data`` (with an optional
PDF file) or ``application/json`` (text/url), per the API contract. PDF text is
extracted synchronously at upload time (fast); the slow embedding work is
enqueued to Celery. Everything is workspace-scoped server-side.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import ValidationError
from sqlalchemy import select
from starlette.datastructures import UploadFile

from app.auth import CurrentWorkspace, DbSession
from app.models import Document
from app.rag.ingest import ingest_document_task
from app.rag.loaders import IngestError, parse_pdf
from app.schemas.documents import (
    DocumentAccepted,
    DocumentCreate,
    DocumentDetail,
    DocumentOut,
)

logger = logging.getLogger("tendari.documents")

router = APIRouter(prefix="/v1/documents", tags=["documents"])

# Application-level upload cap (mirrors the URL fetch cap in loaders.py).
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Documents the multipart form in Swagger (renders a file picker). JSON bodies
# with {title, source_type, content|source_ref} are also accepted.
_CREATE_OPENAPI_EXTRA = {
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["title", "source_type"],
                    "properties": {
                        "title": {"type": "string"},
                        "source_type": {
                            "type": "string",
                            "enum": ["pdf", "text", "url"],
                        },
                        "content": {
                            "type": "string",
                            "description": "required when source_type=text",
                        },
                        "source_ref": {
                            "type": "string",
                            "description": "URL when source_type=url",
                        },
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "PDF file when source_type=pdf",
                        },
                    },
                }
            }
        }
    }
}


class _ParsedUpload:
    __slots__ = ("title", "source_type", "content", "source_ref", "raw_text")

    def __init__(self, title, source_type, content, source_ref, raw_text):
        self.title = title
        self.source_type = source_type
        self.content = content
        self.source_ref = source_ref
        self.raw_text = raw_text


def _bad_request(detail) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


def _too_large() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
    )


async def _parse_json_body(request: Request) -> _ParsedUpload:
    try:
        payload = DocumentCreate.model_validate(await request.json())
    except ValidationError as exc:
        # exc.errors() can contain non-JSON-serializable ctx; exc.json() is safe.
        raise _bad_request(json.loads(exc.json())) from exc
    except ValueError as exc:
        raise _bad_request(f"Invalid JSON body: {exc}") from exc
    raw_text = payload.content if payload.source_type == "text" else None
    return _ParsedUpload(
        payload.title, payload.source_type, payload.content, payload.source_ref, raw_text
    )


async def _parse_multipart(request: Request) -> _ParsedUpload:
    form = await request.form()
    title = (form.get("title") or "").strip()
    source_type = (form.get("source_type") or "").strip()
    content = form.get("content")
    source_ref = form.get("source_ref")
    upload = form.get("file")

    if not title:
        raise _bad_request("title is required")
    if source_type not in ("pdf", "text", "url"):
        raise _bad_request("source_type must be one of: pdf, text, url")

    raw_text: str | None = None
    if source_type == "pdf":
        if not isinstance(upload, UploadFile):
            raise _bad_request("a PDF 'file' is required for source_type=pdf")
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise _too_large()
        try:
            raw_text = await asyncio.to_thread(parse_pdf, data)
        except IngestError as exc:
            raise _bad_request(str(exc)) from exc
        source_ref = source_ref or upload.filename
    elif source_type == "text":
        if not (content and str(content).strip()):
            raise _bad_request("content is required for source_type=text")
        raw_text = str(content)
    elif source_type == "url":
        if not (source_ref and str(source_ref).strip()):
            raise _bad_request("source_ref (a URL) is required for source_type=url")

    return _ParsedUpload(title, source_type, content, source_ref, raw_text)


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentAccepted,
    openapi_extra=_CREATE_OPENAPI_EXTRA,
    summary="Upload/create a document for ingestion",
)
async def create_document(
    request: Request, workspace: CurrentWorkspace, session: DbSession
) -> DocumentAccepted:
    # Early reject of oversized bodies before buffering the upload.
    declared_len = request.headers.get("content-length")
    if declared_len and declared_len.isdigit() and int(declared_len) > MAX_UPLOAD_BYTES:
        raise _too_large()

    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        parsed = await _parse_json_body(request)
    elif "multipart/form-data" in content_type:
        parsed = await _parse_multipart(request)
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Use multipart/form-data (with optional PDF) or application/json.",
        )

    document = Document(
        workspace_id=workspace.id,
        title=parsed.title,
        source_type=parsed.source_type,
        source_ref=parsed.source_ref,
        status="pending",
    )
    session.add(document)
    await session.flush()
    # Commit before enqueueing so the worker can always see the row.
    await session.commit()

    try:
        ingest_document_task.delay(
            str(document.id), parsed.raw_text, parsed.source_type, parsed.source_ref
        )
    except Exception as exc:  # broker unreachable, etc. — don't leave a stuck 'pending'.
        logger.exception("Failed to enqueue ingestion for document %s", document.id)
        document.status = "failed"
        document.error = f"Could not enqueue ingestion: {exc}"
        await session.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not enqueue ingestion; please retry.",
        ) from exc

    return DocumentAccepted(id=document.id, status=document.status)


@router.get("", response_model=list[DocumentOut], summary="List documents")
async def list_documents(workspace: CurrentWorkspace, session: DbSession) -> list[Document]:
    rows = await session.scalars(
        select(Document)
        .where(Document.workspace_id == workspace.id)
        .order_by(Document.created_at.desc())
    )
    return list(rows)


async def _get_owned_document(
    document_id: uuid.UUID, workspace: CurrentWorkspace, session: DbSession
) -> Document:
    document = await session.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.workspace_id == workspace.id,
        )
    )
    if document is None:
        # 404 (not 403) so callers can't probe other workspaces' document ids.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


@router.get("/{document_id}", response_model=DocumentDetail, summary="Get a document")
async def get_document(
    document_id: uuid.UUID, workspace: CurrentWorkspace, session: DbSession
) -> Document:
    return await _get_owned_document(document_id, workspace, session)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document (cascades its chunks)",
)
async def delete_document(
    document_id: uuid.UUID, workspace: CurrentWorkspace, session: DbSession
) -> Response:
    document = await _get_owned_document(document_id, workspace, session)
    await session.delete(document)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
