"""Schemas for the documents (RAG) endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class DocumentCreate(BaseModel):
    """JSON body for text/url ingestion. (PDF is uploaded via multipart.)"""

    title: str
    source_type: Literal["text", "url"]
    content: str | None = None
    source_ref: str | None = None

    @model_validator(mode="after")
    def _require_source_payload(self) -> "DocumentCreate":
        if self.source_type == "text" and not (self.content and self.content.strip()):
            raise ValueError("content is required for source_type='text'")
        if self.source_type == "url" and not (self.source_ref and self.source_ref.strip()):
            raise ValueError("source_ref (a URL) is required for source_type='url'")
        return self


class DocumentAccepted(BaseModel):
    id: uuid.UUID
    status: str


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    status: str
    source_type: str
    created_at: datetime


class DocumentDetail(DocumentOut):
    source_ref: str | None = None
    error: str | None = None
