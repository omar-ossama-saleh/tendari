"""Schemas for the human-in-the-loop pending-actions endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class PendingActionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    action_type: str
    payload: dict[str, Any]
    status: str
    external_ref: str | None = None
    error: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None


class ApproveResponse(BaseModel):
    status: str
    external_ref: str | None = None


class RejectResponse(BaseModel):
    status: str
