"""Schemas for the conversations (agent) endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ConversationCreate(BaseModel):
    customer_email: str | None = None


class ConversationCreated(BaseModel):
    id: uuid.UUID


class MessageRequest(BaseModel):
    content: str
    stream: bool = False


class ToolCallSummary(BaseModel):
    tool: str
    status: str


class UsageSummary(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class MessageResponse(BaseModel):
    message_id: uuid.UUID
    content: str
    tool_calls: list[ToolCallSummary]
    usage: UsageSummary


# --- conversation history ---
class ToolCallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    tool_name: str
    arguments: dict[str, Any]
    result: Any | None = None
    status: str
    error: str | None = None
    latency_ms: int | None = None
    created_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str | None = None
    created_at: datetime
    tool_calls: list[ToolCallOut] = []


class ConversationHistory(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    needs_human: bool
    messages: list[MessageOut]
