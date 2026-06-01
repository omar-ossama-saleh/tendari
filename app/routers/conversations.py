"""Conversations / agent endpoints (the core).

M2: create a conversation, send a message (non-streaming → JSON), read history.
SSE streaming is added in M3.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sse_starlette.sse import EventSourceResponse

from app.agent.engine import run_agent
from app.agent.providers import get_provider
from app.auth import CurrentWorkspace, DbSession
from app.db import SessionLocal
from app.models import Conversation, Customer, Message, Workspace
from app.schemas.conversations import (
    ConversationCreate,
    ConversationCreated,
    ConversationHistory,
    MessageRequest,
    MessageResponse,
    ToolCallSummary,
    UsageSummary,
)
from app.tools import get_registry

logger = logging.getLogger("tendari.conversations")

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])


async def _find_or_create_customer(
    session: DbSession, workspace_id: uuid.UUID, email: str
) -> uuid.UUID:
    # Normalize here so matching is a property of the data layer, not callers.
    email = email.strip().lower()
    customer = await session.scalar(
        select(Customer).where(
            Customer.workspace_id == workspace_id, Customer.email == email
        )
    )
    if customer is None:
        customer = Customer(workspace_id=workspace_id, email=email)
        session.add(customer)
        await session.flush()
    return customer.id


@router.post("", response_model=ConversationCreated, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: ConversationCreate, workspace: CurrentWorkspace, session: DbSession
) -> ConversationCreated:
    customer_id = None
    if body.customer_email:
        customer_id = await _find_or_create_customer(
            session, workspace.id, body.customer_email
        )
    conversation = Conversation(workspace_id=workspace.id, customer_id=customer_id)
    session.add(conversation)
    await session.flush()
    return ConversationCreated(id=conversation.id)


async def _get_owned_conversation(
    conversation_id: uuid.UUID, workspace: CurrentWorkspace, session: DbSession
) -> Conversation:
    conversation = await session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.workspace_id == workspace.id,
        )
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


async def _agent_event_stream(
    workspace: Workspace,
    conversation: Conversation,
    content: str,
) -> AsyncIterator[dict]:
    """Run the agent in a background task and relay its emitted events as SSE.

    The agent runs on its OWN session (not the request session), committed only
    when the run completes cleanly and rolled back on client disconnect / error.
    This avoids committing a half-written turn on disconnect and avoids issuing a
    commit on a session that was cancelled mid-flush.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(event: str, data: dict) -> None:
        await queue.put((event, data))

    async def runner() -> None:
        try:
            async with SessionLocal() as agent_session:
                try:
                    await run_agent(
                        session=agent_session,
                        session_factory=SessionLocal,
                        workspace=workspace,
                        conversation=conversation,
                        user_text=content,
                        registry=get_registry(),
                        provider=get_provider(),
                        stream=True,
                        emit=emit,
                    )
                    await agent_session.commit()
                except asyncio.CancelledError:
                    # Client disconnected — don't persist a partial turn.
                    with contextlib.suppress(Exception):
                        await agent_session.rollback()
                    raise
                except Exception:
                    logger.exception("Streaming agent run failed")
                    with contextlib.suppress(Exception):
                        await agent_session.rollback()
                    await queue.put(("error", {"message": "The agent hit an internal error."}))
        finally:
            await queue.put(None)  # sentinel: stream complete

    task = asyncio.create_task(runner())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            event, data = item
            yield {"event": event, "data": json.dumps(data)}
    finally:
        if not task.done():
            task.cancel()
        # Only our own cancellation should surface here; let real errors raise.
        with contextlib.suppress(asyncio.CancelledError):
            await task


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def post_message(
    conversation_id: uuid.UUID,
    body: MessageRequest,
    workspace: CurrentWorkspace,
    session: DbSession,
):
    """Send a message to the agent.

    ``stream=false`` → JSON MessageResponse. ``stream=true`` → text/event-stream
    with ``token`` / ``tool_call_start`` / ``tool_call_result`` / ``done`` events.
    """
    conversation = await _get_owned_conversation(conversation_id, workspace, session)

    if body.stream:
        # Streaming runs on its own session; the request session only did the
        # ownership read above, so its post-stream commit is a no-op.
        return EventSourceResponse(
            _agent_event_stream(workspace, conversation, body.content)
        )

    result = await run_agent(
        session=session,
        session_factory=SessionLocal,
        workspace=workspace,
        conversation=conversation,
        user_text=body.content,
        registry=get_registry(),
        provider=get_provider(),
        stream=False,
    )

    return MessageResponse(
        message_id=result.final_message_id,
        content=result.text,
        tool_calls=[ToolCallSummary(**tc) for tc in result.tool_calls],
        usage=UsageSummary(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=float(result.cost_usd),
        ),
    )


@router.get("/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(
    conversation_id: uuid.UUID, workspace: CurrentWorkspace, session: DbSession
) -> Conversation:
    conversation = await session.scalar(
        select(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.workspace_id == workspace.id,
        )
        .options(
            selectinload(Conversation.messages).selectinload(Message.tool_calls)
        )
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation
