"""Vertical tool: create_ticket — open a support ticket for the workspace."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.agent.registry import ToolContext, ToolSpec
from app.models import Ticket


class CreateTicketArgs(BaseModel):
    subject: str = Field(..., min_length=1, description="Short ticket subject line.")
    body: str | None = Field(default=None, description="Ticket details / description.")
    priority: Literal["low", "normal", "high"] = "normal"


async def _create_ticket(args: CreateTicketArgs, ctx: ToolContext) -> dict:
    # Side-effecting write committed on its own session (the action stands even
    # if the surrounding turn later rolls back). Scoped to the workspace.
    async with ctx.session_factory() as session:
        ticket = Ticket(
            workspace_id=ctx.workspace.id,
            customer_id=ctx.conversation.customer_id,
            conversation_id=ctx.conversation.id,
            subject=args.subject,
            body=args.body,
            priority=args.priority,
        )
        session.add(ticket)
        await session.commit()
        return {
            "ticket_id": str(ticket.id),
            "priority": ticket.priority,
            "status": ticket.status,
        }


CREATE_TICKET = ToolSpec(
    name="create_ticket",
    description=(
        "Open a support ticket for the customer when an issue needs follow-up by "
        "the team. Provide a concise subject, an optional body with details, and a "
        "priority (low | normal | high). Returns the new ticket id."
    ),
    args_model=CreateTicketArgs,
    handler=_create_ticket,
)
