"""Vertical tool: escalate_to_human — flag the conversation for human follow-up."""

from __future__ import annotations

from pydantic import BaseModel, Field
from sqlalchemy import update

from app.agent.registry import ToolContext, ToolSpec
from app.models import Conversation


class EscalateArgs(BaseModel):
    reason: str = Field(..., min_length=1, description="Why this needs a human.")


async def _escalate_to_human(args: EscalateArgs, ctx: ToolContext) -> dict:
    # Scoped UPDATE on its own session; commits immediately.
    async with ctx.session_factory() as session:
        await session.execute(
            update(Conversation)
            .where(
                Conversation.id == ctx.conversation.id,
                Conversation.workspace_id == ctx.workspace.id,
            )
            .values(needs_human=True)
        )
        await session.commit()
    return {"escalated": True, "reason": args.reason}


ESCALATE_TO_HUMAN = ToolSpec(
    name="escalate_to_human",
    description=(
        "Escalate the conversation to a human teammate when you can't resolve the "
        "request, the customer is upset, or they ask for a person. Provide a brief "
        "reason. This flags the conversation for human follow-up."
    ),
    args_model=EscalateArgs,
    handler=_escalate_to_human,
)
