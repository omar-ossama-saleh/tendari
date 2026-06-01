"""Vertical tool: send_email — idempotent SendGrid send, log-fallback if no key."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re

from pydantic import BaseModel, Field, field_validator

from app.agent.registry import ToolContext, ToolError, ToolSpec
from app.cache import claim_once, release
from app.config import settings

logger = logging.getLogger("tendari.email")

# Pragmatic address check (no email-validator dependency): single @, non-empty
# local/domain parts, a dotted domain, and no whitespace.
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class SendEmailArgs(BaseModel):
    to: str = Field(..., description="Recipient email address.")
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)

    @field_validator("to")
    @classmethod
    def _looks_like_email(cls, v: str) -> str:
        v = v.strip()
        if not _EMAIL_RE.match(v):
            raise ValueError("to must be a valid email address")
        return v


def _idempotency_key(ctx: ToolContext, args: SendEmailArgs) -> str:
    raw = f"{ctx.conversation.id}|{args.to}|{args.subject}|{args.body}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _send_via_sendgrid(to: str, subject: str, body: str) -> None:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail

    message = Mail(
        from_email=settings.sendgrid_from_email,
        to_emails=to,
        subject=subject,
        plain_text_content=body,
    )
    SendGridAPIClient(settings.sendgrid_api_key).send(message)


async def _send_email(args: SendEmailArgs, ctx: ToolContext) -> dict:
    # Idempotency: a send keyed by (conversation, to, subject, body) fires once.
    key = f"email:{_idempotency_key(ctx, args)}"
    if not await claim_once(key):
        return {"delivery": "skipped_duplicate", "to": args.to}

    # Log-fallback when no provider key is configured.
    if not settings.sendgrid_api_key:
        logger.info("EMAIL (logged; no SendGrid key) to=%s subject=%r", args.to, args.subject)
        return {"delivery": "logged", "to": args.to}

    try:
        await asyncio.to_thread(_send_via_sendgrid, args.to, args.subject, args.body)
    except Exception as exc:
        # Release the claim so a later retry can re-attempt the send.
        await release(key)
        raise ToolError(f"Failed to send email: {exc}") from exc
    return {"delivery": "sent", "to": args.to}


SEND_EMAIL = ToolSpec(
    name="send_email",
    description=(
        "Send an email to the customer (e.g. a confirmation or follow-up). Provide "
        "the recipient address, a subject, and a body. Idempotent — the same email "
        "won't be sent twice. If no email provider is configured it is logged."
    ),
    args_model=SendEmailArgs,
    handler=_send_email,
)
