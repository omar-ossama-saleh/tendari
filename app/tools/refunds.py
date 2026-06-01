"""Refunds — the gated destructive action + its human-approved execution.

Two halves:
  * ``initiate_refund`` (tool) NEVER refunds. It authorizes (the order belongs to
    the workspace, is refundable, the amount is valid AND within the order's
    remaining refundable balance), then creates a pending_actions row
    (status=pending_approval) and emits approval_required.
  * ``execute_refund`` / ``reject_action`` run only via the human-approval
    endpoints (app/routers/actions.py).

Safety invariants (security-critical — see handoff §15):
  1. AT MOST ONCE per request — deterministic UNIQUE idempotency_key dedupes
     repeat tool calls; execute uses SELECT...FOR UPDATE + a status guard + the
     same key as Stripe's native idempotency key.
  2. NEVER over-refund an order — both initiate and execute enforce an
     order/payment-level ceiling (sum of refunds must stay within the total).
     execute does this under a SELECT order ... FOR UPDATE lock so concurrent
     approvals on the same order are serialized and summed against PROCESSED
     refunds before any money moves.
  3. Amounts are quantized to the currency's minor units (and converted to minor
     units for Stripe) so 89.5/89.50 are one refund and zero-decimal currencies
     (JPY) aren't multiplied by 100.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import uuid
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.registry import ToolContext, ToolError, ToolSpec
from app.config import settings
from app.models import Order, PendingAction, Workspace

logger = logging.getLogger("tendari.refunds")

# Refunds that count against an order's balance.
_OPEN_OR_DONE = ("pending_approval", "approved", "processed")
_GENERIC_REFUND_ERROR = "The refund could not be processed. Please retry or contact support."

# Stripe currency minor-unit exponents (default 2). Zero-/three-decimal sets.
_ZERO_DECIMAL = {
    "bif", "clp", "djf", "gnf", "jpy", "kmf", "krw", "mga", "pyg", "rwf",
    "ugx", "vnd", "vuv", "xaf", "xof", "xpf",
}
_THREE_DECIMAL = {"bhd", "jod", "kwd", "omr", "tnd"}


class RefundNotFound(Exception):
    """Pending action not found for this workspace (→ 404)."""


class RefundConflict(Exception):
    """Pending action is in a state that can't be approved/rejected (→ 409)."""


# --------------------------------------------------------------------------- #
# money helpers
# --------------------------------------------------------------------------- #
def _minor_unit_exponent(currency: str) -> int:
    c = (currency or "usd").lower()
    if c in _ZERO_DECIMAL:
        return 0
    if c in _THREE_DECIMAL:
        return 3
    return 2


def _quantize(amount: Decimal, currency: str) -> Decimal:
    exp = _minor_unit_exponent(currency)
    return amount.quantize(Decimal(1).scaleb(-exp), rounding=ROUND_HALF_UP)


def _to_minor_units(amount: Decimal, currency: str) -> int:
    exp = _minor_unit_exponent(currency)
    return int(amount.scaleb(exp).to_integral_value(rounding=ROUND_HALF_UP))


# --------------------------------------------------------------------------- #
# initiate_refund (tool) — gate only, never refunds
# --------------------------------------------------------------------------- #
class InitiateRefundArgs(BaseModel):
    order_number: str = Field(..., min_length=1)
    amount: float | None = Field(
        default=None, description="Amount to refund; omit for a full refund of the order total."
    )
    reason: str = Field(..., min_length=1, description="Why the customer wants a refund.")

    @field_validator("amount")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("amount must be a finite number")
        return v


def _refund_idempotency_key(
    workspace_id: uuid.UUID, conversation_id: uuid.UUID, order_number: str, amount: Decimal
) -> str:
    # Reason is deliberately excluded: a paraphrased reason must NOT mint a second
    # refundable row. Identity is (workspace, conversation, order, quantized amount).
    raw = f"refund|{workspace_id}|{conversation_id}|{order_number}|{amount}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _load_order(session: AsyncSession, workspace_id: uuid.UUID, order_number: str) -> Order | None:
    return await session.scalar(
        select(Order).where(
            Order.workspace_id == workspace_id, Order.order_number == order_number
        )
    )


async def _committed_refund_total(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    order_number: str,
    statuses: tuple[str, ...],
    exclude_id: uuid.UUID | None = None,
) -> Decimal:
    """Sum of refund amounts for an order across the given statuses."""
    stmt = select(PendingAction).where(
        PendingAction.workspace_id == workspace_id,
        PendingAction.action_type == "refund",
        PendingAction.status.in_(statuses),
        PendingAction.payload["order_number"].astext == order_number,
    )
    if exclude_id is not None:
        stmt = stmt.where(PendingAction.id != exclude_id)
    total = Decimal("0")
    for action in await session.scalars(stmt):
        try:
            total += Decimal(str(action.payload.get("amount", "0")))
        except (ArithmeticError, ValueError, TypeError):
            continue
    return total


async def _resolve_order_and_amount(
    session: AsyncSession, workspace_id: uuid.UUID, order_number: str, amount_arg: float | None
) -> tuple[Order, Decimal]:
    """Per-request validation: order owned + refundable, amount valid & ≤ total."""
    order = await _load_order(session, workspace_id, order_number)
    if order is None:
        raise ToolError(f"Order '{order_number}' was not found.")
    if not order.stripe_payment_intent_id:
        raise ToolError(f"Order '{order_number}' has no payment on file to refund.")

    currency = order.currency
    total = _quantize(Decimal(str(order.total_amount)), currency)
    if amount_arg is None:
        amount = total
    else:
        amount = _quantize(Decimal(str(amount_arg)), currency)
        if amount <= 0:
            raise ToolError("Refund amount must be positive.")
    if _to_minor_units(amount, currency) <= 0:
        raise ToolError("Refund amount is below the smallest unit for this currency.")
    if amount > total:
        raise ToolError(f"Refund amount {amount} exceeds the order total {total}.")
    return order, amount


async def _authorize_refund(args: InitiateRefundArgs, ctx: ToolContext) -> None:
    """SECURITY GATE — runs before the tool executes.

    Validates ownership, refundability, and per-request amount. The cumulative
    order-level ceiling is enforced when actually creating/executing (where the
    set of open refunds is consistent), not here.
    """
    async with ctx.session_factory() as session:
        await _resolve_order_and_amount(session, ctx.workspace.id, args.order_number, args.amount)


async def _initiate_refund(args: InitiateRefundArgs, ctx: ToolContext) -> dict:
    async with ctx.session_factory() as session:
        order, amount = await _resolve_order_and_amount(
            session, ctx.workspace.id, args.order_number, args.amount
        )
        total = _quantize(Decimal(str(order.total_amount)), order.currency)
        idempotency_key = _refund_idempotency_key(
            ctx.workspace.id, ctx.conversation.id, args.order_number, amount
        )

        action = await session.scalar(
            select(PendingAction).where(PendingAction.idempotency_key == idempotency_key)
        )
        if action is None:
            # Order-level ceiling: this new refund + everything already open/done
            # for the order must stay within the total.
            committed = await _committed_refund_total(
                session, ctx.workspace.id, args.order_number, _OPEN_OR_DONE
            )
            if committed + amount > total:
                raise ToolError(
                    f"This refund would exceed the order total: {committed} of {total} "
                    "is already refunded or pending approval for this order."
                )
            payload = {
                "order_number": args.order_number,
                "amount": str(amount),
                "currency": order.currency,
                "reason": args.reason,
            }
            action = PendingAction(
                workspace_id=ctx.workspace.id,
                conversation_id=ctx.conversation.id,
                action_type="refund",
                payload=payload,
                idempotency_key=idempotency_key,
                status="pending_approval",
            )
            session.add(action)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                action = await session.scalar(
                    select(PendingAction).where(
                        PendingAction.idempotency_key == idempotency_key
                    )
                )
                if action is None:  # extremely unlikely; surface a clean error
                    raise ToolError("Could not record the refund request; please retry.") from None

        action_id = str(action.id)
        action_status = action.status
        event_payload = dict(action.payload)

    event = {"action_id": action_id, "type": "refund", "payload": event_payload, "status": action_status}
    if ctx.emit is not None:
        await ctx.emit("approval_required", event)

    return {
        "status": "pending_approval",
        "action_id": action_id,
        "amount": str(amount),
        "message": (
            "A refund request has been submitted for human review. It is NOT yet "
            "processed — a teammate will approve or reject it."
        ),
    }


INITIATE_REFUND = ToolSpec(
    name="initiate_refund",
    description=(
        "Submit a refund request for human approval. This does NOT refund money — "
        "it creates a pending request that a human must approve. Provide the "
        "order_number, an optional amount (omit for a full refund), and the reason. "
        "Tell the customer the refund has been submitted for review, never that it "
        "is already done."
    ),
    args_model=InitiateRefundArgs,
    handler=_initiate_refund,
    authorizer=_authorize_refund,
)


# --------------------------------------------------------------------------- #
# execution (human-approved) — the only place a refund actually fires
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stripe_refund(payment_intent: str, amount_minor: int, idempotency_key: str, reason_text: str) -> str:
    import stripe

    stripe.api_key = settings.stripe_secret_key
    refund = stripe.Refund.create(
        payment_intent=payment_intent,
        amount=amount_minor,
        reason="requested_by_customer",
        metadata={"tendari_reason": reason_text[:480]},
        idempotency_key=idempotency_key,  # Stripe-native dedupe: one refund per key
    )
    return refund.id


async def _do_refund(
    *, payment_intent: str, amount: Decimal, currency: str, idempotency_key: str, reason_text: str
) -> str:
    if not settings.stripe_secret_key:
        # No key → simulate deterministically so the demo runs offline.
        logger.info("STRIPE (simulated; no key) refund pi=%s amount=%s %s", payment_intent, amount, currency)
        return f"re_sim_{idempotency_key[:24]}"
    return await asyncio.to_thread(
        _stripe_refund, payment_intent, _to_minor_units(amount, currency), idempotency_key, reason_text
    )


async def execute_refund(
    session: AsyncSession, workspace: Workspace, action_id: uuid.UUID
) -> PendingAction:
    """Approve + process a refund exactly once, never over-refunding the order."""
    action = await session.scalar(
        select(PendingAction)
        .where(PendingAction.id == action_id, PendingAction.workspace_id == workspace.id)
        .with_for_update()  # serialize concurrent approvals of THIS action
    )
    if action is None:
        raise RefundNotFound("Pending action not found.")
    if action.action_type != "refund":
        raise RefundConflict("This action is not a refund.")
    if action.status == "processed":
        return action  # idempotent: already done, never refund twice
    if action.status == "rejected":
        raise RefundConflict("This action was rejected and cannot be approved.")

    order_number = action.payload.get("order_number", "")
    # Lock the ORDER so concurrent approvals of different refunds on the same
    # order are serialized and summed before any money moves.
    order = await session.scalar(
        select(Order)
        .where(Order.workspace_id == workspace.id, Order.order_number == order_number)
        .with_for_update()
    )
    if order is None or not order.stripe_payment_intent_id:
        action.status = "failed"
        action.error = "Order or payment intent missing at execution time."
        action.resolved_at = _now()
        await session.commit()
        raise RefundConflict(action.error)

    amount = Decimal(str(action.payload.get("amount") or order.total_amount))
    total = _quantize(Decimal(str(order.total_amount)), order.currency)
    already_processed = await _committed_refund_total(
        session, workspace.id, order_number, ("processed",), exclude_id=action.id
    )
    if already_processed + amount > total:
        action.status = "failed"
        action.error = (
            f"Over-refund blocked: {already_processed} already refunded of {total}; "
            f"this refund of {amount} would exceed the order total."
        )
        action.resolved_at = _now()
        await session.commit()
        raise RefundConflict(action.error)

    try:
        external_ref = await _do_refund(
            payment_intent=order.stripe_payment_intent_id,
            amount=amount,
            currency=order.currency,
            idempotency_key=action.idempotency_key,
            reason_text=action.payload.get("reason", ""),
        )
    except Exception as exc:
        action.status = "failed"
        action.error = str(exc)[:500]  # full detail stored internally only
        action.resolved_at = _now()
        await session.commit()
        logger.exception("Refund execution failed for action %s", action_id)
        raise RefundConflict(_GENERIC_REFUND_ERROR) from exc

    action.status = "processed"
    action.external_ref = external_ref
    action.error = None
    action.resolved_at = _now()
    await session.commit()
    return action


async def reject_action(
    session: AsyncSession, workspace: Workspace, action_id: uuid.UUID
) -> PendingAction:
    action = await session.scalar(
        select(PendingAction)
        .where(PendingAction.id == action_id, PendingAction.workspace_id == workspace.id)
        .with_for_update()
    )
    if action is None:
        raise RefundNotFound("Pending action not found.")
    if action.status == "rejected":
        return action  # idempotent
    if action.status == "processed":
        raise RefundConflict("This action was already processed and cannot be rejected.")
    action.status = "rejected"
    action.resolved_at = _now()
    await session.commit()
    return action
