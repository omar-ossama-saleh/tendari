"""Vertical tool: lookup_order — read order status for the workspace."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select

from app.agent.registry import ToolContext, ToolError, ToolSpec
from app.models import Customer, Order


class LookupOrderArgs(BaseModel):
    order_number: str | None = Field(
        default=None, description="The order number/identifier (e.g. '1002')."
    )
    email: str | None = Field(
        default=None, description="Customer email; returns that customer's orders."
    )

    @model_validator(mode="after")
    def _require_one(self) -> "LookupOrderArgs":
        if self.order_number is not None:
            self.order_number = self.order_number.strip() or None
        if not (self.order_number or self.email):
            raise ValueError("Provide order_number or email.")
        return self


def _order_to_dict(order: Order) -> dict:
    return {
        "order_number": order.order_number,
        "status": order.status,
        "shipping_status": order.shipping_status,
        "tracking_number": order.tracking_number,
        "total_amount": float(order.total_amount) if isinstance(order.total_amount, Decimal) else order.total_amount,
        "currency": order.currency,
        "items": order.items,
        "created_at": order.created_at.isoformat() if isinstance(order.created_at, datetime) else None,
    }


async def _lookup_order(args: LookupOrderArgs, ctx: ToolContext) -> dict:
    # SECURITY: every query is scoped to the caller's workspace; order_number is
    # unique per workspace, so a model can only ever read its own store's orders.
    async with ctx.session_factory() as session:
        if args.order_number:
            order = await session.scalar(
                select(Order).where(
                    Order.workspace_id == ctx.workspace.id,
                    Order.order_number == args.order_number,
                )
            )
            if order is None:
                return {"found": False, "message": f"No order '{args.order_number}' found."}
            return {"found": True, "orders": [_order_to_dict(order)]}

        email = (args.email or "").strip().lower()
        customer = await session.scalar(
            select(Customer).where(
                Customer.workspace_id == ctx.workspace.id, Customer.email == email
            )
        )
        if customer is None:
            return {"found": False, "message": f"No customer found for '{email}'."}
        orders = await session.scalars(
            select(Order)
            .where(
                Order.workspace_id == ctx.workspace.id,
                Order.customer_id == customer.id,
            )
            .order_by(Order.created_at.desc())
        )
        order_list = [_order_to_dict(o) for o in orders]
        if not order_list:
            return {"found": False, "message": f"No orders for '{email}'."}
        return {"found": True, "orders": order_list}


LOOKUP_ORDER = ToolSpec(
    name="lookup_order",
    description=(
        "Look up an order's status, items, totals, shipping status, and tracking "
        "number. Provide the order_number for a specific order, or an email to list "
        "that customer's orders. Use this for any question about where an order is, "
        "its status, or what it contained."
    ),
    args_model=LookupOrderArgs,
    handler=_lookup_order,
)
