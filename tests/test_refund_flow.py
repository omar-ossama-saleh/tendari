"""Refund gate, idempotent creation, idempotent approval, and rejection.

DB-free via a fake session (real SELECT ... FOR UPDATE locking is verified live);
these cover the authorization gate and the status-guard idempotency branches.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

import pytest

from app.agent.providers.mock import MockProvider
from app.agent.registry import ToolContext, ToolError
from app.config import settings
from app.models import Conversation, Order, PendingAction, Workspace
from app.tools.refunds import (
    InitiateRefundArgs,
    RefundConflict,
    RefundNotFound,
    _authorize_refund,
    _initiate_refund,
    _refund_idempotency_key,
    execute_refund,
    reject_action,
)


class FakeSession:
    def __init__(self, results: list[Any] | None = None, scalars_result: list[Any] | None = None) -> None:
        self.results = list(results or [])
        self.scalars_result = list(scalars_result or [])
        self.added: list[Any] = []
        self.committed = False

    async def scalar(self, *_a: Any, **_k: Any) -> Any:
        return self.results.pop(0) if self.results else None

    async def scalars(self, *_a: Any, **_k: Any) -> list[Any]:
        return list(self.scalars_result)

    def add(self, obj: Any) -> None:
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed = True

    async def flush(self) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False


def _ws() -> Workspace:
    return Workspace(id=uuid.uuid4(), name="Acme", api_key_hash="x")


def _order(ws_id: uuid.UUID, *, pi: str | None = "pi_test_1", total: str = "89.50") -> Order:
    return Order(
        id=uuid.uuid4(), workspace_id=ws_id, order_number="1002", status="delivered",
        items=[], total_amount=Decimal(total), currency="USD", stripe_payment_intent_id=pi,
    )


def _ctx(ws: Workspace, factory) -> ToolContext:
    conv = Conversation(id=uuid.uuid4(), workspace_id=ws.id)
    return ToolContext(workspace=ws, conversation=conv, session_factory=factory)


def _pending(ws_id: uuid.UUID, *, status: str = "pending_approval") -> PendingAction:
    return PendingAction(
        id=uuid.uuid4(), workspace_id=ws_id, action_type="refund",
        payload={"order_number": "1002", "amount": "89.50", "currency": "USD", "reason": "damaged"},
        idempotency_key="k1", status=status,
    )


# --------------------------------------------------------------------------- #
# gate (authorize)
# --------------------------------------------------------------------------- #
async def test_authorize_ok_for_owned_refundable_order() -> None:
    ws = _ws()
    ctx = _ctx(ws, lambda: FakeSession([_order(ws.id)]))
    await _authorize_refund(InitiateRefundArgs(order_number="1002", reason="damaged"), ctx)


async def test_authorize_rejects_unknown_order() -> None:
    ws = _ws()
    ctx = _ctx(ws, lambda: FakeSession([None]))
    with pytest.raises(ToolError):
        await _authorize_refund(InitiateRefundArgs(order_number="9999", reason="x"), ctx)


async def test_authorize_rejects_order_without_payment() -> None:
    ws = _ws()
    ctx = _ctx(ws, lambda: FakeSession([_order(ws.id, pi=None)]))
    with pytest.raises(ToolError):
        await _authorize_refund(InitiateRefundArgs(order_number="1002", reason="x"), ctx)


async def test_authorize_rejects_amount_over_total() -> None:
    ws = _ws()
    ctx = _ctx(ws, lambda: FakeSession([_order(ws.id, total="50.00")]))
    with pytest.raises(ToolError):
        await _authorize_refund(InitiateRefundArgs(order_number="1002", amount=999.0, reason="x"), ctx)


async def test_authorize_rejects_nonpositive_amount() -> None:
    ws = _ws()
    ctx = _ctx(ws, lambda: FakeSession([_order(ws.id)]))
    with pytest.raises(ToolError):
        await _authorize_refund(InitiateRefundArgs(order_number="1002", amount=0.0, reason="x"), ctx)


# --------------------------------------------------------------------------- #
# create (gate-only; emits approval_required; full-refund default)
# --------------------------------------------------------------------------- #
async def test_initiate_creates_pending_and_emits_event() -> None:
    ws = _ws()
    events: list[tuple[str, dict]] = []

    async def emit(event: str, data: dict) -> None:
        events.append((event, data))

    # order load, then no existing-by-key; no committed refunds yet.
    ctx = _ctx(ws, lambda: FakeSession([_order(ws.id), None], scalars_result=[]))
    ctx.emit = emit
    result = await _initiate_refund(InitiateRefundArgs(order_number="1002", reason="damaged"), ctx)

    assert result["status"] == "pending_approval"
    assert result["amount"] == "89.50"  # full order total when amount omitted
    assert any(e[0] == "approval_required" for e in events)


def test_idempotency_key_is_deterministic_and_amount_sensitive() -> None:
    wid, cid = uuid.uuid4(), uuid.uuid4()
    k1 = _refund_idempotency_key(wid, cid, "1002", Decimal("89.50"))
    k2 = _refund_idempotency_key(wid, cid, "1002", Decimal("89.50"))
    k3 = _refund_idempotency_key(wid, cid, "1002", Decimal("10.00"))
    assert k1 == k2 and k1 != k3


def test_money_helpers_quantize_and_minor_units() -> None:
    from app.tools.refunds import _quantize, _to_minor_units

    # 89.5 and 89.50 normalize identically (so they produce one idempotency key).
    assert _quantize(Decimal("89.5"), "usd") == _quantize(Decimal("89.50"), "usd")
    assert str(_quantize(Decimal("89.5"), "usd")) == "89.50"
    assert _to_minor_units(Decimal("89.50"), "usd") == 8950
    # Zero-decimal currency must NOT be multiplied by 100.
    assert _to_minor_units(Decimal("100"), "jpy") == 100


def test_nan_and_inf_amounts_rejected() -> None:
    with pytest.raises((ValueError, Exception)):
        InitiateRefundArgs(order_number="1002", amount=float("nan"), reason="x")
    with pytest.raises((ValueError, Exception)):
        InitiateRefundArgs(order_number="1002", amount=float("inf"), reason="x")


async def test_initiate_blocks_over_refund_against_order_total() -> None:
    ws = _ws()
    # Order total 50; 40 already pending/processed; a new 20 would exceed it.
    existing = _pending(ws.id)
    existing.payload = {"order_number": "1002", "amount": "40.00", "currency": "USD", "reason": "r"}
    ctx = _ctx(ws, lambda: FakeSession([_order(ws.id, total="50.00"), None], scalars_result=[existing]))
    with pytest.raises(ToolError):
        await _initiate_refund(
            InitiateRefundArgs(order_number="1002", amount=20.0, reason="more"), ctx
        )


async def test_execute_blocks_over_refund(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "stripe_secret_key", None)
    ws = _ws()
    action = _pending(ws.id)
    action.payload = {"order_number": "1002", "amount": "30.00", "currency": "USD", "reason": "r"}
    already = _pending(ws.id, status="processed")
    already.payload = {"order_number": "1002", "amount": "30.00", "currency": "USD", "reason": "r"}
    # action(30) + processed(30) = 60 > order total 50 → blocked, marked failed.
    session = FakeSession([action, _order(ws.id, total="50.00")], scalars_result=[already])
    with pytest.raises(RefundConflict):
        await execute_refund(session, ws, action.id)
    assert action.status == "failed"


# --------------------------------------------------------------------------- #
# approve (idempotent execution)
# --------------------------------------------------------------------------- #
async def test_approve_processes_once_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "stripe_secret_key", None)  # simulate path
    ws = _ws()
    action = _pending(ws.id)
    session = FakeSession([action, _order(ws.id)])
    result = await execute_refund(session, ws, action.id)
    assert result.status == "processed"
    assert result.external_ref.startswith("re_sim_")


async def test_approve_is_idempotent_when_already_processed() -> None:
    ws = _ws()
    action = _pending(ws.id, status="processed")
    action.external_ref = "re_existing"
    session = FakeSession([action])  # returns early; never touches Stripe/order
    result = await execute_refund(session, ws, action.id)
    assert result.status == "processed"
    assert result.external_ref == "re_existing"


async def test_approve_rejected_action_conflicts() -> None:
    ws = _ws()
    action = _pending(ws.id, status="rejected")
    with pytest.raises(RefundConflict):
        await execute_refund(FakeSession([action]), ws, action.id)


async def test_approve_missing_action_is_not_found() -> None:
    ws = _ws()
    with pytest.raises(RefundNotFound):
        await execute_refund(FakeSession([None]), ws, uuid.uuid4())


# --------------------------------------------------------------------------- #
# reject
# --------------------------------------------------------------------------- #
async def test_reject_pending_action() -> None:
    ws = _ws()
    action = _pending(ws.id)
    result = await reject_action(FakeSession([action]), ws, action.id)
    assert result.status == "rejected"


async def test_reject_processed_action_conflicts() -> None:
    ws = _ws()
    action = _pending(ws.id, status="processed")
    with pytest.raises(RefundConflict):
        await reject_action(FakeSession([action]), ws, action.id)


# --------------------------------------------------------------------------- #
# mock routing
# --------------------------------------------------------------------------- #
async def test_mock_routes_refund_request() -> None:
    tools = [{"name": n, "description": "d", "input_schema": {}}
             for n in ("search_help_docs", "lookup_order", "initiate_refund")]
    resp = await MockProvider().chat(
        system="s",
        messages=[{"role": "user", "content": "I want to return #1002, it arrived damaged."}],
        tools=tools,
    )
    assert resp.tool_calls[0].name == "initiate_refund"
    assert resp.tool_calls[0].arguments["order_number"] == "1002"
