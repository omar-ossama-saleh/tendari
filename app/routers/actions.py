"""Human-in-the-loop: list / approve / reject pending actions (refunds).

Approval is the ONLY path that executes a refund. Everything is workspace-scoped.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status

from app.auth import CurrentWorkspace, DbSession
from app.models import PendingAction
from app.schemas.actions import ApproveResponse, PendingActionOut, RejectResponse
from app.tools.refunds import (
    RefundConflict,
    RefundNotFound,
    execute_refund,
    reject_action,
)
from sqlalchemy import select

router = APIRouter(prefix="/v1/pending-actions", tags=["pending-actions"])


@router.get("", response_model=list[PendingActionOut])
async def list_pending_actions(
    workspace: CurrentWorkspace,
    session: DbSession,
    status_filter: str | None = Query(default=None, alias="status"),
) -> list[PendingAction]:
    stmt = select(PendingAction).where(PendingAction.workspace_id == workspace.id)
    if status_filter:
        stmt = stmt.where(PendingAction.status == status_filter)
    stmt = stmt.order_by(PendingAction.created_at.desc())
    return list(await session.scalars(stmt))


@router.post("/{action_id}/approve", response_model=ApproveResponse)
async def approve_action(
    action_id: uuid.UUID, workspace: CurrentWorkspace, session: DbSession
) -> ApproveResponse:
    try:
        action = await execute_refund(session, workspace, action_id)
    except RefundNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RefundConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ApproveResponse(status=action.status, external_ref=action.external_ref)


@router.post("/{action_id}/reject", response_model=RejectResponse)
async def reject_pending_action(
    action_id: uuid.UUID, workspace: CurrentWorkspace, session: DbSession
) -> RejectResponse:
    try:
        action = await reject_action(session, workspace, action_id)
    except RefundNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RefundConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return RejectResponse(status=action.status)
