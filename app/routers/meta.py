"""Meta / observability endpoints: health and workspace identity.

Tools listing and usage aggregation are added in their respective milestones
(M6). For M0 this proves liveness and that an authed request resolves to a
workspace.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from app.auth import CurrentWorkspace
from app.db import engine

router = APIRouter(tags=["meta"])


class HealthResponse(BaseModel):
    status: str


class WorkspaceIdentity(BaseModel):
    workspace_id: uuid.UUID
    name: str


@router.get("/healthz", response_model=HealthResponse)
async def healthz(response: Response) -> HealthResponse:
    """Liveness + DB connectivity check. Public (no auth)."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded")
    return HealthResponse(status="ok")


@router.get("/v1/me", response_model=WorkspaceIdentity, tags=["meta"])
async def whoami(workspace: CurrentWorkspace) -> WorkspaceIdentity:
    """Return the workspace resolved from the Bearer API key (auth smoke test)."""
    return WorkspaceIdentity(workspace_id=workspace.id, name=workspace.name)
