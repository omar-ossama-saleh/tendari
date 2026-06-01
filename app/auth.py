"""Authentication: `Authorization: Bearer <api_key>` → Workspace.

This is the trust boundary. The dependency resolves the caller's workspace
from their API key; EVERY downstream query and tool MUST scope its data access
by ``workspace.id``. The agent can never reach another workspace's records
because it only ever receives the resolved workspace, never a raw id from the
model or the client.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models.workspace import Workspace
from app.security import hash_api_key

# auto_error=False so we can return a clean, consistent 401 with WWW-Authenticate.
_bearer_scheme = HTTPBearer(auto_error=False, description="Workspace API key")

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Missing or invalid API key.",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_workspace(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Workspace:
    """Resolve the workspace owning the presented API key, or raise 401.

    Identical error for "no token" and "bad token" so callers can't probe which
    keys exist. Lookup is by hash; the raw key is never compared or stored.
    """
    if credentials is None or not credentials.credentials:
        raise _UNAUTHORIZED

    key_hash = hash_api_key(credentials.credentials)
    workspace = await session.scalar(
        select(Workspace).where(Workspace.api_key_hash == key_hash)
    )
    if workspace is None:
        raise _UNAUTHORIZED
    return workspace


# Convenience alias for route signatures: `ws: CurrentWorkspace`.
CurrentWorkspace = Annotated[Workspace, Depends(get_current_workspace)]
DbSession = Annotated[AsyncSession, Depends(get_session)]
