"""Auth trust boundary: Bearer key → workspace, with consistent 401s."""

from __future__ import annotations

import uuid
from collections.abc import Callable

from httpx import AsyncClient

from app.models import Workspace


async def test_missing_token_is_401(client: AsyncClient, session_returns: Callable) -> None:
    session_returns(None)
    resp = await client.get("/v1/me")
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


async def test_invalid_key_is_401(client: AsyncClient, session_returns: Callable) -> None:
    # No workspace matches this key's hash → 401, identical to the missing case.
    session_returns(None)
    resp = await client.get("/v1/me", headers={"Authorization": "Bearer not-a-real-key"})
    assert resp.status_code == 401


async def test_valid_key_resolves_to_workspace(
    client: AsyncClient, session_returns: Callable
) -> None:
    ws = Workspace(id=uuid.uuid4(), name="Acme Outdoors", api_key_hash="dummy")
    session_returns(ws)
    resp = await client.get("/v1/me", headers={"Authorization": "Bearer demo-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Acme Outdoors"
    assert body["workspace_id"] == str(ws.id)


async def test_missing_and_invalid_give_identical_response(
    client: AsyncClient, session_returns: Callable
) -> None:
    session_returns(None)
    missing = await client.get("/v1/me")
    invalid = await client.get("/v1/me", headers={"Authorization": "Bearer x"})
    assert missing.status_code == invalid.status_code == 401
    assert missing.json() == invalid.json()
