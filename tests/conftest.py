"""Shared test fixtures.

The M0 tests exercise the auth trust boundary without a live database by
overriding the session dependency with a tiny fake. Integration tests against a
real pgvector database arrive with later milestones.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.db import get_session
from app.main import app


class FakeSession:
    """Minimal async session stand-in: ``scalar`` returns a preset result."""

    def __init__(self, result: Any = None) -> None:
        self.result = result

    async def scalar(self, *args: Any, **kwargs: Any) -> Any:
        return self.result


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def session_returns() -> Callable[[Any], None]:
    """Make ``get_session`` yield a FakeSession whose ``scalar`` returns ``result``."""

    def _set(result: Any) -> None:
        async def _dep():
            yield FakeSession(result)

        app.dependency_overrides[get_session] = _dep

    return _set
