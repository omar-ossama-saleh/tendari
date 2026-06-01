"""FastAPI application entrypoint: app construction, router mounting, lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.config import settings
from app.db import engine
from app.routers import actions, conversations, documents, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing to warm yet (migrations run via Alembic, not here).
    yield
    # Shutdown: release the connection pool cleanly.
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tendari — AI Support & Operations Agent",
        version=__version__,
        summary="Answers from a store's help docs and takes action on its systems.",
        lifespan=lifespan,
    )

    app.include_router(meta.router)
    app.include_router(documents.router)
    app.include_router(conversations.router)
    app.include_router(actions.router)

    return app


app = create_app()


__all__ = ["app", "settings"]
