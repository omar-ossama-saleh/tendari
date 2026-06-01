"""Workspace = one store / tenant, identified by an API key."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.models._common import created_at_col, uuid_pk


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # We store only a hash of the API key, never the raw key.
    api_key_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # Per-store agent instruction override; falls back to the default prompt.
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = created_at_col()
