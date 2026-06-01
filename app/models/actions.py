"""Human-in-the-loop: pending_actions (gated destructive actions, e.g. refunds)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.models._common import created_at_col, uuid_pk


class PendingAction(Base):
    __tablename__ = "pending_actions"

    id: Mapped[uuid.UUID] = uuid_pk()
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    action_type: Mapped[str] = mapped_column(Text, nullable=False)  # 'refund'
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Guarantees an action never executes twice, even on retry / double-approve.
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # pending_approval | approved | rejected | processed | failed
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="pending_approval",
        server_default="pending_approval",
    )
    external_ref: Mapped[str | None] = mapped_column(Text, nullable=True)  # stripe_refund_id
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = created_at_col()
    resolved_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
