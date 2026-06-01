"""initial schema — all tables, pgvector + pgcrypto, indexes

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-31

Mirrors the source-of-truth DDL in the build handoff (section 5). The vector
dimension is 1536 (OpenAI text-embedding-3-small). Changing the embedding model
to a different dimension requires a NEW migration, not editing this one, and
must match EMBEDDING_DIM in config.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")  # gen_random_uuid()

    # ---------- tenancy ----------
    op.execute(
        """
        CREATE TABLE workspaces (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name          TEXT NOT NULL,
            api_key_hash  TEXT NOT NULL UNIQUE,
            system_prompt TEXT,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )

    # ---------- RAG ----------
    op.execute(
        """
        CREATE TABLE documents (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            title        TEXT NOT NULL,
            source_type  TEXT NOT NULL,
            source_ref   TEXT,
            status       TEXT NOT NULL DEFAULT 'pending',
            error        TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        f"""
        CREATE TABLE chunks (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            chunk_index  INT  NOT NULL,
            content      TEXT NOT NULL,
            embedding    VECTOR({EMBEDDING_DIM}),
            token_count  INT,
            metadata     JSONB DEFAULT '{{}}'::jsonb
        );
        """
    )
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops);"
    )
    op.execute("CREATE INDEX ix_chunks_workspace_id ON chunks (workspace_id);")

    # ---------- e-commerce ----------
    op.execute(
        """
        CREATE TABLE customers (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            email        TEXT NOT NULL,
            name         TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        """
        CREATE TABLE orders (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id  UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            customer_id   UUID REFERENCES customers(id) ON DELETE SET NULL,
            order_number  TEXT NOT NULL,
            status        TEXT NOT NULL,
            items         JSONB NOT NULL DEFAULT '[]'::jsonb,
            total_amount  NUMERIC(12,2) NOT NULL,
            currency      TEXT NOT NULL DEFAULT 'USD',
            shipping_status TEXT,
            tracking_number TEXT,
            stripe_payment_intent_id TEXT,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (workspace_id, order_number)
        );
        """
    )
    op.execute(
        """
        CREATE TABLE tickets (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id    UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            customer_id     UUID REFERENCES customers(id) ON DELETE SET NULL,
            conversation_id UUID,
            subject         TEXT NOT NULL,
            body            TEXT,
            priority        TEXT NOT NULL DEFAULT 'normal',
            status          TEXT NOT NULL DEFAULT 'open',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )

    # ---------- conversations / agent ----------
    op.execute(
        """
        CREATE TABLE conversations (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            customer_id  UUID REFERENCES customers(id) ON DELETE SET NULL,
            title        TEXT,
            needs_human  BOOLEAN NOT NULL DEFAULT false,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        """
        CREATE TABLE messages (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role            TEXT NOT NULL,
            content         TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        """
        CREATE TABLE tool_calls (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            message_id  UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            tool_name   TEXT NOT NULL,
            arguments   JSONB NOT NULL,
            result      JSONB,
            status      TEXT NOT NULL,
            error       TEXT,
            latency_ms  INT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )

    # ---------- human-in-the-loop ----------
    op.execute(
        """
        CREATE TABLE pending_actions (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id    UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
            action_type     TEXT NOT NULL,
            payload         JSONB NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE,
            status          TEXT NOT NULL DEFAULT 'pending_approval',
            external_ref    TEXT,
            error           TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            resolved_at     TIMESTAMPTZ
        );
        """
    )

    # ---------- observability ----------
    op.execute(
        """
        CREATE TABLE usage_records (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id      UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            conversation_id   UUID REFERENCES conversations(id) ON DELETE SET NULL,
            model             TEXT NOT NULL,
            prompt_tokens     INT NOT NULL,
            completion_tokens INT NOT NULL,
            cost_usd          NUMERIC(12,6) NOT NULL,
            latency_ms        INT,
            endpoint          TEXT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        "CREATE INDEX ix_usage_records_workspace_created "
        "ON usage_records (workspace_id, created_at);"
    )


def downgrade() -> None:
    for table in (
        "usage_records",
        "pending_actions",
        "tool_calls",
        "messages",
        "conversations",
        "tickets",
        "orders",
        "customers",
        "chunks",
        "documents",
        "workspaces",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
    # Extensions are left installed; other schemas may rely on them.
