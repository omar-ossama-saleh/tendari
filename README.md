# Tendari

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![Postgres](https://img.shields.io/badge/Postgres-pgvector-4169E1?logo=postgresql&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-worker-37814A?logo=celery&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-idempotency-DC382D?logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-85%20passing-success)

**An AI customer-support & operations agent for e-commerce that does four things
well: grounded answers, real actions, a hard safety gate on money, and visible
cost — all in one Swagger-documented API.**

It answers shopper questions grounded in your store's help docs (RAG), takes real
actions through a governed tool layer (look up orders, file tickets, send email,
escalate), and gates money-moving actions (refunds) behind mandatory human
approval — while logging token usage, cost, and latency for every model call.

*Built to show the patterns a production agent actually needs — not a chat
wrapper: a domain-agnostic agent loop, multi-tenant isolation, a human-in-the-loop
gate on money, RAG over help docs, swappable LLM providers, and per-call cost
accounting.*

<p align="center">
  <img src="docs/demo.png" width="540"
       alt="Tendari demo console — a grounded answer, a live order lookup, and a refund held for human approval, with token cost ticking up in the usage panel." />
</p>

> The built-in **`/demo`** console, end to end: a cited answer from the store's
> help docs → a live `lookup_order` → a refund **parked for human approval** —
> with token cost accruing in the usage panel. Runs on the mock provider with zero
> API keys.

---

## Highlights

- **A domain-agnostic agent engine** — `app/agent/` runs the entire tool-calling
  loop (iteration cap, concurrent tools, context-budget truncation, usage
  accounting) without importing a single e-commerce concept.
- **A trust boundary the model can't cross** — the LLM supplies tool *arguments*
  only, never a workspace id; the workspace is resolved from the API key, so a
  tool physically cannot reach another tenant's data.
- **Money is gated, bounded, and idempotent** — refunds never fire from the
  model; they wait for a human `approve`, can't exceed an order-level ceiling, and
  process exactly once via a Stripe idempotency key.
- **Provider-swappable, config-driven** — Anthropic · OpenAI · a mock provider
  behind one interface; model ids and token prices live in config, never code.
- **Runs offline with zero keys** — no keys ⇒ mock LLM + deterministic
  embeddings, so the full demo *and* the 85-test suite run on nothing but Docker.
- **Cost is a first-class, durable signal** — every model call writes a usage row
  on its own committed transaction (it survives a request rollback); `GET
  /v1/usage` rolls it up by model and by day.

---

## Architecture: a domain-agnostic engine + a vertical tool layer

The single most important design decision: the **agent engine knows nothing
about e-commerce.**

```
            HTTP (FastAPI, Bearer-key auth → workspace)
                              │
        ┌─────────────────────▼─────────────────────┐
        │  app/agent/  — DOMAIN-AGNOSTIC             │
        │  engine.py   run_agent loop: iterate,      │
        │              authorize→execute tools,      │
        │              stream, budget, record usage  │
        │  registry.py tool registration + the       │
        │              authorize()-before-execute    │
        │              security invariant            │
        │  providers/  anthropic | openai | mock     │
        └─────────────────────┬─────────────────────┘
                              │ ToolContext(workspace, …)
        ┌─────────────────────▼─────────────────────┐
        │  app/tools/  — VERTICAL (e-commerce)       │
        │  help_docs · orders · tickets · email ·    │
        │  escalate · refunds                        │
        └────────────────────────────────────────────┘
```

- **The engine** runs the loop, enforces the iteration cap, runs tools
  concurrently, truncates context to a token budget, and records usage. It never
  imports a model or a store concept.
- **The tools** hold all store logic. Swapping verticals (say, to logistics or
  healthcare) means rewriting `app/tools/` — not the engine.
- **Two security invariants live in `registry.py`:** (1) the model supplies only
  tool *arguments*, never a workspace id — the workspace comes from the trusted
  `ToolContext` resolved from the API key, so a tool can only ever touch the
  caller's own data; (2) `authorize()` always runs **before** `execute()`.
- **Money is gated.** `initiate_refund` never refunds — it authorizes, enforces
  an order-level ceiling, and creates a `pending_actions` row. A refund fires
  only via the human `approve` endpoint, exactly once (idempotent), and can
  never exceed the order total. See [`app/tools/refunds.py`](app/tools/refunds.py).

---

## Quickstart (local, zero external keys)

The full stack runs offline: with no API keys it uses a **mock LLM provider** and
**deterministic offline embeddings**, so the demo and the test suite need nothing
but Docker.

```bash
cp .env.example .env          # defaults are fine for local
docker compose up --build     # api + worker + postgres(pgvector) + redis
```

On boot the `api` service applies migrations, seeds a demo workspace, and serves:

- **Swagger UI:** http://localhost:8000/docs
- **Health:** http://localhost:8000/healthz → `{"status":"ok"}`
- **Demo page:** http://localhost:8000/demo/ (tiny built-in chat console)

The seed creates workspace **Acme Outdoors**, 2 customers, 5 orders (incl.
**#1002** with a Stripe test payment intent), and 2 help docs. The demo API key
is `demo-key-tendari-001` — send it as `Authorization: Bearer demo-key-tendari-001`.

> **Real providers:** set `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`,
> `OPENAI_API_KEY` (embeddings), and `STRIPE_SECRET_KEY` (`sk_test_…`) in `.env`.
> Token prices are config, not code — see `LLM_PRICING_JSON` and
> `app/observability/usage.py`.

> **Beyond local:** every process is containerized, so the same four-service
> stack runs anywhere Docker does — no code changes between laptop and cloud.

---

## Try it yourself — the end-to-end demo

Run live against Swagger (or the `/demo` page) with the demo key:

1. **`GET /v1/tools`** — "here's everything the agent can do."
2. **`POST /v1/documents`** — upload a return-policy PDF; poll until it's `ready`.
3. Ask **"What's your return window?"** → streams an answer **cited from the doc**.
4. Ask **"Where's my order #1002?"** → agent calls `lookup_order`, streams real status.
5. Ask **"I want to return #1002, it arrived damaged."** → agent calls
   `initiate_refund`; you see the `approval_required` SSE event and a new row in
   **`GET /v1/pending-actions`**.
6. **`POST /v1/pending-actions/{id}/approve`** → Stripe test-mode refund processes
   once (idempotent); response carries the `external_ref`.
7. **`GET /v1/usage`** → total cost with per-model and per-day breakdown.

---

## API reference

| Method & path | Purpose |
|---|---|
| `GET /healthz` | Liveness + DB check (public) |
| `GET /v1/me` | Workspace resolved from the API key |
| `GET /v1/tools` | Tool catalogue: `{name, description, parameters_schema}` |
| `GET /v1/usage` | `{total_cost_usd, total_tokens, request_count, by_model[], by_day[]}` |
| `POST /v1/documents` · `GET` · `DELETE` | Ingest / list / delete help docs (RAG) |
| `POST /v1/conversations` | Start a conversation |
| `POST /v1/conversations/{id}/messages` | Send a message; `stream:true` → SSE |
| `GET /v1/conversations/{id}` | Full history incl. tool calls |
| `GET /v1/pending-actions` | List pending actions (`?status=…`) |
| `POST /v1/pending-actions/{id}/approve` · `/reject` | Human-in-the-loop refund control |

**SSE events** (`stream:true`): `token`, `tool_call_start`, `tool_call_result`,
`approval_required`, `done` (carries `usage`).

Every `/v1/*` route is workspace-scoped by the Bearer key; there is no
cross-workspace data access.

---

## Observability

Every LLM call writes a `usage_records` row (model, prompt/completion tokens,
cost, latency, endpoint) on its **own committed transaction** — cost is a durable
audit that survives a later request rollback. Cost = tokens × per-model price
from config (`app/observability/usage.py`, overridable via `LLM_PRICING_JSON`).
`GET /v1/usage` aggregates these per workspace, with `by_model` and `by_day`
rollups.

---

## Testing

```bash
pip install -e ".[dev]"
pytest                       # DB-free unit suite (mock provider, fake sessions)
```

The suite covers the agent loop (incl. error path + iteration cap), tool arg
validation + routing, the auth trust boundary, the refund gate → approve →
idempotent execute flow (incl. over-refund ceiling), and the observability
endpoints. DB-dependent SQL (pgvector retrieval, the refund `FOR UPDATE` ceiling,
usage `GROUP BY`) is verified live against Postgres.

---

## Project structure

```
app/
  main.py            FastAPI app, router mounting, lifespan, /demo static
  config.py          pydantic-settings — all env vars
  db.py              async engine + session dependency
  auth.py            Bearer key → workspace (the trust boundary)
  models/            SQLAlchemy 2.0 models
  schemas/           Pydantic request/response models
  agent/             engine.py · registry.py · prompts.py · providers/
  tools/             help_docs · orders · tickets · email · refunds · escalate
  rag/               ingest (Celery) + retrieve (workspace-scoped top-k)
  observability/     usage.py — record_usage + aggregation + pricing config
  routers/           documents · conversations (+SSE) · actions · meta
  tasks.py           Celery app
  seed.py            demo workspace, customers, orders, help docs
tests/               loop, tools, auth, refund flow, meta
alembic/             migrations
docker-compose.yml · Dockerfile · pyproject.toml · .env.example
```

## License

Released under the [MIT License](LICENSE).
