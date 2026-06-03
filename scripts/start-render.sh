#!/usr/bin/env sh
# Worker-less start sequence for the Render free-tier blueprints (render-free.yaml
# and render-neon.yaml). Render's dockerCommand field doesn't reliably handle
# shell `&&` chaining or nested quotes, so the migrate + seed + serve steps live
# here as a single script the blueprint can invoke by name.
#
# All three steps are idempotent, so re-running on every cold start is safe.
set -e

alembic upgrade head
python -m app.seed
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
