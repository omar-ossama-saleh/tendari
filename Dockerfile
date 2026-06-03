FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Editable install needs the package sources at install time, so they are copied
# BEFORE `pip install`. The runtime image ships ONLY runtime deps (no dev extras)
# and no tests/ — the suite runs in CI, not inside the deployed image.
COPY pyproject.toml ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
RUN pip install -e .

# Start script for the worker-less Render free-tier blueprints (migrate + seed +
# serve). Not used by docker-compose, which overrides the command per service.
COPY scripts ./scripts
RUN chmod +x ./scripts/start-render.sh

EXPOSE 8000

# Default command; docker-compose overrides per service (api / worker).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
