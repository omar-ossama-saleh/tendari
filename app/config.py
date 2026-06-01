"""Application configuration.

All runtime configuration is read from the environment via pydantic-settings.
Model ids and token prices are NEVER hardcoded in business logic — they live
here (or in the pricing map in app/observability/usage.py) and are overridable
by env vars, per the build handoff.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["anthropic", "openai", "mock"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- infra ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@db:5432/tendari"
    redis_url: str = "redis://redis:6379/0"

    # --- LLM provider ---
    llm_provider: ProviderName = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    chat_model: str = "claude-haiku-4-5"

    # --- embeddings (OpenAI primary, deterministic offline fallback) ---
    embedding_model: str = "text-embedding-3-small"
    # MUST match embedding_model AND the VECTOR(...) dimension in the applied
    # migration. Changing this requires a NEW migration (see 0001_initial_schema).
    # The ingest path (M1) asserts this matches the DB column and fails loudly.
    embedding_dim: int = 1536

    # --- agent engine ---
    max_agent_iterations: int = 8
    context_token_budget: int = 12_000
    max_output_tokens: int = 1024
    retrieval_top_k: int = 5

    # --- RAG chunking ---
    chunk_target_tokens: int = 800
    chunk_overlap_tokens: int = 100

    # --- external services (optional) ---
    sendgrid_api_key: str | None = None
    sendgrid_from_email: str = "support@tendari.example"
    stripe_secret_key: str | None = None

    # --- demo seed ---
    seed_api_key: str = "demo-key-tendari-001"
    seed_workspace_name: str = "Acme Outdoors"

    # --- pricing override (JSON map) ---
    llm_pricing_json: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_provider(self) -> ProviderName:
        """Fall back to the mock provider when the configured provider has no key.

        Lets the full suite and offline demo run with zero external keys while
        still honouring real keys when present.
        """
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            return "mock"
        if self.llm_provider == "openai" and not self.openai_api_key:
            return "mock"
        return self.llm_provider

    @property
    def pricing_overrides(self) -> dict[str, dict[str, float]]:
        """Parse the optional pricing override map, ignoring malformed/wrong-shape input.

        Must be a JSON object ``{model: {input_per_1k, output_per_1k}}``; anything
        else (a list, a scalar, bad JSON) is treated as "no override" so a config
        typo can never crash the cost calculator.
        """
        if not self.llm_pricing_json:
            return {}
        try:
            parsed = json.loads(self.llm_pricing_json)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {
            model: rates
            for model, rates in parsed.items()
            if isinstance(rates, dict)
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
