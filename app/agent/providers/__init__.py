"""Provider selection. Honors LLM_PROVIDER but falls back to the mock provider
when the configured provider has no API key (so the app runs key-free)."""

from __future__ import annotations

from functools import lru_cache

from app.agent.providers.base import EmitFn, LLMResponse, Provider, ToolCall, Usage
from app.agent.providers.mock import MockProvider
from app.config import settings

__all__ = ["Provider", "LLMResponse", "ToolCall", "Usage", "EmitFn", "get_provider"]


@lru_cache
def get_provider() -> Provider:
    """Process-wide provider (reuses one SDK HTTP client across requests)."""
    provider = settings.effective_provider
    if provider == "anthropic":
        from app.agent.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    if provider == "openai":
        from app.agent.providers.openai import OpenAIProvider

        return OpenAIProvider()
    return MockProvider()
