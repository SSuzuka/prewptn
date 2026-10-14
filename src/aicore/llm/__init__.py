"""LLM layer: one client contract with timeout, retries, rate limit and cost.

Provider choice is a config switch (``llm_backend``): ``stub`` for offline dev
and tests, ``openai`` for any OpenAI-compatible endpoint. Both implement
:class:`~aicore.llm.base.LLMClient`, so callers never branch on the backend.
"""

from __future__ import annotations

from aicore.config import Settings, get_settings
from aicore.llm.base import (
    LLMClient,
    LLMError,
    LLMPermanentError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    LLMUsage,
    RetryPolicy,
    compute_cost_usd,
    run_with_retry,
)

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMPermanentError",
    "LLMRequest",
    "LLMResponse",
    "LLMTransientError",
    "LLMUsage",
    "RetryPolicy",
    "build_llm_client",
    "compute_cost_usd",
    "get_llm_client",
    "reset_llm_client",
    "run_with_retry",
    "set_llm_client",
]


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    """Build a client from settings.

    The OpenAI backend imports ``httpx`` lazily, so a base install that only
    uses the stub never needs the optional ``openai`` extra.
    """
    cfg = settings or get_settings()
    if cfg.llm_backend == "openai":
        from aicore.llm.openai_compat import OpenAICompatibleLLM

        return OpenAICompatibleLLM(settings=cfg)

    from aicore.llm.stub import StubLLM

    return StubLLM(settings=cfg)


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Process-wide client, built on first use (mirrors ``get_database``)."""
    global _client
    if _client is None:
        _client = build_llm_client()
    return _client


def set_llm_client(client: LLMClient) -> None:
    """Override the singleton — used by tests and check scripts."""
    global _client
    _client = client


def reset_llm_client() -> None:
    """Drop the singleton (tests)."""
    global _client
    _client = None
