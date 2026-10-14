"""Deterministic in-process LLM for dev, tests and demos.

No network, no API key, no real spend: the answer and the token counts are
derived from the prompt with a whitespace tokenizer, so ``cost_usd`` is
reproducible. That lets the whole accounting path (tokens -> price -> stored
metrics) be tested end to end without touching a provider.

Two knobs exist only to exercise the resilience paths:
  * ``fail_times`` raises :class:`LLMTransientError` for the first N calls;
  * ``latency_seconds`` sleeps before answering, to trigger timeouts.
"""

from __future__ import annotations

import asyncio
from time import perf_counter

from aicore.config import Settings, get_settings
from aicore.llm.base import (
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    LLMUsage,
    compute_cost_usd,
)
from aicore.telemetry import record_generation, span


def _count_tokens(text: str) -> int:
    """Whitespace token estimate — deterministic and good enough for tests."""
    return len(text.split())


class StubLLM:
    """Echo-style deterministic client."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model: str | None = None,
        fail_times: int = 0,
        latency_seconds: float = 0.0,
    ) -> None:
        cfg = settings or get_settings()
        self.model = model or f"stub:{cfg.llm_model}"
        self._prompt_price = cfg.llm_price_prompt_per_1k
        self._completion_price = cfg.llm_price_completion_per_1k
        self._fail_times = fail_times
        self._latency_seconds = latency_seconds
        self.calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = perf_counter()
        with span("llm.complete", **{"llm.model": self.model, "llm.backend": "stub"}):
            self.calls += 1
            if self.calls <= self._fail_times:
                raise LLMTransientError(f"stub transient failure #{self.calls}")

            if self._latency_seconds > 0:
                await asyncio.sleep(self._latency_seconds)

            text = f"[{self.model}] {request.prompt.strip()}"
            usage = LLMUsage(
                prompt_tokens=_count_tokens(request.prompt) + _count_tokens(request.system or ""),
                completion_tokens=_count_tokens(text),
            )
            latency_ms = int((perf_counter() - started) * 1000)
            response = LLMResponse(
                text=text,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                cost_usd=compute_cost_usd(
                    usage,
                    prompt_per_1k=self._prompt_price,
                    completion_per_1k=self._completion_price,
                ),
            )
            record_generation(
                name="llm.complete",
                model=self.model,
                input={"prompt": request.prompt, "system": request.system},
                output=response.text,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
                metadata={"backend": "stub"},
            )
            return response

    async def aclose(self) -> None:
        """No resources to release."""
        return None
