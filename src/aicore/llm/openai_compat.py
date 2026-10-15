"""OpenAI-compatible chat client with the four stage-3 guarantees.

One class covers the whole LLM contract:
  * **timeout** — ``asyncio.timeout`` bounds every attempt, so one hung provider
    cannot pin a worker forever;
  * **rate limiting** — a process-wide ``asyncio.Semaphore(llm_max_concurrency)``
    caps concurrent in-flight requests;
  * **retries** — exponential backoff + jitter on transient failures only
    (:class:`LLMTransientError`); 4xx errors fail fast;
  * **accounting** — prompt/completion tokens, latency and ``cost_usd`` per call.

The transport is created only when a client is instantiated, so importing this
module needs neither an API key nor a network. ``transport=`` / ``client=`` exist
so tests can inject ``httpx.MockTransport`` and assert the retry/timeout
behaviour offline.
"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

import httpx

from aicore.config import Settings, get_settings
from aicore.llm.base import (
    LLMPermanentError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    LLMUsage,
    RetryPolicy,
    compute_cost_usd,
    run_with_retry,
)
from aicore.llm.http_status import decode_json_object, raise_for_status
from aicore.logging import get_logger
from aicore.metrics import record_llm_call
from aicore.telemetry import record_generation, span

logger = get_logger(__name__)


class OpenAICompatibleLLM:
    """Chat client for any endpoint exposing ``POST {base_url}/chat/completions``."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        cfg = settings or get_settings()
        if not cfg.openai_api_key:
            raise LLMPermanentError("openai_api_key is required when llm_backend=openai")

        self._cfg = cfg
        self.model = cfg.llm_model
        self._base_url = cfg.openai_base_url.rstrip("/")
        self._api_key = cfg.openai_api_key
        self._timeout = cfg.llm_timeout_seconds
        self._policy = RetryPolicy(
            max_retries=cfg.llm_max_retries,
            base_delay=cfg.llm_retry_base_delay,
        )
        self._semaphore = asyncio.Semaphore(cfg.llm_max_concurrency)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=self._timeout, transport=transport)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        with span("llm.complete", **{"llm.model": self.model, "llm.backend": "openai"}):
            response = await run_with_retry(
                lambda: self._attempt(request),
                self._policy,
                on_retry=self._log_retry,
            )
        record_generation(
            name="llm.complete",
            model=response.model,
            input={"prompt": request.prompt, "system": request.system},
            output=response.text,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            metadata={"backend": "openai"},
        )
        record_llm_call(
            backend="openai",
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
        )
        return response

    async def _attempt(self, request: LLMRequest) -> LLMResponse:
        started = perf_counter()
        payload = self._build_payload(request)

        async with self._semaphore:
            try:
                async with asyncio.timeout(self._timeout):
                    http_response = await self._client.post(
                        f"{self._base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
            except TimeoutError as exc:
                raise LLMTransientError(f"llm request timed out after {self._timeout}s") from exc
            except httpx.TransportError as exc:
                raise LLMTransientError(f"llm transport error: {exc}") from exc

        raise_for_status(
            http_response,
            label="llm",
            transient_error=LLMTransientError,
            permanent_error=LLMPermanentError,
        )
        data = decode_json_object(
            http_response,
            label="llm",
            transient_error=LLMTransientError,
            permanent_error=LLMPermanentError,
        )
        text, usage = self._extract(data)
        latency_ms = int((perf_counter() - started) * 1000)
        return LLMResponse(
            text=text,
            model=str(data.get("model") or self.model),
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=compute_cost_usd(
                usage,
                prompt_per_1k=self._cfg.llm_price_prompt_per_1k,
                completion_per_1k=self._cfg.llm_price_completion_per_1k,
            ),
        )

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        return payload

    @staticmethod
    def _extract(data: dict[str, Any]) -> tuple[str, LLMUsage]:
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMPermanentError(f"unexpected llm response shape: {exc}") from exc

        raw_usage = data.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
        )
        return text, usage

    @staticmethod
    def _log_retry(attempt: int, exc: BaseException, delay: float) -> None:
        logger.warning("llm.retry", attempt=attempt, delay=round(delay, 3), error=str(exc))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
