"""Tests for the OpenAI-compatible client, offline via ``httpx.MockTransport``.

No network: a mock transport stands in for the provider, so the timeout, retry,
rate-limit and accounting paths are all exercised deterministically.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from aicore.config import Settings
from aicore.llm.base import LLMPermanentError, LLMRequest, LLMTransientError
from aicore.llm.openai_compat import OpenAICompatibleLLM
from aicore.telemetry import reset_telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry():
    reset_telemetry()
    yield
    reset_telemetry()


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_backend": "openai",
        "openai_api_key": "sk-test",
        "llm_max_retries": 2,
        "llm_retry_base_delay": 0.001,
        "llm_timeout_seconds": 5.0,
    }
    base.update(overrides)
    return Settings(**base)


def _ok_body(
    text: str = "hi",
    model: str = "gpt-test",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _client(handler: Any, settings: Settings | None = None) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        settings=settings or _settings(),
        transport=httpx.MockTransport(handler),
    )


class TestConstruction:
    def test_missing_api_key_rejected(self) -> None:
        with pytest.raises(LLMPermanentError):
            OpenAICompatibleLLM(settings=Settings(llm_backend="openai", openai_api_key=None))


class TestSuccessfulCall:
    @pytest.mark.asyncio
    async def test_parses_content_and_usage(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_body(text="hello", model="gpt-x"))

        resp = await _client(handler).complete(LLMRequest(prompt="q"))
        assert resp.text == "hello"
        assert resp.model == "gpt-x"
        assert resp.usage.prompt_tokens == 11
        assert resp.usage.completion_tokens == 7

    @pytest.mark.asyncio
    async def test_cost_uses_configured_prices(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_body(prompt_tokens=1000, completion_tokens=1000))

        cfg = _settings(llm_price_prompt_per_1k=0.001, llm_price_completion_per_1k=0.002)
        resp = await _client(handler, cfg).complete(LLMRequest(prompt="q"))
        assert resp.cost_usd == pytest.approx(0.003)

    @pytest.mark.asyncio
    async def test_sends_auth_header_and_payload(self) -> None:
        captured: dict[str, Any] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            captured["url"] = str(request.url)
            captured["body"] = request.read().decode()
            return httpx.Response(200, json=_ok_body())

        await _client(handler).complete(LLMRequest(prompt="hello", system="be nice"))
        assert captured["auth"] == "Bearer sk-test"
        assert captured["url"].endswith("/chat/completions")
        assert "be nice" in captured["body"] and "hello" in captured["body"]


class TestRetries:
    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self) -> None:
        calls: list[int] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(429, text="slow down")
            return httpx.Response(200, json=_ok_body())

        resp = await _client(handler).complete(LLMRequest(prompt="q"))
        assert resp.text == "hi"
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_permanent_400_is_not_retried(self) -> None:
        calls: list[int] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(400, text="bad request")

        with pytest.raises(LLMPermanentError):
            await _client(handler).complete(LLMRequest(prompt="q"))
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_exhausted_5xx_raises_transient(self) -> None:
        calls: list[int] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(503, text="unavailable")

        with pytest.raises(LLMTransientError):
            await _client(handler, _settings(llm_max_retries=2)).complete(LLMRequest(prompt="q"))
        assert len(calls) == 3  # first attempt + 2 retries

    @pytest.mark.asyncio
    async def test_timeout_maps_to_transient(self) -> None:
        calls: list[int] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(1)
            await asyncio.sleep(0.5)
            return httpx.Response(200, json=_ok_body())

        cfg = _settings(llm_max_retries=0, llm_timeout_seconds=0.05)
        with pytest.raises(LLMTransientError):
            await _client(handler, cfg).complete(LLMRequest(prompt="q"))
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_invalid_json_is_transient(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json")

        with pytest.raises(LLMTransientError):
            await _client(handler, _settings(llm_max_retries=0)).complete(LLMRequest(prompt="q"))

    @pytest.mark.asyncio
    async def test_unexpected_shape_is_permanent(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"model": "m"})

        with pytest.raises(LLMPermanentError):
            await _client(handler).complete(LLMRequest(prompt="q"))


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_concurrency_is_capped(self) -> None:
        active = 0
        peak = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return httpx.Response(200, json=_ok_body())

        client = _client(handler, _settings(llm_max_concurrency=2))
        await asyncio.gather(*(client.complete(LLMRequest(prompt=f"q{i}")) for i in range(6)))
        assert peak <= 2
