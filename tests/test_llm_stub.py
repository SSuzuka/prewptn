"""Tests for the deterministic StubLLM used in dev and tests."""

from __future__ import annotations

import pytest

from aicore.config import Settings
from aicore.llm.base import LLMRequest, LLMTransientError
from aicore.llm.stub import StubLLM
from aicore.telemetry import reset_telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry():
    reset_telemetry()
    yield
    reset_telemetry()


class TestStubLLM:
    @pytest.mark.asyncio
    async def test_answer_is_deterministic(self) -> None:
        client = StubLLM(settings=Settings(), model="stub-test")
        first = await client.complete(LLMRequest(prompt="ping pong"))
        second = await client.complete(LLMRequest(prompt="ping pong"))
        assert first.text == second.text == "[stub-test] ping pong"

    @pytest.mark.asyncio
    async def test_counts_tokens_from_words(self) -> None:
        client = StubLLM(settings=Settings(), model="m")
        resp = await client.complete(LLMRequest(prompt="one two three", system="sys"))
        assert resp.usage.prompt_tokens == 4  # 3 prompt words + 1 system word
        assert resp.usage.completion_tokens == len(resp.text.split())

    @pytest.mark.asyncio
    async def test_cost_matches_pricing(self) -> None:
        cfg = Settings(llm_price_prompt_per_1k=0.001, llm_price_completion_per_1k=0.002)
        client = StubLLM(settings=cfg, model="m")
        resp = await client.complete(LLMRequest(prompt="a b"))
        expected = (
            resp.usage.prompt_tokens / 1000 * 0.001 + resp.usage.completion_tokens / 1000 * 0.002
        )
        assert resp.cost_usd == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_fail_times_raises_transient_then_recovers(self) -> None:
        client = StubLLM(settings=Settings(), fail_times=1)
        with pytest.raises(LLMTransientError):
            await client.complete(LLMRequest(prompt="x"))
        assert (await client.complete(LLMRequest(prompt="x"))).text

    @pytest.mark.asyncio
    async def test_aclose_is_noop(self) -> None:
        await StubLLM(settings=Settings()).aclose()
