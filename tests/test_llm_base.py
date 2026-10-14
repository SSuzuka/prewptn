"""Tests for the provider-agnostic LLM primitives (usage, cost, retries)."""

from __future__ import annotations

import pytest

from aicore.llm.base import (
    LLMPermanentError,
    LLMTransientError,
    LLMUsage,
    RetryPolicy,
    compute_cost_usd,
    run_with_retry,
)


class TestUsageAndCost:
    def test_total_tokens(self) -> None:
        assert LLMUsage(prompt_tokens=3, completion_tokens=4).total_tokens == 7

    def test_cost_is_sum_of_both_sides(self) -> None:
        usage = LLMUsage(prompt_tokens=1000, completion_tokens=1000)
        cost = compute_cost_usd(usage, prompt_per_1k=0.001, completion_per_1k=0.002)
        assert cost == pytest.approx(0.003)

    def test_zero_tokens_cost_nothing(self) -> None:
        assert compute_cost_usd(LLMUsage(), prompt_per_1k=1.0, completion_per_1k=1.0) == 0.0


class TestRetryPolicy:
    def test_exponential_without_jitter(self) -> None:
        policy = RetryPolicy(max_retries=3, base_delay=1.0, jitter=False)
        assert [policy.delay_for(i) for i in range(3)] == [1.0, 2.0, 4.0]

    def test_delay_is_capped(self) -> None:
        policy = RetryPolicy(base_delay=10.0, max_delay=15.0, jitter=False)
        assert policy.delay_for(5) == 15.0

    @pytest.mark.parametrize("rand", [0.0, 0.5, 1.0])
    def test_jitter_stays_within_half_to_full(self, rand: float) -> None:
        policy = RetryPolicy(base_delay=2.0, jitter=True)
        delay = policy.delay_for(1, rand=lambda: rand)
        assert 2.0 <= delay <= 4.0  # raw = 4 -> spread over [2, 4]


class TestRunWithRetry:
    @pytest.mark.asyncio
    async def test_returns_first_success(self) -> None:
        calls: list[int] = []

        async def op() -> str:
            calls.append(1)
            return "ok"

        assert await run_with_retry(op, RetryPolicy()) == "ok"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_retries_transient_then_succeeds(self) -> None:
        state = {"n": 0}
        slept: list[float] = []

        async def op() -> int:
            state["n"] += 1
            if state["n"] < 3:
                raise LLMTransientError("blip")
            return state["n"]

        async def fake_sleep(delay: float) -> None:
            slept.append(delay)

        policy = RetryPolicy(max_retries=3, base_delay=1.0, jitter=False)
        assert await run_with_retry(op, policy, sleep=fake_sleep) == 3
        assert slept == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_exhausted_reraises_last_transient(self) -> None:
        async def op() -> None:
            raise LLMTransientError("always")

        async def no_sleep(_delay: float) -> None:
            return None

        with pytest.raises(LLMTransientError):
            await run_with_retry(op, RetryPolicy(max_retries=2), sleep=no_sleep)

    @pytest.mark.asyncio
    async def test_permanent_error_is_not_retried(self) -> None:
        calls: list[int] = []

        async def op() -> None:
            calls.append(1)
            raise LLMPermanentError("400")

        async def no_sleep(_delay: float) -> None:
            return None

        with pytest.raises(LLMPermanentError):
            await run_with_retry(op, RetryPolicy(max_retries=5), sleep=no_sleep)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_unexpected_exception_propagates(self) -> None:
        async def op() -> None:
            raise ValueError("boom")

        async def no_sleep(_delay: float) -> None:
            return None

        with pytest.raises(ValueError):
            await run_with_retry(op, RetryPolicy(max_retries=5), sleep=no_sleep)

    @pytest.mark.asyncio
    async def test_on_retry_reports_attempt_and_delay(self) -> None:
        state = {"n": 0}
        seen: list[tuple[int, str, float]] = []

        async def op() -> str:
            state["n"] += 1
            if state["n"] == 1:
                raise LLMTransientError("blip")
            return "ok"

        async def no_sleep(_delay: float) -> None:
            return None

        def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
            seen.append((attempt, type(exc).__name__, delay))

        await run_with_retry(
            op,
            RetryPolicy(max_retries=1, base_delay=2.0, jitter=False),
            on_retry=on_retry,
            sleep=no_sleep,
        )
        assert seen == [(1, "LLMTransientError", 2.0)]
