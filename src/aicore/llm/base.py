"""Provider-agnostic building blocks of the LLM layer.

The rest of the application talks to an :class:`LLMClient` and receives an
:class:`LLMResponse` carrying the answer plus token usage, latency and cost.
Which provider produced it (a deterministic stub in dev, an OpenAI-compatible
endpoint in prod) stays hidden behind the protocol.

Resilience lives here too: :class:`RetryPolicy` and :func:`run_with_retry`
implement exponential backoff with jitter over *transient* failures only.
Keeping that logic provider-agnostic means the OpenAI client and any future
provider share one, well-tested retry loop.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """One chat request, provider-independent."""

    prompt: str
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Token usage of a single call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Result of a completed call, with the metrics stage 3 must account for."""

    text: str
    model: str
    usage: LLMUsage
    latency_ms: int
    cost_usd: float


class LLMError(RuntimeError):
    """Base class for every failure raised by the LLM layer."""


class LLMTransientError(LLMError):
    """Retryable: timeout, connection reset, HTTP 429 or 5xx."""


class LLMPermanentError(LLMError):
    """Not retryable: HTTP 4xx (bad request, auth, unknown model) or bad config."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async chat interface implemented by every backend."""

    model: str

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one completion and return the answer with its metrics."""
        ...

    async def aclose(self) -> None:
        """Release any network resources held by the client."""
        ...


def compute_cost_usd(
    usage: LLMUsage,
    *,
    prompt_per_1k: float,
    completion_per_1k: float,
) -> float:
    """Cost of one call in USD from the token counts and per-1k token prices."""
    cost = (
        usage.prompt_tokens / 1000 * prompt_per_1k
        + usage.completion_tokens / 1000 * completion_per_1k
    )
    return round(cost, 8)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential-backoff policy for transient failures.

    ``max_retries`` counts *retries*, not attempts: the operation runs at most
    ``max_retries + 1`` times.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True

    def delay_for(self, attempt: int, *, rand: Callable[[], float] = random.random) -> float:
        """Delay before retry ``attempt`` (0-based).

        Exponential growth is capped at ``max_delay``. With jitter enabled the
        delay is spread over ``[raw/2, raw]`` — this keeps the ordering while
        avoiding a thundering herd of simultaneous retries against the provider.
        """
        raw = min(self.base_delay * (2**attempt), self.max_delay)
        if self.jitter:
            raw *= 0.5 + 0.5 * rand()
        return raw


async def run_with_retry(
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``operation``, retrying only :class:`LLMTransientError`.

    Permanent errors and any non-LLM exception propagate immediately. Once the
    retries are exhausted the last transient error is re-raised, so the caller
    can decide between a task-level retry and a dead-letter.

    ``sleep`` is injectable so tests exercise the backoff without real waits.
    """
    attempt = 0
    while True:
        try:
            return await operation()
        except LLMTransientError as exc:
            if attempt >= policy.max_retries:
                raise
            delay = policy.delay_for(attempt)
            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            await sleep(delay)
            attempt += 1
