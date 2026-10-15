"""Provider-agnostic building blocks of the embeddings layer.

The rest of the application talks to an :class:`EmbeddingClient` and receives an
:class:`EmbeddingResponse` carrying the vectors plus token usage, latency and
cost. Which provider produced them (a deterministic stub in dev, a local ONNX
model, or an OpenAI-compatible endpoint) stays hidden behind the protocol —
exactly the same shape as :mod:`aicore.llm.base`.

The transient/permanent split and the shared ``run_with_retry`` loop live here
too, so a flaky embedding provider is handled the same way a flaky chat provider
is: retry transient failures with backoff, fail fast on 4xx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """One batch of texts to embed, provider-independent."""

    texts: list[str]


@dataclass(frozen=True, slots=True)
class EmbeddingUsage:
    """Token usage of an embedding call (inputs only, outputs are vectors)."""

    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """Result of an embedding call, with the metrics stage 4 must account for."""

    vectors: list[list[float]] = field(default_factory=list)
    model: str = ""
    dimension: int = 0
    usage: EmbeddingUsage = field(default_factory=EmbeddingUsage)
    latency_ms: int = 0
    cost_usd: float = 0.0


class EmbeddingError(RuntimeError):
    """Base class for every failure raised by the embeddings layer."""


class EmbeddingTransientError(EmbeddingError):
    """Retryable: timeout, connection reset, HTTP 429 or 5xx."""


class EmbeddingPermanentError(EmbeddingError):
    """Not retryable: HTTP 4xx, missing library/credentials, bad config."""


@runtime_checkable
class EmbeddingClient(Protocol):
    """Minimal async interface implemented by every embedding backend."""

    model: str
    dimension: int

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed a batch of texts, preserving input order."""
        ...

    async def aclose(self) -> None:
        """Release any resources held by the client."""
        ...


def compute_embedding_cost_usd(usage: EmbeddingUsage, *, per_1k: float) -> float:
    """Cost of one embedding call in USD from the token count and price."""
    return round(usage.total_tokens / 1000 * per_1k, 8)
