"""Deterministic in-process embeddings for dev, tests and demos.

No network, no API key, no model download: each text is turned into a sparse
bag-of-tokens vector (tokens hashed into buckets), then L2-normalised. The
result is not *good* embeddings — it is *reproducible* embeddings whose cosine
similarity grows with token overlap, which is exactly what tests need to assert
"the right chunk is retrieved" without pulling a model.

Two useful properties fall out of that design:
  * indexing the same text twice yields byte-identical vectors (idempotent);
  * ``dimension`` follows ``embedding_dim``, so the stub and a real model can
    share one Qdrant collection configuration.
"""

from __future__ import annotations

import hashlib
import math
import re
from time import perf_counter

from aicore.config import Settings, get_settings
from aicore.metrics import record_embedding_call
from aicore.rag.base import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    compute_embedding_cost_usd,
)
from aicore.telemetry import span

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class StubEmbeddings:
    """Deterministic hashing embedder (dev/test only)."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model: str | None = None,
        dimension: int | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.model = model or f"stub:{cfg.embedding_model}"
        self.dimension = dimension or cfg.embedding_dim
        self._price = cfg.embedding_price_per_1k

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokens(text):
            # blake2b is stable across runs and platforms (unlike hash()).
            digest = int.from_bytes(
                hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            vector[digest % self.dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # Empty / punctuation-only text: a fixed unit vector keeps cosines
            # well-defined instead of dividing by zero.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        started = perf_counter()
        with span("rag.embed", **{"embedding.model": self.model, "embedding.backend": "stub"}):
            vectors = [self._vector(text) for text in request.texts]
            tokens = sum(len(_tokens(text)) for text in request.texts)
            usage = EmbeddingUsage(total_tokens=tokens)
            latency_ms = int((perf_counter() - started) * 1000)
            cost = compute_embedding_cost_usd(usage, per_1k=self._price)
            record_embedding_call(
                backend="stub",
                model=self.model,
                tokens=tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
            )
            return EmbeddingResponse(
                vectors=vectors,
                model=self.model,
                dimension=self.dimension,
                usage=usage,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

    async def aclose(self) -> None:
        """No resources to release."""
        return None
