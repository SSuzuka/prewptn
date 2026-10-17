"""Real embedding backends: local ONNX (fastembed) and OpenAI-compatible HTTP.

Both implement :class:`~aicore.rag.base.EmbeddingClient`, so the retriever never
branches on the backend. The heavy ``fastembed`` import is lazy (at call time),
so a base install that only uses the stub never needs the ``embeddings`` extra.
``httpx`` (openai backend) is not a concern: it already ships with the base stack
as a hard dependency of ``qdrant-client``.

``local`` is the free, no-account option: it downloads a small ONNX model once
and runs it on CPU with no API key and no per-token cost. ``openai`` targets any
endpoint exposing ``POST {base_url}/embeddings``.

Both share the transient/permanent retry loop from the LLM layer via
``run_with_retry(retry_on=EmbeddingTransientError)``.
"""

from __future__ import annotations

import asyncio
import threading
from time import perf_counter
from typing import Any

import httpx

from aicore.config import Settings, get_settings
from aicore.llm.base import RetryPolicy, run_with_retry
from aicore.llm.http_status import decode_json_object, raise_for_status
from aicore.logging import get_logger
from aicore.metrics import record_embedding_call
from aicore.rag.base import (
    EmbeddingPermanentError,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingTransientError,
    EmbeddingUsage,
    compute_embedding_cost_usd,
)
from aicore.telemetry import span

logger = get_logger(__name__)


def _estimate_tokens(text: str) -> int:
    """Whitespace estimate — providers report real usage; this is a fallback."""
    return len(text.split())


class LocalEmbeddings:
    """CPU embeddings via fastembed (extra ``embeddings``)."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        cfg = settings or get_settings()
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise EmbeddingPermanentError(
                "embedding_backend=local requires fastembed; run `uv sync --extra embeddings`"
            ) from exc

        self._text_embedding_cls = TextEmbedding
        self.model = model or cfg.embedding_model
        self._configured_dimension = cfg.embedding_dim
        self.dimension = cfg.embedding_dim
        self._price = cfg.embedding_price_per_1k
        # Построение модели (скачивание + инициализация ONNX) отложено до
        # первого вызова ``embed``: он идёт через ``asyncio.to_thread``, поэтому
        # скачивание не блокирует event loop и параллельные запросы.
        self._embedder: Any | None = None
        # ``_embed_sync`` исполняется в рабочем потоке, поэтому защищаем ленивую
        # инициализацию потоковым локом: два параллельных ``embed`` иначе
        # построили бы модель дважды (повторное скачивание и лишняя память).
        self._embedder_lock = threading.Lock()

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = self._text_embedding_cls(model_name=self.model)
        vectors = [[float(value) for value in vector] for vector in self._embedder.embed(texts)]
        if vectors:
            actual = len(vectors[0])
            if actual != self._configured_dimension:
                # The Qdrant collection is sized from ``embedding_dim`` before the
                # first write, so a drift here would fail every upsert with an
                # opaque error. Surface it as a permanent, actionable one instead.
                raise EmbeddingPermanentError(
                    f"embedding model {self.model!r} returns {actual}-dim vectors but "
                    f"embedding_dim is {self._configured_dimension}; set "
                    f"embedding_dim={actual} and recreate the collection"
                )
            self.dimension = actual
        return vectors

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        started = perf_counter()
        with span("rag.embed", **{"embedding.model": self.model, "embedding.backend": "local"}):
            # ONNX inference is CPU-bound: keep it off the event loop.
            vectors = await asyncio.to_thread(self._embed_sync, request.texts)
            tokens = sum(_estimate_tokens(text) for text in request.texts)
            usage = EmbeddingUsage(total_tokens=tokens)
            latency_ms = int((perf_counter() - started) * 1000)
            cost = compute_embedding_cost_usd(usage, per_1k=self._price)
            record_embedding_call(
                backend="local",
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
        return None


class OpenAICompatibleEmbeddings:
    """Embeddings for any endpoint exposing ``POST {base_url}/embeddings``."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        cfg = settings or get_settings()
        if not cfg.openai_api_key:
            raise EmbeddingPermanentError(
                "openai_api_key is required when embedding_backend=openai"
            )

        self._cfg = cfg
        self.model = cfg.embedding_model
        self.dimension = cfg.embedding_dim
        self._base_url = cfg.openai_base_url.rstrip("/")
        self._api_key = cfg.openai_api_key
        self._timeout = cfg.embedding_timeout_seconds
        self._policy = RetryPolicy(
            max_retries=cfg.embedding_max_retries,
            base_delay=cfg.embedding_retry_base_delay,
        )
        self._semaphore = asyncio.Semaphore(cfg.embedding_max_concurrency)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=self._timeout, transport=transport)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        with span("rag.embed", **{"embedding.model": self.model, "embedding.backend": "openai"}):
            response = await run_with_retry(
                lambda: self._attempt(request),
                self._policy,
                retry_on=EmbeddingTransientError,
                on_retry=self._log_retry,
            )
        record_embedding_call(
            backend="openai",
            model=response.model,
            tokens=response.usage.total_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
        )
        return response

    async def _attempt(self, request: EmbeddingRequest) -> EmbeddingResponse:
        started = perf_counter()
        async with self._semaphore:
            try:
                async with asyncio.timeout(self._timeout):
                    http_response = await self._client.post(
                        f"{self._base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"model": self.model, "input": request.texts},
                    )
            except TimeoutError as exc:
                raise EmbeddingTransientError(
                    f"embedding request timed out after {self._timeout}s"
                ) from exc
            except httpx.TransportError as exc:
                raise EmbeddingTransientError(f"embedding transport error: {exc}") from exc

        raise_for_status(
            http_response,
            label="embedding",
            transient_error=EmbeddingTransientError,
            permanent_error=EmbeddingPermanentError,
        )
        data = decode_json_object(
            http_response,
            label="embedding",
            transient_error=EmbeddingTransientError,
            permanent_error=EmbeddingPermanentError,
        )
        vectors, usage = self._extract(data)
        latency_ms = int((perf_counter() - started) * 1000)
        return EmbeddingResponse(
            vectors=vectors,
            model=str(data.get("model") or self.model),
            dimension=len(vectors[0]) if vectors else self.dimension,
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=compute_embedding_cost_usd(usage, per_1k=self._cfg.embedding_price_per_1k),
        )

    @staticmethod
    def _extract(data: dict[str, Any]) -> tuple[list[list[float]], EmbeddingUsage]:
        try:
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            vectors = [[float(value) for value in item["embedding"]] for item in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingPermanentError(f"unexpected embedding response shape: {exc}") from exc

        raw_usage = data.get("usage") or {}
        tokens = int(raw_usage.get("total_tokens", 0) or 0)
        if tokens == 0:
            tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
        return vectors, EmbeddingUsage(total_tokens=tokens)

    @staticmethod
    def _log_retry(attempt: int, exc: BaseException, delay: float) -> None:
        logger.warning("embedding.retry", attempt=attempt, delay=round(delay, 3), error=str(exc))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
