"""Тесты слоя эмбеддингов: детерминированный stub и OpenAI-совместимый клиент.

Всё офлайн: stub без сети, openai-клиент на httpx.MockTransport.
"""

from __future__ import annotations

import math

import httpx
import pytest

from aicore.config import Settings
from aicore.rag import EmbeddingPermanentError, EmbeddingRequest, build_embedding_client
from aicore.rag.embeddings import LocalEmbeddings, OpenAICompatibleEmbeddings
from aicore.rag.stub import StubEmbeddings


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestStubEmbeddings:
    @pytest.mark.asyncio
    async def test_deterministic_and_normalised(self) -> None:
        client = StubEmbeddings(settings=Settings(embedding_dim=64))
        first = await client.embed(EmbeddingRequest(texts=["idempotency key"]))
        second = await client.embed(EmbeddingRequest(texts=["idempotency key"]))
        assert first.vectors == second.vectors
        assert first.dimension == 64
        norm = math.sqrt(sum(v * v for v in first.vectors[0]))
        assert math.isclose(norm, 1.0, rel_tol=1e-9)

    @pytest.mark.asyncio
    async def test_similar_text_scores_higher(self) -> None:
        client = StubEmbeddings(settings=Settings(embedding_dim=256))
        resp = await client.embed(
            EmbeddingRequest(texts=["redis stream broker ack", "redis stream broker"])
        )
        similar = _cosine(resp.vectors[0], resp.vectors[1])
        far = await client.embed(EmbeddingRequest(texts=["sourdough bread recipe"]))
        assert similar > _cosine(resp.vectors[0], far.vectors[0])

    @pytest.mark.asyncio
    async def test_usage_and_cost_accounted(self) -> None:
        client = StubEmbeddings(settings=Settings(embedding_price_per_1k=0.02))
        resp = await client.embed(EmbeddingRequest(texts=["one two three"]))
        assert resp.usage.total_tokens == 3
        assert resp.cost_usd > 0

    @pytest.mark.asyncio
    async def test_empty_text_is_unit_vector(self) -> None:
        client = StubEmbeddings(settings=Settings(embedding_dim=8))
        resp = await client.embed(EmbeddingRequest(texts=["!!!"]))
        assert math.isclose(sum(v * v for v in resp.vectors[0]), 1.0, rel_tol=1e-9)


def test_factory_builds_stub() -> None:
    assert isinstance(build_embedding_client(Settings(embedding_backend="stub")), StubEmbeddings)


class TestOpenAICompatibleEmbeddings:
    @pytest.mark.asyncio
    async def test_parses_vectors_and_usage(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/embeddings")
            return httpx.Response(
                200,
                json={
                    "model": "text-embedding-3-small",
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ],
                    "usage": {"total_tokens": 7},
                },
            )

        client = OpenAICompatibleEmbeddings(
            settings=Settings(
                embedding_backend="openai",
                openai_api_key="sk-test",
                embedding_dim=2,
                embedding_price_per_1k=0.02,
            ),
            transport=httpx.MockTransport(handler),
        )
        resp = await client.embed(EmbeddingRequest(texts=["a", "b"]))
        assert resp.vectors == [[1.0, 0.0], [0.0, 1.0]]  # sorted by index
        assert resp.usage.total_tokens == 7
        assert resp.cost_usd > 0
        await client.aclose()

    @pytest.mark.asyncio
    async def test_429_is_retried(self) -> None:
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, text="slow down")
            return httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [1.0]}], "usage": {}}
            )

        client = OpenAICompatibleEmbeddings(
            settings=Settings(
                embedding_backend="openai",
                openai_api_key="sk-test",
                embedding_dim=1,
                embedding_max_retries=2,
                embedding_retry_base_delay=0.001,
            ),
            transport=httpx.MockTransport(handler),
        )
        resp = await client.embed(EmbeddingRequest(texts=["x"]))
        assert calls["n"] == 2
        assert resp.vectors == [[1.0]]
        await client.aclose()

    def test_missing_api_key_rejected(self) -> None:
        with pytest.raises(EmbeddingPermanentError):
            OpenAICompatibleEmbeddings(
                settings=Settings(embedding_backend="openai", openai_api_key=None)
            )


class _FakeEmbedder:
    """Stand-in for fastembed's TextEmbedding: fixed vector size, no download."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dimension for _ in texts]


def _local_client(dimension: int, *, configured: int) -> LocalEmbeddings:
    client = LocalEmbeddings.__new__(LocalEmbeddings)
    client.model = "fake-model"
    client._configured_dimension = configured
    client.dimension = configured
    client._embedder = _FakeEmbedder(dimension)
    return client


class TestLocalEmbeddingsDimension:
    def test_mismatch_is_permanent_error(self) -> None:
        # Модель отдала не embedding_dim: понятная постоянная ошибка вместо
        # непрозрачного падения upsert в Qdrant.
        client = _local_client(768, configured=384)
        with pytest.raises(EmbeddingPermanentError):
            client._embed_sync(["hello"])

    def test_match_returns_vectors(self) -> None:
        client = _local_client(3, configured=3)
        assert client._embed_sync(["hello"]) == [[0.0, 0.0, 0.0]]
