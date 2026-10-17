"""Тесты ретривера: индексация, плотный поиск, гибридный ранкинг, фильтры."""

from __future__ import annotations

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from aicore.config import Settings
from aicore.rag.chunking import chunk_text
from aicore.rag.retriever import Retriever
from aicore.rag.store import VectorStore
from aicore.rag.stub import StubEmbeddings

DOCS = {
    "idempotency": "Idempotency keys prevent duplicate task creation and duplicate LLM spend.",
    "broker": "The redis stream broker supports acknowledge and redelivery when a worker dies.",
    "bread": "Sourdough bread needs flour water salt and a long slow fermentation.",
}


@pytest_asyncio.fixture
async def retriever() -> Retriever:
    cfg = Settings(vector_backend="memory", embedding_backend="stub", embedding_dim=384)
    engine = Retriever(
        embeddings=StubEmbeddings(settings=cfg),
        store=VectorStore(client=AsyncQdrantClient(":memory:"), collection="t", dimension=384),
        settings=cfg,
    )
    for document_id, text in DOCS.items():
        await engine.index(
            document_id=document_id,
            chunks=chunk_text(text, chunk_size=200, chunk_overlap=20),
            title=document_id,
        )
    yield engine
    await engine.aclose()


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_returns_most_relevant_document(self, retriever: Retriever) -> None:
        hits = await retriever.retrieve("redis stream acknowledge redelivery worker", limit=1)
        assert hits and hits[0].document_id == "broker"

    @pytest.mark.asyncio
    async def test_empty_query_returns_nothing(self, retriever: Retriever) -> None:
        assert await retriever.retrieve("   ") == []

    @pytest.mark.asyncio
    async def test_limit_is_respected(self, retriever: Retriever) -> None:
        hits = await retriever.retrieve("bread flour fermentation", limit=1)
        assert len(hits) == 1

    @pytest.mark.asyncio
    async def test_document_filter(self, retriever: Retriever) -> None:
        hits = await retriever.retrieve("anything", limit=5, document_ids=["bread"])
        assert {hit.document_id for hit in hits} == {"bread"}

    @pytest.mark.asyncio
    async def test_lexical_half_promotes_exact_token(self) -> None:
        cfg = Settings(vector_backend="memory", embedding_dim=384)
        engine = Retriever(
            embeddings=StubEmbeddings(settings=cfg),
            store=VectorStore(client=AsyncQdrantClient(":memory:"), collection="t", dimension=384),
            settings=cfg,
        )
        await engine.index(
            document_id="a", chunks=chunk_text("alpha beta gamma delta", chunk_size=200)
        )
        await engine.index(
            document_id="b", chunks=chunk_text("omega psi error E1234", chunk_size=200)
        )
        hits = await engine.retrieve("error code E1234", limit=1)
        assert hits[0].document_id == "b"
        await engine.aclose()


class TestIndex:
    @pytest.mark.asyncio
    async def test_reindex_does_not_duplicate(self) -> None:
        cfg = Settings(vector_backend="memory", embedding_dim=384)
        engine = Retriever(
            embeddings=StubEmbeddings(settings=cfg),
            store=VectorStore(client=AsyncQdrantClient(":memory:"), collection="t", dimension=384),
            settings=cfg,
        )
        text = "one two three four five six seven eight " * 20
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        first = await engine.index(document_id="d", chunks=chunks)
        second = await engine.index(document_id="d", chunks=chunks)
        assert first == second
        assert await engine.store.count() == first
        await engine.aclose()


class TestCyrillicLexical:
    def test_tokens_keep_cyrillic(self) -> None:
        """Регресс: `[a-z0-9]+` выкидывал кириллицу и делал лексику no-op."""
        from aicore.rag.retriever import _tokens

        assert _tokens("Ошибка Авторизации") == {"ошибка", "авторизации"}

    def test_fuse_promotes_cyrillic_lexical_match(self) -> None:
        from aicore.rag.retriever import _fuse
        from aicore.rag.store import SearchHit

        hits = [
            SearchHit(
                id="a",
                score=0.95,
                text="совсем другой текст про выпечку хлеба",
                document_id="a",
                chunk_index=0,
            ),
            SearchHit(
                id="b",
                score=0.9,
                text="ошибка авторизации в базе данных",
                document_id="b",
                chunk_index=0,
            ),
            SearchHit(
                id="c",
                score=0.85,
                text="ошибка авторизации и доступ к данным",
                document_id="c",
                chunk_index=0,
            ),
        ]
        fused = _fuse(hits, "ошибка авторизации", rrf_k=60)
        assert fused[0].id == "b"
