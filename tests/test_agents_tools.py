"""Тесты агентных инструментов (RAG-tool) — офлайн."""

from __future__ import annotations

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from aicore.agents import search_knowledge_base
from aicore.config import Settings
from aicore.rag import chunk_text, reset_retriever, set_retriever
from aicore.rag.retriever import Retriever
from aicore.rag.store import VectorStore
from aicore.rag.stub import StubEmbeddings

DOC = "The redis stream broker supports acknowledge and redelivery if a worker dies."


def _retriever(collection: str) -> Retriever:
    cfg = Settings(vector_backend="memory", embedding_backend="stub", embedding_dim=384)
    return Retriever(
        embeddings=StubEmbeddings(settings=cfg),
        store=VectorStore(
            client=AsyncQdrantClient(":memory:"), collection=collection, dimension=384
        ),
        settings=cfg,
    )


@pytest_asyncio.fixture
async def seeded_retriever():
    retriever = _retriever("tools_seeded")
    set_retriever(retriever)
    await retriever.index(
        document_id="broker",
        chunks=chunk_text(DOC, chunk_size=200, chunk_overlap=20),
        title="broker",
    )
    yield retriever
    await retriever.aclose()
    reset_retriever()


class TestSearchKnowledgeBase:
    @pytest.mark.asyncio
    async def test_returns_grounded_passage(self, seeded_retriever: Retriever) -> None:
        result = await search_knowledge_base.ainvoke({"query": "redis redelivery worker"})
        assert "redelivery" in result
        assert "broker" in result

    @pytest.mark.asyncio
    async def test_empty_index_returns_clear_message(self) -> None:
        retriever = _retriever("tools_empty")
        set_retriever(retriever)
        try:
            result = await search_knowledge_base.ainvoke({"query": "anything at all"})
        finally:
            await retriever.aclose()
            reset_retriever()
        assert "no relevant passages" in result.lower()

    @pytest.mark.asyncio
    async def test_retrieve_failure_degrades_gracefully(self) -> None:
        """Блип ретривера не должен ронять прогон: тул возвращает строку.

        Иначе исключение из тула улетит в `agent_run` как transient → брокер
        ретраит весь агентный прогон, заново сжигая уже оплаченные model-calls.
        """

        class _BrokenRetriever:
            async def retrieve(self, query: str) -> list[object]:
                raise RuntimeError("vector store down")

            async def aclose(self) -> None: ...

        set_retriever(_BrokenRetriever())  # type: ignore[arg-type]
        try:
            result = await search_knowledge_base.ainvoke({"query": "anything"})
        finally:
            reset_retriever()
        assert "temporarily unavailable" in result.lower()
