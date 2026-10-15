"""Тесты обёртки векторного хранилища (Qdrant в режиме :memory:). Офлайн."""

from __future__ import annotations

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from aicore.rag.store import VectorPoint, VectorStore, make_point_id


@pytest_asyncio.fixture
async def store() -> VectorStore:
    client = AsyncQdrantClient(":memory:")
    vector_store = VectorStore(client=client, collection="t", dimension=4)
    yield vector_store
    await vector_store.aclose()


def test_make_point_id_is_deterministic() -> None:
    assert make_point_id("doc", 0) == make_point_id("doc", 0)
    assert make_point_id("doc", 0) != make_point_id("doc", 1)


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_upsert_and_count(self, store: VectorStore) -> None:
        written = await store.upsert(
            [
                VectorPoint(id=make_point_id("d", 0), vector=[1, 0, 0, 0], payload={"text": "a"}),
                VectorPoint(id=make_point_id("d", 1), vector=[0, 1, 0, 0], payload={"text": "b"}),
            ]
        )
        assert written == 2
        assert await store.count() == 2

    @pytest.mark.asyncio
    async def test_search_returns_payload_fields(self, store: VectorStore) -> None:
        await store.upsert(
            [
                VectorPoint(
                    id=make_point_id("doc-1", 0),
                    vector=[1.0, 0.0, 0.0, 0.0],
                    payload={
                        "document_id": "doc-1",
                        "chunk_index": 0,
                        "text": "hello",
                        "title": "T",
                        "source": "S",
                    },
                )
            ]
        )
        hits = await store.search([1.0, 0.0, 0.0, 0.0], limit=1)
        assert len(hits) == 1
        assert hits[0].text == "hello"
        assert hits[0].document_id == "doc-1"
        assert hits[0].title == "T"
        assert hits[0].source == "S"
        assert hits[0].score > 0.99

    @pytest.mark.asyncio
    async def test_search_filters_by_documents(self, store: VectorStore) -> None:
        await store.upsert(
            [
                VectorPoint(
                    id=make_point_id("a", 0),
                    vector=[1.0, 0.0, 0.0, 0.0],
                    payload={"document_id": "a", "chunk_index": 0, "text": "a"},
                ),
                VectorPoint(
                    id=make_point_id("b", 0),
                    vector=[1.0, 0.0, 0.0, 0.0],
                    payload={"document_id": "b", "chunk_index": 0, "text": "b"},
                ),
            ]
        )
        hits = await store.search([1.0, 0.0, 0.0, 0.0], limit=5, document_ids=["a"])
        assert {hit.document_id for hit in hits} == {"a"}

    @pytest.mark.asyncio
    async def test_upsert_same_id_overwrites(self, store: VectorStore) -> None:
        point = VectorPoint(id=make_point_id("d", 0), vector=[1, 0, 0, 0], payload={"text": "v1"})
        await store.upsert([point])
        await store.upsert([VectorPoint(id=point.id, vector=[0, 1, 0, 0], payload={"text": "v2"})])
        assert await store.count() == 1
        hits = await store.search([0.0, 1.0, 0.0, 0.0], limit=1)
        assert hits[0].text == "v2"

    @pytest.mark.asyncio
    async def test_delete_document(self, store: VectorStore) -> None:
        await store.upsert(
            [
                VectorPoint(
                    id=make_point_id("d", 0),
                    vector=[1, 0, 0, 0],
                    payload={"document_id": "d", "chunk_index": 0, "text": "x"},
                )
            ]
        )
        await store.delete_document("d")
        assert await store.count() == 0
