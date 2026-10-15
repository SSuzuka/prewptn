"""API-тесты документов RAG: загрузка (идемпотентно), статус, список. Офлайн."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient

from aicore.api.app import create_app
from aicore.api.deps import reset_rate_limiter
from aicore.config import Settings
from aicore.db.models import DocumentStatus, TaskStatus
from aicore.db.session import Database
from aicore.rag import reset_retriever, set_retriever
from aicore.rag.documents import DocumentService
from aicore.rag.retriever import Retriever
from aicore.rag.store import VectorStore
from aicore.rag.stub import StubEmbeddings
from aicore.tasks.service import TaskService
from aicore.telemetry import reset_telemetry


@pytest_asyncio.fixture
async def client(db: Database, use_db: Database):
    reset_rate_limiter()
    reset_retriever()
    reset_telemetry()
    cfg = Settings(vector_backend="memory", embedding_backend="stub", embedding_dim=384)
    set_retriever(
        Retriever(
            embeddings=StubEmbeddings(settings=cfg),
            store=VectorStore(client=AsyncQdrantClient(":memory:"), collection="t", dimension=384),
            settings=cfg,
        )
    )

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    reset_rate_limiter()
    reset_retriever()
    reset_telemetry()


class TestCreateDocument:
    @pytest.mark.asyncio
    async def test_create_indexes_inline(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/v1/documents",
            json={"title": "Intro", "content": "Idempotency keys avoid duplicate LLM spend."},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == DocumentStatus.INDEXED.value
        assert body["chunk_count"] >= 1
        assert body["content_length"] > 0
        assert body["indexed_at"] is not None

    @pytest.mark.asyncio
    async def test_same_content_is_idempotent(self, client: AsyncClient) -> None:
        payload = {"title": "T", "content": "same body text"}
        first = await client.post("/v1/documents", json=payload)
        second = await client.post("/v1/documents", json=payload)
        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["id"] == second.json()["id"]

    @pytest.mark.asyncio
    async def test_empty_content_rejected(self, client: AsyncClient) -> None:
        resp = await client.post("/v1/documents", json={"title": "T", "content": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_content_length_is_normalised(self, client: AsyncClient) -> None:
        # content_length считается от нормализованного текста (как и content_hash),
        # а не от сырого: иначе «  hello  » и «hello» расходились бы по длине.
        resp = await client.post("/v1/documents", json={"title": "T", "content": "  hello  "})
        assert resp.status_code == 202
        assert resp.json()["content_length"] == len("hello")

    @pytest.mark.asyncio
    async def test_whitespace_only_difference_is_idempotent(self, client: AsyncClient) -> None:
        first = await client.post("/v1/documents", json={"title": "T", "content": "hello"})
        second = await client.post("/v1/documents", json={"title": "T", "content": "\r\nhello  "})
        assert first.json()["id"] == second.json()["id"]
        assert first.json()["content_length"] == second.json()["content_length"]


class TestReadDocument:
    @pytest.mark.asyncio
    async def test_get_after_create(self, client: AsyncClient) -> None:
        created = await client.post("/v1/documents", json={"title": "T", "content": "hello doc"})
        document_id = created.json()["id"]

        resp = await client.get(f"/v1/documents/{document_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == DocumentStatus.INDEXED.value

    @pytest.mark.asyncio
    async def test_unknown_returns_404(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/documents/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_422(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/documents/not-a-uuid")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_list_documents(self, client: AsyncClient) -> None:
        await client.post("/v1/documents", json={"title": "T", "content": "listed doc"})
        resp = await client.get("/v1/documents")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1


async def _failed_task(db: Database, *, task_type: str, payload: dict) -> str:
    async with db.session() as session:
        svc = TaskService(session)
        task, _ = await svc.create_task(task_type=task_type, payload=payload, max_attempts=1)
        await svc.mark_running(task)
        await svc.mark_failed(task, "boom", retryable=False)
    return str(task.id)


class TestRetryDispatch:
    @pytest.mark.asyncio
    async def test_index_document_retry_uses_index_handler(
        self, client: AsyncClient, db: Database
    ) -> None:
        # Регресс: ручной retry всегда шёл в LLM-хендлер, поэтому задача
        # index_document перезапускалась не тем обработчиком.
        content = "retryable indexable content"
        async with db.session() as session:
            document, _ = await DocumentService(session).create_document(
                title="t", content=content
            )
            document_id = str(document.id)

        task_id = await _failed_task(
            db,
            task_type="index_document",
            payload={"document_id": document_id, "title": "t", "content": content},
        )

        resp = await client.post(f"/v1/tasks/{task_id}/retry")
        assert resp.status_code == 200
        assert resp.json()["status"] == TaskStatus.SUCCEEDED.value

        document_status = (await client.get(f"/v1/documents/{document_id}")).json()["status"]
        assert document_status == DocumentStatus.INDEXED.value

    @pytest.mark.asyncio
    async def test_unknown_task_type_retry_is_conflict(
        self, client: AsyncClient, db: Database
    ) -> None:
        task_id = await _failed_task(db, task_type="mystery", payload={})
        resp = await client.post(f"/v1/tasks/{task_id}/retry")
        assert resp.status_code == 409
