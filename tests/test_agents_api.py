"""API-тесты агентов: запуск (идемпотентно), retry, валидация. Офлайн."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient

from aicore.agents import reset_agent
from aicore.api.app import create_app
from aicore.api.deps import reset_rate_limiter
from aicore.config import Settings
from aicore.db.models import TaskStatus
from aicore.db.session import Database
from aicore.rag import chunk_text, reset_retriever, set_retriever
from aicore.rag.retriever import Retriever
from aicore.rag.store import VectorStore
from aicore.rag.stub import StubEmbeddings
from aicore.tasks.service import TaskService
from aicore.telemetry import reset_telemetry

DOC = "The redis stream broker supports acknowledge and redelivery if a worker dies."


@pytest_asyncio.fixture
async def client(db: Database, use_db: Database):
    reset_rate_limiter()
    reset_retriever()
    reset_agent()
    reset_telemetry()
    cfg = Settings(vector_backend="memory", embedding_backend="stub", embedding_dim=384)
    retriever = Retriever(
        embeddings=StubEmbeddings(settings=cfg),
        store=VectorStore(client=AsyncQdrantClient(":memory:"), collection="api", dimension=384),
        settings=cfg,
    )
    set_retriever(retriever)
    await retriever.index(
        document_id="broker",
        chunks=chunk_text(DOC, chunk_size=200, chunk_overlap=20),
        title="broker",
    )

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    await retriever.aclose()
    reset_rate_limiter()
    reset_retriever()
    reset_agent()
    reset_telemetry()


class TestRunAgentApi:
    @pytest.mark.asyncio
    async def test_run_agent_indexes_inline(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/v1/agents", json={"question": "how does the broker handle redelivery"}
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["task_type"] == "run_agent"
        assert body["status"] == TaskStatus.SUCCEEDED.value
        assert "redelivery" in body["result"]["answer"].lower()

    @pytest.mark.asyncio
    async def test_idempotency_key_replays(self, client: AsyncClient) -> None:
        headers = {"Idempotency-Key": "agent-key-1"}
        body = {"question": "redis redelivery worker"}
        first = await client.post("/v1/agents", json=body, headers=headers)
        second = await client.post("/v1/agents", json=body, headers=headers)
        assert first.json()["id"] == second.json()["id"]

    @pytest.mark.asyncio
    async def test_idempotency_key_does_not_collide_across_endpoints(
        self, client: AsyncClient
    ) -> None:
        """Один ключ на /v1/tasks и /v1/agents — разные скоупы, разные задачи.

        Регресс: оба роута писали ключ в скоуп "default", поэтому повторный
        запрос на другой эндпоинт возвращал задачу чужого типа.
        """
        headers = {"Idempotency-Key": "shared-across-endpoints"}
        agent = await client.post(
            "/v1/agents",
            json={"question": "how does the broker handle redelivery"},
            headers=headers,
        )
        llm = await client.post(
            "/v1/tasks", json={"question": "plain llm question"}, headers=headers
        )
        assert agent.status_code == 202
        assert llm.status_code == 202
        assert agent.json()["id"] != llm.json()["id"]
        assert agent.json()["task_type"] == "run_agent"
        assert llm.json()["task_type"] == "answer_question"

    @pytest.mark.asyncio
    async def test_empty_question_rejected(self, client: AsyncClient) -> None:
        resp = await client.post("/v1/agents", json={"question": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_retry_uses_agent_handler(self, client: AsyncClient, db: Database) -> None:
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(
                task_type="run_agent",
                payload={"question": "how does the broker handle redelivery"},
                max_attempts=1,
            )
            await svc.mark_running(task)
            await svc.mark_failed(task, "boom", retryable=False)
            task_id = str(task.id)

        resp = await client.post(f"/v1/tasks/{task_id}/retry")
        assert resp.status_code == 200
        assert resp.json()["status"] == TaskStatus.SUCCEEDED.value
