"""API-тесты задач: создание (идемпотентно), чтение, retry. Офлайн.

Без TestClient: httpx.ASGITransport в async-тесте, БД — in-memory фикстура,
LLM — StubLLM. Покрывает сквозной путь POST -> задача -> результат+метрики.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from aicore.api.app import create_app
from aicore.api.deps import reset_rate_limiter
from aicore.config import Settings
from aicore.db.models import TaskStatus
from aicore.db.session import Database
from aicore.llm import reset_llm_client, set_llm_client
from aicore.llm.stub import StubLLM
from aicore.tasks.service import TaskService
from aicore.telemetry import reset_telemetry


@pytest_asyncio.fixture
async def client(db: Database, use_db: Database):
    reset_rate_limiter()
    reset_llm_client()
    reset_telemetry()
    set_llm_client(StubLLM(settings=Settings(), model="stub-api"))

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    reset_rate_limiter()
    reset_llm_client()
    reset_telemetry()


class TestCreateTask:
    @pytest.mark.asyncio
    async def test_create_returns_202_with_answer(self, client: AsyncClient) -> None:
        resp = await client.post("/v1/tasks", json={"question": "hello there"})
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == TaskStatus.SUCCEEDED.value
        assert body["result"]["answer"].startswith("[stub-api]")
        assert body["prompt_tokens"] > 0
        assert body["completion_tokens"] > 0
        assert body["cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_idempotency_key_returns_same_task(self, client: AsyncClient) -> None:
        first = await client.post(
            "/v1/tasks", json={"question": "q"}, headers={"Idempotency-Key": "k1"}
        )
        second = await client.post(
            "/v1/tasks", json={"question": "q"}, headers={"Idempotency-Key": "k1"}
        )
        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["id"] == second.json()["id"]

    @pytest.mark.asyncio
    async def test_empty_question_rejected(self, client: AsyncClient) -> None:
        resp = await client.post("/v1/tasks", json={"question": ""})
        assert resp.status_code == 422


class TestReadTask:
    @pytest.mark.asyncio
    async def test_get_after_create(self, client: AsyncClient) -> None:
        created = await client.post("/v1/tasks", json={"question": "hi"})
        task_id = created.json()["id"]

        resp = await client.get(f"/v1/tasks/{task_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == TaskStatus.SUCCEEDED.value

    @pytest.mark.asyncio
    async def test_unknown_returns_404(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/tasks/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_422(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/tasks/not-a-uuid")
        assert resp.status_code == 422


class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_succeeded_is_conflict(self, client: AsyncClient) -> None:
        created = await client.post("/v1/tasks", json={"question": "hi"})
        task_id = created.json()["id"]

        resp = await client.post(f"/v1/tasks/{task_id}/retry")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_retry_dead_letter_requeues(self, client: AsyncClient, db: Database) -> None:
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(
                task_type="answer_question", payload={"question": "x"}, max_attempts=1
            )
            await svc.mark_running(task)
            await svc.mark_failed(task, "boom", retryable=False)

        resp = await client.post(f"/v1/tasks/{task.id}/retry")
        assert resp.status_code == 200
        assert resp.json()["status"] == TaskStatus.SUCCEEDED.value
