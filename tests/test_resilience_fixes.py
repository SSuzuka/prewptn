"""Регресс-тесты на исправленные дыры из ревью.

Покрывают:
  * enqueue-failure больше не оставляет задачу в PENDING навсегда (иначе
    /retry отдавал 409, а повтор POST возвращал ту же зависшую задачу);
  * идемпотентность сверяет тело запроса (тот же ключ + другой вопрос → 409);
  * correlation_id выставляется и возвращается в заголовке;
  * CORS preflight проходит для настроенного origin'а;
  * GET-эндпоинты под rate limit (см. test_rate_limit.py).
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from aicore.api.app import create_app
from aicore.api.deps import reset_rate_limiter
from aicore.config import Settings, reset_settings_cache
from aicore.db.models import TaskStatus
from aicore.db.session import Database
from aicore.llm import reset_llm_client, set_llm_client
from aicore.llm.stub import StubLLM
from aicore.tasks.service import TaskService
from aicore.telemetry import reset_telemetry


@pytest_asyncio.fixture
async def client(db: Database, use_db: Database):
    reset_settings_cache()
    reset_rate_limiter()
    reset_llm_client()
    reset_telemetry()
    set_llm_client(StubLLM(settings=Settings(), model="stub-fix"))

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    reset_settings_cache()
    reset_rate_limiter()
    reset_llm_client()
    reset_telemetry()


class TestEnqueueFailure:
    @pytest.mark.asyncio
    async def test_queue_down_marks_task_failed_and_retryable(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom() -> None:
            raise ConnectionError("redis is down")

        monkeypatch.setattr("aicore.tasks.llm.ensure_broker_started", boom)

        resp = await client.post("/v1/tasks", json={"question": "hi"})
        assert resp.status_code == 202
        body = resp.json()
        # Раньше здесь был "pending" навсегда; теперь честный failed.
        assert body["status"] == TaskStatus.FAILED.value
        assert "ConnectionError" in (body["error"] or "")

        # FAILED перезапускаем — ручной retry не должен отдавать 409.
        retry = await client.post(f"/v1/tasks/{body['id']}/retry")
        assert retry.status_code == 200

    @pytest.mark.asyncio
    async def test_does_not_overwrite_claimed_task(self, db: Database) -> None:
        """mark_enqueue_failed не трогает задачу, которую уже взял воркер."""
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(task_type="ping", payload={})
            await svc.mark_running(task)

        async with db.session() as session:
            svc = TaskService(session)
            task = await svc.get_task(task.id)
            assert task is not None
            await svc.mark_enqueue_failed(task, "queue down")
            assert task.status == TaskStatus.RUNNING


class TestIdempotencyBody:
    @pytest.mark.asyncio
    async def test_same_key_different_body_conflicts(self, client: AsyncClient) -> None:
        first = await client.post(
            "/v1/tasks", json={"question": "q1"}, headers={"Idempotency-Key": "k"}
        )
        assert first.status_code == 202

        different = await client.post(
            "/v1/tasks", json={"question": "q2"}, headers={"Idempotency-Key": "k"}
        )
        assert different.status_code == 409

    @pytest.mark.asyncio
    async def test_same_key_same_body_replays(self, client: AsyncClient) -> None:
        first = await client.post(
            "/v1/tasks", json={"question": "q"}, headers={"Idempotency-Key": "k2"}
        )
        second = await client.post(
            "/v1/tasks", json={"question": "q"}, headers={"Idempotency-Key": "k2"}
        )
        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["id"] == second.json()["id"]


class TestCorrelationId:
    @pytest.mark.asyncio
    async def test_response_has_generated_correlation_id(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.headers.get("x-correlation-id")

    @pytest.mark.asyncio
    async def test_incoming_correlation_id_is_echoed(self, client: AsyncClient) -> None:
        resp = await client.get("/health", headers={"X-Correlation-ID": "abc123"})
        assert resp.headers.get("x-correlation-id") == "abc123"


class TestCors:
    @pytest.mark.asyncio
    async def test_preflight_allows_configured_origin(self, client: AsyncClient) -> None:
        resp = await client.options(
            "/v1/tasks",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "idempotency-key",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


@pytest_asyncio.fixture
async def secured_client(db: Database, use_db: Database, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_KEYS", "secret-key")
    reset_settings_cache()
    reset_rate_limiter()
    reset_llm_client()
    reset_telemetry()
    set_llm_client(StubLLM(settings=Settings(), model="stub-auth"))

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    reset_settings_cache()
    reset_rate_limiter()
    reset_llm_client()
    reset_telemetry()


class TestApiKeyAuth:
    @pytest.mark.asyncio
    async def test_missing_key_is_unauthorized(self, secured_client: AsyncClient) -> None:
        resp = await secured_client.get("/v1/tasks/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_is_unauthorized(self, secured_client: AsyncClient) -> None:
        resp = await secured_client.get(
            "/v1/tasks/00000000-0000-0000-0000-000000000000",
            headers={"X-API-Key": "nope"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_metrics_is_protected(self, secured_client: AsyncClient) -> None:
        assert (await secured_client.get("/metrics")).status_code == 401

    @pytest.mark.asyncio
    async def test_valid_key_passes(self, secured_client: AsyncClient) -> None:
        resp = await secured_client.get(
            "/v1/tasks/00000000-0000-0000-0000-000000000000",
            headers={"X-API-Key": "secret-key"},
        )
        # Прошла аутентификация — дальше уже 404 по несуществующей задаче.
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_health_stays_open(self, secured_client: AsyncClient) -> None:
        assert (await secured_client.get("/health")).status_code == 200
