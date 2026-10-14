"""Тесты rate limiting: после лимита — 429 с Retry-After."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from aicore.api.app import create_app
from aicore.api.deps import reset_rate_limiter
from aicore.config import Settings, reset_settings_cache
from aicore.db.session import Database
from aicore.llm import reset_llm_client, set_llm_client
from aicore.llm.stub import StubLLM
from aicore.telemetry import reset_telemetry


@pytest_asyncio.fixture
async def client(db: Database, use_db: Database, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "2")
    reset_settings_cache()
    reset_rate_limiter()
    reset_llm_client()
    reset_telemetry()
    set_llm_client(StubLLM(settings=Settings(), model="stub-rl"))

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    reset_settings_cache()
    reset_rate_limiter()
    reset_llm_client()
    reset_telemetry()


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_exceeding_limit_returns_429(self, client: AsyncClient) -> None:
        first = await client.post("/v1/tasks", json={"question": "a"})
        second = await client.post("/v1/tasks", json={"question": "b"})
        third = await client.post("/v1/tasks", json={"question": "c"})

        assert first.status_code == 202
        assert second.status_code == 202
        assert third.status_code == 429
        assert third.headers.get("retry-after") == "60"

    @pytest.mark.asyncio
    async def test_read_endpoint_is_not_rate_limited(self, client: AsyncClient) -> None:
        for _ in range(5):
            resp = await client.get("/v1/tasks/00000000-0000-0000-0000-000000000000")
            assert resp.status_code == 404
