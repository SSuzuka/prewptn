"""Тесты метрик: /metrics и инструментирование LLM/задач."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from prometheus_client import generate_latest

from aicore.api.app import create_app
from aicore.api.deps import reset_rate_limiter
from aicore.config import Settings
from aicore.db.session import Database
from aicore.llm import LLMRequest, reset_llm_client, set_llm_client
from aicore.llm.stub import StubLLM
from aicore.telemetry import reset_telemetry


def _exposition() -> str:
    return generate_latest().decode("utf-8")


@pytest_asyncio.fixture
async def client(db: Database, use_db: Database):
    reset_rate_limiter()
    reset_llm_client()
    reset_telemetry()
    set_llm_client(StubLLM(settings=Settings(), model="stub-metrics"))

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    reset_rate_limiter()
    reset_llm_client()
    reset_telemetry()


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_prometheus_text(self, client: AsyncClient) -> None:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert "aicore_llm_calls_total" in resp.text

    @pytest.mark.asyncio
    async def test_task_run_is_counted(self, client: AsyncClient) -> None:
        await client.post("/v1/tasks", json={"question": "metrics please"})
        body = (await client.get("/metrics")).text
        assert 'aicore_tasks_total{status="succeeded",task_type="answer_question"}' in body
        assert "aicore_task_latency_ms_count" in body


class TestLlmInstrumentation:
    @pytest.mark.asyncio
    async def test_llm_call_records_tokens_and_cost(self) -> None:
        reset_telemetry()
        client = StubLLM(settings=Settings(), model="metrics-model")
        await client.complete(LLMRequest(prompt="hello metrics"))
        reset_telemetry()

        body = _exposition()
        assert 'aicore_llm_calls_total{backend="stub",model="metrics-model"}' in body
        assert 'aicore_llm_tokens_total{kind="prompt"}' in body
        assert 'aicore_llm_cost_usd_total{model="metrics-model"}' in body


class TestEnqueueFailureMetric:
    def test_records_failure_per_task_type(self) -> None:
        from aicore.metrics import record_enqueue_failed

        record_enqueue_failed("answer_question")
        body = _exposition()
        assert 'aicore_enqueue_failures_total{task_type="answer_question"}' in body
