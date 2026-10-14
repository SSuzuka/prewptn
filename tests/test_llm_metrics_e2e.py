"""End-to-end: one LLM call persists tokens, latency and cost on the TaskRun."""

from __future__ import annotations

import pytest

from aicore.config import Settings
from aicore.db.models import TaskStatus
from aicore.db.session import Database
from aicore.llm import LLMRequest
from aicore.llm.stub import StubLLM
from aicore.tasks.service import TaskService
from aicore.telemetry import reset_telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry():
    reset_telemetry()
    yield
    reset_telemetry()


class TestMetricsAccounting:
    @pytest.mark.asyncio
    async def test_llm_metrics_persisted_on_success(self, db: Database) -> None:
        client = StubLLM(settings=Settings(), model="stub-e2e")
        response = await client.complete(LLMRequest(prompt="how are you today", system="be polite"))

        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(
                task_type="answer_question",
                payload={"question": "how are you today"},
            )
            await svc.mark_running(task)
            await svc.mark_succeeded(
                task,
                result={"answer": response.text},
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cost_usd=response.cost_usd,
            )

        async with db.session() as session:
            reloaded = await TaskService(session).get_task(task.id)

        assert reloaded is not None
        assert reloaded.status is TaskStatus.SUCCEEDED
        assert reloaded.prompt_tokens == response.usage.prompt_tokens
        assert reloaded.completion_tokens == response.usage.completion_tokens
        assert reloaded.cost_usd == pytest.approx(response.cost_usd)
        assert reloaded.cost_usd is not None and reloaded.cost_usd > 0
