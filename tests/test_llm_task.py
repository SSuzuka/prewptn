"""Тесты LLM-задачи: метрики сохраняются, retry vs dead-letter."""

from __future__ import annotations

import pytest

from aicore.config import Settings
from aicore.db.models import TaskStatus
from aicore.db.session import Database
from aicore.llm import LLMPermanentError, LLMRequest, reset_llm_client, set_llm_client
from aicore.llm.stub import StubLLM
from aicore.tasks.llm import answer_question
from aicore.tasks.service import TaskService
from aicore.telemetry import reset_telemetry


class _PermanentClient:
    model = "boom"

    async def complete(self, request: LLMRequest) -> None:
        raise LLMPermanentError("400 bad request")

    async def aclose(self) -> None:
        return None


class _TransientClient:
    model = "flaky"

    async def complete(self, request: LLMRequest) -> None:
        raise RuntimeError("connection reset")

    async def aclose(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset():
    reset_llm_client()
    reset_telemetry()
    yield
    reset_llm_client()
    reset_telemetry()


async def _make_task(db: Database, **payload: object):
    async with db.session() as session:
        task, _ = await TaskService(session).create_task(
            task_type="answer_question", payload=dict(payload)
        )
    return task


async def _reload(db: Database, task_id):
    async with db.session() as session:
        return await TaskService(session).get_task(task_id)


class TestAnswerQuestion:
    @pytest.mark.asyncio
    async def test_success_stores_metrics(self, db: Database, use_db: Database) -> None:
        set_llm_client(StubLLM(settings=Settings(), model="stub-task"))
        task = await _make_task(db, question="what is two plus two", system="be brief")

        result = await answer_question(str(task.id))
        assert result["answer"].startswith("[stub-task]")

        reloaded = await _reload(db, task.id)
        assert reloaded is not None
        assert reloaded.status is TaskStatus.SUCCEEDED
        assert reloaded.prompt_tokens and reloaded.prompt_tokens > 0
        assert reloaded.completion_tokens and reloaded.completion_tokens > 0
        assert reloaded.cost_usd and reloaded.cost_usd > 0

    @pytest.mark.asyncio
    async def test_permanent_error_dead_letters(self, db: Database, use_db: Database) -> None:
        set_llm_client(_PermanentClient())
        task = await _make_task(db, question="x")

        result = await answer_question(str(task.id))
        assert result["status"] == "dead_letter"

        reloaded = await _reload(db, task.id)
        assert reloaded is not None
        assert reloaded.status is TaskStatus.DEAD_LETTER
        assert "LLMPermanentError" in (reloaded.error or "")

    @pytest.mark.asyncio
    async def test_transient_error_returns_to_pending(self, db: Database, use_db: Database) -> None:
        set_llm_client(_TransientClient())
        task = await _make_task(db, question="x")

        with pytest.raises(RuntimeError):
            await answer_question(str(task.id))

        reloaded = await _reload(db, task.id)
        assert reloaded is not None
        assert reloaded.status is TaskStatus.PENDING
        assert reloaded.can_retry is True

    @pytest.mark.asyncio
    async def test_missing_task_is_noop(self, db: Database, use_db: Database) -> None:
        set_llm_client(StubLLM(settings=Settings()))
        result = await answer_question("00000000-0000-0000-0000-000000000000")
        assert result["status"] == "failed"
