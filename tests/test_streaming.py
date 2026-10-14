"""Тесты SSE-потока: генератор прогоняется по шагам (без гонок по времени)."""

from __future__ import annotations

import json

import pytest

from aicore.db.models import TaskStatus
from aicore.db.session import Database
from aicore.streaming.events import stream_task_events
from aicore.tasks.service import TaskService


async def _create(db: Database):
    async with db.session() as session:
        task, _ = await TaskService(session).create_task(
            task_type="answer_question", payload={"question": "hi"}
        )
    return task


class TestStreamTaskEvents:
    @pytest.mark.asyncio
    async def test_unknown_task_emits_error(self, db: Database) -> None:
        events = [
            event
            async for event in stream_task_events(
                "00000000-0000-0000-0000-000000000000", interval=0, db=db
            )
        ]
        assert events[-1]["event"] == "error"
        assert "not found" in events[-1]["data"]

    @pytest.mark.asyncio
    async def test_invalid_id_emits_error(self, db: Database) -> None:
        events = [event async for event in stream_task_events("nope", interval=0, db=db)]
        assert events[-1]["event"] == "error"

    @pytest.mark.asyncio
    async def test_progress_then_done(self, db: Database) -> None:
        task = await _create(db)
        stream = stream_task_events(str(task.id), interval=0, db=db)

        first = await anext(stream)
        assert first["event"] == "status"
        assert json.loads(first["data"])["status"] == TaskStatus.PENDING.value

        async with db.session() as session:
            svc = TaskService(session)
            current = await svc.get_task(task.id)
            await svc.mark_running(current)

        second = await anext(stream)
        assert second["event"] == "status"
        assert json.loads(second["data"])["status"] == TaskStatus.RUNNING.value

        async with db.session() as session:
            svc = TaskService(session)
            current = await svc.get_task(task.id)
            await svc.mark_succeeded(
                current,
                result={"answer": "42"},
                prompt_tokens=3,
                completion_tokens=1,
                cost_usd=0.0001,
            )

        third = await anext(stream)
        assert third["event"] == "status"
        assert json.loads(third["data"])["status"] == TaskStatus.SUCCEEDED.value

        done = await anext(stream)
        assert done["event"] == "done"
        assert json.loads(done["data"])["result"] == {"answer": "42"}
