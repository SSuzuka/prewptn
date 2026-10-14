"""Тесты схемы и жизненного цикла задачи."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import text

from aicore.db.session import Database
from aicore.tasks.service import TaskService, latency_ms


async def table_names(db: Database) -> set[str]:
    """Имена таблиц в БД (локальный helper, чтобы не импортировать tests как пакет)."""
    async with db.session() as session:
        rows = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        return {row[0] for row in rows}


class TestSchema:
    @pytest.mark.asyncio
    async def test_all_tables_created(self, db: Database) -> None:
        assert {"task_runs", "documents", "idempotency_keys"} <= await table_names(db)

    @pytest.mark.asyncio
    async def test_all_datetime_columns_are_timezone_aware(self) -> None:
        """Регресс: все DateTime-колонки должны быть tz=True.

        `utcnow()` возвращает aware-datetime. Колонка без TZ на Postgres это
        TIMESTAMP WITHOUT TIME ZONE, куда asyncpg aware-объект не положит
        корректно. На SQLite расхождение маскируется, поэтому ловится только
        проверкой метаданных.

        Регресс: queued_at/started_at/finished_at/indexed_at были tz=False,
        а created_at/updated_at — tz=True.
        """
        from aicore.db.models import Document, IdempotencyKey, TaskRun

        for model in (TaskRun, Document, IdempotencyKey):
            for column in model.__table__.columns:
                if column.name.endswith("_at"):
                    tz = getattr(column.type, "timezone", None)
                    assert tz is True, f"{model.__tablename__}.{column.name} tz={tz}, ожидался True"

    @pytest.mark.asyncio
    async def test_healthcheck_true(self, db: Database) -> None:
        assert await db.healthcheck() is True

    @pytest.mark.asyncio
    async def test_session_requires_connect(self) -> None:
        with pytest.raises(RuntimeError, match="not initialised"):
            async with Database().session():
                pass

    @pytest.mark.asyncio
    async def test_session_rolls_back_on_error(self, db: Database) -> None:
        from aicore.db.models import TaskRun

        with pytest.raises(ValueError):
            async with db.session() as session:
                session.add(TaskRun(task_type="x", payload={}))
                raise ValueError("boom")

        async with db.session() as session:
            assert await TaskService(session).list_tasks() == []


class TestTaskLifecycle:
    @pytest.mark.asyncio
    async def test_create_defaults(self, db: Database) -> None:
        async with db.session() as session:
            task, created = await TaskService(session).create_task(
                task_type="ping", payload={"a": 1}
            )
        assert created is True
        assert task.attempts == 0
        assert task.max_attempts == 3
        assert task.queued_at is not None

    @pytest.mark.asyncio
    async def test_mark_running_increments_attempts(self, db: Database) -> None:
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(task_type="ping", payload={})
            await svc.mark_running(task)
            assert task.attempts == 1
            assert task.started_at is not None

    @pytest.mark.asyncio
    async def test_success_stores_metrics(self, db: Database) -> None:
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(task_type="ping", payload={})
            await svc.mark_running(task)
            await svc.mark_succeeded(
                task, result={"ok": True}, prompt_tokens=7, completion_tokens=3, cost_usd=0.01
            )
            assert task.status.value == "succeeded"
            assert task.prompt_tokens == 7
            assert task.cost_usd == 0.01

    @pytest.mark.asyncio
    async def test_get_task_accepts_string_id(self, db: Database) -> None:
        """Из брокера id приходит строкой — сервис должен это переварить."""
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(task_type="ping", payload={})
            await session.flush()

        async with db.session() as session:
            found = await TaskService(session).get_task(str(task.id))
        assert found is not None and found.id == task.id

    @pytest.mark.asyncio
    async def test_get_task_missing_returns_none(self, db: Database) -> None:
        from uuid import uuid4

        async with db.session() as session:
            assert await TaskService(session).get_task(uuid4()) is None

    @pytest.mark.asyncio
    async def test_list_filters_by_status(self, db: Database) -> None:
        from aicore.db.models import TaskStatus

        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(task_type="ping", payload={})
            await svc.create_task(task_type="ping", payload={})
            await svc.mark_running(task)

            dead = await svc.list_tasks(status=TaskStatus.DEAD_LETTER)
            running = await svc.list_tasks(status=TaskStatus.RUNNING)
        assert dead == []
        assert len(running) == 1

    def test_latency_ms(self) -> None:
        from aicore.db.models import TaskRun

        task = TaskRun(
            task_type="t",
            payload={},
            started_at=datetime(2026, 1, 1, 12, 0, 0),
            finished_at=datetime(2026, 1, 1, 12, 0, 2),
        )
        assert latency_ms(task) == 2000

    def test_latency_ms_none_when_incomplete(self) -> None:
        from aicore.db.models import TaskRun

        assert latency_ms(TaskRun(task_type="t", payload={})) is None
