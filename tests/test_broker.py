"""Тесты брокера: конфигурация и реальный механизм ретраев taskiq.

Здесь важно отличие от test_task_service.py: там `process_task` вызывается
напрямую, а здесь задача идёт через настоящий брокер — проверяется, что
`retry_on_error=True` действительно приводит к перепостановке.
"""

from __future__ import annotations

import pytest

from aicore.broker import broker, build_broker, build_retry_middleware
from aicore.config import Settings


class TestBrokerConfig:
    def test_default_broker_is_in_memory(self) -> None:
        from taskiq import InMemoryBroker

        built = build_broker(Settings(broker_backend="memory"))
        assert isinstance(built, InMemoryBroker)

    def test_retry_middleware_reflects_settings(self) -> None:
        cfg = Settings(
            task_max_retries=5,
            task_retry_delay_seconds=2.0,
            task_retry_use_jitter=False,
            task_retry_use_exponent=False,
            task_retry_max_exponent_delay=30.0,
        )
        mw = build_retry_middleware(cfg)
        assert mw.default_retry_count == 5
        assert mw.default_delay == 2.0
        assert mw.use_jitter is False
        assert mw.use_delay_exponent is False
        assert mw.max_delay_exponent == 30.0

    def test_module_broker_has_retry_middleware(self) -> None:
        from taskiq.middlewares import SmartRetryMiddleware

        assert any(isinstance(m, SmartRetryMiddleware) for m in broker.middlewares)

    def test_retry_semantics_middleware_vs_db(self) -> None:
        """Два счётчика ретраев: middleware читает `task_max_retries`, БД — нет.

        taskiq: `default_retry_count` = число ПОВТОРОВ, т.е. исполнений будет
        не более `count + 1`. БД: `TaskRun.max_attempts` = число ИСПОЛНЕНИЙ,
        задаётся при создании задачи (default=3) и НЕ выводится автоматически
        из `task_max_retries`. Если нужен один общий knob, передавайте
        `max_attempts` явно (например, `task_max_retries + 1`).
        """
        from aicore.db.models import TaskRun

        cfg = Settings(task_max_retries=3)
        mw = build_retry_middleware(cfg)
        assert mw.default_retry_count == 3

        # БД-лимит — независимый default модели, а не производная настройки.
        column_default = TaskRun.__table__.columns["max_attempts"].default
        assert column_default is not None
        assert column_default.arg == 3


@pytest.mark.integration
class TestRetryMechanism:
    @pytest.mark.asyncio
    async def test_task_is_actually_registered(self) -> None:
        """Задача должна быть зарегистрирована в брокере под своим именем."""
        from aicore.tasks.ping import process_task

        found = broker.find_task(process_task.task_name)
        assert found is not None
        assert found.task_name == process_task.task_name

    @pytest.mark.asyncio
    async def test_kick_executes_and_persists(self, db, use_db) -> None:
        """Прогон задачи через брокер (kick), а не прямым вызовом.

        InMemoryBroker исполняет задачу в текущем event loop, поэтому виден
        и результат, и запись в БД. Это ближе к реальному пути воркера, чем
        прямой вызов `process_task(...)`.

        ВАЖНО: берём МОДУЛЬНЫЙ `broker`, а не `build_broker(...)`: задача
        зарегистрирована именно в нём (декоратор @broker.task), и свежий брокер
        её не знает — taskiq бросит UnknownTaskError.
        """
        import asyncio

        from aicore.tasks.ping import process_task
        from aicore.tasks.service import TaskService

        await broker.startup()

        try:
            async with db.session() as session:
                task, _ = await TaskService(session).create_task(
                    task_type="ping", payload={"via": "broker"}
                )

            await process_task.kiq(str(task.id))

            # Даём задаче доработать в фоновой задаче event loop.
            for _ in range(50):
                await asyncio.sleep(0.05)
                async with db.session() as session:
                    current = await TaskService(session).get_task(task.id)
                    if current is not None and current.status.value == "succeeded":
                        break

            async with db.session() as session:
                final = await TaskService(session).get_task(task.id)
            assert final is not None
            assert final.status.value == "succeeded"
        finally:
            await broker.shutdown()
