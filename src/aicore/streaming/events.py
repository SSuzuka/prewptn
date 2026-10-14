"""SSE-поток событий задачи.

MVP строится на **опросе строки задачи**: работает офлайн, не зависит от
брокера и одинаково ведёт себя с InMemoryBroker и RedisStreamBroker. В проде
этот генератор можно заменить на подписку к Redis Streams, не меняя API:
эндпоинт отдаёт `EventSourceResponse`, а источник событий — деталь реализации.

События:
  * `status` — при смене статуса (pending → running → succeeded/...);
  * `done`   — терминальный статус, поток закрывается;
  * `error`  — задача не найдена/некорректный id/истёк лимит времени.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from aicore.config import get_settings
from aicore.db.models import TaskRun
from aicore.db.session import Database, get_database
from aicore.logging import get_logger
from aicore.tasks.service import TaskService, latency_ms

logger = get_logger(__name__)


def task_snapshot(task: TaskRun) -> dict[str, Any]:
    """Публичное представление задачи для события/ответа API."""
    return {
        "id": str(task.id),
        "task_type": task.task_type,
        "status": task.status.value,
        "attempts": task.attempts,
        "max_attempts": task.max_attempts,
        "result": task.result,
        "error": task.error,
        "prompt_tokens": task.prompt_tokens,
        "completion_tokens": task.completion_tokens,
        "cost_usd": task.cost_usd,
        "latency_ms": latency_ms(task),
    }


async def stream_task_events(
    task_id: str,
    *,
    interval: float = 0.25,
    timeout: float | None = None,
    db: Database | None = None,
) -> AsyncIterator[dict[str, str]]:
    """Yield SSE events for `task_id` until it reaches a terminal status.

    Each iteration opens a short-lived session so the stream never holds a DB
    connection while waiting.
    """
    database = db or get_database()
    await database.ensure_ready()
    limit = timeout if timeout is not None else get_settings().task_timeout_seconds + 30
    deadline = time.monotonic() + limit
    last_status: str | None = None

    while True:
        try:
            async with database.session() as session:
                task = await TaskService(session).get_task(task_id)
        except ValueError:
            yield _event("error", {"error": "invalid task id", "task_id": task_id})
            return

        if task is None:
            yield _event("error", {"error": "task not found", "task_id": task_id})
            return

        status = task.status.value
        if status != last_status:
            last_status = status
            yield _event("status", task_snapshot(task))

        if task.status.is_terminal:
            yield _event("done", task_snapshot(task))
            return

        if time.monotonic() >= deadline:
            yield _event("error", {"error": "stream timeout", "task_id": task_id})
            return

        await asyncio.sleep(interval)


def _event(name: str, payload: dict[str, Any]) -> dict[str, str]:
    return {"event": name, "data": json.dumps(payload, default=str)}
