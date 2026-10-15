"""Shared helpers for task handlers.

`load_task` and `mark_task_failed` are the session-boundary primitives every
task module (`llm`, `rag_index`, `agent_run`) needs: load a task by id without
exploding on a malformed id, and mark a failure in its own short session so a
post-timeout write cannot mask the original error. Keeping one copy means a fix
to that error handling reaches all task types at once.
"""

from __future__ import annotations

from aicore.db.models import TaskRun
from aicore.db.session import get_database
from aicore.logging import get_logger
from aicore.tasks.service import TaskService

logger = get_logger(__name__)


async def load_task(svc: TaskService, task_id: str) -> TaskRun | None:
    """Load a task; None if the id is malformed or the task is missing."""
    try:
        task = await svc.get_task(task_id)
    except ValueError:
        logger.error("task.invalid_id", task_id=task_id)
        return None
    if task is None:
        logger.error("task.not_found", task_id=task_id)
    return task


async def mark_task_failed(task_id: str, error: str, *, retryable: bool) -> None:
    """Mark a failure in its own session (also after a timeout)."""
    db = get_database()
    try:
        async with db.session() as session:
            svc = TaskService(session)
            task = await load_task(svc, task_id)
            if task is not None:
                await svc.mark_failed(task, error, retryable=retryable)
    except Exception:  # noqa: BLE001 — не маскируем исходную ошибку
        logger.exception("task.mark_failed_failed", task_id=task_id)
