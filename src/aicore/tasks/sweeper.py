"""Lease-based reclaim of stuck RUNNING tasks.

When a worker is killed mid-task its message is redelivered, but the DB row is
still ``RUNNING``. The atomic claim in ``TaskService.mark_running`` rejects the
redelivery while the lease is fresh (correct — the work may still be in flight),
and once the lease expires the sweeper re-enqueues the task so a live worker can
re-claim it. Without this loop the "redelivery on worker crash" guarantee that
``RedisStreamBroker`` provides would dead-end at a task stuck in ``RUNNING``
forever: ``/retry`` refuses to touch ``RUNNING`` and no message remains.

Tasks that have no attempts left are moved to ``DEAD_LETTER`` instead of being
re-queued, so a crash on the last attempt still reaches a terminal state.

The loop runs in the worker process (see ``aicore.worker``), where the queue is
reachable; the API only enqueues.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from aicore.config import get_settings
from aicore.db.session import Database, get_database
from aicore.logging import get_logger
from aicore.tasks.service import TaskService

logger = get_logger(__name__)

# Диспетчеризация перезахвата по типу задачи, как в ручном /retry: каждый тип
# перезапускается своим хендлером. Импорт ленивый (внутри функции), иначе
# получим цикл: модули задач импортируют broker, а broker тянет задачи.
_ENQUEUERS: dict[str, Callable[[str], Awaitable[None]]] | None = None


def _enqueuers() -> dict[str, Callable[[str], Awaitable[None]]]:
    global _ENQUEUERS
    if _ENQUEUERS is None:
        from aicore.tasks.agent_run import enqueue_run_agent
        from aicore.tasks.llm import enqueue_answer_question
        from aicore.tasks.rag_index import enqueue_index_document

        _ENQUEUERS = {
            "answer_question": enqueue_answer_question,
            "index_document": enqueue_index_document,
            "run_agent": enqueue_run_agent,
        }
    return _ENQUEUERS


async def reclaim_stale_tasks(
    *,
    db: Database | None = None,
    lease_seconds: float | None = None,
    limit: int = 100,
) -> int:
    """Re-enqueue RUNNING tasks whose lease expired; return how many were handled.

    The candidates are read and decided in one short session, then the DB
    transaction is closed *before* enqueuing: a broker round-trip must not hold a
    connection, and an enqueue failure must leave the row RUNNING so the next
    sweep retries it (``mark_task_enqueue_failed`` only touches PENDING rows).
    """
    database = db or get_database()
    await database.ensure_ready()
    cfg = get_settings()
    lease = cfg.task_lease_seconds if lease_seconds is None else lease_seconds

    reclaimed = 0
    dead = 0
    async with database.session() as session:
        service = TaskService(session)
        stale = await service.list_stale_running(lease_seconds=lease, limit=limit)
        # Материализуем id/type и сразу решаем судьбу каждой задачи, пока
        # сессия открыта: после выхода из блока объекты могут стать detached.
        plan: list[tuple[str, str, bool]] = [
            (str(task.id), task.task_type, task.can_retry) for task in stale
        ]
        for task in stale:
            if not task.can_retry:
                await service.mark_lease_expired(task)
                dead += 1

    for task_id, task_type, can_retry in plan:
        if not can_retry:
            continue
        enqueue = _enqueuers().get(task_type)
        if enqueue is None:
            logger.warning(
                "task.lease_no_enqueuer",
                task_id=task_id,
                task_type=task_type,
            )
            continue
        logger.warning("task.lease_reclaim", task_id=task_id, task_type=task_type)
        # Не глотаем ошибку enqueue: задача остаётся RUNNING, следующий sweep
        # повторит попытку. Иначе потеряли бы её в PENDING/FAILED-ветке.
        try:
            await enqueue(task_id)
        except Exception:  # noqa: BLE001 — best-effort, повтор на следующем цикле
            logger.exception("task.lease_reclaim_failed", task_id=task_id)
            continue
        reclaimed += 1

    if reclaimed or dead:
        logger.info("task.lease_sweep", reclaimed=reclaimed, dead_lettered=dead)
    return reclaimed


async def lease_sweeper_loop(
    stop: asyncio.Event,
    *,
    db: Database | None = None,
    interval: float | None = None,
) -> None:
    """Periodically call :func:`reclaim_stale_tasks` until ``stop`` is set."""
    cfg = get_settings()
    period = cfg.task_lease_sweep_interval_seconds if interval is None else interval
    while not stop.is_set():
        # Ждём перед первым проходом: не бьём в БД сразу на старте воркера, и
        # даём тестам шанс остановить цикл, не запустив ни одного sweep'а.
        try:
            await asyncio.wait_for(stop.wait(), timeout=period)
        except TimeoutError:
            pass
        if stop.is_set():
            break
        try:
            await reclaim_stale_tasks(db=db)
        except Exception:  # noqa: BLE001 — цикл не должен умирать от одной ошибки
            logger.exception("task.lease_sweep_failed")
