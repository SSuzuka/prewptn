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
_ENQUEUERS: dict[str, Callable[..., Awaitable[bool]]] | None = None


def _enqueuers() -> dict[str, Callable[..., Awaitable[bool]]]:
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


async def _enqueue_task(task_id: str, task_type: str, *, mark_failed_on_error: bool = True) -> bool:
    """Re-enqueue one task; return False when no enqueuer exists or it failed.

    `enqueue_*` helpers intentionally swallow transport errors and (by default)
    mark the row FAILED themselves, so the caller cannot rely on exceptions.
    They return False on failure; a None return is treated as success so test
    doubles need not build a full result object.

    The PENDING safety net passes ``mark_failed_on_error=False``: a transient
    broker outage must leave the row PENDING so the next sweep retries it, not
    terminalize work that the sweeper exists to rescue.
    """
    enqueue = _enqueuers().get(task_type)
    if enqueue is None:
        logger.warning(
            "task.lease_no_enqueuer",
            task_id=task_id,
            task_type=task_type,
        )
        return False
    try:
        result = await enqueue(task_id, mark_failed_on_error=mark_failed_on_error)
    except Exception:  # noqa: BLE001 — best-effort, повтор на следующем цикле
        logger.exception("task.lease_reclaim_failed", task_id=task_id)
        return False
    return result is not False


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
        logger.warning("task.lease_reclaim", task_id=task_id, task_type=task_type)
        if await _enqueue_task(task_id, task_type):
            reclaimed += 1

    if reclaimed or dead:
        logger.info("task.lease_sweep", reclaimed=reclaimed, dead_lettered=dead)
    return reclaimed


async def reclaim_stale_pending(
    *,
    db: Database | None = None,
    older_than_seconds: float | None = None,
    limit: int = 100,
) -> int:
    """Safety net for PENDING rows that no worker/scheduler ever picks up.

    This is deliberately slower than the normal retry path
    (``task_pending_sweep_seconds`` > max backoff): it must not race a healthy
    `taskiq scheduler`, only rescue tasks stuck because that process is down or
    an enqueue was lost. After a successful re-enqueue we push ``queued_at``
    forward so the same row is not re-enqueued on every sweep.

    Rows older than ``task_pending_max_age_seconds`` are not resurrected (that
    is forgotten work, not a transient outage); they are moved to DEAD_LETTER so
    they reach a terminal status and become visible/retryable via ``/retry``.
    """
    database = db or get_database()
    await database.ensure_ready()
    cfg = get_settings()
    threshold = cfg.task_pending_sweep_seconds if older_than_seconds is None else older_than_seconds

    expired = 0
    async with database.session() as session:
        service = TaskService(session)
        for abandoned in await service.list_expired_pending(limit=limit):
            await service.mark_pending_expired(abandoned)
            expired += 1
        stale = await service.list_stale_pending(older_than_seconds=threshold, limit=limit)
        plan = [(str(task.id), task.task_type) for task in stale]

    reclaimed = 0
    for task_id, task_type in plan:
        logger.warning("task.pending_reclaim", task_id=task_id, task_type=task_type)
        if not await _enqueue_task(task_id, task_type, mark_failed_on_error=False):
            continue
        # If the task was claimed while enqueueing, this is a no-op; otherwise
        # it prevents a duplicate enqueue on the next sweep.
        async with database.session() as session:
            service = TaskService(session)
            task = await service.get_task(task_id)
            if task is not None:
                await service.mark_pending_requeued(task)
        reclaimed += 1

    if reclaimed or expired:
        logger.info("task.pending_sweep", reclaimed=reclaimed, dead_lettered=expired)
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
        try:
            await reclaim_stale_pending(db=db)
        except Exception:  # noqa: BLE001 — цикл не должен умирать от одной ошибки
            logger.exception("task.pending_sweep_failed")
