"""Пример задачи с полным циклом: retries -> dead-letter.

Задача намеренно умеет падать по флагу из payload — это позволяет в тестах
и демо проверить механику ретраев и DLQ, не дожидаясь этапов 3-5.
"""

from __future__ import annotations

from typing import Any

from taskiq import TaskiqEvents

from aicore.broker import broker
from aicore.config import get_settings
from aicore.db.session import get_database
from aicore.logging import get_logger
from aicore.tasks.service import TaskService, utcnow

logger = get_logger(__name__)


class TransientTaskError(Exception):
    """Временная ошибка — taskiq переставит задачу в очередь."""


class PermanentTaskError(Exception):
    """Постоянная ошибка — смысла в ретрае нет."""


@broker.task(retry_on_error=True)
async def process_task(
    task_id: str,
    *,
    fail_times: int = 0,
    delay_seconds: float = 0.0,
) -> dict[str, Any]:
    """Обработать задачу из БД: running -> succeeded / retry / dead-letter.

    `fail_times` заставляет первые N попыток падать с TransientTaskError —
    так проверяется, что SmartRetryMiddleware действительно переставляет задачу.

    Реальный бизнес-логикой (LLM, RAG) задача наполнится на этапах 3-5; сейчас
    здесь проверенный каркас управления состоянием.
    """
    import asyncio

    # task_timeout_seconds из конфига: без этого одно зависшее задание
    # утащило бы воркер в бесконечное ожидание. Задержка тоже внутри timeout —
    # иначе «медленная» задача успела бы отработать до применения лимита.
    settings = get_settings()
    db = get_database()

    try:
        async with asyncio.timeout(settings.task_timeout_seconds):
            return await _run(task_id, fail_times=fail_times, delay_seconds=delay_seconds)
    except TimeoutError:
        logger.error("task.timeout", task_id=task_id, timeout=settings.task_timeout_seconds)
        async with db.session() as session:
            service = TaskService(session)
            try:
                task = await service.get_task(task_id)
            except ValueError:
                task = None
            if task is not None:
                await service.mark_failed(task, error="TimeoutError: task exceeded deadline")
                await session.commit()
        raise


async def _run(task_id: str, *, fail_times: int, delay_seconds: float = 0.0) -> dict[str, Any]:
    """Тело обработки задачи (вынесено, чтобы обернуть его в timeout)."""
    import asyncio

    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)

    db = get_database()

    async with db.session() as session:
        service = TaskService(session)

        try:
            task = await service.get_task(task_id)
        except ValueError:
            # Некорректный UUID — ретрай бессмысленен.
            logger.error("task.invalid_id", task_id=task_id)
            raise PermanentTaskError(f"invalid task id: {task_id!r}") from None

        if task is None:
            # Задача удалена или не существует — ретрай бессмысленен.
            logger.error("task.not_found", task_id=task_id)
            raise PermanentTaskError(f"task {task_id} not found")

        await service.mark_running(task)

        try:
            if task.attempts <= fail_times:
                raise TransientTaskError(f"simulated transient failure on attempt {task.attempts}")

            result = {
                "task_type": task.task_type,
                "echo": task.payload,
                "processed_at": utcnow().isoformat(),
                "attempts_used": task.attempts,
            }
            await service.mark_succeeded(task, result=result)
            # Успех фиксируем явно: ниже исключение уйдёт в taskiq.
            await session.commit()
            return result

        except Exception as exc:  # noqa: BLE001
            # Сначала СОХРАНЯЕМ новое состояние, и только потом бросаем.
            # Иначе `db.session()` сделает rollback и attempts/статус потеряются.
            transient = isinstance(exc, TransientTaskError)
            await service.mark_failed(task, error=f"{type(exc).__name__}: {exc}")
            await session.commit()

            if transient and task.can_retry:
                # Пусть taskiq переставит задачу (SmartRetryMiddleware).
                raise

            logger.error(
                "task.terminal_failure",
                task_id=task_id,
                status=task.status.value,
                attempts=task.attempts,
            )
            return {
                "status": task.status.value,
                "task_id": task_id,
                "error": f"{type(exc).__name__}: {exc}",
            }


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def _on_worker_startup(state: Any) -> None:  # noqa: ANN401
    """При старте воркера гарантируем, что схема БД существует."""
    db = get_database()
    await db.connect()
    await db.create_all()
    logger.info("worker.startup.complete")
