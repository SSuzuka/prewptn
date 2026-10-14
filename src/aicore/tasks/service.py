"""Управление жизненным циклом задачи в БД.

Сервис знает, как переводить `TaskRun` между статусами и что делать, когда
попытки исчерпаны. Вынесено отдельно от задач, чтобы один и тот же код
использовался и в воркере, и в API, и в тестах.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from aicore.db.models import IdempotencyKey, TaskRun, TaskStatus
from aicore.logging import get_logger

logger = get_logger(__name__)


def utcnow() -> datetime:
    """Timezone-aware текущее время."""
    return datetime.now(UTC)


class TaskService:
    """Операции над задачами и ключами идемпотентности."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ------------------------------------------------------------ создание
    async def create_task(
        self,
        *,
        task_type: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
        idempotency_scope: str = "default",
        max_attempts: int = 3,
        broker_task_id: str | None = None,
    ) -> tuple[TaskRun, bool]:
        """Создать задачу. Возвращает `(task, created)`.

        Если передан `idempotency_key` и задача с таким ключом уже есть,
        возвращается существующая задача и `created=False` — дубль не создаётся
        и токены не жгутся повторно.

        Гонки решаются unique-constraint'ом: при параллельном создании одного
        ключа второй INSERT падает с IntegrityError, и мы перечитываем задачу.
        """
        if idempotency_key:
            existing = await self.get_task_by_idempotency_key(
                idempotency_key, scope=idempotency_scope
            )
            if existing is not None:
                logger.info(
                    "task.idempotent_replay",
                    task_id=str(existing.id),
                    idempotency_key=idempotency_key,
                )
                return existing, False

        task = TaskRun(
            task_type=task_type,
            payload=payload,
            status=TaskStatus.PENDING,
            attempts=0,
            max_attempts=max_attempts,
            idempotency_key=idempotency_key,
            broker_task_id=broker_task_id,
            queued_at=utcnow(),
        )
        self.session.add(task)

        if idempotency_key:
            # ВАЖНО: не передаём task_id вручную. На этот момент task.id ещё
            # не сгенерирован (default=uuid4 срабатывает при flush), и ключ
            # ушёл бы в БД с task_id=NULL — тогда ON DELETE CASCADE не на что
            # вешать. Связь через relationship проставляет FK корректно.
            self.session.add(
                IdempotencyKey(key=idempotency_key, scope=idempotency_scope, task=task)
            )

        try:
            await self.session.flush()
        except IntegrityError:
            # Параллельный запрос успел раньше — отдаём его задачу.
            await self.session.rollback()
            existing = await self.get_task_by_idempotency_key(
                idempotency_key, scope=idempotency_scope
            )
            if existing is None:
                raise
            logger.info("task.idempotent_race_resolved", task_id=str(existing.id))
            return existing, False

        logger.info("task.created", task_id=str(task.id), task_type=task_type)
        return task, True

    # ---------------------------------------------------------------- чтение
    async def get_task(self, task_id: UUID | str) -> TaskRun | None:
        """Найти задачу по id.

        Принимает и `UUID`, и строку: из брокера id приходит строкой, а UUID-колонка
        на SQLite требует объект UUID (иначе `'str' object has no attribute 'hex'`).
        """
        key = task_id if isinstance(task_id, UUID) else UUID(str(task_id))
        result = await self.session.execute(select(TaskRun).where(TaskRun.id == key))
        return result.scalar_one_or_none()

    async def get_task_by_idempotency_key(
        self, key: str, *, scope: str = "default"
    ) -> TaskRun | None:
        result = await self.session.execute(select(TaskRun).where(TaskRun.idempotency_key == key))
        return result.scalar_one_or_none()

    async def list_tasks(
        self, *, status: TaskStatus | None = None, limit: int = 50, offset: int = 0
    ) -> list[TaskRun]:
        stmt = select(TaskRun).order_by(TaskRun.created_at.desc()).limit(limit).offset(offset)
        if status is not None:
            stmt = stmt.where(TaskRun.status == status)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_dead_letters(self, *, limit: int = 50) -> list[TaskRun]:
        """Задачи, исчерпавшие попытки и ждущие ручного вмешательства."""
        return await self.list_tasks(status=TaskStatus.DEAD_LETTER, limit=limit)

    # ------------------------------------------------------------ переходы
    async def mark_running(self, task: TaskRun) -> TaskRun:
        """Отметить начало обработки и инкрементировать счётчик попыток."""
        task.status = TaskStatus.RUNNING
        task.attempts += 1
        task.started_at = utcnow()
        await self.session.flush()
        logger.info(
            "task.running",
            task_id=str(task.id),
            attempt=task.attempts,
            max_attempts=task.max_attempts,
        )
        return task

    async def mark_succeeded(
        self,
        task: TaskRun,
        *,
        result: dict[str, Any] | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> TaskRun:
        """Отметить успех и сохранить метрики LLM (этап 3 начнёт их считать)."""
        task.status = TaskStatus.SUCCEEDED
        task.result = result
        task.error = None
        task.finished_at = utcnow()
        task.prompt_tokens = prompt_tokens
        task.completion_tokens = completion_tokens
        task.cost_usd = cost_usd
        await self.session.flush()
        logger.info(
            "task.succeeded",
            task_id=str(task.id),
            latency_ms=latency_ms(task),
            cost_usd=cost_usd,
        )
        return task

    async def mark_failed(self, task: TaskRun, error: str) -> TaskRun:
        """Отметить неудачу. Если попытки исчерпаны — увести в dead-letter.

        Повторную постановку в очередь делает taskiq; здесь только фиксируем
        факт неудачной попытки и решаем, пора ли в DLQ.
        """
        task.error = error
        task.finished_at = utcnow()

        if task.can_retry:
            task.status = TaskStatus.PENDING
            logger.warning(
                "task.attempt_failed_will_retry",
                task_id=str(task.id),
                attempt=task.attempts,
                max_attempts=task.max_attempts,
            )
        else:
            task.status = TaskStatus.DEAD_LETTER
            logger.error(
                "task.dead_lettered",
                task_id=str(task.id),
                attempts=task.attempts,
                error=error,
            )

        await self.session.flush()
        return task

    async def mark_cancelled(self, task: TaskRun) -> TaskRun:
        """Отменить задачу вручную."""
        task.status = TaskStatus.CANCELLED
        task.finished_at = utcnow()
        await self.session.flush()
        logger.info("task.cancelled", task_id=str(task.id))
        return task

    async def reset_for_retry(self, task: TaskRun) -> TaskRun:
        """Подготовить dead-letter задачу к ручному перезапуску."""
        task.status = TaskStatus.PENDING
        task.attempts = 0
        task.error = None
        task.started_at = None
        task.finished_at = None
        await self.session.flush()
        logger.info("task.reset_for_retry", task_id=str(task.id))
        return task


def latency_ms(task: TaskRun) -> int | None:
    """Латентность обработки в миллисекундах."""
    if task.started_at is None or task.finished_at is None:
        return None
    delta = task.finished_at - task.started_at
    return int(delta.total_seconds() * 1000)
