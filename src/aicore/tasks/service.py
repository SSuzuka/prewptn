"""Управление жизненным циклом задачи в БД.

Сервис знает, как переводить `TaskRun` между статусами и что делать, когда
попытки исчерпаны. Вынесено отдельно от задач, чтобы один и тот же код
использовался и в воркере, и в API, и в тестах.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, and_, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from aicore.config import get_settings
from aicore.db.models import IdempotencyKey, TaskRun, TaskStatus
from aicore.logging import get_logger
from aicore.metrics import record_task_latency, record_task_status

logger = get_logger(__name__)


class IdempotencyConflictError(ValueError):
    """Тот же Idempotency-Key использован с другим телом запроса.

    По семантике идемпотентности (Stripe и др.) это конфликт: ключ уже связан
    с конкретным запросом, и молча отдать ответ на старое тело нельзя — клиент
    получил бы результат не на свой вопрос.
    """


def utcnow() -> datetime:
    """Timezone-aware текущее время."""
    return datetime.now(UTC)


def fingerprint_payload(payload: dict[str, Any]) -> str:
    """Канонический SHA-256 тела запроса, независимый от порядка ключей."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
        request_fingerprint: str | None = None,
    ) -> tuple[TaskRun, bool]:
        """Создать задачу. Возвращает `(task, created)`.

        Если передан `idempotency_key` и задача с таким ключом уже есть,
        возвращается существующая задача и `created=False` — дубль не создаётся
        и токены не жгутся повторно.

        Повтор с тем же ключом, но ДРУГИМ телом запроса — не идемпотентный
        повтор, а конфликт: поднимается `IdempotencyConflictError` (API отдаёт
        409). Иначе клиент молча получил бы ответ на старый вопрос.

        Гонки решаются unique-constraint'ом: при параллельном создании одного
        ключа второй INSERT падает с IntegrityError, и мы перечитываем задачу.
        """
        if idempotency_key:
            if request_fingerprint is None:
                request_fingerprint = fingerprint_payload(payload)
            existing_key = await self._get_idempotency_key(idempotency_key, idempotency_scope)
            if existing_key is not None:
                self._verify_fingerprint(existing_key, request_fingerprint)
                existing = (
                    await self.get_task(existing_key.task_id)
                    if existing_key.task_id is not None
                    else None
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

        try:
            # Savepoint (not a full rollback): a lost race must not discard other
            # work the caller has already staged in this session. The pending
            # INSERTs live inside the savepoint, so a rollback expunges them.
            async with self.session.begin_nested():
                self.session.add(task)

                if idempotency_key:
                    # ВАЖНО: не передаём task_id вручную. На этот момент task.id ещё
                    # не сгенерирован (default=uuid4 срабатывает при flush), и ключ
                    # ушёл бы в БД с task_id=NULL — тогда ON DELETE CASCADE не на что
                    # вешать. Связь через relationship проставляет FK корректно.
                    self.session.add(
                        IdempotencyKey(
                            key=idempotency_key,
                            scope=idempotency_scope,
                            task=task,
                            request_fingerprint=request_fingerprint,
                        )
                    )
                await self.session.flush()
        except IntegrityError:
            # Параллельный запрос успел раньше — отдаём его задачу, но только
            # если тело совпадает: иначе это конфликт ключа, а не повтор.
            if idempotency_key is None:
                # IntegrityError без ключа идемпотентности — не наш race
                # (например, другой unique-constraint): не глотаем его.
                raise
            existing_key = await self._get_idempotency_key(idempotency_key, idempotency_scope)
            if existing_key is None:
                raise
            self._verify_fingerprint(existing_key, request_fingerprint)
            existing = (
                await self.get_task(existing_key.task_id)
                if existing_key.task_id is not None
                else None
            )
            if existing is None:
                raise
            logger.info("task.idempotent_race_resolved", task_id=str(existing.id))
            return existing, False

        logger.info("task.created", task_id=str(task.id), task_type=task_type)
        return task, True

    async def _get_idempotency_key(self, key: str, scope: str) -> IdempotencyKey | None:
        """Найти строку ключа идемпотентности по паре (key, scope)."""
        result = await self.session.execute(
            select(IdempotencyKey).where(IdempotencyKey.key == key, IdempotencyKey.scope == scope)
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _verify_fingerprint(existing: IdempotencyKey, incoming: str | None) -> None:
        """409-конфликт, если тело запроса у повтора отличается.

        `request_fingerprint` может быть NULL у записей, созданных до появления
        колонки, — такие пропускаем, чтобы не ломать старые ключи.
        """
        if existing.request_fingerprint and incoming and existing.request_fingerprint != incoming:
            logger.warning(
                "task.idempotency_conflict",
                idempotency_key=existing.key,
                scope=existing.scope,
            )
            raise IdempotencyConflictError(
                "Idempotency-Key already used with a different request body"
            )

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
        """Найти задачу по ключу идемпотентности В РАМКАХ СКОУПА.

        Читаем через таблицу `idempotency_keys` по паре (key, scope) — это
        источник правды для идемпотентности. Фильтр по `TaskRun.idempotency_key`
        был бы неверным: две разные задачи в разных скоупах могут носить один
        и тот же ключ.

        Регресс, который тут ловился: раньше `scope` принимался, но не
        использовался в запросе — ключ одного тенанта отдавал чужую задачу.
        """
        result = await self.session.execute(
            select(TaskRun)
            .join(IdempotencyKey, IdempotencyKey.task_id == TaskRun.id)
            .where(IdempotencyKey.key == key, IdempotencyKey.scope == scope)
        )
        return result.scalar_one_or_none()

    async def list_tasks(
        self, *, status: TaskStatus | None = None, limit: int = 50, offset: int = 0
    ) -> list[TaskRun]:
        stmt = (
            select(TaskRun)
            # `id` as a tiebreaker: `created_at` can collide (same second on
            # SQLite), and without a total order offset pagination can drop or
            # duplicate rows.
            .order_by(TaskRun.created_at.desc(), TaskRun.id)
            .limit(limit)
            .offset(offset)
        )
        if status is not None:
            stmt = stmt.where(TaskRun.status == status)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_dead_letters(self, *, limit: int = 50) -> list[TaskRun]:
        """Задачи, исчерпавшие попытки и ждущие ручного вмешательства."""
        return await self.list_tasks(status=TaskStatus.DEAD_LETTER, limit=limit)

    # ------------------------------------------------------------ переходы
    async def mark_running(self, task: TaskRun, *, lease_seconds: float | None = None) -> bool:
        """Atomically claim a PENDING task (or a RUNNING one with an expired lease).

        The transition is a single conditional ``UPDATE`` so that under
        at-least-once delivery (taskiq redelivery, several workers) exactly one
        caller wins and the rest see ``rowcount == 0``. Without this, a
        redelivered message could re-run — and overwrite — a task that already
        reached a terminal status, double-charging the LLM/embeddings.

        A RUNNING task is re-claimable only once its *lease* expired, i.e. its
        ``started_at`` is older than ``now - lease_seconds``. This is what keeps
        the two failure modes apart:

        * a worker crashed mid-task and the message was redelivered — the lease
          is stale, so the new worker legitimately takes over (previously such a
          task stayed RUNNING forever: the claim was rejected, the message was
          acked, and neither ``/retry`` nor redelivery could move it);
        * the task is simply still being processed elsewhere — the lease is
          fresh, so the claim is rejected as before (no double payment).

        Returns ``True`` when this caller claimed the task, ``False`` when it was
        neither ``PENDING`` nor a stale ``RUNNING`` (already running/terminal), in
        which case the caller must not execute the work.
        """
        cfg = get_settings()
        lease = cfg.task_lease_seconds if lease_seconds is None else lease_seconds
        cutoff = utcnow() - timedelta(seconds=lease)
        claimed_statuses = or_(
            TaskRun.status == TaskStatus.PENDING,
            and_(
                TaskRun.status == TaskStatus.RUNNING,
                # started_at может быть NULL у криво созданной RUNNING-строки —
                # считаем такую задачу просроченной, а не «вечно живой».
                or_(TaskRun.started_at.is_(None), TaskRun.started_at < cutoff),
            ),
        )
        result = await self.session.execute(
            update(TaskRun)
            .where(
                TaskRun.id == task.id,
                claimed_statuses,
                # Не захватываем сверх лимита попыток: иначе упавший на
                # последней попытке воркер выжигал бы дополнительные прогоны.
                TaskRun.attempts < TaskRun.max_attempts,
            )
            .values(
                status=TaskStatus.RUNNING,
                attempts=TaskRun.attempts + 1,
                started_at=utcnow(),
            )
            # The value for `attempts` is a SQL expression, so let the DB apply it
            # and re-read the row instead of syncing the identity map in Python.
            .execution_options(synchronize_session=False)
        )
        if cast(CursorResult[Any], result).rowcount != 1:
            logger.warning("task.claim_rejected", task_id=str(task.id))
            return False

        await self.session.refresh(task)
        record_task_status(task.task_type, task.status.value)
        logger.info(
            "task.running",
            task_id=str(task.id),
            attempt=task.attempts,
            max_attempts=task.max_attempts,
        )
        return True

    async def list_stale_running(
        self, *, lease_seconds: float | None = None, limit: int = 100
    ) -> list[TaskRun]:
        """RUNNING-задачи с истёкшим lease — кандидаты на перезахват.

        Их находит фоновый purge-цикл воркера (см. `tasks/sweeper.py`): без него
        задача, чей воркер умер уже после claim, осталась бы RUNNING навсегда.
        """
        cfg = get_settings()
        lease = cfg.task_lease_seconds if lease_seconds is None else lease_seconds
        cutoff = utcnow() - timedelta(seconds=lease)
        result = await self.session.execute(
            select(TaskRun)
            .where(
                TaskRun.status == TaskStatus.RUNNING,
                or_(TaskRun.started_at.is_(None), TaskRun.started_at < cutoff),
            )
            .order_by(TaskRun.started_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_stale_pending(
        self, *, older_than_seconds: float | None = None, limit: int = 100
    ) -> list[TaskRun]:
        """PENDING-задачи, которые слишком долго не забрал ни один воркер.

        Штатно PENDING живёт миллисекунды (создание → очередь → claim) либо
        ждёт отложенный ретрай не дольше `task_retry_max_exponent_delay`.
        Если процесс `taskiq scheduler` не запущен, задача остаётся PENDING
        навсегда; этот список — safety net для такого случая.
        """
        cfg = get_settings()
        threshold = (
            cfg.task_pending_sweep_seconds if older_than_seconds is None else older_than_seconds
        )
        cutoff = utcnow() - timedelta(seconds=threshold)
        oldest = utcnow() - timedelta(seconds=cfg.task_pending_max_age_seconds)
        result = await self.session.execute(
            select(TaskRun)
            .where(
                TaskRun.status == TaskStatus.PENDING,
                TaskRun.queued_at.is_not(None),
                TaskRun.queued_at < cutoff,
                # Не воскрешаем задачи, брошенные недели назад: они уже не
                # «застряли из-за минутного сбоя», а забыты.
                TaskRun.queued_at >= oldest,
            )
            .order_by(TaskRun.queued_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_expired_pending(
        self, *, max_age_seconds: float | None = None, limit: int = 100
    ) -> list[TaskRun]:
        """PENDING-задачи, которые заброшены дольше `task_pending_max_age_seconds`.

        Их бессмысленно переставлять в очередь (см. `list_stale_pending`), но
        оставлять навсегда в PENDING тоже нельзя: `/retry` принимает только
        FAILED/DEAD_LETTER, поэтому такую строку нельзя было ни увидеть в
        списке dead-letter, ни перезапустить вручную. Sweeper переводит их в
        DEAD_LETTER.
        """
        cfg = get_settings()
        max_age = cfg.task_pending_max_age_seconds if max_age_seconds is None else max_age_seconds
        cutoff = utcnow() - timedelta(seconds=max_age)
        result = await self.session.execute(
            select(TaskRun)
            .where(
                TaskRun.status == TaskStatus.PENDING,
                TaskRun.queued_at.is_not(None),
                TaskRun.queued_at < cutoff,
            )
            .order_by(TaskRun.queued_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_pending_requeued(self, task: TaskRun) -> bool:
        """Сдвинуть `queued_at`, чтобы pending-sweeper не спамил повторами.

        Условный UPDATE: если задачу уже забрал воркер (статус сменился), не
        трогаем её. Возвращает True, если строка всё ещё была PENDING.
        """
        result = await self.session.execute(
            update(TaskRun)
            .where(TaskRun.id == task.id, TaskRun.status == TaskStatus.PENDING)
            .values(queued_at=utcnow())
            .execution_options(synchronize_session=False)
        )
        await self.session.refresh(task)
        return cast(CursorResult[Any], result).rowcount == 1

    async def mark_lease_expired(self, task: TaskRun) -> TaskRun:
        """Увести просроченную RUNNING-задачу без попыток в DEAD_LETTER.

        Вызывается purge-циклом, когда перезахватить нельзя (`attempts >=
        max_attempts`): перезапуск уже невозможен, и задачу нужно явно завершить,
        а не оставлять в RUNNING до бесконечности.
        """
        result = await self.session.execute(
            update(TaskRun)
            .where(TaskRun.id == task.id, TaskRun.status == TaskStatus.RUNNING)
            .values(
                status=TaskStatus.DEAD_LETTER,
                error="lease expired with no attempts left",
                finished_at=utcnow(),
            )
            .execution_options(synchronize_session=False)
        )
        await self.session.refresh(task)
        if cast(CursorResult[Any], result).rowcount == 1:
            record_task_status(task.task_type, task.status.value)
            logger.error("task.lease_expired_dead_letter", task_id=str(task.id))
        return task

    async def mark_pending_expired(self, task: TaskRun) -> TaskRun:
        """Увести заброшенную PENDING-задачу в DEAD_LETTER.

        PENDING старше `task_pending_max_age_seconds` уже не переставляется в
        очередь (это не минутный сбой, а забытая работа). Раньше такая строка
        оставалась PENDING навсегда: без терминального статуса она не попадала
        в dead-letter и не перезапускалась через `/retry`. Условный UPDATE не
        перетирает статус, если воркер успел забрать задачу параллельно.
        """
        result = await self.session.execute(
            update(TaskRun)
            .where(TaskRun.id == task.id, TaskRun.status == TaskStatus.PENDING)
            .values(
                status=TaskStatus.DEAD_LETTER,
                error="pending too long without being claimed",
                finished_at=utcnow(),
            )
            .execution_options(synchronize_session=False)
        )
        await self.session.refresh(task)
        if cast(CursorResult[Any], result).rowcount == 1:
            record_task_status(task.task_type, task.status.value)
            logger.error("task.pending_expired_dead_letter", task_id=str(task.id))
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
        record_task_status(task.task_type, task.status.value)
        measured = latency_ms(task)
        if measured is not None:
            record_task_latency(task.task_type, measured)
        logger.info(
            "task.succeeded",
            task_id=str(task.id),
            latency_ms=latency_ms(task),
            cost_usd=cost_usd,
        )
        return task

    async def mark_failed(
        self,
        task: TaskRun,
        error: str,
        *,
        retryable: bool = True,
    ) -> TaskRun:
        """Отметить неудачу. Если попытки исчерпаны — увести в dead-letter.

        Повторную постановку в очередь делает taskiq; здесь только фиксируем
        факт неудачной попытки и решаем, пора ли в DLQ.

        `retryable=False` означает постоянную ошибку (например, 4xx от LLM):
        ретрай бессмысленен, поэтому задача идёт сразу в DEAD_LETTER, даже если
        попытки остались. Иначе без этого флага постоянная ошибка на первой
        попытке вернула бы задачу в PENDING навсегда.

        Регресс: раньше ветвление шло только по `task.can_retry`, и постоянная
        ошибка с остатком попыток зацикливалась в PENDING.
        """
        task.error = error
        task.finished_at = utcnow()

        if retryable and task.can_retry:
            task.status = TaskStatus.PENDING
            # queued_at — время последней постановки в очередь, а не только
            # создания. Pending-sweeper использует его, чтобы не переставить
            # задачу, которая ждёт штатный backoff, и чтобы после аварийного
            # re-enqueue не делать это на каждом цикле.
            task.queued_at = utcnow()
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
                retryable=retryable,
                error=error,
            )

        await self.session.flush()
        record_task_status(task.task_type, task.status.value)
        return task

    async def mark_cancelled(self, task: TaskRun) -> TaskRun:
        """Отменить задачу вручную."""
        task.status = TaskStatus.CANCELLED
        task.finished_at = utcnow()
        await self.session.flush()
        record_task_status(task.task_type, task.status.value)
        logger.info("task.cancelled", task_id=str(task.id))
        return task

    async def mark_enqueue_failed(self, task: TaskRun, error: str) -> TaskRun:
        """Задача не попала в очередь — перевести её из PENDING в FAILED.

        `enqueue_*` ловит ошибку брокера/транспорта (например, Redis недоступен).
        Раньше задача оставалась в PENDING навсегда: ручной /retry отдавал 409,
        потому что перезапускаемы только FAILED и DEAD_LETTER. Здесь переводим
        её в FAILED, но только если её ещё НИКТО не заклеймил — условный
        `UPDATE ... WHERE status='pending'`, чтобы не перетереть статус задачи,
        которую параллельно успел взять воркер.
        """
        result = await self.session.execute(
            update(TaskRun)
            .where(TaskRun.id == task.id, TaskRun.status == TaskStatus.PENDING)
            .values(status=TaskStatus.FAILED, error=error, finished_at=utcnow())
            .execution_options(synchronize_session=False)
        )
        await self.session.refresh(task)
        if cast(CursorResult[Any], result).rowcount == 1:
            record_task_status(task.task_type, task.status.value)
            logger.error("task.enqueue_failed", task_id=str(task.id), error=error)
        return task

    async def reset_for_retry(self, task: TaskRun) -> TaskRun:
        """Подготовить задачу к ручному перезапуску.

        Перезапускать имеет смысл только задачу, завершившуюся неудачей:
        FAILED или DEAD_LETTER. SUCCEEDED трогать нельзя (результат уже
        отдан клиенту), а RUNNING/PENDING перезапуск только сломает.
        """
        if task.status not in {TaskStatus.FAILED, TaskStatus.DEAD_LETTER}:
            raise ValueError(
                f"cannot reset task {task.id}: status {task.status.value} is not retryable"
            )

        task.status = TaskStatus.PENDING
        task.attempts = 0
        task.error = None
        task.started_at = None
        task.finished_at = None
        await self.session.flush()
        logger.info("task.reset_for_retry", task_id=str(task.id))
        return task


def latency_ms(task: TaskRun) -> int | None:
    """Латентность обработки в миллисекундах.

    Значения из БД могут прийти naive: SQLite не хранит таймзону, поэтому
    после round-trip `started_at`/`finished_at` теряют tzinfo. Считаем такие
    значения UTC, иначе вычитание aware и naive падает с TypeError.
    """
    if task.started_at is None or task.finished_at is None:
        return None
    started = task.started_at if task.started_at.tzinfo else task.started_at.replace(tzinfo=UTC)
    finished = task.finished_at if task.finished_at.tzinfo else task.finished_at.replace(tzinfo=UTC)
    return int((finished - started).total_seconds() * 1000)
