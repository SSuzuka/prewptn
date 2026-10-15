"""Брокер задач (taskiq) и его мидлвари.

Выбор бэкенда по `broker_backend`:
  * `memory` — InMemoryBroker. Только для dev и тестов: задачи исполняются
    в текущем процессе, без сети, без ack. При падении воркера задача
    **теряется**, поэтому в прод недопустим.
  * `redis`  — RedisStreamBroker (extra `redis`). Production-вариант:
    Redis Streams поддерживают acknowledge и переотдачу сообщения, если
    воркер умер.

Ретраи настраиваются через SmartRetryMiddleware: exponential backoff + jitter,
чтобы синхронные ретраи не били пиками по провайдеру.

ВАЖНО про отложенные ретраи (backoff): SmartRetryMiddleware переставляет задачу
через schedule_source с меткой `delay`. InMemoryBroker не умеет отложенную
доставку и игнорирует её, поэтому в dev-режиме backoff/jitter фактически
no-op, а ретрай происходит только когда taskiq повторно исполняет задачу
сразу (или когда её переставили вручную). Настоящий backoff работает на
RedisStreamBroker + RedisScheduleSource. Это задокументированное ограничение
dev-режима, а не баг.
"""

from __future__ import annotations

from taskiq import AsyncBroker, InMemoryBroker
from taskiq.middlewares import SmartRetryMiddleware

from aicore.config import Settings, get_settings
from aicore.logging import get_logger

logger = get_logger(__name__)


def build_retry_middleware(settings: Settings | None = None) -> SmartRetryMiddleware:
    """Собрать SmartRetryMiddleware из настроек.

    Сигнатура сверена с taskiq 0.13.0 (`default_retry_count`, `default_delay`,
    `use_jitter`, `use_delay_exponent`, `max_delay_exponent`).

    Примечание: `default_retry_count` — это число ПОВТОРОВ (taskiq semantics):
    задача исполнится не более `default_retry_count + 1` раз. В БД при этом
    свой счётчик `TaskRun.attempts/max_attempts`. Источник правды о лимите —
    БД (`max_attempts`); middleware-счётчик служит предохранителем на стороне
    брокера. Оба выставляются из одного поля `task_max_retries`, чтобы не
    расходиться, но семантика у них разная — см. тест test_retry_semantics.
    """
    cfg = settings or get_settings()
    return SmartRetryMiddleware(
        default_retry_count=cfg.task_max_retries,
        default_delay=cfg.task_retry_delay_seconds,
        use_jitter=cfg.task_retry_use_jitter,
        use_delay_exponent=cfg.task_retry_use_exponent,
        max_delay_exponent=cfg.task_retry_max_exponent_delay,
    )


def build_broker(settings: Settings | None = None) -> AsyncBroker:
    """Собрать брокер согласно настройкам.

    Для `memory` возвращается InMemoryBroker с await_inplace=True, чтобы в
    скриптах и тестах задача гарантированно дорабатывала до конца в рамках
    одного event loop.

    Для `redis` требуется extra `redis`; импорт делается лениво, чтобы базовый
    `uv sync` оставался лёгким и проект работал без Redis.

    Возвращаемый тип — AsyncBroker: в ветке redis это RedisStreamBroker,
    в ветке memory — InMemoryBroker. Аннотация была бы неверной как
    InMemoryBroker для обоих случаев.
    """
    cfg = settings or get_settings()

    if cfg.broker_backend == "redis":
        from taskiq_redis import RedisAsyncResultBackend, RedisStreamBroker

        broker: AsyncBroker = RedisStreamBroker(
            url=cfg.redis_url,
            queue_name=cfg.taskiq_queue,
        ).with_result_backend(
            # expire time обязателен: иначе результаты копятся в Redis вечно.
            RedisAsyncResultBackend(
                redis_url=cfg.redis_url,
                result_ex_time=3600,
            )
        )
        logger.info("broker.using_redis_streams", url=cfg.redis_url)
    else:
        broker = InMemoryBroker(await_inplace=True)
        logger.warning(
            "broker.using_in_memory",
            note="dev/test only: no ack, no network, task lost if worker dies",
        )

    # Мидлвари регистрируются ПОСЛЕ создания брокера.
    broker.with_middlewares(build_retry_middleware(cfg))
    return broker


broker = build_broker()

_broker_started = False


async def ensure_broker_started() -> None:
    """Start the broker exactly once per process.

    Single guard shared by every `enqueue_*` helper: separate per-module flags
    would call `broker.startup()` (and its worker-startup hooks) once per task
    type. In production the worker starts the broker itself; this is for the
    API/dev path where tasks are enqueued inline.
    """
    global _broker_started
    if not _broker_started:
        await broker.startup()
        _broker_started = True


# Регистрируем задачи на этом брокере. Импорт в конце файла — иначе цикл:
# модули задач импортируют `broker` из этого модуля. Без этого воркер
# (`taskiq worker aicore.broker:broker`) не знал бы ни одной задачи.
from aicore.tasks import agent_run as _agent_run  # noqa: E402,F401
from aicore.tasks import llm as _llm  # noqa: E402,F401
from aicore.tasks import ping as _ping  # noqa: E402,F401
from aicore.tasks import rag_index as _rag_index  # noqa: E402,F401
