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
"""

from __future__ import annotations

from taskiq import InMemoryBroker
from taskiq.middlewares import SmartRetryMiddleware

from aicore.config import Settings, get_settings
from aicore.logging import get_logger

logger = get_logger(__name__)


def build_retry_middleware(settings: Settings | None = None) -> SmartRetryMiddleware:
    """Собрать SmartRetryMiddleware из настроек.

    Сигнатура сверена с taskiq 0.13.0 (`default_retry_count`, `default_delay`,
    `use_jitter`, `use_delay_exponent`, `max_delay_exponent`).
    """
    cfg = settings or get_settings()
    return SmartRetryMiddleware(
        default_retry_count=cfg.task_max_retries,
        default_delay=cfg.task_retry_delay_seconds,
        use_jitter=cfg.task_retry_use_jitter,
        use_delay_exponent=cfg.task_retry_use_exponent,
        max_delay_exponent=cfg.task_retry_max_exponent_delay,
    )


def build_broker(settings: Settings | None = None) -> InMemoryBroker:
    """Собрать брокер согласно настройкам.

    Для `memory` возвращается InMemoryBroker с await_inplace=True, чтобы в
    скриптах и тестах задача гарантированно дорабатывала до конца в рамках
    одного event loop.

    Для `redis` требуется extra `redis`; импорт делается лениво, чтобы базовый
    `uv sync` оставался лёгким и проект работал без Redis.
    """
    cfg = settings or get_settings()

    if cfg.broker_backend == "redis":
        from taskiq_redis import RedisAsyncResultBackend, RedisStreamBroker

        broker: InMemoryBroker = RedisStreamBroker(
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
