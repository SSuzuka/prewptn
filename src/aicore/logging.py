"""Структурированное логирование на structlog.

Даёт JSON-логи в проде и читаемый вывод в dev, а также прокидывает
`correlation_id` через contextvars, чтобы один request/task был склеен
во все свои логи.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Mapping, MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog

from aicore.config import Settings, get_settings

# correlation_id текущего запроса/задачи. Через ContextVar, поэтому
# в asyncio он не протекает между конкурентными задачами.
correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "secret",
        "token",
        "password",
        "authorization",
        "langfuse_secret_key",
        "langfuse_public_key",
        "openai_api_key",
    }
)


def new_correlation_id() -> str:
    """Сгенерировать новый correlation_id."""
    return uuid.uuid4().hex


def set_correlation_id(value: str | None) -> object:
    """Установить correlation_id; верните токен, чтобы потом откатить."""
    return correlation_id.set(value or new_correlation_id())


def get_correlation_id() -> str | None:
    return correlation_id.get()


def reset_correlation_id(token: object) -> None:
    correlation_id.reset(token)  # type: ignore[arg-type]


def add_correlation_id(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    """Процессор: кладёт correlation_id в каждую запись лога."""
    cid = correlation_id.get()
    if cid is not None:
        event_dict.setdefault("correlation_id", cid)
    return event_dict


def redact_secrets(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    """Процессор: маскирует значения, похожие на секреты.

    Защита от случайной утечки ключей в логи — включена всегда.
    """
    for key in list(event_dict):
        if key.lower() in _SECRET_KEYS:
            event_dict[key] = "***"
    return event_dict


def _build_processors(*, as_json: bool) -> list[Any]:
    """Цепочка процессоров. as_json=True для прода, False для dev."""
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        add_correlation_id,
        redact_secrets,
    ]
    if as_json:
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer(ensure_ascii=False))
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    return processors


def configure_logging(settings: Settings | None = None) -> None:
    """Настроить structlog и stdlib-logging.

    Вызывается один раз при старте приложения. Повторный вызов безопасен —
    он просто перенастраивает рендереры под текущие настройки.
    """
    cfg = settings or get_settings()
    as_json = cfg.log_json
    level = getattr(logging, cfg.log_level, logging.INFO)

    structlog.configure(
        processors=_build_processors(as_json=as_json),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Направляем stdlib-логи (uvicorn, sqlalchemy, taskiq) в тот же формате,
    # чтобы в проде не было смеси JSON и «человеческого» текста.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    # Сторонние библиотеки шумят на DEBUG/INFO — приглушаем.
    for noisy in ("uvicorn.access", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Получить логгер. Привязанный correlation_id добавится автоматически."""
    return structlog.stdlib.get_logger(name)  # type: ignore[no-any-return]
