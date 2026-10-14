"""Централизованная конфигурация через pydantic-settings.

Все переключатели читаются из окружения (`.env`). Значения по умолчанию
намеренно in-memory: проект должен запускаться без Docker, PostgreSQL,
Redis, Qdrant и без LLM-ключа.

Секреты (API-ключи) живут только в окружении и никогда не хардкодятся.
`redacted_dict()` возвращает снимок конфигурации для логов без секретов.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BrokerBackend = Literal["memory", "redis"]
VectorBackend = Literal["memory", "qdrant"]
LLMBackend = Literal["stub", "openai"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ALLOWED_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"}


class Settings(BaseSettings):
    """Настройки сервиса."""

    model_config = SettingsConfigDict(
        env_file=(".env", str(PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- очередь
    broker_backend: BrokerBackend = Field(
        default="memory",
        description="memory - InMemoryBroker (dev/test), redis - RedisStreamBroker (prod)",
    )
    redis_url: str = Field(default="redis://localhost:6379/0")
    taskiq_queue: str = Field(default="aicore:tasks")

    # ---------------------------------------------------------- база данных
    db_url: str = Field(
        default="sqlite+aiosqlite:///./local.db",
        description="sqlite+aiosqlite для dev, postgresql+asyncpg для prod",
    )
    db_echo: bool = Field(default=False)

    # -------------------------------------------------- векторное хранилище
    vector_backend: VectorBackend = Field(
        default="memory",
        description="memory - AsyncQdrantClient(':memory:'), qdrant - отдельный сервер",
    )
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="aicore_documents")
    embedding_dim: int = Field(default=384, ge=1)

    # ----------------------------------------------------------------- LLM
    llm_backend: LLMBackend = Field(
        default="stub",
        description="stub - детерминированный без ключей, openai - совместимый endpoint",
    )
    llm_model: str = Field(default="gpt-4o-mini")
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_concurrency: int = Field(default=8, ge=1)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_retry_base_delay: float = Field(default=1.0, gt=0)

    # Цена за 1000 токенов в USD — для метрик стоимости.
    llm_price_prompt_per_1k: float = Field(default=0.00015, ge=0)
    llm_price_completion_per_1k: float = Field(default=0.0006, ge=0)

    # --------------------------------------------------------- надёжность
    task_max_retries: int = Field(default=3, ge=0)
    task_retry_delay_seconds: float = Field(default=5.0, gt=0)
    task_retry_use_jitter: bool = Field(default=True)
    task_retry_use_exponent: bool = Field(default=True)
    task_retry_max_exponent_delay: float = Field(default=60.0, gt=0)
    task_timeout_seconds: float = Field(default=120.0, gt=0)

    # ----------------------------------------------------------------- API
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_rate_limit_per_minute: int = Field(default=60, ge=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --------------------------------------------------- наблюдаемость
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True)

    otel_service_name: str = Field(default="aicore")
    otel_exporter_otlp_endpoint: str | None = Field(default=None)

    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_base_url: str = Field(default="https://cloud.langfuse.com")

    openai_api_key: str | None = Field(default=None)
    openai_base_url: str = Field(default="https://api.openai.com/v1")

    # ------------------------------------------------ утилиты и валидация
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> object:
        """Принимает как список, так и строку через запятую."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        upper = value.upper()
        if upper not in _ALLOWED_LOG_LEVELS:
            allowed = sorted(_ALLOWED_LOG_LEVELS)
            raise ValueError(f"log_level must be one of {allowed}, got {value!r}")
        return upper

    @field_validator("openai_api_key")
    @classmethod
    def _warn_openai_key_in_stub_mode(cls, value: str | None) -> str | None:
        """Ключ допустим, но при llm_backend=stub он просто не используется."""
        return value

    # -------------------------------------------------------------- свойства
    @property
    def langfuse_enabled(self) -> bool:
        """Трейсинг включается только когда заданы оба ключа.

        Без ключей SDK работает как no-op, поэтому локальный запуск и тесты
        не зависят от наличия сервера.
        """
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def is_production(self) -> bool:
        """Прод-режим = хотя бы один не in-memory бэкенд."""
        return (
            self.broker_backend == "redis"
            or self.vector_backend == "qdrant"
            or self.llm_backend == "openai"
        )

    @property
    def uses_sqlite(self) -> bool:
        return self.db_url.startswith("sqlite")

    def redacted_dict(self) -> dict[str, object]:
        """Снимок конфигурации для логов — без секретов."""
        data: dict[str, object] = self.model_dump()
        for key in ("langfuse_secret_key", "langfuse_public_key", "openai_api_key"):
            if data.get(key):
                data[key] = "***"
        db_url = data.get("db_url")
        if isinstance(db_url, str) and "@" in db_url:
            scheme, _, rest = db_url.partition("://")
            creds, _, host = rest.rpartition("@")
            user = creds.split(":", 1)[0]
            data["db_url"] = f"{scheme}://{user}:***@{host}"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Кэшированный синглтон настроек."""
    return Settings()


def reset_settings_cache() -> None:
    """Сброс кэша — нужно в тестах при подмене env."""
    get_settings.cache_clear()


def is_testing() -> bool:
    """True, если запущены тесты (pytest выставляет PYTEST_CURRENT_TEST)."""
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))
