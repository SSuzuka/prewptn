"""Сборка FastAPI-приложения.

Пока это каркас: health-check и настройки. Роуты `/v1/tasks`, `/v1/documents`,
`/metrics` и монтирование MCP-сервера добавляются в следующих этапах.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aicore.api.routes import health
from aicore.config import get_settings
from aicore.logging import configure_logging, get_correlation_id, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Настройка логирования на старте и мягкое завершение на стопе."""
    configure_logging()
    settings = get_settings()
    logger.info(
        "app.startup",
        version=app.version,
        broker_backend=settings.broker_backend,
        llm_backend=settings.llm_backend,
        vector_backend=settings.vector_backend,
        is_production=settings.is_production,
    )
    yield
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    """Фабрика приложения (удобно для тестов)."""
    app = FastAPI(
        title="AI Agent Platform",
        version="0.1.0",
        description=(
            "Async task pipeline over LangGraph agents with RAG, MCP tools and observability."
        ),
        lifespan=lifespan,
    )

    app.include_router(health.router, tags=["health"])

    logger.info("app.configured", correlation_id=get_correlation_id())
    return app


app = create_app()
