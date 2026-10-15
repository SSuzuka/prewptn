"""Сборка FastAPI-приложения.

Монтируются роуты health, задач (`/v1/tasks`), документов RAG
(`/v1/documents`) и `/metrics`. MCP-сервер (этап 6) добавляется позже.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aicore.api.routes import documents, health, metrics, tasks
from aicore.config import get_settings
from aicore.db.session import get_database
from aicore.logging import configure_logging, get_correlation_id, get_logger
from aicore.telemetry import configure_telemetry

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Настройка логирования и телеметрии на старте, мягкое завершение на стопе."""
    configure_logging()
    configure_telemetry()
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
    db = get_database()
    if db.is_connected:
        await db.disconnect()
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
    app.include_router(tasks.router, prefix="/v1", tags=["tasks"])
    app.include_router(documents.router, prefix="/v1", tags=["documents"])
    app.include_router(metrics.router, tags=["metrics"])

    logger.info("app.configured", correlation_id=get_correlation_id())
    return app


app = create_app()
