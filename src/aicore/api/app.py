"""Сборка FastAPI-приложения.

Монтируются роуты health, задач (`/v1/tasks`), документов RAG
(`/v1/documents`), агентов (`/v1/agents`) и `/metrics`. MCP-сервер (этап 6)
добавляется позже.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aicore.api.deps import require_api_key
from aicore.api.middleware import CorrelationIdMiddleware
from aicore.api.routes import agents, documents, health, metrics, tasks
from aicore.broker import shutdown_broker
from aicore.config import get_settings
from aicore.db.session import get_database
from aicore.llm import close_llm_client
from aicore.logging import configure_logging, get_correlation_id, get_logger
from aicore.rag import close_retriever
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
    # Порядок завершения не важен, но каждый ресурс закрываем защищённо: сбой
    # одного не должен помешать закрыть остальные (и не уронить shutdown).
    for name, closer in (
        ("broker", shutdown_broker),
        ("llm", close_llm_client),
        ("retriever", close_retriever),
    ):
        try:
            await closer()
        except Exception:  # noqa: BLE001 — shutdown best-effort
            logger.exception("app.shutdown_resource_failed", resource=name)
    db = get_database()
    if db.is_connected:
        await db.disconnect()
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    """Фабрика приложения (удобно для тестов)."""
    settings = get_settings()
    app = FastAPI(
        title="AI Agent Platform",
        version="0.1.0",
        description=(
            "Async task pipeline over LangGraph agents with RAG, MCP tools and observability."
        ),
        lifespan=lifespan,
    )

    # CORS: без него браузерный фронтенд (например, localhost:3000) не пройдёт
    # preflight. Список origin'ов — из настроек; пустой список = middleware нет.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # correlation_id из X-Correlation-ID/X-Request-ID (или новый) + в ответе.
    app.add_middleware(CorrelationIdMiddleware)

    app.include_router(health.router, tags=["health"])
    # Аутентификация по X-API-Key (если `api_keys` заданы) — на всех /v1 и
    # /metrics. Health/ready остаются открытыми для проб оркестратора.
    protected = [Depends(require_api_key)]
    app.include_router(tasks.router, prefix="/v1", tags=["tasks"], dependencies=protected)
    app.include_router(documents.router, prefix="/v1", tags=["documents"], dependencies=protected)
    app.include_router(agents.router, prefix="/v1", tags=["agents"], dependencies=protected)
    app.include_router(metrics.router, tags=["metrics"], dependencies=protected)

    logger.info("app.configured", correlation_id=get_correlation_id())
    return app


app = create_app()
