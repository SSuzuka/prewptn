"""Health-check эндпоинты.

Отдаёт liveness и снимок активных бэкендов — по нему сразу видно, в каком
режиме запущен сервис (in-memory или production).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from aicore.config import get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Ответ `/health`."""

    status: str = Field(default="ok")
    service: str
    version: str
    environment: str
    backends: dict[str, str]


class ReadinessResponse(BaseModel):
    """Ответ `/ready`."""

    status: str = Field(default="ready")
    detail: str


@router.get("/health", response_model=HealthResponse, summary="Liveness + активные бэкенды")
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        service=settings.otel_service_name,
        version="0.1.0",
        environment="production" if settings.is_production else "development",
        backends={
            "broker": settings.broker_backend,
            "db": "sqlite" if settings.uses_sqlite else "postgres",
            "vector": settings.vector_backend,
            "llm": settings.llm_backend,
        },
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness")
async def ready() -> ReadinessResponse:
    """Заглушка: реальные проверки БД/очереди появятся вместе с их клиентами."""
    return ReadinessResponse(detail="no external dependencies checked yet")


@router.get("/version", summary="Версия сервиса")
async def version() -> dict[str, Any]:
    return {"name": "aicore", "version": "0.1.0"}
