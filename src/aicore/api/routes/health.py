"""Health-check эндпоинты.

Отдаёт liveness и снимок активных бэкендов — по нему сразу видно, в каком
режиме запущен сервис (in-memory или production).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from aicore.config import get_settings
from aicore.db.session import get_database

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


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness (проверяет БД и наличие схемы)",
    responses={503: {"model": ReadinessResponse, "description": "Not ready"}},
)
async def ready(response: Response) -> ReadinessResponse:
    """Readiness: реально проверяет БД и наличие схемы.

    Раньше эндпоинт всегда отвечал `ready`, даже когда БД недоступна или
    миграции не накатаны — и оркестратор слал трафик в инстанс, который на
    первом же запросе отдавал 500 (`relation "task_runs" does not exist`).
    Теперь: подключение + `SELECT 1` + проверка наличия таблицы `task_runs`.
    Неготовность = 503, чтобы пробу было чем остановить rollout.

    Health/ready намеренно не под rate limit и не под API-ключом.
    """
    db = get_database()
    try:
        await db.ensure_ready()
        connected = await db.healthcheck()
        schema_ok = connected and await db.schema_ready()
    except Exception:  # noqa: BLE001 — проба не должна падать, только сигналить
        connected = schema_ok = False

    if not schema_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        detail = "database reachable but schema missing" if connected else "database unreachable"
        return ReadinessResponse(status="not_ready", detail=detail)
    return ReadinessResponse(detail="database and schema ok")


@router.get("/version", summary="Версия сервиса")
async def version() -> dict[str, Any]:
    return {"name": "aicore", "version": "0.1.0"}
