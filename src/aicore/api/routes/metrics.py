"""Роут `/metrics` — Prometheus-экспозиция."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from aicore.api.deps import rate_limit

router = APIRouter()


@router.get(
    "/metrics",
    dependencies=[Depends(rate_limit)],
    summary="Prometheus-метрики (latency, счётчики, стоимость)",
)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
