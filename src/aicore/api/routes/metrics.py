"""Роут `/metrics` — Prometheus-экспозиция."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from aicore.api.deps import rate_limit
from aicore.metrics import multiprocess_registry

router = APIRouter()


@router.get(
    "/metrics",
    dependencies=[Depends(rate_limit)],
    summary="Prometheus-метрики (latency, счётчики, стоимость)",
)
def metrics() -> Response:
    registry = multiprocess_registry() or REGISTRY
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
