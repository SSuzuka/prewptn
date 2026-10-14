"""Observability wiring: OpenTelemetry tracer + Langfuse client.

Both integrations are optional and no-op safe:

  * without an OTLP endpoint nothing is exported — spans are still created,
    but never leave the process;
  * without Langfuse keys the client is never created (the SDK would log an
    authentication error and disable itself), so :func:`record_generation` is
    a cheap no-op and local runs and tests stay fully offline.

Telemetry must never break a real call: :func:`record_generation` swallows and
logs any exporter error instead of propagating it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from aicore.config import Settings, get_settings
from aicore.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Telemetry:
    """Resolved observability handles for the current process."""

    service_name: str
    tracer: trace.Tracer
    provider: TracerProvider
    langfuse: Any | None = None


_telemetry: Telemetry | None = None
# `trace.set_tracer_provider` may only run once per process; later calls would
# only emit a warning, so we set it on the first configure and reuse afterwards.
_global_provider_set = False


def _build_provider(cfg: Settings) -> TracerProvider:
    """Tracer provider with an OTLP exporter only when an endpoint is configured."""
    provider = TracerProvider(resource=Resource.create({"service.name": cfg.otel_service_name}))
    if cfg.otel_exporter_otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter = OTLPSpanExporter(endpoint=cfg.otel_exporter_otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except Exception as exc:  # noqa: BLE001
            logger.warning("telemetry.otlp_exporter_unavailable", error=str(exc))
    return provider


def _build_langfuse(cfg: Settings) -> Any | None:
    """Langfuse client, or ``None`` when tracing is disabled.

    ``get_client()`` reads the ``LANGFUSE_*`` environment variables. Settings may
    have come from a ``.env`` file, which pydantic-settings does not export, so we
    mirror them with ``setdefault`` (real environment values still win).
    """
    if not cfg.langfuse_enabled:
        return None

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", cfg.langfuse_public_key or "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", cfg.langfuse_secret_key or "")
    os.environ.setdefault("LANGFUSE_HOST", cfg.langfuse_base_url)
    try:
        from langfuse import get_client

        return get_client()
    except Exception as exc:  # noqa: BLE001
        logger.warning("telemetry.langfuse_unavailable", error=str(exc))
        return None


def configure_telemetry(settings: Settings | None = None) -> Telemetry:
    """Build the observability handles from settings."""
    global _telemetry, _global_provider_set
    cfg = settings or get_settings()

    provider = _build_provider(cfg)
    if not _global_provider_set:
        try:
            trace.set_tracer_provider(provider)
            _global_provider_set = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry.provider_already_set", error=str(exc))

    langfuse_client = _build_langfuse(cfg)
    _telemetry = Telemetry(
        service_name=cfg.otel_service_name,
        tracer=provider.get_tracer(cfg.otel_service_name),
        provider=provider,
        langfuse=langfuse_client,
    )
    logger.info(
        "telemetry.configured",
        langfuse=langfuse_client is not None,
        otlp=bool(cfg.otel_exporter_otlp_endpoint),
    )
    return _telemetry


def get_telemetry() -> Telemetry:
    """Return the process telemetry, configuring it on first use."""
    global _telemetry
    if _telemetry is None:
        _telemetry = configure_telemetry()
    return _telemetry


def reset_telemetry() -> None:
    """Drop the cached telemetry (tests)."""
    global _telemetry
    _telemetry = None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """Create an OTel span, setting the given attributes when not ``None``."""
    with get_telemetry().tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def record_generation(
    *,
    name: str,
    model: str,
    input: Any,
    output: Any,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_ms: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log one LLM call to Langfuse as a generation. No-op when disabled."""
    langfuse_client = get_telemetry().langfuse
    if langfuse_client is None:
        return
    try:
        with langfuse_client.start_as_current_observation(
            as_type="generation", name=name, model=model, input=input
        ) as generation:
            generation.update(
                output=output,
                usage_details={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "total": prompt_tokens + completion_tokens,
                },
                cost_details={"total": cost_usd},
                metadata={**(metadata or {}), "latency_ms": latency_ms},
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("telemetry.generation_record_failed", error=str(exc))
