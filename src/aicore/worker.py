"""Worker runtime: schema readiness, metrics exporter, retry source, sweeper.

The taskiq worker process needs more than "start the broker": on startup it must
ensure the schema is reachable (SQLite dev creates it; Postgres is Alembic's job),
start the delayed-retry schedule source, expose its own Prometheus metrics (the
API process cannot see the worker's counters), and run the lease sweeper that
reclaims tasks orphaned by a crashed worker. All of it is wired from the
``WORKER_STARTUP``/``WORKER_SHUTDOWN`` taskiq events.

Kept out of ``tasks/ping.py`` so the sample task stays about task semantics and
this process-level plumbing is testable on its own.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aicore.broker import ensure_schedule_source_started, shutdown_schedule_source
from aicore.config import Settings, get_settings
from aicore.db.session import get_database
from aicore.logging import get_logger
from aicore.tasks.sweeper import lease_sweeper_loop

logger = get_logger(__name__)

_stop: asyncio.Event | None = None
_sweeper_task: asyncio.Task[None] | None = None
_metrics_server: Any = None
_metrics_thread: Any = None


async def start_worker_runtime(settings: Settings | None = None) -> None:
    """Bring up everything the worker process needs, in a safe order."""
    global _stop, _sweeper_task
    cfg = settings or get_settings()

    # Schema: connect always; auto-create only for SQLite (dev/tests). On
    # Postgres `create_all()` racing with `alembic upgrade head` used to fail
    # with `type "document_status" already exists`, so it must not run there.
    db = get_database()
    await db.ensure_ready()

    _start_metrics_exporter(cfg)
    try:
        await ensure_schedule_source_started(cfg)
    except Exception:  # noqa: BLE001 — retry/scheduler is best-effort at startup
        logger.exception("worker.schedule_source_start_failed")

    # A fresh Event per start: a module-level Event binds to the first event
    # loop it is awaited on and then refuses reuse from another loop (which is
    # exactly what happens across test cases / restarts).
    _stop = asyncio.Event()
    if _sweeper_task is None or _sweeper_task.done():
        _sweeper_task = asyncio.create_task(lease_sweeper_loop(_stop), name="lease-sweeper")

    logger.info("worker.startup.complete", metrics_port=cfg.worker_metrics_port)


async def stop_worker_runtime() -> None:
    """Tear down the background pieces started by :func:`start_worker_runtime`."""
    global _sweeper_task, _stop, _metrics_server, _metrics_thread
    if _stop is not None:
        _stop.set()
    if _sweeper_task is not None:
        try:
            await asyncio.wait_for(_sweeper_task, timeout=5)
        except (TimeoutError, asyncio.CancelledError):
            _sweeper_task.cancel()
        _sweeper_task = None
    _stop = None

    _stop_metrics_exporter()

    try:
        await shutdown_schedule_source()
    except Exception:  # noqa: BLE001 — shutdown best-effort
        logger.exception("worker.schedule_source_shutdown_failed")

    logger.info("worker.shutdown.complete")


def _start_metrics_exporter(cfg: Settings) -> None:
    """Serve the worker's Prometheus registry on its own port.

    Counters for tasks and LLM calls are incremented in the worker process; the
    API's `/metrics` cannot see them. Prometheus scrapes this second target.
    A bind conflict (several workers on one host) is non-fatal: we warn and
    keep the worker running rather than crash it over an observability port.
    """
    global _metrics_server, _metrics_thread
    if cfg.worker_metrics_port <= 0 or _metrics_server is not None:
        return
    from prometheus_client import start_http_server

    try:
        result = start_http_server(cfg.worker_metrics_port, addr=cfg.worker_metrics_host)
    except OSError as exc:
        logger.warning("worker.metrics_exporter_bind_failed", error=str(exc))
        return

    # prometheus-client >= 0.23 returns ``(httpd, thread)``; older versions
    # returned the server or nothing. Accept both so the exporter still stops
    # cleanly on the pinned line.
    if isinstance(result, tuple):
        _metrics_server, _metrics_thread = result[0], result[1]
    else:
        _metrics_server, _metrics_thread = result, None
    logger.info(
        "worker.metrics_exporter_started",
        port=cfg.worker_metrics_port,
        host=cfg.worker_metrics_host,
    )


def _stop_metrics_exporter() -> None:
    """Shut down the exporter server and its daemon thread, if any."""
    global _metrics_server, _metrics_thread
    if _metrics_server is not None:
        try:
            _metrics_server.shutdown()
            _metrics_server.server_close()
        except Exception:  # noqa: BLE001 — best-effort teardown
            logger.exception("worker.metrics_exporter_stop_failed")
        _metrics_server = None
    if _metrics_thread is not None:
        _metrics_thread.join(timeout=2)
        _metrics_thread = None
