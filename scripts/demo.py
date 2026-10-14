"""Демонстрационный сценарий «из коробки» — без Docker, БД, очереди и LLM-ключа.

Показывает то, что уже работает на текущем этапе:
  1. загрузку конфигурации и активные бэкенды;
  2. структурированное логирование + correlation_id;
  3. что секреты не утекают в логи;
  4. живой HTTP-запрос к health-check собранного приложения.

Запуск:
    uv run python scripts/demo.py

Что здесь НЕ демонстрируется (это следующие этапы, см. README):
очередь, воркер, RAG, агенты, MCP, SSE-стрим.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from aicore.api.app import create_app  # noqa: E402
from aicore.config import get_settings  # noqa: E402
from aicore.logging import (  # noqa: E402
    configure_logging,
    get_logger,
    new_correlation_id,
    reset_correlation_id,
    set_correlation_id,
)


def main() -> int:
    configure_logging(get_settings())
    log = get_logger("demo")

    print("=" * 68)
    print("AI Agent Platform — демо конфигурации, логирования и health-check")
    print("=" * 68)

    # ---------------------------------------------------------------- 1. config
    cfg = get_settings()
    print("\n[1] Активная конфигурация (секреты замаскированы)")
    print("-" * 68)
    for key, value in sorted(cfg.redacted_dict().items()):
        print(f"    {key:32s} = {value}")

    print(f"\n    broker    : {cfg.broker_backend}")
    print(f"    database  : {'sqlite' if cfg.uses_sqlite else 'postgres'}")
    print(f"    vector    : {cfg.vector_backend}")
    print(f"    llm       : {cfg.llm_backend}")
    print(f"    langfuse  : {'enabled' if cfg.langfuse_enabled else 'disabled (no-op)'}")
    print(f"    режим     : {'production' if cfg.is_production else 'development'}")

    # --------------------------------------------------------------- 2. logging
    print("\n[2] Структурированное логирование с correlation_id")
    print("-" * 68)

    token = set_correlation_id(new_correlation_id())
    try:
        configure_logging(cfg.model_copy(update={"log_json": True}))
        log.info("demo.step.started", stage=1, note="JSON-формат логов")

        # Ключ в лог-событии должен быть замаскирован процессором redact_secrets.
        log.warning(
            "demo.secret.redaction.check",
            api_key="sk-THIS-MUST-NOT-APPEAR",
            password="also-masked",
            tokens=1234,
        )

        configure_logging(cfg.model_copy(update={"log_json": False}))
        log.info("demo.step.finished", stage=1, note="console-формат логов")
    finally:
        reset_correlation_id(token)

    # ------------------------------------------------------------------ 3. HTTP
    print("\n[3] Живой HTTP-запрос к собранному приложению")
    print("-" * 68)
    configure_logging(cfg)

    with TestClient(create_app()) as client:
        for path in ("/health", "/ready", "/version", "/openapi.json"):
            resp = client.get(path)
            print(f"    GET {path:15s} -> {resp.status_code}")

        health = client.get("/health").json()
        print("\n    /health:")
        for key, value in health.items():
            print(f"      {key:14s} = {value}")

        print("\n    Swagger UI доступен после старта:")
        print("      uv run uvicorn aicore.api.app:app --reload")
        print(f"      http://{cfg.api_host}:{cfg.api_port}/docs")

    log.info("demo.completed")
    print("\n" + "=" * 68)
    print("Демо завершено. Следующие этапы: очередь -> LLM -> RAG -> агенты -> MCP")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
