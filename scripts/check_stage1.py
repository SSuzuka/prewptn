"""Проверка config.py и logging.py без внешней инфраструктуры.

Запуск:
    uv run python scripts/check_stage1.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aicore.config import (  # noqa: E402
    Settings,
    get_settings,
    is_testing,
    reset_settings_cache,
)
from aicore.logging import (  # noqa: E402
    configure_logging,
    get_correlation_id,
    get_logger,
    new_correlation_id,
    reset_correlation_id,
    set_correlation_id,
)

ok = True


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok
    mark = "OK  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        ok = False


print("== config ==")
cfg = get_settings()
check("singleton cached", get_settings() is cfg)
check("default broker is memory", cfg.broker_backend == "memory", cfg.broker_backend)
check("default llm is stub", cfg.llm_backend == "stub", cfg.llm_backend)
check("default vector is memory", cfg.vector_backend == "memory", cfg.vector_backend)
check("db uses sqlite by default", cfg.uses_sqlite, cfg.db_url)
check("not production by default", cfg.is_production is False)
check("langfuse disabled without keys", cfg.langfuse_enabled is False)

# Секреты не должны утекать в redacted_dict.
secret_cfg = Settings(
    openai_api_key="sk-SECRET-VALUE",
    langfuse_secret_key="sk-lf-SECRET",
    langfuse_public_key="pk-lf-PUBLIC",
    db_url="postgresql+asyncpg://bob:hunter2@db.internal:5432/aicore",
)
snapshot = secret_cfg.redacted_dict()
blob = repr(snapshot)
check("openai_api_key masked", "sk-SECRET-VALUE" not in blob)
check("langfuse secret masked", "sk-lf-SECRET" not in blob)
check("langfuse public masked", "pk-lf-PUBLIC" not in blob)
check("db password masked", "hunter2" not in blob, snapshot["db_url"])
check("db username kept", "bob" in blob)
check("langfuse_enabled with both keys", secret_cfg.langfuse_enabled is True)
# Внимание: наличие ключа само по себе НЕ делает прод — смотрят на бэкенды.
check("key alone is not production", secret_cfg.is_production is False)
check(
    "is_production with openai backend",
    Settings(llm_backend="openai").is_production is True,
)
check("is_production with redis broker", Settings(broker_backend="redis").is_production is True)
check("is_production with qdrant", Settings(vector_backend="qdrant").is_production is True)

# Валидация env.
os.environ["AICORE_LOG_LEVEL_OVERRIDE"] = "1"
try:
    Settings(log_level="verbose")
    check("invalid log_level rejected", False)
except Exception as exc:  # pydantic ValidationError
    check("invalid log_level rejected", True, type(exc).__name__)

check("cors from comma string", Settings(cors_origins="a, b ,c").cors_origins == ["a", "b", "c"])
check("log_level upper-cased", Settings(log_level="debug").log_level == "DEBUG")
check("is_testing() false here", is_testing() is False)

reset_settings_cache()

print()
print("== logging ==")
configure_logging(Settings(log_level="DEBUG", log_json=True))
log = get_logger("stage1")

token = set_correlation_id(new_correlation_id())
try:
    cid = get_correlation_id()
    check("correlation_id set", isinstance(cid, str) and len(cid) == 32, str(cid)[:8] + "...")

    log.info("hello.from.json.logger", answer=42)
    log.warning("secret.should.be.masked", api_key="sk-SHOULD-NOT-APPEAR")
finally:
    reset_correlation_id(token)

check("correlation_id reset", get_correlation_id() is None)

configure_logging(Settings(log_level="INFO", log_json=False))
get_logger("stage1").info("hello.from.console.logger")

# stdlib-логгеры тоже должны быть перенаправлены, а не сыпать своим форматом.
root = logging.getLogger()
handler_name = type(root.handlers[0]).__name__ if root.handlers else "none"
check("root has handler", len(root.handlers) == 1, handler_name)

print()
print("STAGE 1 " + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
