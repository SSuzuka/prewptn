# ============================================================================
# aicore — окружение разработки (uv)
# ============================================================================
#
# ВАЖНО (Windows): venv по умолчанию создаётся в ./.venv, но на этой машине
# интерпретатор туда не кладётся корректно, поэтому uv используется с явным
# путём окружения. Команды ниже работают в PowerShell и cmd.
#
# Один раз создать окружение:
#   uv venv --python 3.12 c:\uv-envs\uop
#
# Каждый запуск (PowerShell):
#   $env:UV_PROJECT_ENVIRONMENT = "c:\uv-envs\uop"; uv sync
#   $env:UV_PROJECT_ENVIRONMENT = "c:\uv-envs\uop"; uv run pytest
#
# Каждый запуск (cmd):
#   set "UV_PROJECT_ENVIRONMENT=c:\uv-envs\uop" && uv sync
#
# ВНИМАНИЕ про кавычки в cmd: `set VAR=value &&` захватывает пробелы в значение
# и uv падает с ошибкой "not a compatible environment". Кавычки обязательны.
# ============================================================================

# --- зависимости -----------------------------------------------------------
uv sync                                        # базовый набор (offline-first)
uv sync --extra redis --extra postgres         # + прод-брокер и прод-БД
uv sync --group dev                            # pytest, httpx

# --- проверки --------------------------------------------------------------
uv run python -c "import aicore; print('OK')"
uv run pytest
uv run ruff check src tests scripts migrations
uv run ruff format --check src tests scripts migrations
uv run alembic upgrade head
uv run alembic downgrade base
uv run alembic upgrade head

# --- запуск ----------------------------------------------------------------
docker compose up -d                           # postgres + redis + qdrant
docker compose --profile app up -d --build     # migrate + api + worker + scheduler
uv run python scripts/demo.py                  # демо: конфиг, логи, health-check
uv run python scripts/check_stage1.py          # проверка config + logging
uv run python scripts/check_stage2.py          # проверка БД, брокера, retries, DLQ
uv run python scripts/check_server.py          # поднимает uvicorn и бьёт по API
uv run uvicorn aicore.api.app:app --reload     # API на 127.0.0.1:8000
uv run taskiq worker aicore.broker:broker      # воркер (прод-режим)
uv run taskiq scheduler aicore.broker:scheduler --update-interval 1  # backoff

# --- git -------------------------------------------------------------------
git add -A
git commit -m "..."
git push
