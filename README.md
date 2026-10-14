# AI Agent Platform — backend для AI/Agent-платформы

Backend-сервис, который принимает AI-задачу, проводит её через асинхронный
**task-processing pipeline**, запускает **агентский workflow** (RAG + tools + MCP),
стримит ход выполнения и отдаёт метрики по latency и стоимости LLM-вызовов.

> **Статус реализации.** Легенда: ✅ реализовано · 🚧 в работе · 📋 запланировано.
>
> | Этап | Что | Статус |
> |---|---|---|
> | 0 | Каркас репозитория, пины, env-переключатели, README | ✅ |
> | 1 | `config.py`, `logging.py`, health-check + live-сервер | ✅ |
> | 2 | БД + брокер: статусы, retries, idempotency, DLQ | ✅ |
> | 3 | LLM-слой: timeouts, rate limit, учёт токенов и стоимости | 📋 |
> | 4 | RAG: chunking → embeddings → Qdrant → контекст | 📋 |
> | 5 | Агенты: `create_agent` + supervisor-граф на LangGraph | 📋 |
> | 6 | MCP-сервер + монтирование в FastAPI | 📋 |
> | 7 | API + SSE-стриминг + `/metrics` | 📋 |
> | 9 | `evals/`: faithfulness / relevance / стоимость | 📋 фаза 2 |
>
> Тесты защищают от молчаливого ухода на breaking-линии: если uv подтянет
> `fastapi 0.143+`, `langfuse 5+`, `mcp 2+` или `sqlalchemy 2.1+`, они упадут
> (см. `tests/test_smoke_imports.py`).

---

## Быстрый старт (без Docker)

Нужны только **Python 3.11+** и **[uv](https://docs.astral.sh/uv/)**.
Docker, PostgreSQL, Redis, Qdrant и LLM-ключ **не требуются** — по умолчанию
все бэкенды работают in-process.

```bash
uv sync                              # venv + точные версии из uv.lock
uv run python -c "import aicore"     # проверка, что пакет импортируется
uv run pytest                        # тесты, 0 зависимостей от инфраструктуры
uv run python scripts/check_stage1.py    # проверка config + logging
uv run python scripts/check_server.py    # поднимает uvicorn и бьёт по API
uv run uvicorn aicore.api.app:app --reload
```

## Проверки

Всё проходит офлайн — без Docker, Redis, Postgres, Qdrant и LLM-ключа.

```bash
uv run pytest                        # 101 тест
uv run ruff check src tests scripts  # линтер
uv run ruff format --check src tests scripts
uv run python scripts/demo.py            # конфиг, логи, health-check
uv run python scripts/check_stage1.py    # config + logging
uv run python scripts/check_stage2.py    # БД, брокер, retries, DLQ
uv run python scripts/check_server.py    # поднимает uvicorn и бьёт по API
```

> **Про `ruff`:** он объявлен в `[dependency-groups] dev`, поэтому нужен
> `uv sync --group dev` (или `uv run ruff ...`, который подтянет группу сам).
> Markdown намеренно исключён из форматирования (`exclude = ["*.md"]` в
> `[tool.ruff]`): python-блоки в README выровнены по колонкам для читаемости,
> а ruff снимает любое ручное выравнивание.
>
> **Про локальный прокси:** `check_server.py` намеренно использует opener с
> пустым `ProxyHandler({})`. Иначе системные `HTTP_PROXY`/`HTTPS_PROXY` уводят
> запросы к `127.0.0.1` в прокси, и все роуты отвечают 503 при открытом порте.

Swagger UI после старта сервера: <http://127.0.0.1:8000/docs> · OpenAPI-схема: `/openapi.json`

### Требования к окружению

| Инструмент | Версия | Зачем | Обязателен? |
|---|---|---|---|
| Python | 3.11 / 3.12 | рантайм | ✅ да |
| uv | ≥0.5 | venv, зависимости, запуск | ✅ да |
| git | любая | версионирование | ✅ да |
| Docker Desktop | — | Postgres/Redis/Qdrant для prod-режима | ❌ нет |

---

## Архитектура

```
                       POST /v1/tasks
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │              FastAPI (API)               │
        │  Pydantic-валидация · Idempotency-Key    │
        │  rate limiting · correlation_id          │
        └───────────────┬──────────────────────────┘
                        │ .kiq()
                        ▼
        ┌──────────────────────────────────────────┐
        │     Очередь  (broker.py)                 │
        │  InMemoryBroker  (dev)                   │
        │  RedisStreamBroker (prod: ack + retry)   │
        └───────────────┬──────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────┐
        │   Воркер  (taskiq worker)                │
        │  SmartRetryMiddleware: backoff + jitter  │
        │  timeout · исчерпал retries -> DLQ       │
        └───────────────┬──────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────┐
        │   LangGraph supervisor  (agents/)        │
        │   ┌────────────┐      ┌────────────┐     │
        │   │ RAG-агент  │      │ Tool-агент │     │
        │   └─────┬──────┘      └──────┬─────┘     │
        └─────────┼────────────────────┼───────────┘
                  │                    │
        ┌─────────▼────────┐  ┌────────▼─────────┐
        │   RAG  (rag/)    │  │  MCP  (mcp/)     │
        │ embeddings       │  │ инструменты и    │
        │ Qdrant :memory:  │  │ внешние системы  │
        └──────────────────┘  └──────────────────┘
                  │
                  ▼
        ┌──────────────────────────────────────────┐
        │   LLM  (llm/)                            │
        │  StubLLM (dev) | OpenAI-совместимый      │
        │  timeout · семафор · токены · $/вызов    │
        └───────────────┬──────────────────────────┘
                        │
        ┌───────────────┴──────────────────────────┐
        ▼                                          ▼
  Redis Streams ──SSE──▶ /v1/tasks/{id}/stream      Наблюдаемость
                                                      structlog (JSON)
                                                      Prometheus /metrics
                                                      Langfuse (OTel)
```

**Поток одной задачи, пошагово:**

1. `POST /v1/tasks` с заголовком `Idempotency-Key` — повторный запрос с тем же
   ключом вернёт **существующую** задачу, а не создаст дубль и не сожжёт токены.
2. Задача пишется в БД со статусом `pending` и уходит в очередь.
3. Воркер забирает её, статус → `running`, события пишутся в Redis Streams.
4. LangGraph-supervisor решает, какие подагенты вызвать (RAG и/или tools).
5. LLM-вызов идёт через обёртку: `asyncio.timeout`, семафор на concurrency,
   ретраи с backoff, фиксация `latency_ms` / токенов / `cost_usd`.
6. Клиент читает прогресс через SSE, не удерживая соединение.
7. Итог → `succeeded`. Если retries исчерпаны → `dead_letter` +
   ручной `POST /v1/tasks/{id}/retry`.

---

## API

Легенда: ✅ реализовано · 📋 запланировано.

**Доступно сейчас:**

| Метод | Путь | Назначение | Статус |
|---|---|---|---|
| `GET` | `/health` | liveness + снимок активных бэкендов | ✅ |
| `GET` | `/ready` | readiness (проверки БД/очереди появятся вместе с ними) | ✅ |
| `GET` | `/version` | версия сервиса | ✅ |
| `GET` | `/openapi.json` | OpenAPI-схема | ✅ |
| `GET` | `/docs` | Swagger UI | ✅ |

**Целевой API (этапы 2–7):**

| Метод | Путь | Назначение | Статус |
|---|---|---|---|
| `GET` | `/metrics` | Prometheus-метрики (latency, счётчики, стоимость) | 📋 этап 7 |
| `POST` | `/v1/documents` | загрузка документа → задача индексации в RAG | 📋 этап 4 |
| `GET` | `/v1/documents/{id}` | статус индексации документа | 📋 этап 4 |
| `POST` | `/v1/tasks` | постановка AI-задачи (идемпотентно, `Idempotency-Key`) | 📋 этап 7 |
| `GET` | `/v1/tasks/{id}` | статус: `pending` / `running` / `succeeded` / `failed` / `dead_letter` | 📋 этап 7 |
| `GET` | `/v1/tasks/{id}/stream` | **SSE**-поток событий выполнения агента | 📋 этап 7 |
| `POST` | `/v1/tasks/{id}/retry` | ручной перезапуск упавшей задачи | 📋 этап 7 |
| `/mcp` | — | Streamable HTTP эндпоинт MCP-сервера | 📋 этап 6 |

Проверить то, что уже есть:

```bash
# Поднять сервер
uv run uvicorn aicore.api.app:app --reload

# Доступные сейчас эндпоинты
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
curl -sS http://127.0.0.1:8000/version

# Пример целевого API (станет рабочим на этапе 7)
#   curl -sS -X POST http://127.0.0.1:8000/v1/tasks \
#     -H 'Content-Type: application/json' \
#     -H 'Idempotency-Key: demo-001' \
#     -d '{"question": "..."}'
#   curl -N http://127.0.0.1:8000/v1/tasks/<id>/stream
```

---

## Конфигурация

Всё управляется переменными окружения (`.env`, образец — `.env.example`).
Значения по умолчанию полностью in-memory, поэтому `scripts/demo.py` и тесты
работают сразу «из коробки». Полный список полей — в
[`.env.example`](.env.example); ниже ключевые.

```python
# --- бэкенды (переключатели среды) ---
broker_backend:  Literal["memory", "redis"]  = "memory"
db_url:          str = "sqlite+aiosqlite:///./local.db"
vector_backend:  Literal["memory", "qdrant"] = "memory"
llm_backend:     Literal["stub", "openai"]   = "stub"
db_echo:         bool = False          # SQL-лог (шумно, только для отладки)

# --- очередь и ретраи ---
task_max_retries:             int   = 3
task_retry_delay_seconds:     float = 5.0
task_retry_use_jitter:        bool  = True
task_retry_use_exponent:      bool  = True    # экспоненциальный backoff
task_retry_max_exponent_delay: float = 60.0   # потолок задержки
task_timeout_seconds:         float = 120.0

# --- LLM ---
llm_model:            str   = "gpt-4o-mini"
llm_timeout_seconds:  float = 60.0
llm_max_concurrency:  int   = 8      # семафор на одновременные вызовы
llm_max_retries:      int   = 3
llm_retry_base_delay: float = 1.0
llm_price_prompt_per_1k:     float = 0.00015   # USD, для метрик стоимости
llm_price_completion_per_1k: float = 0.0006

# --- API ---
api_host:  str = "127.0.0.1"
api_port:  int = 8000
api_rate_limit_per_minute: int = 60
cors_origins: list[str] = ["http://localhost:3000"]   # или строка через запятую

# --- секреты (только из окружения, в коде не хардкодятся) ---
openai_api_key:       str | None = None
langfuse_public_key:  str | None = None   # нет ключа -> SDK работает как no-op
langfuse_secret_key:  str | None = None
```

Остальные поля (адреса инфраструктуры и observability) — по умолчанию
работоспособны, менять нужно только при переходе в прод-режим:

| Поле | Default | Когда менять |
|---|---|---|
| `redis_url` | `redis://localhost:6379/0` | `broker_backend=redis` |
| `taskiq_queue` | `aicore:tasks` | роутинг задач по очередям |
| `qdrant_url` | `http://localhost:6333` | `vector_backend=qdrant` |
| `qdrant_collection` | `aicore_documents` | имя коллекции векторов |
| `embedding_dim` | `384` | под конкретную модель эмбеддингов |
| `openai_base_url` | `https://api.openai.com/v1` | совместимый endpoint / прокси |
| `log_level` | `INFO` | отладка (`DEBUG`) |
| `log_json` | `True` | `False` для читаемого вывода в dev |
| `langfuse_base_url` | `https://cloud.langfuse.com` | self-hosted Langfuse |
| `otel_service_name` | `aicore` | имя сервиса в трейсах |
| `otel_exporter_otlp_endpoint` | `None` | пусто = трейсы не экспортируются наружу |

### Разделение dev и prod — принципиальное

Важно понимать разницу между dev-фолбэком и production-стеком. Значения по
умолчанию выбраны так, чтобы проект запускался без инфраструктуры, но в прод
**часть из них недопустима**:

| Компонент | dev / тесты | production |
|---|---|---|
| Очередь | `InMemoryBroker` | **только** `RedisStreamBroker` |
| БД | `sqlite+aiosqlite` | **только** PostgreSQL + `asyncpg` |
| Векторный поиск | `AsyncQdrantClient(":memory:")` | Qdrant-сервер |
| LLM | `StubLLM` (детерминированный, без ключей) | OpenAI-совместимый endpoint |

**Почему `InMemoryBroker` нельзя брать в прод.** Он исполняет задачи в текущем
процессе: сообщения не ходят по сети, нет acknowledge и redelivery, часть
middleware-хуков не вызывается. То есть при падении воркера задача **теряется**.
Для AI-задач, где каждый прогон стоит денег, это недопустимо — отсюда выбор
`RedisStreamBroker` (Redis Streams) как единственного прод-брокера: он
поддерживает ack и переотдачу сообщения, если воркер умер.

---

## Production setup

```bash
cp .env.example .env          # выставить prod-значения
docker compose up -d          # postgres + redis + qdrant
uv run alembic upgrade head   # миграции
uv run taskiq worker aicore.broker:broker          # воркер
uv run uvicorn aicore.api.app:app --workers 4      # API
```

В `.env` для прода:

```bash
BROKER_BACKEND=redis
DB_URL=postgresql+asyncpg://user:pass@localhost:5432/aicore
VECTOR_BACKEND=qdrant
LLM_BACKEND=openai
```

Опциональные extras (чтобы базовый `uv sync` оставался лёгким):

```bash
uv sync --extra redis      # taskiq-redis, прод-брокер
uv sync --extra postgres   # asyncpg, прод-БД
```

`docker-compose.yml` в MVP намеренно **не запускается** — он лежит как
демонстрация понимания прод-инфраструктуры и готовый путь переключения.
Отладка полного стека (Langfuse, Prometheus, OTel Collector) отложена в фазу 2.

---

## Design decisions и trade-offs

**`RedisStreamBroker`, а не `ListQueueBroker` или `PubSubBroker`.** Из трёх
брокеров taskiq-redis только stream-вариант поддерживает acknowledge и
redelivery при падении воркера. `ListQueue` (LPUSH/BRPOP) и `PubSub` ack не
поддерживают — при убийстве воркера сообщение теряется безвозвратно.

**Retries: exponential backoff + jitter.** `SmartRetryMiddleware(default_delay=5,
use_jitter=True, use_delay_exponent=True, max_delay_exponent=60)`. Jitter
разводит по времени одновременные ретраи, экспонента снижает давление на
провайдера. Сигнатура сверена с исходником taskiq 0.13.0.

**Idempotency через `Idempotency-Key` + unique-constraint.** Повторный
`POST /v1/tasks` с тем же ключом возвращает существующую задачу. Без этого
дубль от ретрая на стороне клиента означает двойную оплату LLM-вызовов.

**Dead-letter queue.** Задача, исчерпавшая `max_retries`, не пропадает:
статус `dead_letter` + ручной `POST /v1/tasks/{id}/retry`. Ошибка при этом
логируется и попадает в трейс, а не уходит в никуда.

**Timeout и concurrency на уровне LLM-клиента.** `asyncio.timeout()` на каждый
вызов плюс `asyncio.Semaphore(llm_max_concurrency)`: один зависший провайдер не
должен утащить весь пул воркеров в ожидание.

**Учёт стоимости LLM.** Каждый вызов фиксирует `prompt_tokens`,
`completion_tokens`, `latency_ms` и `cost_usd` → Prometheus + Langfuse.
Оптимизация latency и стоимости должна быть измеримой, иначе это не оптимизация.

**Идемпотентность изолирована по скоупу.** Источник правды — таблица
`idempotency_keys` с уникальностью по паре `(key, scope)`, а не глобальная
уникальность ключа. Иначе два тенанта с одним `Idempotency-Key` схлопнулись бы
в одну задачу, и ключ одного клиента отдавал бы чужую задачу. `TaskRun.idempotency_key`
намеренно **не** уникален глобально — он для поиска и отладки.

**Таймаут задачи.** `asyncio.timeout(task_timeout_seconds)` оборачивает всю
обработку, включая задержки. Иначе одно зависшее задание утащило бы воркер в
бесконечное ожидание. По истечении задача помечается failed с `TimeoutError`.

**Langfuse 4.x на OpenTelemetry, а не 5.x.** API `get_client()` +
`start_as_current_observation(as_type="generation")` присутствует уже в 4.x и
хорошо знаком. Версия 5.0.1 вышла 14.10.2026 с breaking changes (`api.scores.create`
принимает список, требуется сервер Langfuse v4.59+). Для MVP это лишний риск без
выигрыша в функциональности.

**`fastapi==0.142.x`, а не 0.143.x.** В 0.143.0 настройка OTel-экспортёров из
переменных `OTEL_*` стала opt-in (breaking change: нужен
`FASTAPI_OTEL_AUTO_CONFIGURE=true` или `FastAPI(telemetry={"auto_configure": True})`).
Поскольку проект строит трейсинг на OTel, ветка 0.142.x стабильнее и не требует
этого флага.

**MCP 1.x (`FastMCP`), а не 2.x.** В SDK v2 модуль `mcp.server.fastmcp` удалён
целиком, wire-типы переехали в `mcp-types`, убран handshake и session. Это
требует переписывания сервера. Захват версии `mcp>=1.28,<2` оставляет рабочим
привычный API с `streamable_http_app()`. Миграция на v2 — в фазу 2.

**Тесты без инфраструктуры.** `uv run pytest` не требует Docker, Redis,
Postgres или LLM-ключа: используется `InMemoryBroker`, `sqlite+aiosqlite`,
`AsyncQdrantClient(":memory:")` и `StubLLM`. Тесты, которым нужна реальная
инфраструктура, помечены маркером `@pytest.mark.integration` и по умолчанию
пропускаются.

---

## Технологический стек

Версии заданы диапазонами по стабильным API-линиям; точные версии фиксируются
в `uv.lock` (проверено на PyPI 14.10.2026).

| Слой | Технология | Диапазон |
|---|---|---|
| Runtime | Python | 3.11 – 3.12 |
| API | FastAPI · uvicorn · pydantic · pydantic-settings | `>=0.142,<0.143` · `>=0.30,<0.55` · `>=2.9,<3` · `>=2.5,<3` |
| Очередь | taskiq · taskiq-redis *(extra)* | `>=0.13,<0.14` · `>=1.2,<2` |
| БД | SQLAlchemy · aiosqlite · asyncpg *(extra)* | `>=2.0,<2.1` · `>=0.20,<0.23` · `>=0.30,<0.33` |
| Агенты | langchain · langgraph | `>=1.0,<2` · `>=1.0,<2` |
| RAG | qdrant-client | `>=1.10,<2` |
| Наблюдаемость | langfuse · structlog · prometheus-client · opentelemetry-sdk | `>=4.17,<5` · `>=24,<27` · `>=0.20,<1` · `>=1.27,<2` |
| Инструменты | mcp | `>=1.28,<2` |
| Streaming | sse-starlette | `>=3.0,<4` |
| Тесты | pytest · httpx | `>=8,<10` · `>=0.27,<0.29` |

Верхние границы исключают линии с breaking changes: `fastapi 0.143.0` (OTel opt-in),
`langfuse 5.x` (новый API и требования к серверу), `mcp 2.x` (удалён `FastMCP`),
`sqlalchemy 2.1.x` (слишком свежая async-линия).

---

## Структура проекта

Легенда: ✅ существует · 📋 целевое дерево.

```
uop/
├── README.md                        ✅ этот файл
├── pyproject.toml                   ✅ зависимости, extras, настройки pytest/ruff
├── uv.lock                          ✅ фиксация точных версий
├── .env.example                     ✅ все переключатели с комментариями
├── .gitignore                       ✅
├── docker-compose.yml               ✅ лежит, но в MVP не запускается (см. комментарий в файле)
├── Makefile.ps1                     ✅ шпаргалка по командам
├── scripts/
│   ├── demo.py                      ✅ демо: конфиг, логи, health-check
│   ├── check_stage1.py              ✅ проверка config + logging
│   └── check_server.py              ✅ поднимает uvicorn и бьёт по API
├── src/aicore/
│   ├── __init__.py                  ✅
│   ├── config.py                    ✅ pydantic-settings, env-переключатели
│   ├── logging.py                   ✅ structlog, JSON, correlation_id, redaction
│   ├── api/
│   │   ├── app.py                   ✅ фабрика приложения + lifespan
│   │   ├── __init__.py              ✅
│   │   └── routes/
│   │       ├── __init__.py          ✅
│   │       └── health.py            ✅ /health /ready /version
│   ├── broker.py                    ✅ InMemory | RedisStream + SmartRetryMiddleware
│   ├── db/                          ✅ этап 2
│   │   ├── __init__.py              ✅
│   │   ├── base.py                  ✅ Base, naming_convention, url_is_sqlite
│   │   ├── models.py                ✅ TaskRun, Document, IdempotencyKey
│   │   └── session.py               ✅ async-движок, сессии, create_all, PRAGMA FK
│   ├── tasks/                       ✅ этап 2
│   │   ├── __init__.py              ✅
│   │   ├── service.py               ✅ статусы, идемпотентность, retries, DLQ
│   │   └── ping.py                  ✅ задача-образец с ретраями
│   ├── telemetry.py                 📋 OpenTelemetry + langfuse get_client() — этап 3
│   ├── llm/                         📋 этап 3
│   │   ├── base.py openai_compat.py stub.py  # timeout, retry, rate limit, токены/$
│   ├── rag/                         📋 этап 4
│   │   ├── chunking.py embeddings.py store.py retriever.py context.py
│   ├── agents/                      📋 этап 5
│   │   ├── state.py graph.py tools.py memory.py  # create_agent + supervisor
│   ├── mcp/                         📋 этап 6
│   │   └── server.py                     # FastMCP + streamable_http_app
│   ├── api/routes/                  📋 этапы 4/6/7
│   │   └── {documents,agents,tasks,metrics}.py
│   └── streaming/                   📋 этап 7
│       └── events.py                     # Redis Streams -> SSE
├── tests/                           ✅ pytest, без внешней инфраструктуры
│   ├── conftest.py                  ✅ фикстуры (SQLite :memory: + StaticPool)
│   ├── test_smoke_imports.py        ✅
│   ├── test_config_logging.py       ✅
│   ├── test_api_health.py           ✅
│   ├── test_db_schema.py            ✅
│   ├── test_task_service.py         ✅
│   ├── test_idempotency_scope.py    ✅ изоляция по scope (регресс)
│   └── test_broker.py               ✅
└── evals/                           📋 датасет + раннер качества — фаза 2
```

---

## Roadmap

**Фаза 2:**

- `evals/` — датасет и раннер: faithfulness / relevance / latency / стоимость прогона.
- Полноценный supervisor с двумя подагентами (в MVP — один tool-using агент).
- Alembic-миграции вместо `create_all()`.
- Kafka-брокер (`taskiq-aio-kafka`) как альтернатива Redis Streams.
- WebSocket как второй транспорт стриминга рядом с SSE.
- Langfuse Cloud / Prometheus / OTel Collector в `docker-compose.yml`.
- Миграция MCP-сервера на SDK v2 (`MCPServer`).

---

## Лицензия

MIT
