# AI Agent Platform — backend для AI/Agent-платформы

[![CI/CD](https://github.com/SSuzuka/prewptn/actions/workflows/ci.yml/badge.svg)](https://github.com/SSuzuka/prewptn/actions/workflows/ci.yml)

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
> | 3 | LLM-слой: timeouts, rate limit, учёт токенов и стоимости | ✅ |
> | 4 | RAG: chunking → embeddings → Qdrant → контекст | ✅ |
> | 5 | Агенты: `create_agent` + supervisor (subagents-as-tools) | ✅ |
> | 6 | MCP-сервер + монтирование в FastAPI | 📋 |
> | 7 | API + SSE-стриминг + `/metrics` | 🚧 срез |
> | 9 | `evals/`: trajectory + outcome (детерминированные грейдеры) | 🚧 срез |
>
> Тесты защищают от молчаливого ухода на breaking-линии: если uv подтянет
> `fastapi 0.143+`, `langfuse 5+`, `mcp 2+` или `sqlalchemy 2.1+`, они упадут
> (см. `tests/test_smoke_imports.py`).
>
> Реализован сквозной срез: `/v1/tasks` (создание/статус/retry), SSE-стриминг,
> `/metrics`, RAG (`/v1/documents`) и агенты (`/v1/agents`: supervisor +
> subagents, RAG как tool) — офлайн, на детерминированной stub-модели и
> stub-эмбеддере. MCP (6) отложен.

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
uv run python scripts/check_server.py    # uvicorn + сквозной путь задачи (POST/SSE/metrics)
uv run uvicorn aicore.api.app:app --reload
```

## Проверки

Всё проходит офлайн — без Docker, Redis, Postgres, Qdrant и LLM-ключа.

```bash
uv run pytest                        # 306 тестов
uv run ruff check src tests scripts migrations  # линтер
uv run ruff format --check src tests scripts migrations
uv run python scripts/demo.py            # конфиг, логи, health-check
uv run python scripts/check_stage1.py    # config + logging
uv run python scripts/check_stage2.py    # БД, брокер, retries, DLQ
uv run python scripts/check_stage3.py    # LLM: timeout, retries, токены, стоимость
uv run python scripts/check_stage4.py    # RAG: chunking, embeddings, Qdrant, retrieval
uv run python scripts/check_stage5.py    # агенты: supervisor + subagents, tools, evals
uv run python scripts/check_server.py    # uvicorn + сквозной путь задачи (POST/SSE/metrics)
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
| Python | 3.11 – 3.13 | рантайм | ✅ да |
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
        │   create_agent supervisor  (agents/)     │
        │   └─ research_knowledge_base (subagent)  │
        └───────────────┬──────────────────────────┘
                        │ tool call
        ┌───────────────▼──────────────────────────┐
        │   knowledge subagent                     │
        │   └─ search_knowledge_base (RAG tool)    │
        └───────────────┬──────────────────────────┘
                        │
        ┌───────────────▼─────────┐   ┌──────────────┐
        │   RAG  (rag/)           │   │ MCP  (mcp/)  │
        │ embeddings · Qdrant     │   │  📋 этап 6   │
        └─────────────────────────┘   └──────────────┘
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

**Поток одной задачи, пошагово** (реализованный срез):

1. `POST /v1/tasks` с заголовком `Idempotency-Key` — повторный запрос с тем же
   ключом вернёт **существующую** задачу, а не создаст дубль и не сожжёт токены.
2. Задача пишется в БД со статусом `pending` и уходит в очередь.
3. Воркер забирает её, статус → `running` (в dev это InMemoryBroker в процессе).
4. LLM-вызов идёт через обёртку: `asyncio.timeout`, семафор на concurrency,
   ретраи с backoff, фиксация `latency_ms` / токенов / `cost_usd`.
5. Клиент читает прогресс через SSE (`GET /v1/tasks/{id}/stream`).
6. Итог → `succeeded` (с метриками). Если retries исчерпаны → `dead_letter` +
   ручной `POST /v1/tasks/{id}/retry`.

RAG (этап 4) реализован как отдельная подсистема: документы индексируются через
`/v1/documents`, а ответ можно «заземлить» контекстом, поставив `use_rag=true`
(тогда перед вызовом LLM подмешиваются retrieved-чанки; retrieval-ошибка
деградирует до ответа без контекста, не роняя задачу).

Агенты (этап 5) идут отдельным маршрутом `/v1/agents`: задача типа `run_agent`
прогоняет граф `create_agent` — supervisor делегирует вопрос knowledge-подагенту
через tool `research_knowledge_base`, а тот вызывает `search_knowledge_base`
(тот же гибридный retrieval из RAG). Токены/стоимость суммируются по всем шагам
модели в `TaskRun`, поэтому `/metrics` и API показывают реальную цену прогона.
Прямой LLM-путь `/v1/tasks` (`use_rag`) остаётся без агентского графа.

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
| `POST` | `/v1/tasks` | постановка AI-задачи (идемпотентно, `Idempotency-Key`, rate limit) | ✅ |
| `GET` | `/v1/tasks/{id}` | статус задачи + метрики LLM (токены, latency, `cost_usd`) | ✅ |
| `GET` | `/v1/tasks/{id}/stream` | **SSE**-поток событий выполнения (`status`/`done`/`error`) | ✅ |
| `POST` | `/v1/tasks/{id}/retry` | ручной перезапуск упавшей задачи | ✅ |
| `GET` | `/metrics` | Prometheus-метрики (latency, счётчики, стоимость) | ✅ |
| `POST` | `/v1/documents` | загрузка документа → задача индексации в RAG (идемпотентно по содержимому) | ✅ |
| `GET` | `/v1/documents` | список документов RAG | ✅ |
| `GET` | `/v1/documents/{id}` | статус индексации документа | ✅ |
| `POST` | `/v1/agents` | запуск агента (supervisor + subagents, RAG как tool), идемпотентно | ✅ |

Все `/v1`-эндпоинты и `/metrics` под rate limit; если задан `api_keys`,
они требуют заголовок `X-API-Key` (иначе 401). `/health`, `/ready`, `/version`
остаются открытыми для проб оркестратора. Каждый ответ несёт заголовок
`X-Correlation-ID` (принятый из запроса или сгенерированный) — тот же id идёт
во все логи запроса.

**Целевой API (этап 6):**

| Метод | Путь | Назначение | Статус |
|---|---|---|---|
| `/mcp` | — | Streamable HTTP эндпоинт MCP-сервера | 📋 этап 6 |

Сквозной сценарий (работает уже сейчас, офлайн):

```bash
# Поднять сервер
uv run uvicorn aicore.api.app:app --reload

curl -sS http://127.0.0.1:8000/health

# Поставить задачу (идемпотентно), затем статус и SSE-поток
curl -sS -X POST http://127.0.0.1:8000/v1/tasks \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: demo-001' \
  -d '{"question": "Explain idempotency in one sentence"}'

curl -sS http://127.0.0.1:8000/v1/tasks/<id>
curl -N  http://127.0.0.1:8000/v1/tasks/<id>/stream

curl -sS http://127.0.0.1:8000/metrics | grep aicore_
```

Сквозной RAG-сценарий (индексация + ответ с контекстом):

```bash
# Загрузить документ — вернётся задача индексации, статус документа в фоне
curl -sS -X POST http://127.0.0.1:8000/v1/documents \
  -H 'Content-Type: application/json' \
  -d '{"title": "Intro", "content": "Idempotency keys avoid duplicate LLM spend."}'

curl -sS http://127.0.0.1:8000/v1/documents/<id>   # status: indexed, chunk_count

# Ответ, «заземлённый» retrieved-контекстом
curl -sS -X POST http://127.0.0.1:8000/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"question": "how to avoid duplicate spend?", "use_rag": true}'
```

Сквозной агентный сценарий (supervisor + subagents):

```bash
# Запустить агента; статус/SSE/метрики — через общий /v1/tasks/{id}
curl -sS -X POST http://127.0.0.1:8000/v1/agents \
  -H 'Content-Type: application/json' \
  -d '{"question": "how does the broker handle redelivery?"}'
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
embedding_backend: Literal["stub", "local", "openai"] = "stub"
agent_backend:   Literal["stub", "openai"]   = "stub"   # LangChain-модель для агентов
db_echo:         bool = False          # SQL-лог (шумно, только для отладки)

# --- очередь и ретраи ---
task_max_retries:             int   = 3
task_retry_delay_seconds:     float = 5.0
task_retry_use_jitter:        bool  = True
task_retry_use_exponent:      bool  = True    # экспоненциальный backoff
task_retry_max_exponent_delay: float = 60.0   # потолок задержки
task_retry_schedule_prefix:   str   = "aicore:retry-schedules"  # ключи Redis для отложенных ретраев
task_timeout_seconds:         float = 120.0
task_lease_seconds:           float = 300.0  # > task_timeout_seconds: lease «живой» RUNNING-задачи
task_lease_sweep_interval_seconds: float = 30.0  # период перезахвата просроченных RUNNING

# --- LLM ---
llm_model:            str   = "gpt-4o-mini"
llm_timeout_seconds:  float = 60.0
llm_max_concurrency:  int   = 8      # семафор на одновременные вызовы
llm_max_retries:      int   = 3
llm_retry_base_delay: float = 1.0
llm_price_prompt_per_1k:     float = 0.00015   # USD, для метрик стоимости
llm_price_completion_per_1k: float = 0.0006

# --- RAG ---
embedding_model:       str   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # для local (fastembed)
embedding_dim:         int   = 384      # размер вектора; должен совпасть с моделью
embedding_timeout_seconds: float = 30.0
embedding_max_retries: float = 3
rag_chunk_size:        int   = 1000     # символы (~4 символа на токен)
rag_chunk_overlap:     int   = 100
rag_retrieve_limit:    int   = 5        # топ-k чанков в контекст
rag_retrieve_fetch_k:  int   = 20       # пул кандидатов до ранжирования
rag_hybrid_enabled:    bool  = True     # плотный поиск + лексический ранк (RRF)
rag_context_max_chars: int   = 6000     # бюджет контекста в промпте

# --- агенты (create_agent: supervisor + subagents) ---
agent_model:          str   = "gpt-4o-mini"
agent_temperature:    float = 0.0
agent_max_model_calls: int  = 10        # guardrail: лимит вызовов модели за прогон
agent_max_tool_calls:  int  = 15        # guardrail: лимит вызовов инструментов
agent_recursion_limit: int  = 25        # лимит шагов графа
agent_summarize_trigger_tokens: int = 0 # >0 — сжимать историю (SummarizationMiddleware)
agent_max_threads:    int   = 256       # предел веток (thread_id) в in-memory чекпойнтере

# --- API ---
api_host:  str = "127.0.0.1"
api_port:  int = 8000
api_rate_limit_per_minute: int = 60
api_keys: list[str] = []                                # пусто = auth выключен; иначе X-API-Key
trust_proxy_headers: bool = False                       # брать IP из X-Forwarded-For (за прокси)
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
| `worker_metrics_port` | `9100` | порт Prometheus-экспортёра воркера (`0` = выключен) |
| `worker_metrics_host` | `0.0.0.0` | адрес привязки экспортёра воркера |
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
| Агенты | `StubChatModel` (детерминированный, tool calling) | `ChatOpenAI` (extra `openai`) |
| Эмбеддинги | `StubEmbeddings` (хэш, без сети и ключей) | `local` (fastembed, бесплатно) или OpenAI-совместимый `/embeddings` |

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
uv run taskiq scheduler aicore.broker:scheduler    # отложенные ретраи (backoff)
uv run uvicorn aicore.api.app:app --workers 4      # API
```

> **Scheduler обязателен в проде.** Без процесса `taskiq scheduler` ретраи
> происходят **немедленно**: middleware кладёт отложенную задачу в
> `ListRedisScheduleSource`, но исполняет её именно планировщик. Отсутствие
> scheduler'а = «3 попытки за 0,3 с» вместо backoff, то есть 429 от провайдера
> сжигает все попытки почти сразу.
>
> **Метрики воркера.** Счётчики задач/LLM (`aicore_tasks_total`,
> `aicore_llm_calls_total`, …) растут в процессе воркера, а `/metrics` отдаёт
> API. Воркер поднимает собственный Prometheus-экспортёр на
> `WORKER_METRICS_PORT` (по умолчанию `9100`) — добавьте его второй целью
> скрейпа. Альтернатива для нескольких воркеров на одном хосте —
> multiprocess-режим `prometheus_client` (фаза 2) или уникальный порт на процесс.

В `.env` для прода:

```bash
BROKER_BACKEND=redis
DB_URL=postgresql+asyncpg://user:pass@localhost:5432/aicore
VECTOR_BACKEND=qdrant
LLM_BACKEND=openai
AGENT_BACKEND=openai      # ChatOpenAI для агентов (тот же endpoint)
EMBEDDING_BACKEND=local   # бесплатные локальные эмбеддинги (fastembed), либо openai
```

`EMBEDDING_BACKEND=local` скачивает ONNX-модель один раз при первом использовании
(кэш — `FASTEMBED_CACHE_PATH`, в образе уже writable) и работает без API-ключа и
без оплаты за токены; `openai` шлёт в OpenAI-совместимый `/embeddings`.

Опциональные extras (чтобы базовый `uv sync` оставался лёгким):

```bash
uv sync --extra redis      # taskiq-redis, прод-брокер
uv sync --extra postgres   # asyncpg, прод-БД
uv sync --extra openai     # httpx + langchain-openai: LLM-транспорт и ChatOpenAI для агентов
uv sync --extra embeddings # fastembed (ONNX): бесплатные локальные эмбеддинги
```

`docker-compose.yml` в MVP намеренно **не запускается** — он лежит как
демонстрация понимания прод-инфраструктуры и готовый путь переключения.
Отладка полного стека (Langfuse, Prometheus, OTel Collector) отложена в фазу 2.

---

## CI/CD (GitHub Actions)

Один workflow `.github/workflows/ci.yml` с тремя джобами; порядок задаётся через
`needs`, поэтому публикация образа не может обогнать тесты.

| Job | Триггер | Что делает |
|---|---|---|
| `test` | push, PR | `ruff check`, `ruff format --check` (src/tests/scripts/migrations), `pytest` на Python 3.11, 3.12 и 3.13 (офлайн) |
| `docker` | push, PR | сборка `Dockerfile` без пуша — ловит ошибки образа на PR |
| `publish` | push в `main`/теги, `needs: [test, docker]` | собирает и пушит образ в GHCR |

Почему так: отдельные workflow с одним триггером GitHub запускает **параллельно** —
между файлами нет зависимостей, и CD мог опубликовать образ раньше (или вопреки)
упавшим тестам. `needs` внутри одного workflow гарантирует, что `publish` стартует
только после успешных `test` и `docker`.

Образ из `Dockerfile` (multi-stage, uv) содержит **оба** процесса — API и воркер;
прод-extras (`redis`/`postgres`/`openai`/`embeddings`) уже внутри. Теги: `main`/`latest`,
semver-тег и короткий `sha`.

```bash
# API
docker run --rm -p 8000:8000 --env-file .env ghcr.io/ssuzuka/prewptn:latest

# Воркер — тот же образ, другая команда
docker run --rm --env-file .env ghcr.io/ssuzuka/prewptn:latest \
  taskiq worker aicore.broker:broker
```

Пуш в GHCR использует встроенный `GITHUB_TOKEN` — отдельных секретов не нужно
(права `packages: write` выданы в workflow). Реальный деплой на хост — за
пределами репозитория: берёте образ из GHCR и запускаете рядом с
Postgres/Redis/Qdrant.

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

**Повтор ключа с другим телом — это 409, а не молчаливый старый ответ.**
Вместе с ключом хранится SHA-256 канонического тела запроса
(`IdempotencyKey.request_fingerprint`). Тот же ключ с другим вопросом — не
идемпотентный повтор, а конфликт: по семантике Stripe клиент получает 409,
иначе он бы получил результат не на свой вопрос.

**Ошибка постановки в очередь не оставляет задачу в `pending`.** Если
брокер/транспорт недоступен (Redis лёг), `enqueue_*` ловит исключение и
переводит задачу в `failed` условным `UPDATE ... WHERE status='pending'`
(задачу, которую уже взял воркер, не трогает). Раньше задача зависала в
`pending` навсегда: `/retry` отдавал 409, а повтор POST возвращал ту же
зависшую задачу.

**Dead-letter queue.** Задача, исчерпавшая `max_retries`, не пропадает:
статус `dead_letter` + ручной `POST /v1/tasks/{id}/retry`. Ошибка при этом
логируется и попадает в трейс, а не уходит в никуда.

**Timeout и concurrency на уровне LLM-клиента.** `asyncio.timeout()` на каждый
вызов плюс `asyncio.Semaphore(llm_max_concurrency)`: один зависший провайдер не
должен утащить весь пул воркеров в ожидание.

**Учёт стоимости LLM.** Каждый вызов фиксирует `prompt_tokens`,
`completion_tokens`, `latency_ms` и `cost_usd` → Prometheus + Langfuse.
Оптимизация latency и стоимости должна быть измеримой, иначе это не оптимизация.

**Один контракт на все провайдеры.** Весь код зависит от протокола `LLMClient`
и возвращает `LLMResponse` (текст + токены + latency + `cost_usd`), а не от
конкретного SDK. `stub` и `openai` взаимозаменяемы через `build_llm_client`,
поэтому вызовы не ветвятся по бэкенду, а dev/тесты идут на детерминированном
`StubLLM` без сети и ключей.

**Ретрай только для transient-ошибок.** Ошибки разделены на
`LLMTransientError` (timeout, connection reset, 429, 5xx) и
`LLMPermanentError` (4xx: bad request, auth, неизвестная модель). Повторяется
только первое с exponential backoff + jitter; постоянная ошибка падает сразу.
Иначе 401 от провайдера бессмысленно долбить ретраями, а 429 — наоборот, нужно.

**Timeout, rate limit и ретраи — в одном месте.** `asyncio.timeout` на попытку,
`asyncio.Semaphore(llm_max_concurrency)` на одновременные запросы и общий
retry-цикл живут в клиенте, а не в вызывающем коде: один зависший провайдер не
утащит пул воркеров, а лимит concurrency не нужно дублировать в каждом агенте.

**Стоимость считается из токенов и цены за 1k.** `compute_cost_usd(usage,
prompt_per_1k, completion_per_1k)` — единственная точка расчёта `$`, из неё
метрика уходит в `TaskRun.cost_usd`, логи и Langfuse. Цены — конфиг, а не
хардкод: смена модели/провайдера не требует правки кода.

**Telemetry no-op safe.** Langfuse-клиент создаётся только когда заданы оба
ключа (иначе SDK логирует auth-ошибку и сам себя отключает), а OTLP-экспортёр
подключается лишь при заданном endpoint. `record_generation` глотает и логирует
ошибку экспортёра — телеметрия не должна ломать сам вызов LLM.

**Идемпотентность изолирована по скоупу.** Источник правды — таблица
`idempotency_keys` с уникальностью по паре `(key, scope)`, а не глобальная
уникальность ключа. Иначе два тенанта с одним `Idempotency-Key` схлопнулись бы
в одну задачу, и ключ одного клиента отдавал бы чужую задачу. `TaskRun.idempotency_key`
намеренно **не** уникален глобально — он для поиска и отладки.

**Таймаут задачи.** `asyncio.timeout(task_timeout_seconds)` оборачивает всю
обработку, включая задержки. Иначе одно зависшее задание утащило бы воркер в
бесконечное ожидание. По истечении задача помечается failed с `TimeoutError`.

**`retryable` отдельно от «остались попытки».** `mark_failed(..., retryable=False)`
отправляет постоянную ошибку (например, 4xx от LLM) сразу в `DEAD_LETTER`, даже
если попытки остались. Без этого флага постоянная ошибка на первой попытке
вернула бы задачу в `PENDING` навсегда.

**Схема в проде создаётся Alembic, не `create_all()`.** `create_all()` из
нескольких процессов (`uvicorn --workers 4` + воркер) гоняется за созданием
таблиц и enum-типов, а миграций для эволюции схемы он не даёт. Поэтому
авто-создание оставлено только для SQLite (dev/тесты), а в проде схему
накатывает `uv run alembic upgrade head` (DSN берётся из тех же настроек).

**Все DateTime-колонки timezone-aware.** `utcnow()` возвращает aware-datetime,
поэтому каждая колонка объявлена `DateTime(timezone=True)`. На Postgres колонка
без TZ (`TIMESTAMP WITHOUT TIME ZONE`) такой объект испортит, а на SQLite
расхождение маскируется — поэтому есть регресс-тест на метаданные.

**Ограничение dev-режима ретраев.** `SmartRetryMiddleware` переставляет задачу
через `schedule_source` с меткой `delay`. В проде брокеру передаётся
`ListRedisScheduleSource`, и отложенную доставку исполняет процесс
`taskiq scheduler` — без него задержка игнорируется и ретрай идёт немедленно.
В dev `InMemoryBroker` не умеет отложенную доставку, поэтому backoff/jitter
фактически no-op. Источник правды о лимите попыток — БД (`TaskRun.max_attempts`),
middleware-счётчик служит предохранителем на стороне брокера.

**Lease против «навсегда RUNNING» после падения воркера.** Атомарный claim
(`UPDATE ... WHERE status='pending'`) защищает от двойной оплаты, но у него был
побочный эффект: если воркер убить посреди задачи, redelivered-сообщение
отклонялось (статус уже `RUNNING`), сообщение подтверждалось, и задача висела в
`RUNNING` навсегда (SSE до таймаута, `/retry` → 409). Теперь у `RUNNING`-задачи
есть lease: пока `now - started_at < task_lease_seconds`, перезахват запрещён
(защита от дублей), а просроченную задачу фоновый purge-цикл воркера
(`tasks/sweeper.py`) переставляет обратно в очередь, и живой воркер её
перезахватывает. `task_lease_seconds` обязан быть больше `task_timeout_seconds`,
иначе живая медленная задача была бы перехвачена и оплачена дважды (валидируется
на старте). Задача без остатка попыток уводится в `DEAD_LETTER`, а не крутится
вечно.

**Метрики воркера видны через отдельный экспортёр.** Счётчики задач/LLM
инкрементируются в процессе воркера, а `/metrics` отдаёт API-процесс — раньше
`aicore_tasks_total` и `aicore_llm_calls_total` не появлялись ни в одном скрейпе.
Воркер поднимает собственный Prometheus-экспортёр на `worker_metrics_port`
(по умолчанию 9100; `0` выключает). Конфликт порта при нескольких воркерах на
хосте не фатален: логируется предупреждение, процесс продолжает работу.

**`/ready` действительно проверяет БД.** Раньше эндпоинт всегда отвечал
`ready`, даже если БД недоступна или миграции не накатаны, и оркестратор слал
трафик в инстанс, падавший на первом запросе. Теперь `/ready` делает
`SELECT 1` **и** читает `task_runs`: без схемы — 503. Это ловит «подняли API,
забыли `alembic upgrade head`».

**Не-ASCII `X-API-Key` — это 401, а не 500.** `secrets.compare_digest` для
`str` требует ASCII-only и на ключе вроде `café` бросал `TypeError`. Сравниваем
`bytes` (`.encode("utf-8")`) — сравнение остаётся constant-time, а нештатный
ключ корректно отклоняется.

**`X-Forwarded-For`: доверяем правому адресу.** В цепочке `client, proxy1, ...`
каждый узел дописывает источник справа. Левые адреса задаёт клиент, поэтому
брать `split(",", 1)[0]` (как было раньше) — значит позволять обойти rate limit
подделкой заголовка. Теперь берётся последний элемент — тот, что дописал
ближайший доверенный прокси.

**Текст документа хранится на строке `Document`, а не в payload задачи.**
До 1 МБ текста ехало в сообщении очереди (Redis) и в JSONB `task_runs.payload`
на каждую индексацию. Колонка `Document.content` (nullable) — источник правды;
payload несёт только `document_id`. Индексер читает текст из строки, а старый
payload-support оставлен fallback'ом.

**Модель общих API-ключей — без изоляции тенантов.** `api_keys` — это список
разделённых ключей: все они дают доступ ко всем задачам и документам. Поэтому
`bob-key` может прочитать задачу `alice-key` (и по тому же `Idempotency-Key`
скоупа не изолированы по ключу-владельцу). Это осознанный предел MVP, а не баг;
настоящая изоляция — per-key identity/scope (фаза 2).

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

**Эмбеддинги — отдельный контракт, не часть LLM-слоя.** RAG зависит от
протокола `EmbeddingClient` с тем же устройством, что `LLMClient`: три бэкенда
(`stub` детерминированный, `local` fastembed ONNX — бесплатно и без ключа,
`openai` — любой `/embeddings`), разделение transient/permanent ошибок и общий
цикл ретраев (`run_with_retry(retry_on=...)`). Вызывающий код не ветвится по
бэкенду, а `fastembed` живёт в extra `embeddings`, чтобы базовый `uv sync`
оставался лёгким.

**Чанкинг влияет на качество не меньше модели.** Размер и границы чанка задают
потолок того, что ретривер вообще сможет найти. По умолчанию — рекурсивное
разбиение по абзацам/предложениям/словам в диапазон ~200–800 токенов
(1000/100 символов ≈ 250/25 токенов, ~4 символа на токен), с малым
перекрытием: большое перекрытие в основном дублирует индекс и жжёт эмбеддинги
без измеримого прироста recall.

**Детерминированный point id → идемпотентная реиндексация.** id чанка —
`uuid5(document_id, chunk_index)`, поэтому повторная индексация документа
**перезаписывает** его точки, а не плодит дубли (та же идея, что
`Idempotency-Key` у задач и `content_hash` у документа).

**Гибридный поиск через RRF, а не «чистая семантика».** Плотный поиск
пропускает точные токены (коды ошибок, идентификаторы, имена функций), а
лексический — общую семантику. Ранжирования объединяются reciprocal rank
fusion: не нужна калибровка скоров, работает детерминированно и офлайн, в том
числе на `:memory:`-хранилище. Полноценные sparse-векторы/BM25 внутри Qdrant —
апгрейд фазы 2.

**Контекст — это шаг с бюджетом, а не `join(top_k)`.** Наивная склейка top-k
раздувает промпт до 10k–30k токенов и роняет качество («context cliff»).
`build_context` дедуплицирует чанки, нумерует их для цитирования и не
превышает `rag_context_max_chars`; обязательное реранжирование — не догма (в
публичных бенчмарках оно местами вредит), поэтому здесь его нет, есть гибрид.

**`Document` хранит метаданные и текст на строке (не в payload).** Чанки живут
в векторном хранилище, а `Document.content` (nullable) держит исходный текст,
чтобы до 1 МБ не ехало в очереди и в `task_runs.payload`. Идемпотентность
загрузки — по `content_hash` (уникален), поэтому повторная загрузка того же
текста не жжёт эмбеддинги.

**Агенты — subagents-as-tools, а не `create_supervisor`.** С октября 2025
`langgraph-supervisor`/`create_supervisor` больше не поддерживается; официальный
паттерн в LangChain 1.x — supervisor на `create_agent`, который вызывает
подагента обычным tool'ом. Так и сделано: `research_knowledge_base` — обёртка над
отдельным `create_agent`-графом knowledge-агента, а RAG доступен ему через
`search_knowledge_base`. Лишний multi-agent там, где хватает одного агента, не
строится: supervisor делегирует только retrieval-вопросы.

**Guardrails — middleware, а не форк цикла.** `ModelCallLimitMiddleware` и
`ToolCallLimitMiddleware` (плюс `agent_recursion_limit`) ограничивают прогон.
Учёт токенов/стоимости — не middleware, а `AgentUsageCallbackHandler`, который
прокидывается в `callbacks` вызова: он видит и шаги супервизора, и вложенный
прогон подагента (тот же `usage_metadata`), и пишет те же Prometheus-метрики,
что и прямой LLM-вызов. Собственный цикл агента не нужен.

**Память агента ограничена.** Общий чекпойнтер — `BoundedInMemorySaver`
(`agent_max_threads`): старейшие ветки `thread_id` вытесняются, чтобы клиентский
id не раздувал память воркера. Подагент своего чекпойнтера не держит (он
одноразовый), поэтому не копит ветки на каждый вызов. Изоляция бесед по
вызывающему — задача аутентификации (фаза 2); в MVP `thread_id` — общий
клиентский ключ.

**Детерминированная LangChain-модель для оффлайна.** `create_agent` требует
LangChain `BaseChatModel`; чтобы граф (модель → tools → модель) гонялся в CI и
локально без сети и ключей, есть `StubChatModel` с настоящим tool calling.
Инструменты читаются из kwargs вызова (`bind(tools=...)`), поэтому один инстанс
безопасно делят supervisor и подагент.

**Асинхронные инструменты.** Граф исполняется через `ainvoke` (воркер и так
асинхронный), а sync-путь LangGraph падает на асинхронных `StructuredTool`.
Поэтому все tools объявлены `async`.

**CORS, correlation_id и rate limit — на краю API.** CORS-мидлварь включена
по `cors_origins` (иначе браузер не пройдёт preflight). `correlation_id`
ставится чистым ASGI-мидлварем (не `BaseHTTPMiddleware` — тот ломает SSE):
берётся из `X-Correlation-ID`/`X-Request-ID` или генерируется и возвращается
в заголовке. Rate limit стоит на всех `/v1`-эндпоинтах, включая чтение и SSE;
за доверенным прокси (`trust_proxy_headers=true`) клиент берётся из **правого**
адреса `X-Forwarded-For` (его дописал прокси, левые задаёт клиент).

---

## Технологический стек

Версии заданы диапазонами по стабильным API-линиям; точные версии фиксируются
в `uv.lock` (проверено на PyPI 14.10.2026).

| Слой | Технология | Диапазон |
|---|---|---|
| Runtime | Python | 3.11 – 3.13 |
| API | FastAPI · uvicorn · pydantic · pydantic-settings | `>=0.142,<0.143` · `>=0.30,<0.55` · `>=2.9,<3` · `>=2.5,<3` |
| Очередь | taskiq · taskiq-redis *(extra)* | `>=0.13,<0.14` · `>=1.2,<2` |
| БД | SQLAlchemy · aiosqlite · asyncpg *(extra)* | `>=2.0,<2.1` · `>=0.20,<0.23` · `>=0.30,<0.33` |
| Агенты | langchain · langgraph | `>=1.0,<2` · `>=1.0,<2` |
| Агенты (prod) | langchain-openai *(extra `openai`)* | `>=1.0,<2` |
| RAG | qdrant-client | `>=1.10,<2` |
| RAG эмбеддинги | fastembed *(extra)* | `>=0.4,<1` |
| LLM | httpx *(extra `openai`)* | `>=0.27,<0.29` |
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
├── alembic.ini                      ✅ конфиг миграций (DSN берётся из настроек)
├── migrations/                      ✅ Alembic: env.py (async) + versions/
├── Dockerfile                       ✅ образ API+воркер (сборка в GHCR)
├── .dockerignore                    ✅
├── .github/workflows/               ✅ ci.yml: test → docker build → publish (GHCR)
├── Makefile.ps1                     ✅ шпаргалка по командам
├── scripts/
│   ├── demo.py                      ✅ демо: конфиг, логи, health-check
│   ├── check_stage1.py              ✅ проверка config + logging
│   ├── check_stage2.py              ✅ проверка БД, брокера, retries, DLQ
│   ├── check_stage3.py              ✅ проверка LLM: timeout, retries, токены, $
│   ├── check_stage4.py              ✅ проверка RAG: chunking, embeddings, Qdrant, retrieval
│   ├── check_stage5.py              ✅ проверка агентов: supervisor + subagents, evals
│   └── check_server.py              ✅ поднимает uvicorn и бьёт по API
├── src/aicore/
│   ├── __init__.py                  ✅
│   ├── config.py                    ✅ pydantic-settings, env-переключатели
│   ├── logging.py                   ✅ structlog, JSON, correlation_id, redaction
│   ├── api/
│   │   ├── app.py                   ✅ фабрика приложения + lifespan
│   │   ├── __init__.py              ✅
│   │   ├── schemas.py               ✅ Pydantic-схемы задач
│   │   ├── deps.py                  ✅ rate limiting по клиенту
│   │   └── routes/
│   │       ├── __init__.py          ✅
│   │       ├── health.py            ✅ /health /ready /version
│   │       ├── tasks.py             ✅ /v1/tasks (+retry, SSE)
│   │       ├── documents.py         ✅ /v1/documents (загрузка/статус/список)
│   │       ├── agents.py            ✅ /v1/agents (запуск агента)
│   │       └── metrics.py           ✅ /metrics
│   ├── broker.py                    ✅ InMemory | RedisStream + SmartRetryMiddleware + schedule_source
│   ├── worker.py                    ✅ рантайм воркера: метрики, schedule source, lease-sweeper
│   ├── db/                          ✅ этап 2
│   │   ├── __init__.py              ✅
│   │   ├── base.py                  ✅ Base, naming_convention, url_is_sqlite
│   │   ├── dsn.py                   ✅ escape_percent для Alembic
│   │   ├── models.py                ✅ TaskRun, Document (+content), IdempotencyKey
│   │   └── session.py               ✅ async-движок, сессии, create_all (dev), PRAGMA FK
│   ├── tasks/                       ✅ этап 2
│   │   ├── __init__.py              ✅
│   │   ├── service.py               ✅ статусы, идемпотентность, retries, DLQ, lease-claim
│   │   ├── sweeper.py               ✅ перезахват просроченных RUNNING-задач
│   │   ├── ping.py                  ✅ задача-образец с ретраями
│   │   ├── llm.py                   ✅ answer_question: LLM + метрики в TaskRun
│   │   ├── rag_index.py             ✅ index_document: chunking → embed → Qdrant
│   │   └── agent_run.py             ✅ run_agent: граф агента + суммарные метрики
│   ├── telemetry.py                 ✅ OpenTelemetry + langfuse get_client() — этап 3
│   ├── metrics.py                   ✅ Prometheus: счётчики/гистограммы задач и LLM
│   ├── llm/                         ✅ этап 3
│   │   ├── __init__.py              ✅ фабрика build_llm_client + синглтон
│   │   ├── base.py                  ✅ LLMRequest/Response, ошибки, cost, retry-политика
│   │   ├── stub.py                  ✅ детерминированный StubLLM (dev/тесты)
│   │   └── openai_compat.py         ✅ timeout, retry, семафор, токены/$
│   ├── rag/                         ✅ этап 4
│   │   ├── __init__.py              ✅ фабрика эмбеддера + синглтоны
│   │   ├── base.py                  ✅ EmbeddingClient, ошибки, стоимость
│   │   ├── stub.py                  ✅ детерминированные хэш-эмбеддинги (dev/тесты)
│   │   ├── embeddings.py            ✅ local (fastembed) + OpenAI-совместимый
│   │   ├── chunking.py              ✅ рекурсивный сплиттер + estimate_tokens
│   │   ├── store.py                 ✅ Qdrant (memory | server), point ids, delete
│   │   ├── retriever.py             ✅ index + гибридный поиск (RRF)
│   │   ├── context.py               ✅ сборка контекста с бюджетом и цитатами
│   │   └── documents.py             ✅ DocumentService: статусы, content_hash
│   ├── agents/                      ✅ этап 5
│   │   ├── __init__.py              ✅ фабрика build_agent + синглтон
│   │   ├── chat_model.py            ✅ LangChain-модель: stub (tool calling) | ChatOpenAI
│   │   ├── tools.py                 ✅ search_knowledge_base (RAG как tool)
│   │   ├── subagents.py             ✅ knowledge-подагент + обёртка в tool
│   │   ├── usage.py                 ✅ AgentUsageCallbackHandler + итоги токенов/$ + final_answer
│   │   └── agent.py                 ✅ create_agent supervisor + guardrails + bounded checkpointer
│   ├── evals/                       ✅ этап 5 (срез)
│   │   ├── dataset.py               ✅ офлайн-датасет (вопрос → tool → grounding)
│   │   ├── graders.py               ✅ детерминированные грейдеры (trajectory/outcome)
│   │   └── runner.py                ✅ прогон + агрегат (success, routing, $)
│   ├── mcp/                         📋 этап 6
│   │   └── server.py                     # FastMCP + streamable_http_app
│   └── streaming/                   ✅ этап 7 (срез)
│       └── events.py                ✅ SSE: опрос TaskRun -> status/done/error
├── tests/                           ✅ pytest, без внешней инфраструктуры
│   ├── conftest.py                  ✅ фикстуры (SQLite :memory: + StaticPool)
│   ├── test_smoke_imports.py        ✅
│   ├── test_config_logging.py       ✅
│   ├── test_api_health.py           ✅
│   ├── test_db_schema.py            ✅
│   ├── test_task_service.py         ✅
│   ├── test_idempotency_scope.py    ✅ изоляция по scope (регресс)
│   ├── test_broker.py               ✅
│   ├── test_llm_base.py             ✅ usage/cost + retry-политика
│   ├── test_llm_stub.py             ✅ детерминированный клиент
│   ├── test_llm_openai_compat.py    ✅ timeout/retry/семафор на MockTransport
│   ├── test_llm_factory.py          ✅ выбор бэкенда + синглтон
│   ├── test_llm_metrics_e2e.py      ✅ токены/$ сохраняются в TaskRun
│   ├── test_telemetry.py            ✅ no-op-safe OTel + langfuse
│   ├── test_llm_task.py             ✅ LLM-задача: метрики, retry vs DLQ
│   ├── test_api_tasks.py            ✅ API задач (создание/идемпотентность/retry)
│   ├── test_streaming.py            ✅ SSE-поток по шагам
│   ├── test_metrics.py              ✅ /metrics + инструментирование
│   ├── test_rate_limit.py           ✅ 429 после лимита
│   ├── test_rag_chunking.py         ✅ чанкинг: границы, размер, перекрытие
│   ├── test_rag_embeddings.py       ✅ stub + OpenAI-совместимый эмбеддер
│   ├── test_rag_store.py            ✅ Qdrant :memory: (upsert/search/delete)
│   ├── test_rag_retriever.py        ✅ индексация + гибридный поиск (RRF)
│   ├── test_rag_context.py          ✅ бюджет контекста, дедуп, цитаты
│   ├── test_rag_index_task.py       ✅ index_document: статусы, retry/DLQ
│   ├── test_documents_api.py        ✅ /v1/documents API
│   ├── test_agents_chat_model.py    ✅ детерминированная LangChain-модель
│   ├── test_agents_tools.py         ✅ RAG-tool
│   ├── test_agents_agent.py         ✅ supervisor: subagent-as-tool, grounding, память
│   ├── test_agent_task.py           ✅ run_agent: метрики, failed/DLQ
│   ├── test_agents_api.py           ✅ /v1/agents (идемпотентность, retry, валидация)
│   └── test_evals.py                ✅ грейдеры + eval-раннер
```

> **Соответствие исходному скелету.** В первой версии карты требований были
> `tasks/rag_index.py`, `tasks/agent_run.py`, `tasks/dlq.py`. Сейчас:
> `agent_run.py` → `tasks/agent_run.py` (граф агента, этап 5), а LLM-исполнение
> задачи → `tasks/llm.py`; `dlq.py` → логика DLQ в `tasks/service.py`;
> `rag_index.py` → `tasks/rag_index.py` (индексация RAG, этап 4).

---

## Roadmap

**Фаза 2:**

- `evals/`: LLM-judge поверх детерминированных грейдеров (faithfulness / relevance)
  и RAG-специфичные метрики; прогон на реальной модели.
- Агенты: long-term memory через `store` (cross-thread), `HumanInTheLoopMiddleware`
  + resume, второй специализированный подагент, `SummarizationMiddleware` в проде.
- RAG: нативные sparse-векторы/BM25 внутри Qdrant (Query API + fusion) вместо
  лексического RRF; опциональный кросс-энкодер-реранкер (по замеру, не по умолчанию).
- RAG: загрузка из URL/файлов, `DELETE /v1/documents/{id}` и явный reindex.
- Postgres-checkpointer для графа агента (в MVP — `InMemorySaver`).
- Kafka-брокер (`taskiq-aio-kafka`) как альтернатива Redis Streams.
- WebSocket как второй транспорт стриминга рядом с SSE.
- Langfuse Cloud / Prometheus / OTel Collector в `docker-compose.yml`.
- Миграция MCP-сервера на SDK v2 (`MCPServer`).

---

## Лицензия

MIT — см. [LICENSE](LICENSE).
