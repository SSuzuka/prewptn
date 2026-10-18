"""Регресс-тесты на дыры из глубокого аудита (lease, backoff, метрики, edge-кейсы).

Покрывают:
  * lease: падение воркера больше не оставляет задачу в RUNNING навсегда;
  * backoff в проде: middleware получает schedule_source (иначе задержка no-op);
  * экспортёр метрик воркера (иначе счётчики задач/LLM невидимы в /metrics API);
  * воркерный startup не гонится с Alembic (create_all только для SQLite);
  * `%` в DB_URL не роняет Alembic;
  * /ready реально проверяет схему и отдаёт 503;
  * не-ASCII X-API-Key → 401, а не 500;
  * X-Forwarded-For: берётся правый адрес (левые задаёт клиент);
  * кириллица токенизируется в stub-эмбеддере;
  * текст документа не едет в payload задачи.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from starlette.requests import Request
from taskiq.abc.schedule_source import ScheduleSource

from aicore.api.app import create_app
from aicore.api.deps import _client_key, reset_rate_limiter
from aicore.broker import build_retry_middleware
from aicore.config import Settings, is_testing, reset_settings_cache
from aicore.db.dsn import escape_percent
from aicore.db.models import DocumentStatus, TaskStatus
from aicore.db.session import Database
from aicore.rag.base import EmbeddingRequest
from aicore.rag.documents import DocumentService
from aicore.rag.stub import StubEmbeddings
from aicore.tasks import sweeper
from aicore.tasks.service import TaskService, utcnow
from aicore.telemetry import reset_telemetry


# --------------------------------------------------------------- lease / sweep
async def _pending(db: Database, *, task_type: str = "answer_question", max_attempts: int = 3):  # noqa: ANN202
    async with db.session() as session:
        task, _ = await TaskService(session).create_task(
            task_type=task_type, payload={"question": "q"}, max_attempts=max_attempts
        )
    return task.id


class TestLeaseClaim:
    @pytest.mark.asyncio
    async def test_fresh_running_is_not_reclaimed(self, db: Database) -> None:
        """Пока lease свеж, повторный claim отклоняется (защита от двойной оплаты)."""
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(task_type="answer_question", payload={"q": 1})
            assert await svc.mark_running(task) is True
            assert await svc.mark_running(task) is False
            assert task.attempts == 1

    @pytest.mark.asyncio
    async def test_expired_lease_is_reclaimed(self, db: Database) -> None:
        """Просроченный RUNNING перезахватывается — это чинит зависшие задачи."""
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(task_type="answer_question", payload={"q": 1})
            assert await svc.mark_running(task) is True
            task.started_at = utcnow() - timedelta(seconds=10_000)
            await session.flush()

            assert await svc.mark_running(task, lease_seconds=1.0) is True
            assert task.attempts == 2
            assert task.status is TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_exhausted_attempts_not_reclaimed(self, db: Database) -> None:
        """Без остатка попыток перезахват запрещён — задачу уведёт sweeper."""
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(
                task_type="answer_question", payload={"q": 1}, max_attempts=1
            )
            assert await svc.mark_running(task) is True
            task.started_at = utcnow() - timedelta(seconds=10_000)
            await session.flush()
            assert await svc.mark_running(task, lease_seconds=1.0) is False


class TestSweeper:
    @pytest.mark.asyncio
    async def test_stale_running_is_reenqueued(
        self, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        enqueued: list[str] = []

        async def fake_enqueue(task_id: str) -> None:
            enqueued.append(task_id)

        monkeypatch.setattr(sweeper, "_ENQUEUERS", {"answer_question": fake_enqueue})

        task_id = await _pending(db)
        async with db.session() as session:
            svc = TaskService(session)
            task = await svc.get_task(task_id)
            assert task is not None
            await svc.mark_running(task)
            task.started_at = utcnow() - timedelta(seconds=10_000)

        count = await sweeper.reclaim_stale_tasks(db=db, lease_seconds=1.0)
        assert count == 1
        assert enqueued == [str(task_id)]

        async with db.session() as session:
            task = await TaskService(session).get_task(task_id)
        assert task is not None and task.status is TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stale_without_attempts_goes_dead_letter(
        self, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        enqueued: list[str] = []

        async def fake_enqueue(task_id: str) -> None:
            enqueued.append(task_id)

        monkeypatch.setattr(sweeper, "_ENQUEUERS", {"answer_question": fake_enqueue})

        task_id = await _pending(db, max_attempts=1)
        async with db.session() as session:
            svc = TaskService(session)
            task = await svc.get_task(task_id)
            assert task is not None
            await svc.mark_running(task)
            task.started_at = utcnow() - timedelta(seconds=10_000)

        await sweeper.reclaim_stale_tasks(db=db, lease_seconds=1.0)
        assert enqueued == []
        async with db.session() as session:
            task = await TaskService(session).get_task(task_id)
        assert task is not None and task.status is TaskStatus.DEAD_LETTER

    @pytest.mark.asyncio
    async def test_fresh_running_is_left_alone(
        self, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sweeper, "_ENQUEUERS", {"answer_question": _noop_enqueue})
        task_id = await _pending(db)
        async with db.session() as session:
            svc = TaskService(session)
            task = await svc.get_task(task_id)
            assert task is not None
            await svc.mark_running(task)

        assert await sweeper.reclaim_stale_tasks(db=db, lease_seconds=300.0) == 0


async def _noop_enqueue(task_id: str) -> None:
    return None


# ------------------------------------------------------------ retry backoff
class _FakeScheduleSource(ScheduleSource):
    async def get_schedules(self) -> list[object]:
        return []

    async def startup(self) -> None:  # pragma: no cover - not used here
        return None

    async def shutdown(self) -> None:  # pragma: no cover - not used here
        return None


class TestRetryBackoffWiring:
    def test_middleware_without_schedule_source(self) -> None:
        mw = build_retry_middleware(Settings())
        assert mw.schedule_source is None

    def test_middleware_receives_schedule_source(self) -> None:
        """Регресс: без schedule_source taskiq крутит ретрай немедленно."""
        source = _FakeScheduleSource()
        mw = build_retry_middleware(Settings(), schedule_source=source)
        assert mw.schedule_source is source

    @pytest.mark.asyncio
    async def test_ensure_broker_started_skips_in_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """В воркере broker уже поднят taskiq — sweeper не должен стартовать его снова."""
        from aicore import broker as broker_mod

        started: list[str] = []

        async def fake_startup() -> None:
            started.append("startup")

        monkeypatch.setattr(broker_mod, "_broker_started", False)
        monkeypatch.setattr(broker_mod.broker, "startup", fake_startup)
        monkeypatch.setattr(broker_mod.broker, "is_worker_process", True)

        await broker_mod.ensure_broker_started()
        assert started == []


# ------------------------------------------------------- worker metrics exporter
class TestWorkerMetricsExporter:
    def test_disabled_when_port_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aicore import worker

        calls: list[tuple] = []
        monkeypatch.setattr("prometheus_client.start_http_server", lambda *a, **k: calls.append(a))
        monkeypatch.setattr(worker, "_metrics_server", None)
        worker._start_metrics_exporter(Settings(worker_metrics_port=0))
        assert calls == []
        assert worker._metrics_server is None

    def test_enabled_binds_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aicore import worker

        calls: list[tuple] = []
        fake_server = SimpleNamespace(shutdown=lambda: None, server_close=lambda: None)
        monkeypatch.setattr(
            "prometheus_client.start_http_server",
            lambda port, addr: (calls.append((port, addr)), fake_server)[1],
        )
        monkeypatch.setattr(worker, "_metrics_server", None)
        worker._start_metrics_exporter(
            Settings(worker_metrics_port=9301, worker_metrics_host="1.2.3.4")
        )
        assert calls == [(9301, "1.2.3.4")]
        assert worker._metrics_server is fake_server
        monkeypatch.setattr(worker, "_metrics_server", None)

    def test_bind_failure_is_non_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aicore import worker

        def boom(port: int, addr: str) -> None:
            raise OSError("address in use")

        monkeypatch.setattr("prometheus_client.start_http_server", boom)
        monkeypatch.setattr(worker, "_metrics_server", None)
        worker._start_metrics_exporter(Settings(worker_metrics_port=9302))
        assert worker._metrics_server is None

    def test_tuple_return_is_unwrapped_and_stopped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """prometheus-client >= 0.23 returns (httpd, thread); both must be closed."""
        from aicore import worker

        events: list[str] = []

        class FakeServer:
            def shutdown(self) -> None:
                events.append("shutdown")

            def server_close(self) -> None:
                events.append("close")

        class FakeThread:
            def join(self, timeout: float | None = None) -> None:
                events.append("join")

        monkeypatch.setattr(
            "prometheus_client.start_http_server",
            lambda port, addr: (FakeServer(), FakeThread()),
        )
        monkeypatch.setattr(worker, "_metrics_server", None)
        monkeypatch.setattr(worker, "_metrics_thread", None)

        worker._start_metrics_exporter(Settings(worker_metrics_port=9303))
        worker._stop_metrics_exporter()
        assert events == ["shutdown", "close", "join"]
        assert worker._metrics_server is None and worker._metrics_thread is None


class TestWorkerStartupSchema:
    @pytest.mark.asyncio
    async def test_hook_is_noop_outside_worker_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API/тесты не должны поднимать экспортёр метрик и sweeper."""
        from aicore import worker as worker_mod
        from aicore.tasks import ping

        called: list[str] = []
        monkeypatch.setattr(worker_mod, "start_worker_runtime", lambda *a, **k: called.append("x"))
        await ping._on_worker_startup(None)
        assert called == []

    @pytest.mark.asyncio
    async def test_start_uses_ensure_ready_not_create_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_all только для SQLite; в проде схему накатывает Alembic."""
        from aicore import worker

        class FakeDB:
            def __init__(self) -> None:
                self.ensure_ready_called = False
                self.create_all_called = False

            async def ensure_ready(self) -> None:
                self.ensure_ready_called = True

            async def create_all(self) -> None:
                self.create_all_called = True

        fake = FakeDB()
        monkeypatch.setattr(worker, "get_database", lambda: fake)
        monkeypatch.setattr(worker, "_start_metrics_exporter", lambda cfg: None)

        async def noop() -> None:
            return None

        monkeypatch.setattr(worker, "ensure_schedule_source_started", noop)
        monkeypatch.setattr(worker, "shutdown_schedule_source", noop)

        cfg = Settings(worker_metrics_port=0, task_lease_sweep_interval_seconds=999.0)
        await worker.start_worker_runtime(cfg)
        try:
            assert fake.ensure_ready_called is True
            assert fake.create_all_called is False
        finally:
            await worker.stop_worker_runtime()


# --------------------------------------------------------------- alembic DSN
class TestAlembicDsn:
    def test_percent_is_escaped(self) -> None:
        assert (
            escape_percent("postgresql+asyncpg://u:p%40ss@h/db")
            == "postgresql+asyncpg://u:p%%40ss@h/db"
        )

    def test_plain_url_unchanged(self) -> None:
        url = "sqlite+aiosqlite:///./local.db"
        assert escape_percent(url) == url


# ------------------------------------------------------------------- /ready
class TestReadiness:
    @pytest.mark.asyncio
    async def test_503_when_schema_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aicore.api.routes import health

        class FakeDB:
            async def ensure_ready(self) -> None:
                return None

            async def healthcheck(self) -> bool:
                return True

            async def schema_ready(self) -> bool:
                return False

        monkeypatch.setattr(health, "get_database", lambda: FakeDB())
        reset_settings_cache()
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"

    @pytest.mark.asyncio
    async def test_200_when_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aicore.api.routes import health

        class FakeDB:
            async def ensure_ready(self) -> None:
                return None

            async def healthcheck(self) -> bool:
                return True

            async def schema_ready(self) -> bool:
                return True

        monkeypatch.setattr(health, "get_database", lambda: FakeDB())
        reset_settings_cache()
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


# ------------------------------------------------------------------- auth
@pytest_asyncio.fixture
async def nonascii_key_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_KEYS", "café")
    reset_settings_cache()
    reset_rate_limiter()
    reset_telemetry()
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    reset_settings_cache()
    reset_rate_limiter()
    reset_telemetry()


class TestApiKeyUnicode:
    @pytest.mark.asyncio
    async def test_non_ascii_key_is_401_not_500(self, nonascii_key_client: AsyncClient) -> None:
        # Сырые байты: httpx не отправляет не-ASCII в str-заголовке, а реальный
        # клиент/прокси вполне может прислать такие байты. Раньше
        # secrets.compare_digest(str, str) бросал TypeError -> 500 вместо 401.
        resp = await nonascii_key_client.get("/metrics", headers={"X-API-Key": b"nop\xe9"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_non_ascii_valid_key_passes(self, nonascii_key_client: AsyncClient) -> None:
        # b"caf\xe9" -> latin-1 "café" -> utf-8 совпадает с env-ключом "café".
        resp = await nonascii_key_client.get("/metrics", headers={"X-API-Key": b"caf\xe9"})
        assert resp.status_code == 200


# --------------------------------------------------------------- X-Forwarded-For
def _request_with_xff(xff: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"x-forwarded-for", xff.encode())],
        "client": ("9.9.9.9", 12345),
    }
    return Request(scope)


class TestForwardedFor:
    def test_rightmost_address_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
        reset_settings_cache()
        # Левые адреса задаёт клиент; доверяем правому (его дописал прокси).
        assert _client_key(_request_with_xff("1.1.1.1, 10.0.0.7")) == "10.0.0.7"
        reset_settings_cache()

    def test_no_header_falls_back_to_socket_peer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
        reset_settings_cache()
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "client": ("8.8.8.8", 1),
        }
        assert _client_key(Request(scope)) == "8.8.8.8"
        reset_settings_cache()

    @pytest.mark.asyncio
    async def test_forged_leftmost_does_not_bypass_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "2")
        monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
        reset_settings_cache()
        reset_rate_limiter()
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            statuses = []
            for i in range(6):
                # Разный левый адрес, один и тот же правый: обхода быть не должно.
                resp = await c.get(
                    "/v1/tasks/00000000-0000-0000-0000-000000000000",
                    headers={"X-Forwarded-For": f"1.2.3.{i}, 10.0.0.7"},
                )
                statuses.append(resp.status_code)
            assert 429 in statuses

            # А если меняется правый (реальный) адрес — лимит на каждого свой.
            statuses2 = []
            for i in range(4):
                resp = await c.get(
                    "/v1/tasks/00000000-0000-0000-0000-000000000000",
                    headers={"X-Forwarded-For": f"1.2.3.4, 10.0.0.{i}"},
                )
                statuses2.append(resp.status_code)
            assert 429 not in statuses2
        reset_settings_cache()
        reset_rate_limiter()


# ------------------------------------------------------------------- embeddings
class TestStubEmbeddingsUnicode:
    @pytest.mark.asyncio
    async def test_cyrillic_chunks_get_distinct_vectors(self) -> None:
        emb = StubEmbeddings(settings=Settings(embedding_dim=384))
        resp = await emb.embed(EmbeddingRequest(texts=["привет мир", "совсем другой текст"]))
        assert resp.vectors[0] != resp.vectors[1]

    @pytest.mark.asyncio
    async def test_same_cyrillic_text_is_deterministic(self) -> None:
        emb = StubEmbeddings(settings=Settings(embedding_dim=384))
        a = await emb.embed(EmbeddingRequest(texts=["идентичный текст"]))
        b = await emb.embed(EmbeddingRequest(texts=["идентичный текст"]))
        assert a.vectors[0] == b.vectors[0]


# ------------------------------------------------------------- document content
class TestDocumentContentStorage:
    @pytest.mark.asyncio
    async def test_content_stored_on_row(self, db: Database, use_db: Database) -> None:
        async with db.session() as session:
            document, _ = await DocumentService(session).create_document(
                title="T", content="  stored text  "
            )
            assert document.content == "stored text"

    @pytest.mark.asyncio
    async def test_index_task_payload_has_no_content(self, db: Database, use_db: Database) -> None:
        from qdrant_client import AsyncQdrantClient

        from aicore.rag import reset_retriever, set_retriever
        from aicore.rag.retriever import Retriever
        from aicore.rag.store import VectorStore

        reset_settings_cache()
        reset_rate_limiter()
        reset_telemetry()
        reset_retriever()
        cfg = Settings(vector_backend="memory", embedding_backend="stub", embedding_dim=384)
        set_retriever(
            Retriever(
                embeddings=StubEmbeddings(settings=cfg),
                store=VectorStore(
                    client=AsyncQdrantClient(":memory:"), collection="t", dimension=384
                ),
                settings=cfg,
            )
        )
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/v1/documents",
                json={"title": "T", "content": "payload-free document body"},
            )
            assert resp.status_code == 202
            assert resp.json()["status"] == DocumentStatus.INDEXED.value

        async with db.session() as session:
            tasks = await TaskService(session).list_tasks()
        index_tasks = [t for t in tasks if t.task_type == "index_document"]
        assert index_tasks, "expected an index_document task"
        payload = index_tasks[0].payload or {}
        # Текст не должен ехать в очереди/JSONB: он лежит в Document.content.
        assert "content" not in payload
        assert payload.get("document_id")
        reset_settings_cache()
        reset_rate_limiter()
        reset_retriever()
        reset_telemetry()


# --------------------------------------------------------------- config guards
class TestConfigGuards:
    def test_lease_must_exceed_timeout(self) -> None:
        with pytest.raises(ValidationError):
            Settings(task_timeout_seconds=100.0, task_lease_seconds=100.0)

    def test_defaults_are_consistent(self) -> None:
        cfg = Settings()
        assert cfg.task_lease_seconds > cfg.task_timeout_seconds


def test_is_testing_true_under_pytest() -> None:
    assert is_testing() is True
