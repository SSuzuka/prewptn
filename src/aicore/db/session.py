"""Управление async-сессиями SQLAlchemy и создание схемы.

На dev/test используется `sqlite+aiosqlite`, в проде — `postgresql+asyncpg`
(extra `postgres`). На SQLite схема создаётся через `create_all()` для
удобства dev/тестов; в проде (Postgres) схему создаёт **Alembic**
(`uv run alembic upgrade head`): `create_all()` из нескольких процессов
гоняется за таблицами и enum-типами, а эволюция схемы без миграций невозможна.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from aicore.config import Settings, get_settings
from aicore.db.base import Base, url_is_sqlite

# Импорт моделей регистрирует таблицы в Base.metadata — без него create_all()
# создаст пустую схему.
from aicore.db.models import Document, IdempotencyKey, TaskRun  # noqa: F401
from aicore.logging import get_logger

logger = get_logger(__name__)

_ = (TaskRun, Document, IdempotencyKey)


class Database:
    """Обёртка над async-движком и фабрикой сессий."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Database not initialised. Call `connect()` first.")
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("Database not initialised. Call `connect()` first.")
        return self._session_factory

    @property
    def is_connected(self) -> bool:
        """True once `connect()` has built the engine."""
        return self._engine is not None

    def _build_engine(self) -> AsyncEngine:
        """Создать движок с параметрами под конкретный диалект."""
        url = self.settings.db_url
        kwargs: dict[str, Any] = {"echo": self.settings.db_echo}

        if url.startswith("sqlite"):
            # aiosqlite работает в отдельном потоке на соединение; без NullPool
            # возможны "database is locked" при конкурентной записи.
            kwargs["connect_args"] = {"check_same_thread": False}
            kwargs["poolclass"] = NullPool
        else:
            # Postgres: пул + проверка соединения перед использованием.
            kwargs["pool_pre_ping"] = True
            kwargs["pool_size"] = 5
            kwargs["max_overflow"] = 10

        return create_async_engine(url, **kwargs)

    async def connect(self) -> None:
        """Создать движок и фабрику сессий."""
        if self._engine is not None:
            return

        self._engine = self._build_engine()
        if url_is_sqlite(self.settings.db_url):
            self.enable_sqlite_foreign_keys(self._engine)

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,  # объекты остаются читаемыми после commit
            autoflush=False,
        )
        logger.info("db.connected", dialect=self._engine.dialect.name)

    async def disconnect(self) -> None:
        """Закрыть движок."""
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("db.disconnected")
        self._engine = None
        self._session_factory = None

    # --------------------------------------------------------------- sessions
    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Контекстный менеджер сессии: commit при успехе, rollback при ошибке."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialised. Call `connect()` first.")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # ------------------------------------------------------------------ schema
    async def create_all(self) -> None:
        """Создать все таблицы (MVP-замена Alembic)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("db.schema_created", tables=sorted(Base.metadata.tables))

    async def ensure_ready(self) -> None:
        """Подключиться и создать схему один раз (удобство для dev/тестов).

        Вызывается из API-зависимости и воркером, чтобы не требовать отдельного
        startup-хука. Уже подключённая БД остаётся как есть (тесты подменяют
        синглтон на готовую in-memory базу).

        Авто-создание схемы — только для SQLite (dev/test). В проде на Postgres
        это делает Alembic: `create_all()` из нескольких воркеров/процессов
        гоняется за созданием таблиц и enum-типов.
        """
        if self._engine is None:
            await self.connect()
            if url_is_sqlite(self.settings.db_url):
                await self.create_all()

    async def drop_all(self) -> None:
        """Удалить все таблицы — только для тестов."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def healthcheck(self) -> bool:
        """Проверка живости БД."""
        try:
            async with self.session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("db.healthcheck_failed", error=str(exc))
            return False

    async def schema_ready(self) -> bool:
        """Проверка, что схема накатана (есть таблица `task_runs`).

        `SELECT 1` отвечает и на пустой БД без миграций, поэтому readiness по
        одному соединению врал бы «ready» на инстансе, который на первом же
        запросе падает с `relation "task_runs" does not exist`. Читаем из
        таблицы, чтобы поймать отсутствие миграций.
        """
        try:
            async with self.session_factory() as session:
                await session.execute(text("SELECT 1 FROM task_runs LIMIT 1"))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("db.schema_not_ready", error=str(exc))
            return False

    @staticmethod
    def enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
        """Включить PRAGMA foreign_keys=ON для SQLite.

        SQLAlchemy не делает этого по умолчанию, а без него ON DELETE CASCADE
        в `idempotency_keys` молча не срабатывает.
        """

        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn: Any, _record: Any) -> None:  # noqa: ANN401
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()


# ------------------------------------------------------- синглтон приложения
_database: Database | None = None


def get_database() -> Database:
    """Получить синглтон Database."""
    global _database
    if _database is None:
        _database = Database()
    return _database


def set_database(database: Database) -> None:
    """Подменить синглтон Database.

    Нужно в тестах и чек-скриптах: задачи (см. `tasks/`) берут БД через
    `get_database()`, поэтому без инъекции их не проверить на своей БД.
    """
    global _database
    _database = database


def reset_database() -> None:
    """Сбросить синглтон (для тестов)."""
    global _database
    _database = None


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI-зависимость: отдаёт сессию из синглтона (с ленивым connect)."""
    db = get_database()
    await db.ensure_ready()
    async with db.session() as session:
        yield session
