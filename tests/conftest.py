"""Общие фикстуры для тестов слоя данных.

Всё офлайн: SQLite в памяти через StaticPool, без Docker/Redis/Postgres.

ВАЖНО про StaticPool: без него каждый новый коннект получает СВОЮ пустую
in-memory БД, и таблицы, созданные в одном соединении, не видны в другом
(ошибка "no such table"). StaticPool держит одно соединение на все сессии.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from aicore.db.base import Base
from aicore.db.session import Database

MEMORY_DSN = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def db() -> Database:
    """База с созданной схемой в памяти."""
    engine = create_async_engine(
        MEMORY_DSN,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Повторяем прод-настройку: без PRAGMA foreign_keys=ON SQLite игнорирует
    # ON DELETE CASCADE, и тест на каскадное удаление прошёл бы ложно.
    Database.enable_sqlite_foreign_keys(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    database = Database.__new__(Database)
    database._engine = engine
    database._session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    yield database

    await engine.dispose()


@pytest.fixture
def use_db(db: Database, monkeypatch: pytest.MonkeyPatch) -> Database:
    """Подменить синглтон Database, чтобы фоновые задачи видели тестовую БД."""
    import aicore.db.session as session_module

    monkeypatch.setattr(session_module, "_database", db, raising=False)
    return db


async def table_names(db: Database) -> set[str]:
    async with db.session() as session:
        rows = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        return {row[0] for row in rows}
