"""Базовый декларативный класс SQLAlchemy и общие соглашения об именовании.

Соглашения по именам ограничений фиксированы явно, иначе Alembic-автогенерация
миграций генерирует мусорные имена и падает на SQLite (фаза 2).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Соглашение об именовании constraint'ов — нужно для Alembic в проде.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def url_is_sqlite(url: str) -> bool:
    """Проверка, что DSN указывает на SQLite (async- или sync-драйвер)."""
    return url.startswith("sqlite")


class UUIDPrimaryKeyMixin:
    """UUID в качестве первичного ключа."""

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)


class TimestampMixin:
    """Метки создания и обновления."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
