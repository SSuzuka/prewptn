"""ORM-модели: задачи, документы, ключи идемпотентности.

Три таблицы закрывают требования по состоянию пайплайна:
  * `TaskRun`      — задача и её жизненный цикл (статусы, попытки, ошибки);
  * `Document`     — документ для RAG и состояние его индексации;
  * `IdempotencyKey` — защита от повторного выполнения одного и того же запроса.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy import (
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from aicore.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class TaskStatus(enum.StrEnum):
    """Жизненный цикл AI-задачи.

    `DEAD_LETTER` — задача исчерпала все ретраи и ждёт ручного вмешательства.
    StrEnum (Python 3.11+) вместо `(str, Enum)`: значение уже является строкой,
    поэтому `.value` избыточен, а JSON-сериализация работает без костылей.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Достигнуто ли финальное состояние."""
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.DEAD_LETTER,
            TaskStatus.CANCELLED,
        }


class DocumentStatus(enum.StrEnum):
    """Состояние индексации документа в RAG."""

    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class TaskRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Запуск AI-задачи."""

    __tablename__ = "task_runs"

    # Что делать: например "answer_question", "index_document".
    task_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    status: Mapped[TaskStatus] = mapped_column(
        SAEnum(TaskStatus, name="task_status", values_callable=lambda e: [m.value for m in e]),
        default=TaskStatus.PENDING,
        nullable=False,
        index=True,
    )

    # Входные параметры задачи. JSONB на Postgres, обычный JSON на SQLite.
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        default=dict,
        nullable=False,
    )

    # Результат успешного выполнения.
    result: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )

    # Счётчик попыток. Инкрементируется воркером перед каждой обработкой.
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)

    # Текст ошибки последней неудачной попытки.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Метрики LLM для анализа latency и стоимости (этап 3).
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(nullable=True)

    # Время постановки/старта/завершения — для метрик латентности.
    queued_at: Mapped[datetime | None] = mapped_column(nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Ключ идемпотентности для быстрого поиска/отладки. НЕ уникален глобально:
    # уникальность живёт в `idempotency_keys` по паре (key, scope), поэтому
    # разные тененты вправе использовать один и тот же ключ.
    # Источник правды для проверки идемпотентности — таблица idempotency_keys.
    idempotency_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # Идентификатор сообщения в очереди (для трассировки через брокер).
    broker_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    __table_args__ = (
        # Частой выборке "все задачи со статусом X" — композитный индекс.
        Index("ix_task_runs_status_created_at", "status", "created_at"),
    )

    @property
    def can_retry(self) -> bool:
        """Можно ли перезапустить задачу (остались попытки)."""
        return self.attempts < self.max_attempts

    def __repr__(self) -> str:
        return (
            f"<TaskRun id={self.id} type={self.task_type} "
            f"status={self.status} attempts={self.attempts}>"
        )


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Документ, проиндексированный для RAG.

    `content_hash` уникален: повторная загрузка того же документа не создаёт
    дубль в векторном хранилище и не жжёт эмбеддинги.
    """

    __tablename__ = "documents"

    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # SHA-256 от нормализованного содержимого — основа идемпотентности.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(
            DocumentStatus,
            name="document_status",
            values_callable=lambda e: [m.value for m in e],
        ),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
    )

    # Сколько чанков ушло в векторное хранилище.
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    content_length: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    indexed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    def __repr__(self) -> str:
        return f"<Document id={self.id} status={self.status} chunks={self.chunk_count}>"


class IdempotencyKey(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Ключ идемпотентности запроса.

    Хранит соответствие "ключ из заголовка -> созданная задача", поэтому
    повторный `POST /v1/tasks` с тем же ключом возвращает существующую задачу
    вместо создания дубля (дубль = двойная оплата LLM-вызовов).
    """

    __tablename__ = "idempotency_keys"

    # Уникален в разрезе скоупа, чтобы разные клиенты не конфликтовали.
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    scope: Mapped[str] = mapped_column(String(64), default="default", nullable=False)

    # Задача, созданная по этому ключу. ON DELETE CASCADE: удаление задачи
    # освобождает ключ.
    task_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("task_runs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Связь через relationship, а не через task_id вручную: на момент создания
    # ключа task.id ещё не сгенерирован (default=uuid4 срабатывает при flush),
    # и FK ушёл бы в БД как NULL — тогда CASCADE не на что вешать.
    task: Mapped[TaskRun | None] = relationship("TaskRun", foreign_keys=[task_id])

    # Снимок ответа, отданного клиенту (чтобы повтор вернул ровно тот же ответ).
    response_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )

    __table_args__ = (UniqueConstraint("key", "scope", name="uq_idempotency_keys_key_scope"),)

    def __repr__(self) -> str:
        return f"<IdempotencyKey key={self.key!r} scope={self.scope} task_id={self.task_id}>"
