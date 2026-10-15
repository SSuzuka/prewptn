"""Pydantic-схемы API задач."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from aicore.db.models import Document, TaskRun
from aicore.tasks.service import latency_ms


class TaskCreateRequest(BaseModel):
    """Тело `POST /v1/tasks`."""

    question: str = Field(min_length=1, max_length=8000, description="Вопрос к LLM")
    system: str | None = Field(default=None, max_length=4000, description="Системный промпт")
    max_tokens: int | None = Field(default=None, ge=1, le=32000)
    use_rag: bool = Field(
        default=False,
        description="Подмешать retrieved-контекст из RAG (этап 4) перед вызовом LLM",
    )


class DocumentCreateRequest(BaseModel):
    """Тело `POST /v1/documents`."""

    title: str = Field(min_length=1, max_length=512, description="Название документа")
    content: str = Field(min_length=1, max_length=1_000_000, description="Текст документа")
    source: str | None = Field(default=None, max_length=1024, description="Источник/URL")


class DocumentView(BaseModel):
    """Публичное представление документа RAG и статуса его индексации."""

    id: str
    title: str
    source: str | None = None
    status: str
    chunk_count: int
    content_length: int
    content_hash: str
    error: str | None = None
    indexed_at: datetime | None = None
    created_at: datetime

    @classmethod
    def from_document(cls, document: Document) -> DocumentView:
        return cls(
            id=str(document.id),
            title=document.title,
            source=document.source,
            status=document.status.value,
            chunk_count=document.chunk_count,
            content_length=document.content_length,
            content_hash=document.content_hash,
            error=document.error,
            indexed_at=document.indexed_at,
            created_at=document.created_at,
        )


class TaskView(BaseModel):
    """Публичное представление задачи с метриками LLM."""

    id: str
    task_type: str
    status: str
    attempts: int
    max_attempts: int
    result: dict[str, Any] | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    created_at: datetime
    finished_at: datetime | None = None

    @classmethod
    def from_task(cls, task: TaskRun) -> TaskView:
        return cls(
            id=str(task.id),
            task_type=task.task_type,
            status=task.status.value,
            attempts=task.attempts,
            max_attempts=task.max_attempts,
            result=task.result,
            error=task.error,
            prompt_tokens=task.prompt_tokens,
            completion_tokens=task.completion_tokens,
            cost_usd=task.cost_usd,
            latency_ms=latency_ms(task),
            created_at=task.created_at,
            finished_at=task.finished_at,
        )
