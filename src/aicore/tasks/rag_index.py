"""Задача индексации документа в RAG.

Повторяет структуру LLM-задачи: три фазы с короткими БД-сессиями, чтобы не
держать соединение во время эмбеддингов. Документ проходит
``pending → indexing → indexed``; постоянная ошибка эмбеддера уводит и документ
(``failed``), и задачу (dead-letter) — временные ретраит брокер.

Текст документа приходит в payload задачи (у ``Document`` нет колонки content):
это входной параметр задания, а сами чанки живут в векторном хранилище.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aicore.broker import broker, ensure_broker_started
from aicore.config import get_settings
from aicore.db.session import get_database
from aicore.logging import get_logger
from aicore.rag import DocumentService, EmbeddingPermanentError, chunk_text, get_retriever
from aicore.tasks._common import load_task, mark_task_failed
from aicore.tasks.service import TaskService

logger = get_logger(__name__)


@broker.task(retry_on_error=True)
async def index_document(task_id: str) -> dict[str, Any]:
    """Проиндексировать документ RAG и обновить его статус."""
    settings = get_settings()
    try:
        async with asyncio.timeout(settings.task_timeout_seconds):
            return await _run(task_id)
    except TimeoutError:
        logger.error("document.timeout", task_id=task_id)
        await mark_task_failed(task_id, "TimeoutError: indexing exceeded deadline", retryable=True)
        raise


async def _run(task_id: str) -> dict[str, Any]:
    db = get_database()
    await db.ensure_ready()

    # Фаза 1: загрузить задачу и документ, отметить running/indexing.
    async with db.session() as session:
        svc = TaskService(session)
        task = await load_task(svc, task_id)
        if task is None:
            return {"status": "failed", "task_id": task_id, "error": "task not found"}

        await svc.mark_running(task)
        payload = dict(task.payload or {})
        document_id = str(payload.get("document_id") or "")

        docsvc = DocumentService(session)
        document = await docsvc.get_document(document_id) if document_id else None
        if document is None:
            await svc.mark_failed(task, "document not found", retryable=False)
            return {"status": "dead_letter", "task_id": task_id, "error": "document not found"}
        await docsvc.mark_indexing(document)

    # Фаза 2: чанкинг + эмбеддинги без удержания БД-соединения.
    cfg = get_settings()
    try:
        chunks = chunk_text(
            str(payload.get("content") or ""),
            chunk_size=cfg.rag_chunk_size,
            chunk_overlap=cfg.rag_chunk_overlap,
        )
        chunk_count = await get_retriever().index(
            document_id=document_id,
            chunks=chunks,
            title=payload.get("title"),
            source=payload.get("source"),
        )
    except (EmbeddingPermanentError, ValueError) as exc:
        # Постоянная ошибка (4xx от эмбеддера, битый конфиг чанкинга): ретрай
        # бессмысленен, уводим и документ (failed), и задачу (dead-letter).
        error = f"{type(exc).__name__}: {exc}"
        await _mark_document_failed(document_id, error)
        await mark_task_failed(task_id, error, retryable=False)
        return {"status": "dead_letter", "task_id": task_id, "error": error}
    except Exception as exc:  # noqa: BLE001 — transient: пусть брокер ретраит
        await mark_task_failed(task_id, f"{type(exc).__name__}: {exc}", retryable=True)
        raise

    # Фаза 3: сохранить результат в документе и задаче.
    result = {"document_id": document_id, "chunks": chunk_count}
    async with db.session() as session:
        svc = TaskService(session)
        docsvc = DocumentService(session)
        document = await docsvc.get_document(document_id)
        if document is not None:
            await docsvc.mark_indexed(document, chunk_count=chunk_count)
        task = await load_task(svc, task_id)
        if task is not None:
            await svc.mark_succeeded(task, result=result)
    return result


async def _mark_document_failed(document_id: str, error: str) -> None:
    if not document_id:
        return
    db = get_database()
    try:
        async with db.session() as session:
            document = await DocumentService(session).get_document(document_id)
            if document is not None:
                await DocumentService(session).mark_failed(document, error)
    except Exception:  # noqa: BLE001
        logger.exception("document.mark_failed_failed", document_id=document_id)


async def enqueue_index_document(task_id: str) -> None:
    """Поставить задачу индексации в брокер."""
    await ensure_broker_started()
    try:
        await index_document.kiq(task_id)
    except Exception:  # noqa: BLE001 — ошибку уже зафиксировали в БД
        logger.exception("task.enqueue_failed", task_id=task_id)
